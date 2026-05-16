import cv2
import time
import threading
import os
import json
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from datetime import datetime
from detector import YOLODetector, TrafficLightTracker
from video_source import VideoSource
from metrics import MetricsTracker
from violation_engine import ViolationEngine
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
app = Flask(__name__)


# 全局状态
detector = None
video_source = None
metrics = MetricsTracker()
violation_engine = ViolationEngine()
light_tracker = TrafficLightTracker()

current_frame = None
frame_lock = threading.Lock()
is_running = False
detection_thread = None
current_zones = []       # 当前配置的违规区域列表

UPLOAD_FOLDER = 'uploads'
SNAPSHOT_FOLDER = 'static/snapshots'


def _record_alert_events(events):
    for vio in events:
        metrics.add_alert({
            'id': id(vio),
            'type': vio['type'],
            'level': 'danger',
            'message': f"{vio['type']} — {vio['class_name']}",
            'time': vio['time'],
            'count': 1,
            'snapshot': None,
        })


def _resolve_light_color(frame, results):
    tl_dets = [r for r in results if r.class_name == 'traffic light']
    if tl_dets:
        tl_dets.sort(key=lambda d: d.confidence, reverse=True)
        for tl in tl_dets:
            c = tl.extra.get('light_color', 'unknown')
            if c != 'unknown':
                return c, 'detector'
        return tl_dets[0].extra.get('light_color', 'unknown'), 'detector'

    for zone in current_zones:
        if zone.get('type') == 'traffic_light_roi':
            pts = zone.get('pts', [])
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                c = detector._detect_light_color(
                    frame, [min(xs), min(ys), max(xs), max(ys)])
                if c != 'unknown':
                    return c, 'roi'
            break
    return 'unknown', 'none'


def _overlay_light_status(annotated, light_color):
    if light_color == 'unknown':
        return
    bgr = {'red': (30, 30, 220), 'green': (50, 200, 50),
           'yellow': (30, 200, 220)}.get(light_color, (180, 180, 180))
    _, w_a = annotated.shape[:2]
    label = f'LIGHT: {light_color.upper()}'
    (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    px, py = w_a - lw - 16, 36
    cv2.rectangle(annotated, (px - 6, py - lh - 6), (px + lw + 6, py + bl + 2), bgr, -1)
    cv2.putText(annotated, label, (px, py - bl),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def detection_loop():
    global current_frame, is_running, detector, video_source

    while is_running:
        if video_source is None or detector is None:
            time.sleep(0.1)
            continue

        ret, frame = video_source.read()
        if not ret:
            if video_source.source_type == 'file':
                video_source.reset()
            time.sleep(0.05)
            continue

        t0 = time.time()
        results = detector.detect(frame)
        fps = 1.0 / (time.time() - t0) if time.time() > t0 else 0
        metrics.update(fps, results)

        raw_color, source = _resolve_light_color(frame, results)
        light_color = light_tracker.update(raw_color, source=source)

        vio_events = violation_engine.check(results, light_color)
        _record_alert_events(vio_events)

        annotated = detector.draw_results(frame, results, violation_zones=current_zones)
        _overlay_light_status(annotated, light_color)

        with frame_lock:
            current_frame = annotated.copy()

        time.sleep(0.01)


def generate_frames():
    global current_frame
    while True:
        with frame_lock:
            if current_frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, 'Waiting for video source...', (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            else:
                frame = current_frame.copy()

        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    resp = Response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start', methods=['POST'])
def start_detection():
    global detector, video_source, is_running, detection_thread, current_frame

    data = request.json
    source_type = data.get('source_type', 'camera')
    source_value = data.get('source_value', '0')
    model_size = data.get('model_size', 'n')
    conf_threshold = float(data.get('conf_threshold', 0.5))
    target_classes = data.get('target_classes', [])

    try:
        # 微调模型：从 model_stats.json 读取训练好的权重路径
        if model_size == 'finetune':
            try:
                with open('model_stats.json', encoding='utf-8') as _f:
                    _ms = json.load(_f)
                model_size = _ms.get('model', 'runs/detect/traffic_finetune/weights/best.pt')
            except FileNotFoundError:
                return jsonify({'success': False, 'message': '未找到微调模型，请先运行 train.py 完成训练'})

        # 初始化检测器
        detector = YOLODetector(
            model_size=model_size,
            conf_threshold=conf_threshold,
            target_classes=target_classes if target_classes else None
        )

        # 初始化视频源
        video_source = VideoSource(source_type, source_value)
        if not video_source.open():
            return jsonify({'success': False, 'message': '无法打开视频源，请检查摄像头或文件路径'})

        current_frame = None
        metrics.reset()
        light_tracker.reset()
        violation_engine.set_zones(current_zones)
        is_running = True

        if detection_thread is None or not detection_thread.is_alive():
            detection_thread = threading.Thread(target=detection_loop, daemon=True)
            detection_thread.start()

        return jsonify({'success': True, 'message': '检测已启动'})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/stop', methods=['POST'])
def stop_detection():
    global is_running, video_source, current_frame
    is_running = False
    if video_source:
        video_source.release()
        video_source = None
    current_frame = None
    return jsonify({'success': True, 'message': '检测已停止'})


@app.route('/api/status')
def get_status():
    # 返回最近 10 条违规报警，时间倒序
    recent_alerts = list(metrics.alert_log)[-10:][::-1]
    return jsonify({
        'running': is_running,
        'metrics': metrics.get_stats(),
        'alerts': recent_alerts,
    })


@app.route('/api/snapshot', methods=['POST'])
def take_snapshot():
    global current_frame
    with frame_lock:
        if current_frame is None:
            return jsonify({'success': False, 'message': '无可用画面'})
        frame = current_frame.copy()

    filename = f'snap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
    path = os.path.join(SNAPSHOT_FOLDER, filename)
    cv2.imwrite(path, frame)
    return jsonify({'success': True, 'filename': filename, 'url': f'/static/snapshots/{filename}'})


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '未选择文件'})
    f = request.files['file']
    if f.filename == '':
        return jsonify({'success': False, 'message': '文件名为空'})
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)
    return jsonify({'success': True, 'path': path, 'filename': f.filename})


@app.route('/api/set_zones', methods=['POST'])
def set_zones():
    global current_zones
    data = request.json
    zones = data.get('zones', [])
    current_zones = zones
    violation_engine.set_zones(zones)
    return jsonify({'success': True, 'zone_count': len(zones)})


@app.route('/api/get_zones', methods=['GET'])
def get_zones():
    return jsonify({'zones': current_zones})



@app.route('/api/alerts/clear', methods=['POST'])
def clear_alerts():
    metrics.alert_log.clear()
    return jsonify({'success': True})


@app.route('/api/model_stats')
def get_model_stats():
    """返回训练后保存的 mAP 等模型指标（由 train.py 写入 model_stats.json）。"""
    try:
        with open('model_stats.json', encoding='utf-8') as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({'map50': None, 'map50_95': None})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
