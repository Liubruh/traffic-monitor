import cv2
import time
import threading
import os
import json
import numpy as np
from collections import Counter
from flask import Flask, render_template, Response, jsonify, request, make_response
from datetime import datetime
from detector import YOLODetector
from video_source import VideoSource
from metrics import MetricsTracker
from violation_engine import ViolationEngine
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
app = Flask(__name__)


class TrafficLightTracker:
    """
    时序平滑红绿灯状态。

    单帧颜色检测在目标小、光照变化时容易出错或漏检，本类通过：
      1. 保留近 history_len 帧的检测颜色
      2. 当某颜色在历史记录中出现次数 >= min_votes 时才更新稳定状态
      3. 若超过 persist_secs 秒未检测到红绿灯，稳定状态才重置为 unknown

    这样即使 YOLO 连续几帧漏检，violation_engine 仍能拿到上一个有效颜色。
    """
    def __init__(self, history_len=20, min_votes=5, persist_secs=6.0):
        self._history     = []
        self._history_len = history_len
        self._min_votes   = min_votes
        self._persist_s   = persist_secs
        self._stable      = 'unknown'
        self._last_seen   = 0.0

    def update(self, raw_color: str, source: str = 'detector') -> str:
        now = time.time()
        if raw_color != 'unknown':
            self._last_seen = now
            self._history.append(raw_color)
            if len(self._history) > self._history_len:
                self._history.pop(0)

            # 首次锁定或来自检测框的颜色，直接同步，避免开头几帧 unknown
            if self._stable == 'unknown' or source == 'detector':
                self._stable = raw_color
                return self._stable

        # 超时未看到红绿灯 → 状态失效
        if now - self._last_seen > self._persist_s:
            self._stable  = 'unknown'
            self._history = []
            return self._stable

        # ROI 等弱来源颜色使用多数投票，降低抖动
        if self._history:
            top, cnt = Counter(self._history).most_common(1)[0]
            if cnt >= self._min_votes:
                self._stable = top

        return self._stable

    def reset(self):
        self._history   = []
        self._stable    = 'unknown'
        self._last_seen = 0.0


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

        start_time = time.time()
        results = detector.detect(frame)
        elapsed = time.time() - start_time

        fps = 1.0 / elapsed if elapsed > 0 else 0
        metrics.update(fps, results)

        # 违规检测（需先通过 /api/set_zones 配置区域）
        # 1. 从 YOLO 检测结果取红绿灯颜色
        raw_light_color = 'unknown'
        light_source = 'none'
        tl_dets = [r for r in results if r.class_name == 'traffic light']

        # 红绿灯检测框优先：全局状态与检测框颜色保持同步
        if tl_dets:
            tl_dets.sort(key=lambda d: d.confidence, reverse=True)
            for tl in tl_dets:
                c = tl.extra.get('light_color', 'unknown')
                if c != 'unknown':
                    raw_light_color = c
                    break
            if raw_light_color == 'unknown':
                raw_light_color = tl_dets[0].extra.get('light_color', 'unknown')
            light_source = 'detector'

        # 2. 仅在未检测到红绿灯框时，才使用 traffic_light_roi 兜底
        if not tl_dets and raw_light_color == 'unknown' and detector is not None:
            for zone in current_zones:
                if zone.get('type') == 'traffic_light_roi':
                    pts = zone.get('pts', [])
                    if len(pts) >= 2:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        roi_color = detector._detect_light_color(
                            frame, [min(xs), min(ys), max(xs), max(ys)])
                        if roi_color != 'unknown':
                            raw_light_color = roi_color
                            light_source = 'roi'
                    break

        # 3. 时序平滑 → 稳定颜色
        light_color = light_tracker.update(raw_light_color, source=light_source)

        vio_events = violation_engine.check(results, light_color)
        _record_alert_events(vio_events)

        annotated = detector.draw_results(frame, results, violation_zones=current_zones)

        # 在画面右上角叠加当前稳定红绿灯状态（即使本帧未检测到红绿灯也会显示）
        if light_color != 'unknown':
            color_bgr = {'red': (30, 30, 220), 'green': (50, 200, 50),
                         'yellow': (30, 200, 220)}.get(light_color, (180, 180, 180))
            _, w_a = annotated.shape[:2]
            label = f'LIGHT: {light_color.upper()}'
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            px, py = w_a - lw - 16, 36
            cv2.rectangle(annotated, (px - 6, py - lh - 6), (px + lw + 6, py + bl + 2),
                          color_bgr, -1)
            cv2.putText(annotated, label, (px, py - bl),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

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

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    resp = make_response(render_template('index.html'))
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

    if not os.path.isdir(SNAPSHOT_FOLDER):
        return jsonify({'success': False, 'message': f'截图目录不存在: {SNAPSHOT_FOLDER}'})

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
    if not os.path.isdir(UPLOAD_FOLDER):
        return jsonify({'success': False, 'message': f'上传目录不存在: {UPLOAD_FOLDER}'})
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
