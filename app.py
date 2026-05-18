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


class TrafficMonitorApp:

    UPLOAD_FOLDER = 'uploads'
    SNAPSHOT_FOLDER = 'static/snapshots'

    def __init__(self):
        self.app = Flask(__name__)

        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.SNAPSHOT_FOLDER, exist_ok=True)

        self.detector = None
        self.video_source = None
        self.metrics = MetricsTracker()
        self.violation_engine = ViolationEngine()
        self.light_tracker = TrafficLightTracker()

        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = False
        self.detection_thread = None
        self.current_zones = []

        self._register_routes()

    # ── 路由注册 ─────────────────────────────────────────────────────────

    def _register_routes(self):
        rules = [
            ('/', 'index', self.index, ['GET']),
            ('/video_feed', 'video_feed', self.video_feed, ['GET']),
            ('/api/start', 'start_detection', self.start_detection, ['POST']),
            ('/api/stop', 'stop_detection', self.stop_detection, ['POST']),
            ('/api/status', 'get_status', self.get_status, ['GET']),
            ('/api/snapshot', 'take_snapshot', self.take_snapshot, ['POST']),
            ('/api/upload_video', 'upload_video', self.upload_video, ['POST']),
            ('/api/set_zones', 'set_zones', self.set_zones, ['POST']),
            ('/api/get_zones', 'get_zones', self.get_zones, ['GET']),
            ('/api/alerts/clear', 'clear_alerts', self.clear_alerts, ['POST']),
            ('/api/model_stats', 'get_model_stats', self.get_model_stats, ['GET']),
        ]
        for rule, endpoint, view_func, methods in rules:
            self.app.add_url_rule(rule, endpoint, view_func, methods=methods)

    # ── 内部方法 ─────────────────────────────────────────────────────────

    def _record_alert_events(self, events):
        for vio in events:
            self.metrics.add_alert({
                'id': id(vio),
                'type': vio['type'],
                'level': 'danger',
                'message': f"{vio['type']} — {vio['class_name']}",
                'time': vio['time'],
                'count': 1,
                'snapshot': None,
            })

    def _resolve_light_color(self, frame, results):
        tl_dets = [r for r in results if r.class_name == 'traffic light']
        if tl_dets:
            tl_dets.sort(key=lambda d: d.confidence, reverse=True)
            for tl in tl_dets:
                c = tl.extra.get('light_color', 'unknown')
                if c != 'unknown':
                    return c, 'detector'
            return tl_dets[0].extra.get('light_color', 'unknown'), 'detector'

        for zone in self.current_zones:
            if zone.get('type') == 'traffic_light_roi':
                pts = zone.get('pts', [])
                if len(pts) >= 2:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    c = self.detector._detect_light_color(
                        frame, [min(xs), min(ys), max(xs), max(ys)])
                    if c != 'unknown':
                        return c, 'roi'
                break
        return 'unknown', 'none'

    @staticmethod
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

    def _detection_loop(self):
        while self.is_running:
            if self.video_source is None or self.detector is None:
                time.sleep(0.1)
                continue

            ret, frame = self.video_source.read()
            if not ret:
                if self.video_source.source_type == 'file':
                    self.video_source.reset()
                time.sleep(0.05)
                continue

            t0 = time.time()
            results = self.detector.detect(frame)
            fps = 1.0 / (time.time() - t0) if time.time() > t0 else 0
            self.metrics.update(fps, results)

            raw_color, source = self._resolve_light_color(frame, results)
            light_color = self.light_tracker.update(raw_color, source=source)

            vio_events = self.violation_engine.check(results, light_color)
            self._record_alert_events(vio_events)

            annotated = self.detector.draw_results(frame, results, violation_zones=self.current_zones)
            self._overlay_light_status(annotated, light_color)

            with self.frame_lock:
                self.current_frame = annotated.copy()

            time.sleep(0.01)

    def _generate_frames(self):
        while True:
            with self.frame_lock:
                if self.current_frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, 'Waiting for video source...', (80, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                else:
                    frame = self.current_frame.copy()

            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.033)

    # ── 路由处理 ─────────────────────────────────────────────────────────

    def index(self):
        resp = Response(render_template('index.html'))
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp

    def video_feed(self):
        return Response(self._generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def start_detection(self):
        data = request.json
        source_type = data.get('source_type', 'camera')
        source_value = data.get('source_value', '0')
        model_size = data.get('model_size', 'n')
        conf_threshold = float(data.get('conf_threshold', 0.5))
        target_classes = data.get('target_classes', [])

        try:
            if model_size == 'finetune':
                try:
                    with open('model_stats.json', encoding='utf-8') as _f:
                        _ms = json.load(_f)
                    model_size = _ms.get('model', 'runs/detect/traffic_finetune/weights/best.pt')
                except FileNotFoundError:
                    return jsonify({'success': False, 'message': '未找到微调模型，请先运行 train.py 完成训练'})

            self.detector = YOLODetector(
                model_size=model_size,
                conf_threshold=conf_threshold,
                target_classes=target_classes if target_classes else None
            )

            self.video_source = VideoSource(source_type, source_value)
            if not self.video_source.open():
                return jsonify({'success': False, 'message': '无法打开视频源，请检查摄像头或文件路径'})

            self.current_frame = None
            self.metrics.reset()
            self.light_tracker.reset()
            self.violation_engine.set_zones(self.current_zones)
            self.is_running = True

            if self.detection_thread is None or not self.detection_thread.is_alive():
                self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
                self.detection_thread.start()

            return jsonify({'success': True, 'message': '检测已启动'})

        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

    def stop_detection(self):
        self.is_running = False
        if self.video_source:
            self.video_source.release()
            self.video_source = None
        self.current_frame = None
        return jsonify({'success': True, 'message': '检测已停止'})

    def get_status(self):
        recent_alerts = list(self.metrics.alert_log)[-10:][::-1]
        return jsonify({
            'running': self.is_running,
            'metrics': self.metrics.get_stats(),
            'alerts': recent_alerts,
        })

    def take_snapshot(self):
        with self.frame_lock:
            if self.current_frame is None:
                return jsonify({'success': False, 'message': '无可用画面'})
            frame = self.current_frame.copy()

        filename = f'snap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        path = os.path.join(self.SNAPSHOT_FOLDER, filename)
        cv2.imwrite(path, frame)
        return jsonify({'success': True, 'filename': filename, 'url': f'/static/snapshots/{filename}'})

    def upload_video(self):
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '未选择文件'})
        f = request.files['file']
        if f.filename == '':
            return jsonify({'success': False, 'message': '文件名为空'})
        path = os.path.join(self.UPLOAD_FOLDER, f.filename)
        f.save(path)
        return jsonify({'success': True, 'path': path, 'filename': f.filename})

    def set_zones(self):
        data = request.json
        zones = data.get('zones', [])
        self.current_zones = zones
        self.violation_engine.set_zones(zones)
        return jsonify({'success': True, 'zone_count': len(zones)})

    def get_zones(self):
        return jsonify({'zones': self.current_zones})

    def clear_alerts(self):
        self.metrics.alert_log.clear()
        return jsonify({'success': True})

    def get_model_stats(self):
        try:
            with open('model_stats.json', encoding='utf-8') as f:
                return jsonify(json.load(f))
        except FileNotFoundError:
            return jsonify({'map50': None, 'map50_95': None})

    def run(self):
        self.app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    TrafficMonitorApp().run()
