import json
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from detector import TrafficLightTracker, YOLODetector
from metrics import MetricsTracker
from video_source import VideoSource
from violation_engine import ViolationEngine


class TrafficMonitorApp:
    UPLOAD_FOLDER = "uploads"
    SNAPSHOT_FOLDER = "static/snapshots"

    def __init__(self):
        self.app = Flask(__name__)

        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.SNAPSHOT_FOLDER, exist_ok=True)

        self.detector = None  # 目标检测器对象，基于YOLO模型
        self.video_source = None  # 视频源对象，支持摄像头和视频文件
        self.metrics = MetricsTracker()
        self.violation_engine = ViolationEngine()
        self.light_tracker = (
            TrafficLightTracker()
        )  # 交通灯状态跟踪器，融合检测结果和ROI分析

        self.current_frame = None
        self.frame_lock = threading.Lock()  # 用于保护current_frame的线程安全访问
        self.is_running = False
        self.detection_thread = None  # 负责视频读取、检测和违规分析的后台线程
        self.current_zones = []  # 用户自定义的违规区域，eg：斑马线、红绿灯等

        self._register_routes()

    # ── 路由注册 ─────────────────────────────────────────────────────────

    def _register_routes(self):
        rules = [
            ("/", "index", self.index, ["GET"]),
            ("/video_feed", "video_feed", self.video_feed, ["GET"]),
            ("/api/start", "start_detection", self.start_detection, ["POST"]),
            ("/api/stop", "stop_detection", self.stop_detection, ["POST"]),
            ("/api/status", "get_status", self.get_status, ["GET"]),
            ("/api/snapshot", "take_snapshot", self.take_snapshot, ["POST"]),
            ("/api/upload_video", "upload_video", self.upload_video, ["POST"]),
            ("/api/set_zones", "set_zones", self.set_zones, ["POST"]),
            ("/api/get_zones", "get_zones", self.get_zones, ["GET"]),
            ("/api/alerts/clear", "clear_alerts", self.clear_alerts, ["POST"]),
            ("/api/model_stats", "get_model_stats", self.get_model_stats, ["GET"]),
        ]
        for rule, endpoint, view_func, methods in rules:
            self.app.add_url_rule(rule, endpoint, view_func, methods=methods)

    # ── 内部方法 ─────────────────────────────────────────────────────────

    def _record_alert_events(self, events):
        """将违规事件记录到指标追踪器中，供前端展示和统计分析使用"""
        for vio in events:
            self.metrics.add_alert(
                {  # 对应alert形参格式
                    "id": id(vio),
                    "type": vio["type"],
                    "level": "danger",
                    "message": f"{vio['type']} — {vio['class_name']}",
                    "time": vio["time"],
                    "count": 1,
                    "snapshot": None,
                }
            )

    def _resolve_light_color(self, frame, results):
        """
        解析当前交通灯状态，优先使用检测结果中的交通灯类别和置信度信息，如果没有明确结果则退回到ROI分析
        """
        tl_dets = [r for r in results if r.class_name == "traffic light"]
        """
        tl_dets = []
        for r in results:
            if r.class_name == 'traffic light':
                tl_dets.append(r)
        """
        if tl_dets:
            tl_dets.sort(key=lambda d: d.confidence, reverse=True)
            """
            tl_dets[0] 是置信度最高的交通灯检测结果对象，d.confidence 是该检测结果的置信度分数，范围通常在0到1之间，表示模型对该检测结果的信心程度。通过 sort() 方法将 tl_dets 列表按照置信度从高到低排序，可以确保在后续分析中优先考虑置信度最高的交通灯检测结果，从而提高颜色识别的准确性。
            def get_confidence(d):
                return d.confidence
            tl_dets.sort(key=get_confidence, reverse=True)
            """
            for tl in tl_dets:
                c = tl.extra.get(
                    "light_color", "unknown"
                )  # extra是字典，检测到红绿灯后，优先使用extra中的light_color字段，如果没有则返回unknown
                if c != "unknown":
                    return (
                        c,
                        "detector",
                    )  # 至少有一盏灯颜色明确，数据标签为detector，表示来源于检测结果
            return (
                tl_dets[0].extra.get("light_color", "unknown"),
                "detector",
            )  # 有灯，但是颜色全为unknown，返回置信度最高的那个的颜色（可能是unknown）

        for zone in self.current_zones:
            """
                `type`:`crosswalk_roi`
                `pts`:[[x1,y1],[x2,y2],...]
                不规则图形，多顶点，需要找到最小外接矩形进行分析
            """
            if (
                zone.get("type") == "traffic_light_roi"
            ):  # 如果用户定义了交通灯ROI区域，且当前帧没有检测到交通灯，则使用该区域进行颜色分析
                pts = zone.get("pts", [])
                if len(pts) >= 2:
                    xs = [p[0] for p in pts]  # 所有的x坐标列表
                    ys = [p[1] for p in pts]  # 所有的y坐标列表
                    c = self.detector._detect_light_color(
                        frame, [min(xs), min(ys), max(xs), max(ys)]
                    )  # 传入最小外接矩形的坐标，返回颜色
                    if c != "unknown":
                        return (
                            c,
                            "roi",
                        )  # 至少有一盏灯颜色明确，数据标签为roi，表示来源于用户定义的ROI分析
                break
        return "unknown", "none"

    @staticmethod
    def _overlay_light_status(annotated, light_color):
        """
        在当前帧上叠加交通灯状态信息，显示在右上角，颜色和文本根据light_color动态调整，如果unknown则不显示
        """
        if light_color == "unknown":
            return
        bgr = {
            "red": (30, 30, 220),
            "green": (50, 200, 50),
            "yellow": (30, 200, 220),
        }.get(light_color, (180, 180, 180))
        _, w_a = annotated.shape[:2]
        label = f"LIGHT: {light_color.upper()}"
        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        px, py = w_a - lw - 16, 36
        cv2.rectangle(
            annotated, (px - 6, py - lh - 6), (px + lw + 6, py + bl + 2), bgr, -1
        )
        cv2.putText(
            annotated,
            label,
            (px, py - bl),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _detection_loop(
        self,
    ):  # 核心检测循环，持续从视频源读取帧，进行目标检测和违规分析，并更新当前帧供前端展示，同时记录性能指标和违规事件
        while self.is_running:
            if self.video_source is None or self.detector is None:
                time.sleep(0.1)
                continue

            ret, frame = (
                self.video_source.read()
            )  # ret是布尔值，表示是否成功读取到帧；frame是读取到的帧图像（如果成功的话）
            if not ret:
                if self.video_source.source_type == "file":
                    self.video_source.reset()  # 循环播放视频文件
                time.sleep(0.05)
                continue

            t0 = time.time()
            results = self.detector.detect(
                frame
            )  # 得到当前帧的检测结果列表，每个元素是一个 DetectionResult 对象，包含了检测框坐标、类别ID、类别名称、置信度、轨迹ID（如果启用Bytetrack）和额外信息（如红绿灯颜色）
            fps = 1.0 / (time.time() - t0) if time.time() > t0 else 0
            self.metrics.update(fps, results)

            raw_color, source = self._resolve_light_color(
                frame, results
            )  # 得到当前帧的交通灯颜色和数据来源（detector或roi），如果无法确定则为unknown和none
            light_color = self.light_tracker.update(raw_color, source=source)
            # vio_events 是一个列表，包含所有检测到的违规事件，每个事件是一个字典，包含类型、置信度、位置等信息
            vio_events = self.violation_engine.check(results, light_color)
            self._record_alert_events(vio_events)
            # 绘制检测结果和违规区域，包括交通灯状态和违规事件的标注
            annotated = self.detector.draw_results(
                frame, results, violation_zones=self.current_zones
            )
            self._overlay_light_status(annotated, light_color)

            with self.frame_lock:  # 将处理后的帧复制到current_frame，供前端展示
                self.current_frame = annotated.copy()

            time.sleep(0.01)

    def _generate_frames(self):
        while True:
            with self.frame_lock:
                if self.current_frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        "Waiting for video source...",
                        (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (100, 100, 100),
                        2,
                    )
                else:
                    frame = self.current_frame.copy()

            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (  # yield 生成器，每次返回一个帧的JPEG编码数据
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            time.sleep(0.033)

    # ── 路由处理 ─────────────────────────────────────────────────────────

    def index(self):
        resp = Response(render_template("index.html"))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    def video_feed(self):
        return Response(
            self._generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def start_detection(self):
        data = request.json
        source_type = data.get("source_type", "camera")
        source_value = data.get("source_value", "0")
        model_size = data.get("model_size", "n")
        conf_threshold = float(data.get("conf_threshold", 0.5))
        target_classes = data.get("target_classes", [])

        try:
            if model_size == "finetune":
                try:
                    with open("model_stats.json", encoding="utf-8") as _f:
                        _ms = json.load(_f)
                    model_size = _ms.get(
                        "model", "runs/detect/traffic_finetune/weights/best.pt"
                    )
                except FileNotFoundError:
                    return jsonify(
                        {
                            "success": False,
                            "message": "未找到微调模型，请先运行 train.py 完成训练",
                        }
                    )

            self.detector = YOLODetector(
                model_size=model_size,
                conf_threshold=conf_threshold,
                target_classes=target_classes if target_classes else None,
            )

            self.video_source = VideoSource(source_type, source_value)
            if not self.video_source.open():
                return jsonify(
                    {
                        "success": False,
                        "message": "无法打开视频源，请检查摄像头或文件路径",
                    }
                )

            self.current_frame = None
            self.metrics.reset()
            self.light_tracker.reset()
            self.violation_engine.set_zones(self.current_zones)
            self.is_running = True

            if self.detection_thread is None or not self.detection_thread.is_alive():
                self.detection_thread = threading.Thread(
                    target=self._detection_loop, daemon=True
                )
                self.detection_thread.start()

            return jsonify({"success": True, "message": "检测已启动"})

        except Exception as e:
            return jsonify({"success": False, "message": str(e)})

    def stop_detection(self):
        self.is_running = False
        if self.video_source:
            self.video_source.release()
            self.video_source = None
        self.current_frame = None
        return jsonify({"success": True, "message": "检测已停止"})

    def get_status(self):
        recent_alerts = list(self.metrics.alert_log)[-10:][::-1]
        return jsonify(
            {
                "running": self.is_running,
                "metrics": self.metrics.get_stats(),
                "alerts": recent_alerts,
            }
        )

    def take_snapshot(self):
        with self.frame_lock:  # 确保在访问current_frame时线程安全
            if self.current_frame is None:
                return jsonify({"success": False, "message": "无可用画面"})
            frame = self.current_frame.copy()

        filename = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(self.SNAPSHOT_FOLDER, filename)
        cv2.imwrite(path, frame)
        return jsonify(
            {
                "success": True,
                "filename": filename,
                "url": f"/static/snapshots/{filename}",
            }
        )

    def upload_video(self):
        if "file" not in request.files:
            return jsonify({"success": False, "message": "未选择文件"})
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"success": False, "message": "文件名为空"})
        path = os.path.join(self.UPLOAD_FOLDER, f.filename)
        f.save(path)
        return jsonify({"success": True, "path": path, "filename": f.filename})

    def set_zones(self):
        data = request.json
        zones = data.get("zones", [])
        self.current_zones = zones
        self.violation_engine.set_zones(zones)
        return jsonify({"success": True, "zone_count": len(zones)})

    def get_zones(self):
        return jsonify({"zones": self.current_zones})

    def clear_alerts(self):
        self.metrics.alert_log.clear()
        return jsonify({"success": True})

    def get_model_stats(self):
        try:
            with open("model_stats.json", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except FileNotFoundError:
            return jsonify({"map50": None, "map50_95": None})

    def run(self):
        self.app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    TrafficMonitorApp().run()
