import cv2
import numpy as np
import datetime
import os

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# BGR 颜色
CLASS_COLORS = {
    'person':        (200, 130, 60),
    'car':           (80,  180, 80),
    'truck':         (60,  140, 200),
    'bus':           (200, 120, 60),
    'motorcycle':    (160, 80,  200),
    'bicycle':       (60,  200, 200),
    'traffic light': (60,  60,  220),
    'stop sign':     (40,  80,  220),
}
DEFAULT_COLOR   = (130, 180, 130)
VIOLATION_COLOR = (30,  30,  220)   # 红色 BGR

_TL_CLS_ID      = 9                 # COCO traffic light class id
_TL_CONF_MIN    = 0.25              # 红绿灯专用低置信度阈值（远距离小目标）


class DetectionResult:
    def __init__(self, bbox, class_id, class_name, confidence, extra=None, track_id=None):
        self.bbox       = bbox
        self.class_id   = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.extra      = extra or {}
        self.track_id   = track_id
        self.violation  = None  # 违法类型字符串，None 表示正常


class YOLODetector:
    def __init__(self, model_size='n', conf_threshold=0.5,
                 target_classes=None, use_gpu=True,
                 enable_bytetrack=True, tracker_cfg='bytetrack.yaml'):
        self.model_size      = model_size
        self.conf_threshold  = conf_threshold
        self.target_classes  = target_classes
        self.use_gpu         = use_gpu
        self.enable_bytetrack = enable_bytetrack
        self.tracker_cfg      = tracker_cfg
        self.backend         = None
        self.model_loaded    = False
        self.device_info     = 'CPU'
        self._infer_device   = 'cpu'   # ultralytics 路径使用
        self._load_model()

    def _resolve_model_path(self):
        """如果 model_size 是完整路径（含 / 或 .pt）则直接返回，否则拼接标准名称。"""
        ms = self.model_size
        if os.sep in ms or '/' in ms or ms.endswith('.pt'):
            return ms
        return f'yolov8{ms}.pt'

    # ── 模型加载 ──────────────────────────────────────────────────────
    def _load_model(self):
        try:
            from ultralytics import YOLO
            import torch

            if self.use_gpu:
                # 1. NVIDIA CUDA
                if torch.cuda.is_available():
                    self.yolo = YOLO(self._resolve_model_path())
                    self.yolo.to('cuda')
                    self._infer_device = 'cuda'
                    self.device_info   = f'CUDA · {torch.cuda.get_device_name(0)}'
                    self.backend       = 'ultralytics'
                    self.model_loaded  = True
                    print(f'[Detector] 使用 CUDA GPU: {torch.cuda.get_device_name(0)}')
                    print(f'[Detector] YOLOv8{self.model_size} 加载完成  [{self.device_info}]')
                    return

            # 2. CPU 兜底
            self.yolo = YOLO(self._resolve_model_path())
            self._infer_device = 'cpu'
            self.device_info   = 'CPU'
            self.backend       = 'ultralytics'
            self.model_loaded  = True
            reason = '已禁用 GPU' if not self.use_gpu else 'GPU 不可用'
            print(f'[Detector] {reason}，使用 CPU')
            print(f'[Detector] YOLOv8{self.model_size} 加载完成  [{self.device_info}]')

        except Exception as e:
            raise RuntimeError(f'ultralytics 加载失败: {e}')

    # ── 推理路由 ──────────────────────────────────────────────────────
    def detect(self, frame):
        if not self.model_loaded:
            return []
        return self._detect_ultralytics(frame)

    def _detect_ultralytics(self, frame):
        results = []
        try:
            # 以较低阈值推理，确保远距离小红绿灯不被漏掉
            run_conf = min(_TL_CONF_MIN, self.conf_threshold)
            infer_kwargs = {
                'conf': run_conf,
                'device': self._infer_device,
                'verbose': False,
            }
            infer_fn = self.yolo
            if self.enable_bytetrack:
                infer_fn = self.yolo.track
                infer_kwargs['persist'] = True
                infer_kwargs['tracker'] = self.tracker_cfg

            preds = infer_fn(frame, **infer_kwargs)[0]
            for box in preds.boxes:
                cls_id   = int(box.cls[0])
                cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else 'unknown'
                if self.target_classes and cls_name not in self.target_classes:
                    continue
                conf = float(box.conf[0])
                # 非红绿灯目标仍用原始阈值过滤
                if cls_name != 'traffic light' and conf < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = None
                if self.enable_bytetrack and hasattr(box, 'id') and box.id is not None:
                    try:
                        track_id = int(box.id[0])
                    except Exception:
                        track_id = None

                det = DetectionResult([x1, y1, x2, y2], cls_id, cls_name, conf,
                                      track_id=track_id)
                if cls_name == 'traffic light':
                    det.extra['light_color'] = self._detect_light_color(
                        frame, [x1, y1, x2, y2])
                results.append(det)
        except Exception as e:
            print(f'[Detector] 推理错误: {e}')
        return results

    # ── 红绿灯颜色识别 ────────────────────────────────────────────────
    def _detect_light_color(self, frame, bbox):
        """
        综合两种策略识别当前亮起的灯色：
          1. 竖向分区：上 1/3 = 红灯区，中 1/3 = 黄灯区，下 1/3 = 绿灯区
             ——针对竖向标准红绿灯，位置信息大幅降低背景干扰
          2. 全局颜色比率：整个 bounding box 内各颜色像素占比
        最终得分 = 分区比率 × 0.6 + 全局比率 × 0.4
        """
        x1, y1, x2, y2 = [max(0, v) for v in bbox]
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        bh = y2 - y1
        bw = x2 - x1
        if bw < 4 or bh < 4:
            return 'unknown'

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total = max(bh * bw, 1)

        # 全局各色像素数
        r_all = (cv2.countNonZero(cv2.inRange(hsv, (0,   120, 100), (10,  255, 255))) +
                 cv2.countNonZero(cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))))
        g_all = cv2.countNonZero(cv2.inRange(hsv, (40,  80,  80),  (90,  255, 255)))
        y_all = cv2.countNonZero(cv2.inRange(hsv, (15,  100, 100), (35,  255, 255)))

        # 竖向分区（上/中/下各 1/3）
        th = max(bh // 3, 1)
        top_hsv = hsv[:th,      :]
        mid_hsv = hsv[th:2*th,  :]
        bot_hsv = hsv[2*th:,    :]
        tp = max(top_hsv.shape[0] * top_hsv.shape[1], 1)
        mp = max(mid_hsv.shape[0] * mid_hsv.shape[1], 1)
        bp = max(bot_hsv.shape[0] * bot_hsv.shape[1], 1)

        r_top = (cv2.countNonZero(cv2.inRange(top_hsv, (0,   120, 100), (10,  255, 255))) +
                 cv2.countNonZero(cv2.inRange(top_hsv, (160, 100, 100), (180, 255, 255)))) / tp
        y_mid = cv2.countNonZero(cv2.inRange(mid_hsv, (15,  100, 100), (35,  255, 255))) / mp
        g_bot = cv2.countNonZero(cv2.inRange(bot_hsv, (40,   80,  80), (90,  255, 255))) / bp

        # 综合得分
        scores = {
            'red':    r_top * 0.6 + (r_all / total) * 0.4,
            'yellow': y_mid * 0.6 + (y_all / total) * 0.4,
            'green':  g_bot * 0.6 + (g_all / total) * 0.4,
        }
        best  = max(scores, key=scores.get)
        # 远距离小灯体像素少，降低阈值可减少首轮 unknown
        min_side = min(bw, bh)
        if min_side < 12:
            threshold = 0.025
        elif min_side < 20:
            threshold = 0.04
        else:
            threshold = 0.06
        if scores[best] < threshold:
            return 'unknown'
        return best

    # ── 绘制检测结果 ──────────────────────────────────────────────────
    def draw_results(self, frame, results, violation_zones=None):
        output = frame.copy()
        h, w   = output.shape[:2]

        if violation_zones:
            self._draw_zones(output, violation_zones)

        for det in results:
            x1, y1, x2, y2 = det.bbox
            is_violation    = det.violation is not None
            color           = VIOLATION_COLOR if is_violation else \
                              CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)

            # 红绿灯颜色覆盖
            if det.class_name == 'traffic light':
                lc    = det.extra.get('light_color', 'unknown')
                color = {'red':(30,30,220),'green':(50,200,50),
                         'yellow':(30,200,220)}.get(lc, CLASS_COLORS['traffic light'])

            # 违法半透明填充
            if is_violation:
                ov_buf = output.copy()
                cv2.rectangle(ov_buf, (x1,y1), (x2,y2), VIOLATION_COLOR, -1)
                cv2.addWeighted(ov_buf, 0.18, output, 0.82, 0, output)

            thick = 3 if is_violation else 2
            cv2.rectangle(output, (x1,y1), (x2,y2), color, thick)

            # 标签文字
            parts = [det.class_name, f'{det.confidence:.0%}']
            if det.track_id is not None:
                parts.append(f'ID:{det.track_id}')
            if det.class_name == 'traffic light':
                lc = det.extra.get('light_color','')
                if lc and lc != 'unknown':
                    parts.append(lc.upper())
            if is_violation:
                parts.append(f'! {det.violation}')
            label = '  '.join(parts)

            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            ly = max(y1 - 4, lh + 6)
            cv2.rectangle(output, (x1, ly-lh-bl-3), (x1+lw+8, ly+bl-2), color, -1)
            cv2.putText(output, label, (x1+4, ly-bl),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255,255,255),
                        1, cv2.LINE_AA)

        # 状态栏
        ts   = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        info = f'{ts}    objects: {len(results)}    {self.device_info}'
        cv2.rectangle(output, (0, h-24), (w, h), (245,245,245), -1)
        cv2.putText(output, info, (8, h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1, cv2.LINE_AA)
        return output

    def _draw_zones(self, frame, zones):
        overlay = frame.copy()
        h, w    = frame.shape[:2]
        color_map = {
            'stop_line':        (30,  30,  220),
            'crosswalk':        (50,  160, 50),

            'traffic_light_roi':(30,  180, 220),  # 琥珀黄
        }
        for zone in zones:
            ztype = zone.get('type', 'stop_line')
            color = color_map.get(ztype, (150, 150, 150))

            if ztype == 'stop_line':
                pts_sl = zone.get('pts')
                if pts_sl and len(pts_sl) >= 2:
                    p1 = (int(pts_sl[0][0]), int(pts_sl[0][1]))
                    p2 = (int(pts_sl[1][0]), int(pts_sl[1][1]))
                    cv2.line(frame, p1, p2, color, 3)
                    cv2.putText(frame, 'STOP LINE', (p1[0]+6, p1[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                else:
                    # 兼容旧格式
                    y  = int(zone.get('y', h//2))
                    x1 = int(zone.get('x1', 0))
                    x2 = int(zone.get('x2', w))
                    cv2.line(frame, (x1, y), (x2, y), color, 3)
                    cv2.putText(frame, 'STOP LINE', (x1+6, y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            elif ztype == 'crosswalk':
                pts = zone.get('pts')
                if pts:
                    arr = np.array(pts, np.int32)
                    cv2.fillPoly(overlay, [arr], color)
                    cv2.polylines(frame, [arr], True, color, 2)
                    cv2.putText(frame, 'CROSSWALK',
                                (pts[0][0]+4, pts[0][1]+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


            elif ztype == 'traffic_light_roi':
                pts = zone.get('pts')
                if pts:
                    arr = np.array(pts, np.int32)
                    cv2.polylines(frame, [arr], True, color, 2)
                    cv2.putText(frame, 'LIGHT ROI',
                                (pts[0][0]+4, pts[0][1]+16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)
