import datetime
import os
import time
from collections import Counter

import cv2
import numpy as np

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# BGR 颜色e
CLASS_COLORS = {
    "person": (200, 130, 60),
    "car": (80, 180, 80),
    "truck": (60, 140, 200),
    "bus": (200, 120, 60),
    "motorcycle": (160, 80, 200),
    "bicycle": (60, 200, 200),
    "traffic light": (60, 60, 220),
    "stop sign": (40, 80, 220),
}
DEFAULT_COLOR = (130, 180, 130)
VIOLATION_COLOR = (30, 30, 220)  # 红色 BGR

_TL_CLS_ID = 9  # COCO traffic light class id
_TL_CONF_MIN = 0.25  # 红绿灯专用低置信度阈值（远距离小目标）


class DetectionResult:
    """
    功能：封装检测结果的类，包含检测框坐标、类别ID、类别名称、置信度、额外信息（如红绿灯颜色）和轨迹ID（启用Bytetrack时）。同时包含一个violation字段用于标记是否违法，默认为None表示正常。
    """

    def __init__(
        self, bbox, class_id, class_name, confidence, extra=None, track_id=None
    ):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id  # 类型ID，整数
        self.class_name = class_name  # 类型名称，字符串
        self.confidence = confidence  # 置信度，0~1浮点数
        self.extra = extra or {}  # 额外信息字典，存储红绿灯颜色等非标准字段
        self.track_id = track_id  # 轨迹ID，整数或None，启用Bytetrack时有效
        self.violation = None  # 违法类型字符串，None 表示正常


class YOLODetector:
    def __init__(
        self,
        model_size="n",
        conf_threshold=0.5,
        target_classes=None,
        use_gpu=True,
        enable_bytetrack=True,
        tracker_cfg="bytetrack.yaml",
    ):
        self.model_size = model_size
        self.conf_threshold = (
            conf_threshold  # 通用阈值，前端可调整，红绿灯检测内部使用更低的_TL_CONF_MIN
        )
        self.target_classes = target_classes  # 目标检测类别列表，None表示检测所有类别，eg：['car','person']，前端可调整
        self.use_gpu = use_gpu
        self.enable_bytetrack = enable_bytetrack  # 是否启用Bytetrack多目标跟踪，启用后会增加track_id字段用于区分不同个体，tracker_cfg是bytetrack的配置文件路径
        self.tracker_cfg = tracker_cfg
        self.backend = None
        self.model_loaded = False
        self.device_info = "CPU"
        self._infer_device = "cpu"  # ultralytics 路径使用
        self._load_model()

    def _resolve_model_path(self):
        """如果 model_size 是完整路径（含 / 或 .pt）则直接返回，否则拼接标准名称。"""
        ms = self.model_size
        if os.sep in ms or "/" in ms or ms.endswith(".pt"):
            return ms
        return f"yolov8{ms}.pt"

    # ── 模型加载 ──────────────────────────────────────────────────────
    def _load_model(self):
        try:
            import torch
            from ultralytics import YOLO

            if self.use_gpu:
                # 1. NVIDIA CUDA
                if torch.cuda.is_available():
                    self.yolo = YOLO(
                        self._resolve_model_path()
                    )  # 加载模型到默认设备（通常是CPU）
                    self.yolo.to("cuda")
                    self._infer_device = "cuda"
                    self.device_info = f"CUDA · {torch.cuda.get_device_name(0)}"
                    self.backend = "ultralytics"
                    self.model_loaded = True
                    print(f"[Detector] 使用 CUDA GPU: {torch.cuda.get_device_name(0)}")
                    print(
                        f"[Detector] YOLOv8{self.model_size} 加载完成  [{self.device_info}]"
                    )
                    return

            # 2. CPU 兜底
            self.yolo = YOLO(self._resolve_model_path())
            self._infer_device = "cpu"
            self.device_info = "CPU"
            self.backend = "ultralytics"
            self.model_loaded = True
            reason = "已禁用 GPU" if not self.use_gpu else "GPU 不可用"
            print(f"[Detector] {reason}，使用 CPU")
            print(f"[Detector] YOLOv8{self.model_size} 加载完成  [{self.device_info}]")

        except Exception as e:
            raise RuntimeError(f"ultralytics 加载失败: {e}")

    # ── 推理路由 ──────────────────────────────────────────────────────
    def detect(self, frame):  # frame 是当前帧图像
        if not self.model_loaded:
            return []
        return self._detect_ultralytics(frame)

    def _detect_ultralytics(self, frame):
        """
        使用 ultralytics YOLOv8 进行检测，返回 DetectionResult 列表
        """
        run_conf = min(_TL_CONF_MIN, self.conf_threshold)
        infer_kwargs = {
            "conf": run_conf,
            "device": self._infer_device,
            "verbose": False,
        }  # verbose=False 关闭 ultralytics 内部日志输出
        infer_fn = self.yolo
        # bytetrack 追踪模式需要保持状态，传入 persist=True 和 tracker 配置，且推理函数改为 track()，以启用多目标跟踪和轨迹ID分配。启用后每个检测结果会包含 track_id 字段，用于区分不同个体的轨迹。
        if self.enable_bytetrack:
            infer_fn = self.yolo.track
            """
            infer_fn内容：
                - track() 是 ultralytics YOLOv8 中的一个方法，用于在视频或连续帧中进行对象检测和跟踪。启用 Bytetrack 追踪模式后，推理函数改为 track()，以保持状态并分配轨迹ID。
                - persist=True 参数告诉模型在连续帧之间保持跟踪状态，这样它可以识别出同一对象在不同帧中的连续出现，并为其分配相同的 track_id。
                - tracker_cfg 是 Bytetrack 的配置文件路径，包含了跟踪算法的参数设置，如匹配策略、距离度量、最大失踪时间等。传入该配置后，模型会根据配置使用 Bytetrack 进行跟踪。
                bbox,class,confidence,track_id = infer_fn(frame, conf=run_conf, device=self._infer_device, verbose=False, persist=True, tracker=self.tracker_cfg)[0].boxes
            """
            infer_kwargs["persist"] = True
            infer_kwargs["tracker"] = self.tracker_cfg

        results = []
        for box in infer_fn(frame, **infer_kwargs)[0].boxes:
            """
            infer_fn(frame, **infer_kwargs) 返回一个包含检测结果 Result 的列表，取第一个元素（当前帧的结果），再访问其 boxes 属性获取检测框信息
            [0]: 代表当前帧的检测结果，boxes 是一个包含所有检测框信息的属性，每个 box 包含了坐标、类别、置信度等信息
            .boxes: 是 ultralytics YOLOv8 中的一个属性，包含了当前帧所有检测框的信息。每个 box 对象通常具有以下属性：
                - xyxy: 检测框的坐标，格式为 [x1, y1, x2, y2]，表示左上角和右下角的坐标。
                - cls: 检测框的类别ID，整数，表示检测到的对象的类别。
                - conf: 检测框的置信度，0~1之间的浮点数，表示模型对该检测结果的信心程度。
                - id: （启用 Bytetrack 时）检测框的轨迹ID，整数，用于区分不同个体的轨迹。
            eg.
                Results[0](
                    orig_img=frame,        # 原始图像 (720, 1280, 3)
                    boxes=Boxes(            # 检测框集合
                        data=tensor([
                            #  cls  x1   y1   x2   y2   conf  id(如果有track)
                            [  2, 320, 180, 450, 300, 0.92,  1  ],  # 车
                            [  9, 1100, 50, 1200, 120, 0.65, 2  ],  # 交通灯
                            [  0, 500, 400, 560, 550, 0.88,  3  ],  # 行人
                        ])
                    )
                )

            """
            cls_id = int(box.cls[0])  # 类别ID，整数
            """
            box.cls        # tensor([2.])     ← 还是个张量，虽然里面只有一个数
            box.cls[0]     # tensor(2.)       ← 取第 0 个元素，变成标量 tensor
            int(box.cls[0]) # 2               ← 转成 Python int
            """
            cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "unknown"
            if self.target_classes and cls_name not in self.target_classes:
                continue
            conf = float(box.conf[0])
            if cls_name != "traffic light" and conf < self.conf_threshold:
                continue  # 非交通灯且置信度低于通用阈值，过滤掉

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = None
            """
            hasattr(box, 'id') 检查 box 对象是否具有 id 属性，启用 Bytetrack 追踪模式后，box 对象会包含一个 id 属性，表示该检测框的轨迹ID。通过检查该属性，可以确定是否启用了跟踪功能，并获取对应的 track_id。
            """
            if self.enable_bytetrack and hasattr(box, "id") and box.id is not None:
                track_id = int(box.id[0])

            det = DetectionResult(
                [x1, y1, x2, y2], cls_id, cls_name, conf, track_id=track_id
            )
            if cls_name == "traffic light":
                det.extra["light_color"] = self._detect_light_color(
                    frame, [x1, y1, x2, y2]
                )
            results.append(det)
        return results  # 所有的检测结果列表，每个元素是一个 DetectionResult 对象，包含了检测框坐标、类别ID、类别名称、置信度、轨迹ID（如果启用Bytetrack）和额外信息（如红绿灯颜色）

    # ── 红绿灯颜色识别 ────────────────────────────────────────────────
    def _detect_light_color(
        self, frame, bbox
    ):  # bbox = [x1, y1, x2, y2] frame = 当前帧图像
        """
        综合两种策略识别当前亮起的灯色：
          1. 竖向分区：上 1/3 = 红灯区，中 1/3 = 黄灯区，下 1/3 = 绿灯区
             ——针对竖向标准红绿灯，位置信息大幅降低背景干扰
          2. 全局颜色比率：整个 bounding box 内各颜色像素占比
        最终得分 = 分区比率 × 0.6 + 全局比率 × 0.4
        """
        x1, y1, x2, y2 = [max(0, v) for v in bbox]  # 确保坐标非负
        x2 = min(frame.shape[1], x2)  # 确保坐标不超出图像边界
        y2 = min(frame.shape[0], y2)
        bh = y2 - y1  # bbox height
        bw = x2 - x1  # bbox width
        if bw < 4 or bh < 4:  # 过小的检测框难以判断颜色，直接返回 unknown
            return "unknown"

        roi = frame[
            y1:y2, x1:x2
        ]  # 取一张图片的y1到y2行，x1到x2列的区域，也就是检测框内的图像区域
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        """
        hsv格式：
            与BGR类似
            hsv.shape = (bh, bw, 3) # 高、宽、通道数
            hsv.dtype = uint8 # 数据类型为无符号8位整数
            hsv.shape[0] = bh # bbox的高度
            hsv.shape[1] = bw # bbox的宽度

        cv2.cvtColor() 是 OpenCV 库中的一个函数，用于转换图像的颜色空间。在这里，cv2.COLOR_BGR2HSV 参数指定了从 BGR（蓝绿红）颜色空间转换到 HSV（色调、饱和度、明度）颜色空间。
        转换到 HSV 色彩空间，HSV 色彩空间更适合进行颜色分割和识别，其中 H（色调）表示颜色类型，S（饱和度）表示颜色的纯度，V（明度）表示颜色的亮度。通过在 HSV 空间中定义颜色范围，可以更准确地识别红绿灯的颜色。
        HSV 格式的好处：
            - 色调（H）直接对应颜色类型，便于定义红绿灯的颜色范围，eg. 红灯通常在 H=0~10 和 H=160~180，绿灯在 H=40~90，黄灯在 H=15~35
            - 饱和度（S）和明度（V）可以帮助过滤掉过于暗淡或过于灰白的区域，减少误识别
        """
        total = max(bh * bw, 1)  # bbox 内总像素数，避免除零错误

        # 全局各色像素数
        """
        HSV 色彩空间的 H（色调）范围通常是 0~179，S（饱和度）和 V（明度）的范围是 0~255。
        通过定义不同颜色的 HSV 范围，可以使用 cv2.inRange() 函数来创建二值掩码，从而统计特定颜色的像素数量。例如：
            -   红色：由于红色在 HSV 色彩空间中分布在 H=0~10 和 H=160~180 两个范围内，因此需要分别统计这两个范围内的像素数量，并将它们相加得到红色像素的总数。
            -   绿色：绿色通常在 H=40~90 的范围内
            -   黄色：黄色通常在 H=15~35 的范围内
        countNonZero() 函数用于统计二值图像中非零像素的数量，结合 inRange() 创建的掩码，可以得到特定颜色的像素数量。
        """
        # 全区域红绿黄像素数
        r_all = cv2.countNonZero(
            cv2.inRange(hsv, (0, 120, 100), (10, 255, 255))
        ) + cv2.countNonZero(cv2.inRange(hsv, (160, 100, 100), (180, 255, 255)))
        g_all = cv2.countNonZero(cv2.inRange(hsv, (40, 80, 80), (90, 255, 255)))
        y_all = cv2.countNonZero(cv2.inRange(hsv, (15, 100, 100), (35, 255, 255)))

        # 竖向分区（上/中/下各 1/3），统计各区红绿黄像素数占比，减少背景干扰
        th = max(bh // 3, 1)  # 分区高度，至少为1像素，避免切分时出现空区导致除零错误
        top_hsv = hsv[:th, :]
        mid_hsv = hsv[th : 2 * th, :]
        bot_hsv = hsv[2 * th :, :]
        tp = max(top_hsv.shape[0] * top_hsv.shape[1], 1)
        mp = max(mid_hsv.shape[0] * mid_hsv.shape[1], 1)
        bp = max(bot_hsv.shape[0] * bot_hsv.shape[1], 1)

        r_top = (
            cv2.countNonZero(cv2.inRange(top_hsv, (0, 120, 100), (10, 255, 255)))
            + cv2.countNonZero(cv2.inRange(top_hsv, (160, 100, 100), (180, 255, 255)))
        ) / tp
        y_mid = (
            cv2.countNonZero(cv2.inRange(mid_hsv, (15, 100, 100), (35, 255, 255))) / mp
        )
        g_bot = (
            cv2.countNonZero(cv2.inRange(bot_hsv, (40, 80, 80), (90, 255, 255))) / bp
        )

        # 综合得分
        scores = {
            "red": r_top * 0.6 + (r_all / total) * 0.4,
            "yellow": y_mid * 0.6 + (y_all / total) * 0.4,
            "green": g_bot * 0.6 + (g_all / total) * 0.4,
        }
        best = max(scores, key=scores.get)
        """
        get() 方法返回指定键的值，这里是获取 scores 字典中分数最高的颜色类别
        max() 函数用于返回 scores 字典中分数最高的颜色类别，key=scores.get 表示根据字典的值进行比较，最终 best 变量将存储分数最高的颜色类别名称（'red'、'yellow' 或 'green'）。
        """
        # 远距离小灯体像素少，降低阈值可减少首轮 unknown
        min_side = min(bw, bh)
        if min_side < 12:
            threshold = 0.025
        elif min_side < 20:
            threshold = 0.04
        else:
            threshold = 0.06
        if scores[best] < threshold:
            return "unknown"
        return best

    def draw_results(self, frame, results, violation_zones=None):
        """绘制检测结果，包括检测框、类别名称和置信度，以及违规区域的高亮显示"""
        output = frame.copy()
        h, w = output.shape[:2]  # 获取图像的高度和宽度

        # 用户自定义ROI
        if violation_zones:
            self._draw_zones(output, violation_zones)

        for det in results:
            x1, y1, x2, y2 = det.bbox
            is_violation = det.violation is not None
            color = (
                VIOLATION_COLOR
                if is_violation
                else CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)
            )

            # 红绿灯颜色覆盖
            if det.class_name == "traffic light":
                lc = det.extra.get("light_color", "unknown")
                color = {
                    "red": (30, 30, 220),
                    "green": (50, 200, 50),
                    "yellow": (30, 200, 220),
                }.get(lc, CLASS_COLORS["traffic light"])

            # 违法半透明填充
            if is_violation:
                ov_buf = output.copy()
                cv2.rectangle(ov_buf, (x1, y1), (x2, y2), VIOLATION_COLOR, -1)
                cv2.addWeighted(ov_buf, 0.18, output, 0.82, 0, output)

            thick = 3 if is_violation else 2
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thick)

            # 标签文字
            parts = [det.class_name, f"{det.confidence:.0%}"]
            if det.track_id is not None:
                parts.append(f"ID:{det.track_id}")
            if det.class_name == "traffic light":
                lc = det.extra.get("light_color", "")
                if lc and lc != "unknown":
                    parts.append(lc.upper())
            if is_violation:
                vmap = {
                    "闯红灯": "RED-LIGHT",
                    "占用斑马线": "BLOCK XWALK",
                    "未礼让行人": "NO YIELD",
                }
                parts.append(f"! {vmap.get(det.violation, det.violation)}")
            label = "  ".join(parts)

            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            ly = max(y1 - 4, lh + 6)
            cv2.rectangle(
                output, (x1, ly - lh - bl - 3), (x1 + lw + 8, ly + bl - 2), color, -1
            )
            cv2.putText(
                output,
                label,
                (x1 + 4, ly - bl),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 状态栏
        ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        info = f"{ts}    objects: {len(results)}    {self.device_info}"
        cv2.rectangle(output, (0, h - 24), (w, h), (245, 245, 245), -1)
        cv2.putText(
            output,
            info,
            (8, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        return output

    def _draw_zones(self, frame, zones):
        """绘制违规区域的高亮显示"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        color_map = {
            "stop_line": (30, 30, 220),
            "crosswalk": (50, 160, 50),
            "traffic_light_roi": (30, 180, 220),  # 琥珀黄
        }
        for zone in zones:
            ztype = zone.get("type", "stop_line")  # 第二个参数是默认值
            color = color_map.get(ztype, (150, 150, 150))

            if ztype == "stop_line":
                pts_sl = zone.get("pts")  # pts得到用户自定义的ROI顶点列表
                if pts_sl and len(pts_sl) >= 2:
                    p1 = (int(pts_sl[0][0]), int(pts_sl[0][1]))
                    p2 = (int(pts_sl[1][0]), int(pts_sl[1][1]))
                    cv2.line(frame, p1, p2, color, 3)
                    cv2.putText(
                        frame,
                        "STOP LINE",
                        (p1[0] + 6, p1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                    )
                else:
                    # 兼容旧格式
                    y = int(zone.get("y", h // 2))
                    x1 = int(zone.get("x1", 0))
                    x2 = int(zone.get("x2", w))
                    cv2.line(frame, (x1, y), (x2, y), color, 3)
                    cv2.putText(
                        frame,
                        "STOP LINE",
                        (x1 + 6, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                    )

            elif ztype == "crosswalk":
                pts = zone.get("pts")
                if pts:
                    arr = np.array(
                        pts, np.int32
                    )  # arr数据格式：[[x1, y1], [x2, y2], ...]
                    cv2.fillPoly(overlay, [arr], color)  # 填充多边形区域
                    cv2.polylines(frame, [arr], True, color, 2)  # 绘制多边形边界
                    cv2.putText(
                        frame,
                        "CROSSWALK",
                        (pts[0][0] + 4, pts[0][1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        color,
                        1,
                    )

            elif ztype == "traffic_light_roi":
                pts = zone.get("pts")
                if pts:
                    arr = np.array(pts, np.int32)
                    cv2.polylines(frame, [arr], True, color, 2)
                    cv2.putText(
                        frame,
                        "LIGHT ROI",
                        (pts[0][0] + 4, pts[0][1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        color,
                        1,
                    )

        # 将叠加层与原始帧混合，检测框和原画面结合，得到最终输出，alpha-混合因子控制叠加层的透明度，beta-混合因子控制原始帧的透明度
        cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)


class TrafficLightTracker:
    """
    时序平滑红绿灯状态，避免单帧抖动和短暂漏检。
        - history_len: 记录最近几帧的状态，默认20帧
        - min_votes: 判定稳定状态所需的最少票数，默认5票
        - persist_secs: 超过该时间未见有效状态则重置为 unknown，默认6秒
        - update(raw_color, source): 更新当前帧的原始颜色状态，返回平滑后的稳定状态。source 参数用于指示调用来源（如 'detector'），以决定是否强制更新稳定状态。
        - reset(): 重置内部状态，清空历史记录和计时器。
        ROI（Region of Interest，感兴趣区域）就是用户在前端画的红绿灯分析区域，启用后检测到的红绿灯如果在该区域内才进行颜色识别和状态更新，这样可以避免画面中其他位置的红绿灯干扰分析结果。
    """

    def __init__(self, history_len=20, min_votes=5, persist_secs=6.0):
        self._history = []  # 最近几帧的原始颜色状态列表，长度不超过 history_len，每个元素是 'red'、'yellow'、'green' 或 'unknown'
        self._history_len = history_len
        self._min_votes = min_votes
        self._persist_s = persist_secs  # 超过该时间未见有效状态则重置为 unknown
        self._stable = "unknown"  # 当前稳定状态，初始为 unknown，真实的颜色状态
        self._last_seen = 0.0

    def update(self, raw_color, source="detector"):
        """
        更新当前帧的原始颜色状态，返回平滑后的稳定状态。
        - raw_color: 当前帧检测到的原始颜色状态，'red'、'yellow'、'green' 或 'unknown'
        - source: 调用来源字符串，默认为 'detector'，如果是 'detector' 则即使当前稳定状态未知也会被新状态覆盖，否则只有在已有稳定状态的情况下才会被更新
        """
        now = time.time()
        if raw_color != "unknown":
            self._last_seen = now
            self._history.append(raw_color)
            if len(self._history) > self._history_len:
                self._history.pop(0)  # 保持历史记录长度不超过 history_len
            if self._stable == "unknown" or source == "detector":
                self._stable = raw_color  # 如果当前稳定状态未知，或者调用来源是 detector，则直接更新稳定状态为当前原始状态，这样可以快速响应状态变化；否则只有在已有稳定状态的情况下才会被更新，以增加稳定性
                return self._stable

        if now - self._last_seen > self._persist_s:
            self._stable = "unknown"
            self._history = []
            return self._stable

        if self._history:
            """
            当且仅当当前检测到的状态不是 'unknown' 时，并且采用ROI分析时，需要使用投票机制
            统计历史记录中出现次数最多的颜色状态，如果该状态的票数超过 min_votes 则更新稳定状态
            Counter(self._history) 会统计历史记录中每个颜色状态出现的次数，most_common(1) 会返回出现次数最多的那个状态及其次数，[0] 取第一个元素（因为可能有多个状态出现相同次数），[0] 再取状态名称，[1] 取票数
            Counter(self._history) = [(),()]
            Counter(self._history).most_common(1) = [()]
            top, cnt = Counter(self._history).most_common(1)[0] = ('red', 3) 元组
            """
            top, cnt = Counter(self._history).most_common(1)[0]
            if cnt >= self._min_votes:
                self._stable = top

        return self._stable

    def reset(self):
        self._history = []
        self._stable = "unknown"
        self._last_seen = 0.0
