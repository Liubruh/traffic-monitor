import cv2
import numpy as np
import datetime
import os
import glob

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

TRAFFIC_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
                   'traffic light', 'stop sign']

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

_OV_INPUT_SIZE  = 640               # YOLOv8 标准输入尺寸
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
        self._ov_compiled    = None    # OpenVINO 原生路径使用
        self._first_infer    = True    # 首帧打印诊断信息
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

                # 2. Apple MPS
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.yolo = YOLO(self._resolve_model_path())
                    self.yolo.to('mps')
                    self._infer_device = 'mps'
                    self.device_info   = 'Apple MPS'
                    self.backend       = 'ultralytics'
                    self.model_loaded  = True
                    print('[Detector] 使用 Apple MPS GPU')
                    print(f'[Detector] YOLOv8{self.model_size} 加载完成  [{self.device_info}]')
                    return

                # 3. Intel 核显 / CPU — 直接用 OpenVINO Core API（绕过 ultralytics 的 CUDA 检查）
                if self._try_load_openvino():
                    print(f'[Detector] YOLOv8{self.model_size} 加载完成  [{self.device_info}]')
                    return

            # 4. 普通 CPU 兜底
            self.yolo = YOLO(self._resolve_model_path())
            self._infer_device = 'cpu'
            self.device_info   = 'CPU'
            self.backend       = 'ultralytics'
            self.model_loaded  = True
            reason = '已禁用 GPU' if not self.use_gpu else 'GPU 不可用'
            print(f'[Detector] {reason}，使用 CPU')
            print(f'[Detector] YOLOv8{self.model_size} 加载完成  [{self.device_info}]')
            return

        except Exception as e:
            print(f'[Detector] ultralytics 不可用: {e}')

        # Demo 模式
        print('[Detector] 使用 Demo 演示模式')
        self.backend      = 'demo'
        self.model_loaded = True
        self.device_info  = 'Demo Mode'

    def _try_load_openvino(self):
        """
        使用 OpenVINO Core API 直接编译模型并运行在 Intel GPU/CPU 上。
        成功返回 True 并设置 self._ov_compiled / backend / device_info。
        """
        try:
            import openvino as ov
            from ultralytics import YOLO

            ov_dir = f'yolov8{self.model_size}_openvino_model'

            # 首次运行：导出 OpenVINO IR 格式
            if not os.path.exists(ov_dir):
                print('[Detector] 首次运行：正在导出 OpenVINO 格式'
                      '（约需 30–60 秒，之后会缓存）...')
                tmp = YOLO(self._resolve_model_path())
                tmp.export(format='openvino')
                print('[Detector] OpenVINO 模型导出完成')

            # 查找 XML 文件
            xml_files = glob.glob(os.path.join(ov_dir, '*.xml'))
            if not xml_files:
                print(f'[Detector] 未在 {ov_dir} 找到 .xml 文件，跳过 OpenVINO')
                return False

            core      = ov.Core()
            available = core.available_devices
            print(f'[Detector] OpenVINO 可用设备: {available}')

            ov_model = core.read_model(xml_files[0])

            # 优先尝试 Intel iGPU，失败则降级 CPU
            for device, label in [('GPU', 'Intel iGPU · OpenVINO'),
                                   ('CPU', 'Intel CPU · OpenVINO')]:
                if device not in available and device == 'GPU':
                    continue
                try:
                    # GPU 默认 FP16，大模型激活值易溢出成 NaN，强制 FP32
                    config = {'INFERENCE_PRECISION_HINT': 'f32'} if device == 'GPU' else {}
                    compiled = core.compile_model(ov_model, device, config=config)

                    # 用随机图像测试（零输入无法暴露 FP16 溢出）
                    dummy = (np.random.rand(1, 3, _OV_INPUT_SIZE, _OV_INPUT_SIZE)
                             .astype(np.float32))
                    test_out = compiled(dummy)[compiled.output(0)]
                    if np.isnan(test_out).any():
                        print(f'[Detector] OpenVINO {device} 推理结果含 NaN，跳过')
                        continue

                    self._ov_compiled = compiled
                    self.device_info  = label
                    self.backend      = 'openvino_native'
                    self.model_loaded = True
                    print(f'[Detector] OpenVINO {device} 测试推理成功 ✓')
                    print(f'[Detector] 使用 {label}')
                    return True
                except Exception as err:
                    print(f'[Detector] OpenVINO {device} 失败: {err}')

            return False

        except ImportError:
            print('[Detector] OpenVINO 未安装，跳过。'
                  '如需 Intel 核显加速请运行: pip install openvino')
            return False
        except Exception as e:
            print(f'[Detector] OpenVINO 初始化失败: {e}')
            return False

    # ── 推理路由 ──────────────────────────────────────────────────────
    def detect(self, frame):
        if not self.model_loaded:
            return []
        if self.backend == 'ultralytics':
            return self._detect_ultralytics(frame)
        if self.backend == 'openvino_native':
            return self._detect_openvino_native(frame)
        return self._detect_demo(frame)

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

    def _detect_openvino_native(self, frame):
        """直接调用 OpenVINO Core 推理，完整处理预/后处理。"""
        h, w = frame.shape[:2]
        size = _OV_INPUT_SIZE

        # ── 预处理：Letterbox 缩放 + 填充 ──────────────────────────────
        scale   = min(size / h, size / w)
        new_h   = int(h * scale)
        new_w   = int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded  = np.zeros((size, size, 3), dtype=np.uint8)
        pad_y   = (size - new_h) // 2
        pad_x   = (size - new_w) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # BGR → RGB，归一化，HWC → CHW，加 batch 维度
        inp = padded[:, :, ::-1].astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[np.newaxis]          # (1, 3, 640, 640)

        # ── 推理 ────────────────────────────────────────────────────────
        try:
            ov_result = self._ov_compiled(inp)
            raw = ov_result[self._ov_compiled.output(0)]
        except Exception as e:
            print(f'[Detector] OpenVINO 推理错误: {e}')
            return []

        # ── 首帧诊断：打印输出张量信息 ───────────────────────────────────
        if self._first_infer:
            self._first_infer = False
            print(f'[Detector][诊断] raw.shape={raw.shape}  '
                  f'dtype={raw.dtype}  max={float(np.nanmax(raw)):.4f}')

        # ── NaN 保护 ────────────────────────────────────────────────────
        if np.isnan(raw).any():
            print('[Detector] 推理结果含 NaN（FP16 溢出），本帧跳过')
            return []

        # ── 自动识别输出张量方向 ─────────────────────────────────────────
        # YOLOv8 标准输出有两种可能：
        #   (1, 84, 8400) — channels-first：需要转置
        #   (1, 8400, 84) — channels-last：无需转置
        # 判断依据：8400 >> 84，哪个轴更大就是 anchor 轴
        if raw.ndim != 3:
            print(f'[Detector] 意外输出维度: {raw.shape}，跳过')
            return []

        _, a, b = raw.shape
        if a < b:
            # (1, 84, 8400) → 转置为 (8400, 84)
            preds = raw[0].T
        else:
            # (1, 8400, 84) → 已经是 (8400, 84)
            preds = raw[0]

        n_attrs = preds.shape[1]          # 84 = 4 坐标 + 80 类别
        if n_attrs < 5:
            print(f'[Detector] 输出属性数异常: {n_attrs}，跳过')
            return []

        boxes_cxcywh = preds[:, :4]       # cx, cy, w, h（640 空间内像素值）
        class_scores = preds[:, 4:]       # (8400, n_classes)

        try:
            max_scores = class_scores.max(axis=1)
            class_ids  = class_scores.argmax(axis=1)
        except Exception as e:
            print(f'[Detector] 分数解析错误: {e}  shape={preds.shape}')
            return []

        # 红绿灯使用更低的置信度阈值，防止远距离小目标漏检
        tl_conf = min(_TL_CONF_MIN, self.conf_threshold)
        mask = ((class_ids == _TL_CLS_ID) & (max_scores >= tl_conf)) | \
               ((class_ids != _TL_CLS_ID) & (max_scores >= self.conf_threshold))
        if not mask.any():
            return []

        boxes_f  = boxes_cxcywh[mask]
        scores_f = max_scores[mask]
        cls_f    = class_ids[mask]

        # cx,cy,w,h → x1,y1,x2,y2（640 空间）
        x1 = boxes_f[:, 0] - boxes_f[:, 2] / 2
        y1 = boxes_f[:, 1] - boxes_f[:, 3] / 2
        x2 = boxes_f[:, 0] + boxes_f[:, 2] / 2
        y2 = boxes_f[:, 1] + boxes_f[:, 3] / 2

        # NMS — 用红绿灯低阈值作为 score_threshold，避免候选框被提前丢弃
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        nms_conf   = min(_TL_CONF_MIN, self.conf_threshold)
        indices    = cv2.dnn.NMSBoxes(boxes_xywh, scores_f.tolist(),
                                      nms_conf, 0.45)
        if indices is None or len(indices) == 0:
            return []

        # ── 后处理：坐标还原到原始帧 ────────────────────────────────────
        results = []
        for idx in np.array(indices).flatten():
            cls_id   = int(cls_f[idx])
            cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else 'unknown'
            if self.target_classes and cls_name not in self.target_classes:
                continue
            # 非红绿灯目标补充过滤（NMS 已用低阈值运行）
            if cls_id != _TL_CLS_ID and float(scores_f[idx]) < self.conf_threshold:
                continue

            bx1 = int((x1[idx] - pad_x) / scale)
            by1 = int((y1[idx] - pad_y) / scale)
            bx2 = int((x2[idx] - pad_x) / scale)
            by2 = int((y2[idx] - pad_y) / scale)

            # 裁剪到帧范围内
            bx1 = max(0, min(bx1, w - 1))
            by1 = max(0, min(by1, h - 1))
            bx2 = max(0, min(bx2, w))
            by2 = max(0, min(by2, h))

            det = DetectionResult([bx1, by1, bx2, by2],
                                  cls_id, cls_name, float(scores_f[idx]))
            if cls_name == 'traffic light':
                det.extra['light_color'] = self._detect_light_color(
                    frame, [bx1, by1, bx2, by2])
            results.append(det)

        return results

    def _detect_demo(self, frame):
        h, w  = frame.shape[:2]
        seed  = int(np.mean(frame[::10, ::10])) % 100
        np.random.seed(seed // 10)
        demo  = self.target_classes or TRAFFIC_CLASSES
        results = []
        for _ in range(np.random.randint(1, 6)):
            cls_name = np.random.choice(demo)
            cls_id   = COCO_CLASSES.index(cls_name) if cls_name in COCO_CLASSES else 0
            conf     = round(np.random.uniform(0.55, 0.97), 2)
            bw = np.random.randint(80, max(81, w // 3))
            bh = np.random.randint(60, max(61, h // 3))
            bx = np.random.randint(0, max(1, w - bw))
            by = np.random.randint(0, max(1, h - bh))
            det = DetectionResult([bx, by, bx+bw, by+bh], cls_id, cls_name, conf)
            if cls_name == 'traffic light':
                det.extra['light_color'] = np.random.choice(
                    ['red', 'green', 'yellow'], p=[0.4, 0.4, 0.2])
            results.append(det)
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

    def analyze_region_color(self, frame, bbox):
        """公共接口：对指定区域做红绿灯颜色分析（供 app.py ROI 路径调用）。"""
        return self._detect_light_color(frame, bbox)

    def _draw_zones(self, frame, zones):
        overlay = frame.copy()
        h, w    = frame.shape[:2]
        color_map = {
            'stop_line':        (30,  30,  220),
            'crosswalk':        (50,  160, 50),
            'lane_line':        (30,  140, 220),
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

            elif ztype == 'lane_line':
                pts = zone.get('pts', [])
                for i in range(len(pts)-1):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), color, 2)

            elif ztype == 'traffic_light_roi':
                pts = zone.get('pts')
                if pts:
                    arr = np.array(pts, np.int32)
                    cv2.polylines(frame, [arr], True, color, 2)
                    cv2.putText(frame, 'LIGHT ROI',
                                (pts[0][0]+4, pts[0][1]+16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)
