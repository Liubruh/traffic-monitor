import time
from collections import deque, defaultdict


class MetricsTracker:
    def __init__(self, window=60):
        self.fps_history = deque(maxlen=window)
        self.detection_history = deque(maxlen=window) # 每帧检测到的对象数量历史
        self.total_frames = 0 # 总帧数
        self.alert_log = deque(maxlen=100) # 双端队列 - 违法记录
        self.total_violations = 0
        self.start_time = time.time()
        self.last_results = [] # 上一帧的检测结果列表

    def update(self, fps, results):
        self.fps_history.append(fps)
        count = len(results)
        self.detection_history.append(count)
        self.total_frames += 1
        self.last_results = results

    def add_alert(self, alert):
        self.alert_log.append(alert)
        self.total_violations += 1

    def get_stats(self):
        # 计算平均FPS和平均检测数量
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        avg_det = sum(self.detection_history) / len(self.detection_history) if self.detection_history else 0
        elapsed = time.time() - self.start_time

        # 当前帧类别统计
        current_classes = {}
        for r in self.last_results:
            current_classes[r.class_name] = current_classes.get(r.class_name, 0) + 1

        return {
            'fps': round(avg_fps, 1),
            'avg_detections': round(avg_det, 1),
            'total_frames': self.total_frames,
            'total_violations': self.total_violations,
            'elapsed': round(elapsed, 0),
            'current_classes': current_classes, # 当前帧的类别统计
            'fps_history': list(self.fps_history)[-30:],
        }

    def reset(self):
        self.fps_history.clear()
        self.detection_history.clear()
        self.total_frames = 0
        self.total_violations = 0
        self.alert_log.clear()
        self.last_results = []
        self.start_time = time.time()
