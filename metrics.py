import time
from collections import deque, defaultdict


class MetricsTracker:
    def __init__(self, window=60):
        self.fps_history = deque(maxlen=window)
        self.detection_history = deque(maxlen=window)
        self.class_counts = defaultdict(int)
        self.total_frames = 0
        self.total_detections = 0
        self.alert_log = deque(maxlen=100)
        self.violation_counts = defaultdict(int)   # 各类型违规累计次数
        self.total_violations = 0
        self.start_time = time.time()
        self.last_results = []

    def update(self, fps, results):
        self.fps_history.append(fps)
        count = len(results)
        self.detection_history.append(count)
        self.total_frames += 1
        self.total_detections += count
        self.last_results = results
        for r in results:
            self.class_counts[r.class_name] += 1

    def add_alert(self, alert):
        self.alert_log.append(alert)
        self.violation_counts[alert['type']] += 1
        self.total_violations += 1

    def get_stats(self):
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        avg_det = sum(self.detection_history) / len(self.detection_history) if self.detection_history else 0
        elapsed = time.time() - self.start_time

        # 当前帧类别统计
        current_classes = {}
        for r in self.last_results:
            current_classes[r.class_name] = current_classes.get(r.class_name, 0) + 1

        # top class counts
        top_classes = sorted(self.class_counts.items(), key=lambda x: -x[1])[:8]

        return {
            'fps': round(avg_fps, 1),
            'avg_detections': round(avg_det, 1),
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'total_violations': self.total_violations,
            'violation_counts': dict(self.violation_counts),
            'elapsed': round(elapsed, 0),
            'current_classes': current_classes,
            'top_classes': dict(top_classes),
            'fps_history': list(self.fps_history)[-30:],
            'det_history': list(self.detection_history)[-30:],
            'alert_count': len(self.alert_log),
        }

    def reset(self):
        self.fps_history.clear()
        self.detection_history.clear()
        self.class_counts.clear()
        self.total_frames = 0
        self.total_detections = 0
        self.total_violations = 0
        self.violation_counts.clear()
        self.alert_log.clear()
        self.last_results = []
        self.start_time = time.time()
