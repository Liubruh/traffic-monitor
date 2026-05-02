import time
from datetime import datetime

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}


def _bbox_bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, y2)


def _point_in_polygon(pt, polygon):
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _bbox_overlaps_polygon(bbox, polygon):
    if not polygon:
        return False
    x1, y1, x2, y2 = bbox
    test_pts = [
        _bbox_bottom_center(bbox),
        ((x1 + x2) // 2, (y1 + y2) // 2),
    ]
    return any(_point_in_polygon(p, polygon) for p in test_pts)


class PedestrianYieldEngine:
    """
    车辆礼让行人检测（简化版）。

    规则：
      - 当斑马线上有行人时，车辆若进入同一斑马线区域，判定为“未礼让行人”。
      - 使用 ByteTrack track_id 做粘性标记，避免闪烁。
    """

    def __init__(self):
        self.zones = []
        self.cooldown = 4.0
        self.track_ttl = 2.5
        self._last_alert = {}
        self._tracked_vio = {}

    def set_zones(self, zones):
        self.zones = zones or []

    def _cleanup_tracked(self, now):
        stale = [tid for tid, item in self._tracked_vio.items()
                 if now - item['last_seen'] > self.track_ttl]
        for tid in stale:
            self._tracked_vio.pop(tid, None)

    def _touch_sticky(self, det, now):
        track_id = getattr(det, 'track_id', None)
        if track_id is None:
            return
        sticky = self._tracked_vio.get(track_id)
        if not sticky:
            return
        sticky['last_seen'] = now
        if det.violation is None:
            det.violation = sticky['type']

    def check(self, results):
        now = time.time()
        ts = datetime.now().strftime('%H:%M:%S')
        self._cleanup_tracked(now)

        crosswalks = [z for z in self.zones if z.get('type') == 'crosswalk']
        if not crosswalks:
            for det in results:
                self._touch_sticky(det, now)
            return []

        # 找到每个斑马线里的行人
        ped_on_crosswalk = {}
        for idx, cw in enumerate(crosswalks):
            pts = cw.get('pts', [])
            if not pts:
                continue
            ped_on_crosswalk[idx] = [
                det for det in results
                if det.class_name == 'person' and _bbox_overlaps_polygon(det.bbox, pts)
            ]

        violations = []
        for det in results:
            if det.class_name not in VEHICLE_CLASSES:
                continue

            track_id = getattr(det, 'track_id', None)
            vio_type = None

            # 同一斑马线上存在行人 + 车辆进入斑马线 => 未礼让行人
            for idx, cw in enumerate(crosswalks):
                if not ped_on_crosswalk.get(idx):
                    continue
                if _bbox_overlaps_polygon(det.bbox, cw.get('pts', [])):
                    vio_type = '未礼让行人'
                    break

            if vio_type:
                if det.violation is None:
                    det.violation = vio_type

                if track_id is not None:
                    self._tracked_vio[track_id] = {
                        'type': vio_type,
                        'last_seen': now,
                    }

                if track_id is not None:
                    key = f'{vio_type}_track_{track_id}'
                else:
                    key = f'{vio_type}_{det.bbox[0]//80}_{det.bbox[1]//80}'

                if now - self._last_alert.get(key, 0) > self.cooldown:
                    self._last_alert[key] = now
                    violations.append({
                        'type': vio_type,
                        'class_name': det.class_name,
                        'confidence': det.confidence,
                        'time': ts,
                        'bbox': det.bbox,
                    })
            else:
                self._touch_sticky(det, now)

        return violations
