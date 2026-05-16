"""
violation_engine.py
-------------------
交通违法检测引擎，支持：
  1. 闯红灯（车辆在红灯时正在穿越停止线）
  2. 车辆占用斑马线（车辆在斑马线区域内静止超过 1.5 秒）
  3. 未礼让行人（斑马线上有行人时车辆进入斑马线区域）

停止线格式：
  旧格式：{"type": "stop_line", "y": 400, "x1": 100, "x2": 700}
  新格式：{"type": "stop_line", "pts": [[100, 400], [700, 410]]}  ← 支持斜线
"""

import numpy as np
import time
from datetime import datetime

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}


# ── 几何工具 ─────────────────────────────────────────────────────────

def _bbox_bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, y2)


def _point_in_polygon(pt, polygon):
    """射线法判断点是否在多边形内"""
    x, y   = pt
    n      = len(polygon)
    inside = False
    j      = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _bbox_overlaps_polygon(bbox, polygon):
    x1, y1, x2, y2 = bbox
    test_pts = [
        _bbox_bottom_center(bbox),
        ((x1 + x2) // 2, (y1 + y2) // 2),
    ]
    return any(_point_in_polygon(p, polygon) for p in test_pts)





def _stop_line_endpoints(sl):
    """统一解析停止线端点，兼容新旧两种格式。"""
    pts = sl.get('pts')
    if pts and len(pts) >= 2:
        return pts[0][0], pts[0][1], pts[1][0], pts[1][1]
    # 旧格式
    y = sl.get('y', 0)
    return sl.get('x1', 0), y, sl.get('x2', 9999), y


def _bbox_crosses_stop_line(bbox, sl):
    """
    判断车辆是否正在穿越停止线（从下往上行驶的摄像头视角）。

    穿越定义：车辆边界框同时覆盖停止线两侧：
      by1 <= 停止线在该 X 处的 Y ≤ by2

    - by1 = bbox 顶部（车头方向），by2 = bbox 底部（车尾方向）
    - 只有车辆"骑跨"停止线时才触发，避免将已过线/未到线的车都标红
    - 支持斜线停止线（插值计算任意 x 处的 y）
    """
    bx1, by1, bx2, by2 = bbox
    lx1, ly1, lx2, ly2 = _stop_line_endpoints(sl)

    mid_x = (bx1 + bx2) // 2
    x_min, x_max = min(lx1, lx2), max(lx1, lx2)
    if not (x_min <= mid_x <= x_max):
        return False

    # 插值：停止线在 mid_x 处的 Y 值（支持斜线）
    if lx2 != lx1:
        t = (mid_x - lx1) / (lx2 - lx1)
        line_y = ly1 + t * (ly2 - ly1)
    else:
        line_y = ly1

    # 穿越条件：bbox 顶部在线上方（≤），底部在线下方（≥）
    return by1 <= line_y <= by2


# ── 引擎主体 ─────────────────────────────────────────────────────────

class ViolationEngine:

    def __init__(self):
        self.zones            = []
        self.cooldown         = 4.0
        self.track_ttl        = 2.5
        self.crosswalk_static_secs = 1.5
        self.static_move_tol_px    = 8.0
        self._last_vio: dict  = {}
        self._tracked_vio: dict = {}
        self._crosswalk_wait: dict = {}

    def _cleanup_tracked(self, now: float):
        stale = [tid for tid, item in self._tracked_vio.items()
                 if now - item['last_seen'] > self.track_ttl]
        for tid in stale:
            self._tracked_vio.pop(tid, None)

        stale_wait = [k for k, item in self._crosswalk_wait.items()
                      if now - item['last_seen'] > self.track_ttl]
        for k in stale_wait:
            self._crosswalk_wait.pop(k, None)


    def _det_key(self, det):
        track_id = getattr(det, 'track_id', None)
        if track_id is not None:
            return f'track_{track_id}'
        x1, y1, x2, y2 = det.bbox
        return f'bbox_{x1//80}_{y1//80}_{x2//80}_{y2//80}'

    def set_zones(self, zones: list):
        self.zones = zones

    def check(self, results, light_color: str = 'unknown') -> list:
        violations = []
        now = time.time()
        ts  = datetime.now().strftime('%H:%M:%S')
        self._cleanup_tracked(now)

        stop_lines = [z for z in self.zones if z.get('type') == 'stop_line']
        crosswalks = [z for z in self.zones if z.get('type') == 'crosswalk']

        for det in results:
            vio_type = None
            track_id = getattr(det, 'track_id', None)

            # ── 1. 闯红灯 ────────────────────────────────────────────
            # 条件：红灯 + 车辆正在穿越停止线（bbox 骑跨停止线）
            if light_color == 'red' and det.class_name in VEHICLE_CLASSES:
                for sl in stop_lines:
                    if _bbox_crosses_stop_line(det.bbox, sl):
                        vio_type = '闯红灯'
                        break

            # ── 2. 车辆占用斑马线（静止超过 1.5 秒）────────────────────
            if vio_type is None and det.class_name in VEHICLE_CLASSES:
                in_crosswalk = any(
                    _bbox_overlaps_polygon(det.bbox, cw.get('pts', []))
                    for cw in crosswalks
                )
                det_key = self._det_key(det)

                if in_crosswalk:
                    cx, cy = _bbox_bottom_center(det.bbox)
                    state = self._crosswalk_wait.get(det_key)
                    if state is None:
                        self._crosswalk_wait[det_key] = {
                            'last_center': (cx, cy),
                            'last_move': now,
                            'last_seen': now,
                        }
                    else:
                        px, py = state['last_center']
                        move_dist = float(np.hypot(cx - px, cy - py))
                        if move_dist > self.static_move_tol_px:
                            state['last_move'] = now
                        state['last_center'] = (cx, cy)
                        state['last_seen'] = now

                        if now - state['last_move'] >= self.crosswalk_static_secs:
                            vio_type = '占用斑马线'
                else:
                    self._crosswalk_wait.pop(det_key, None)

            # ── 写入违法标记 ─────────────────────────────────────────
            if vio_type:
                # 视觉标记每帧更新（无冷却），保证红框持续显示
                det.violation = vio_type

                # 使用 ByteTrack track_id 粘性跟随违法目标
                if track_id is not None:
                    self._tracked_vio[track_id] = {
                        'type': vio_type,
                        'last_seen': now,
                    }

                # 报警事件有冷却，防止同一辆车刷屏
                if track_id is not None:
                    vio_key = f'{vio_type}_track_{track_id}'
                else:
                    vio_key = f'{vio_type}_{det.bbox[0]//80}_{det.bbox[1]//80}'
                if now - self._last_vio.get(vio_key, 0) > self.cooldown:
                    self._last_vio[vio_key] = now
                    violations.append({
                        'type':       vio_type,
                        'class_name': det.class_name,
                        'confidence': det.confidence,
                        'time':       ts,
                        'bbox':       det.bbox,
                    })
            elif track_id is not None and track_id in self._tracked_vio:
                sticky = self._tracked_vio[track_id]
                sticky['last_seen'] = now
                det.violation = sticky['type']

        return violations


# ── 未礼让行人引擎 ─────────────────────────────────────────────────────

class PedestrianYieldEngine:
    """
    车辆礼让行人检测（简化版）。

    规则：
      - 当斑马线上有行人时，车辆若进入同一斑马线区域，判定为"未礼让行人"。
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
