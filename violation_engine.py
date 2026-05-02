"""
violation_engine.py
-------------------
交通违法检测引擎，支持：
  1. 闯红灯（车辆在红灯时正在穿越停止线）
    2. 压线行驶（轮胎区域触线并持续超过阈值）
    3. 车辆占用斑马线（车辆在斑马线区域内静止超过 1.5 秒）

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


def _segment_intersects_rect(p1, p2, rect):
    """判断线段是否与轴对齐矩形相交。"""
    rx1, ry1, rx2, ry2 = rect

    def in_rect(pt):
        return rx1 <= pt[0] <= rx2 and ry1 <= pt[1] <= ry2

    if in_rect(p1) or in_rect(p2):
        return True

    edges = [
        ((rx1, ry1), (rx2, ry1)),
        ((rx2, ry1), (rx2, ry2)),
        ((rx2, ry2), (rx1, ry2)),
        ((rx1, ry2), (rx1, ry1)),
    ]
    return any(_segment_intersects(p1, p2, e1, e2) for e1, e2 in edges)


def _segment_intersects(a, b, c, d, eps=1e-6):
    """判断线段 AB 与 CD 是否相交（含端点与共线重叠）。"""

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p, q, r):
        return (min(p[0], q[0]) - eps <= r[0] <= max(p[0], q[0]) + eps and
                min(p[1], q[1]) - eps <= r[1] <= max(p[1], q[1]) + eps)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and \
       (o3 > eps and o4 < -eps or o3 < -eps and o4 > eps):
        return True

    if abs(o1) <= eps and on_seg(a, b, c):
        return True
    if abs(o2) <= eps and on_seg(a, b, d):
        return True
    if abs(o3) <= eps and on_seg(c, d, a):
        return True
    if abs(o4) <= eps and on_seg(c, d, b):
        return True

    return False


def _bbox_crosses_line(bbox, p1, p2):
    """判断车辆是否压线：车道线段与车辆底部轮胎带区域相交。"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False

    # 轮胎带：底部约 22% 高度，中间 64% 宽度，减少“车身边缘碰线”误报
    pad_x = int(w * 0.18)
    strip_h = max(6, int(h * 0.22))
    rx1 = x1 + pad_x
    rx2 = x2 - pad_x
    if rx2 <= rx1:
        return False
    ry1 = y2 - strip_h
    ry2 = y2 + 2

    return _segment_intersects_rect(p1, p2, (rx1, ry1, rx2, ry2))


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
        self.lane_hold_secs   = 0.8
        self.crosswalk_static_secs = 1.5
        self.static_move_tol_px    = 8.0
        self._last_vio: dict  = {}
        self._tracked_vio: dict = {}
        self._lane_touch: dict = {}
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

        stale_lane = [k for k, item in self._lane_touch.items()
                      if now - item['last_seen'] > self.track_ttl]
        for k in stale_lane:
            self._lane_touch.pop(k, None)

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
        lane_lines = [z for z in self.zones if z.get('type') == 'lane_line']

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

            # ── 2. 压线行驶 ──────────────────────────────────────────
            if vio_type is None and det.class_name in VEHICLE_CLASSES:
                det_key = self._det_key(det)
                lane_touched = False
                for ll in lane_lines:
                    pts = ll.get('pts', [])
                    if len(pts) >= 2:
                        for i in range(len(pts) - 1):
                            p1 = (pts[i][0], pts[i][1])
                            p2 = (pts[i+1][0], pts[i+1][1])
                            if _bbox_crosses_line(det.bbox, p1, p2):
                                lane_touched = True
                                break
                    if lane_touched:
                        break

                if lane_touched:
                    state = self._lane_touch.get(det_key)
                    if state is None:
                        self._lane_touch[det_key] = {'since': now, 'last_seen': now}
                    else:
                        state['last_seen'] = now
                    if now - self._lane_touch[det_key]['since'] >= self.lane_hold_secs:
                        vio_type = '压线行驶'
                else:
                    self._lane_touch.pop(det_key, None)

            # ── 3. 车辆占用斑马线（静止超过 1.5 秒）────────────────────
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
