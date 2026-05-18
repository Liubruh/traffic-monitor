"""
violation_engine.py — 交通违法检测引擎

支持：闯红灯 / 占用斑马线 / 未礼让行人
"""

import numpy as np
import time
from datetime import datetime


class ViolationEngine:

    VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

    def __init__(self):
        self.zones = []
        self.cooldown = 4.0
        self.track_ttl = 2.5
        self.crosswalk_static_secs = 1.5
        self.static_move_tol_px = 8.0
        self.yield_track_len = 10
        self.yield_threat_ratio = 3.0
        self._last_alert = {}
        self._tracked_vio = {}
        self._crosswalk_wait = {}
        self._vehicle_tracks = {}

    # ── 几何工具（静态方法）──────────────────────────────────────────────

    @staticmethod
    def _bbox_bottom_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)

    @staticmethod
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

    @classmethod
    def _bbox_overlaps_polygon(cls, bbox, polygon):
        x1, y1, x2, y2 = bbox
        return any(cls._point_in_polygon(p, polygon) for p in [
            cls._bbox_bottom_center(bbox),
            ((x1 + x2) // 2, (y1 + y2) // 2),
        ])

    @staticmethod
    def _stop_line_endpoints(sl):
        pts = sl.get('pts')
        if pts and len(pts) >= 2:
            return pts[0][0], pts[0][1], pts[1][0], pts[1][1]
        y = sl.get('y', 0)
        return sl.get('x1', 0), y, sl.get('x2', 9999), y

    @classmethod
    def _bbox_crosses_stop_line(cls, bbox, sl):
        bx1, by1, bx2, by2 = bbox
        lx1, ly1, lx2, ly2 = cls._stop_line_endpoints(sl)
        mid_x = (bx1 + bx2) // 2
        if not (min(lx1, lx2) <= mid_x <= max(lx1, lx2)):
            return False
        if lx2 != lx1:
            t = (mid_x - lx1) / (lx2 - lx1)
            line_y = ly1 + t * (ly2 - ly1)
        else:
            line_y = ly1
        return by1 <= line_y <= by2

    # ── 基础设施 ──────────────────────────────────────────────────────

    def set_zones(self, zones):
        self.zones = zones or []

    def _det_key(self, det):
        track_id = getattr(det, 'track_id', None)
        if track_id is not None:
            return f'track_{track_id}'
        x1, y1, x2, y2 = det.bbox
        return f'bbox_{x1 // 80}_{y1 // 80}_{x2 // 80}_{y2 // 80}'

    def _cleanup_dict(self, d, now):
        stale = [k for k, v in d.items() if now - v['last_seen'] > self.track_ttl]
        for k in stale:
            d.pop(k, None)

    def _cleanup(self, now):
        self._cleanup_dict(self._tracked_vio, now)
        self._cleanup_dict(self._crosswalk_wait, now)
        self._cleanup_dict(self._vehicle_tracks, now)

    # ── 未礼让行人辅助 ──────────────────────────────────────────────

    def _update_vehicle_track(self, det, now):
        track_id = getattr(det, 'track_id', None)
        if track_id is None:
            return
        cx, cy = self._bbox_bottom_center(det.bbox)
        entry = self._vehicle_tracks.get(track_id)
        if entry is None:
            entry = {'positions': [], 'last_seen': now}
            self._vehicle_tracks[track_id] = entry
        entry['positions'].append((cx, cy, now))
        if len(entry['positions']) > self.yield_track_len:
            entry['positions'].pop(0)
        entry['last_seen'] = now
        return track_id

    def _vehicle_is_moving(self, track_id):
        entry = self._vehicle_tracks.get(track_id)
        if entry is None or len(entry['positions']) < 3:
            return False
        pos = entry['positions']
        total_disp = float(np.hypot(pos[-1][0] - pos[0][0], pos[-1][1] - pos[0][1]))
        return total_disp > self.static_move_tol_px

    def _vehicle_direction(self, track_id):
        entry = self._vehicle_tracks.get(track_id)
        if entry is None or len(entry['positions']) < 2:
            return 0.0, 0.0
        pos = entry['positions']
        dx = pos[-1][0] - pos[0][0]
        dy = pos[-1][1] - pos[0][1]
        mag = float(np.hypot(dx, dy)) + 1e-9
        return dx / mag, dy / mag

    def _ped_in_vehicle_path(self, track_id, ped_bbox):
        vx, vy = self._vehicle_direction(track_id)
        if vx == 0.0 and vy == 0.0:
            return False
        entry = self._vehicle_tracks.get(track_id)
        if entry is None:
            return False
        v_cx, v_cy, _ = entry['positions'][-1]
        p_cx, p_cy = self._bbox_bottom_center(ped_bbox)
        to_ped_x = p_cx - v_cx
        to_ped_y = p_cy - v_cy
        return (to_ped_x * vx + to_ped_y * vy) > 0

    def _ped_within_threat(self, det_bbox, ped_bbox):
        v_cx, v_cy = self._bbox_bottom_center(det_bbox)
        p_cx, p_cy = self._bbox_bottom_center(ped_bbox)
        dist = float(np.hypot(p_cx - v_cx, p_cy - v_cy))
        _, _, _, vehicle_h = det_bbox
        _, _, _, ped_h = ped_bbox
        ref_h = max(vehicle_h, ped_h, 1)
        return dist < ref_h * self.yield_threat_ratio

    def _emit(self, det, vio_type, now, ts, violations):
        det.violation = vio_type
        track_id = getattr(det, 'track_id', None)

        if track_id is not None:
            self._tracked_vio[track_id] = {'type': vio_type, 'last_seen': now}
            key = f'{vio_type}_track_{track_id}'
        else:
            key = f'{vio_type}_{det.bbox[0] // 80}_{det.bbox[1] // 80}'

        if now - self._last_alert.get(key, 0) > self.cooldown:
            self._last_alert[key] = now
            violations.append({
                'type':       vio_type,
                'class_name': det.class_name,
                'confidence': det.confidence,
                'time':       ts,
                'bbox':       det.bbox,
            })

    def _apply_sticky(self, det, now):
        track_id = getattr(det, 'track_id', None)
        if track_id is None:
            return
        sticky = self._tracked_vio.get(track_id)
        if sticky:
            sticky['last_seen'] = now
            if det.violation is None:
                det.violation = sticky['type']

    # ── 检测主入口 ────────────────────────────────────────────────────

    def check(self, results, light_color='unknown'):
        now = time.time()
        ts = datetime.now().strftime('%H:%M:%S')
        self._cleanup(now)

        stop_lines = [z for z in self.zones if z.get('type') == 'stop_line']
        crosswalks = [z for z in self.zones if z.get('type') == 'crosswalk']

        ped_on_cw = {}
        for i, cw in enumerate(crosswalks):
            pts = cw.get('pts', [])
            if pts:
                ped_on_cw[i] = [d for d in results
                                if d.class_name == 'person'
                                and self._bbox_overlaps_polygon(d.bbox, pts)]

        violations = []

        for det in results:
            if det.class_name not in self.VEHICLE_CLASSES:
                continue

            track_id = self._update_vehicle_track(det, now)
            vio_type = None

            # 1. 闯红灯
            if light_color == 'red':
                for sl in stop_lines:
                    if self._bbox_crosses_stop_line(det.bbox, sl):
                        vio_type = '闯红灯'
                        break

            # 2. 占用斑马线（静止超过阈值）
            if vio_type is None:
                in_cw = any(self._bbox_overlaps_polygon(det.bbox, cw.get('pts', []))
                            for cw in crosswalks)
                det_key = self._det_key(det)

                if in_cw:
                    cx, cy = self._bbox_bottom_center(det.bbox)
                    state = self._crosswalk_wait.get(det_key)
                    if state is None:
                        self._crosswalk_wait[det_key] = {
                            'last_center': (cx, cy),
                            'last_move': now,
                            'last_seen': now,
                        }
                    else:
                        px, py = state['last_center']
                        if float(np.hypot(cx - px, cy - py)) > self.static_move_tol_px:
                            state['last_move'] = now
                        state['last_center'] = (cx, cy)
                        state['last_seen'] = now
                        if now - state['last_move'] >= self.crosswalk_static_secs:
                            vio_type = '占用斑马线'
                else:
                    self._crosswalk_wait.pop(det_key, None)

            # 3. 未礼让行人
            if vio_type is None and crosswalks:
                for i, cw in enumerate(crosswalks):
                    peds = ped_on_cw.get(i)
                    if not peds:
                        continue
                    if not self._bbox_overlaps_polygon(det.bbox, cw.get('pts', [])):
                        continue
                    if track_id is None or not self._vehicle_is_moving(track_id):
                        continue
                    for ped in peds:
                        if self._ped_in_vehicle_path(track_id, ped.bbox) and \
                           self._ped_within_threat(det.bbox, ped.bbox):
                            vio_type = '未礼让行人'
                            break
                    if vio_type:
                        break

            if vio_type:
                self._emit(det, vio_type, now, ts, violations)
            else:
                self._apply_sticky(det, now)

        return violations
