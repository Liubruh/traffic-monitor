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
        self.zones = [] # 监控区域列表，每个元素是一个字典，包含 type（'stop_line' 或 'crosswalk'）和对应的几何信息，例如 pts（多边形顶点列表）或坐标参数
        self.cooldown = 4.0
        self.track_ttl = 2.5
        self.crosswalk_static_secs = 1.5
        self.static_move_tol_px = 8.0
        self.yield_track_len = 10 # 最长跟踪位置个数
        self.yield_threat_ratio = 3.0
        self._last_alert = {}
        self._tracked_vio = {}
        self._crosswalk_wait = {}
        self._vehicle_tracks = {} # 车辆轨迹字典，键是 track_id，值是一个包含 positions（位置列表）和 last_seen（最后一次看到该车辆的时间戳）的字典，例如 {1: {'positions': [(x1, y1, t1), (x2, y2, t2)], 'last_seen': t2}, ...}

    # ── 几何工具（静态方法）──────────────────────────────────────────────

    @staticmethod
    def _bbox_bottom_center(bbox):
        """
        计算边界框的底部中心点坐标，返回一个 (x, y) 元组，其中 x 是边界框左右边界的平均值，y 是边界框的下边界坐标
        这种方法适用于检测车辆是否进入了斑马线区域或是否越过了停止线，因为车辆的底部中心点通常是最接近地面的部分，更能反映车辆的位置和是否进入了特定区域，而不像边界框的中心点可能会因为车辆的高度和姿态而产生较大偏移，尤其是在斜视图或车辆较高的情况下
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)

    @staticmethod
    def _point_in_polygon(pt, polygon):
        """
        判断一个点是否在一个多边形内，使用射线法（ray casting algorithm），通过计算从该点向右水平发出的一条射线与多边形边界的交点数量来判断，如果交点数量是奇数则在内部，偶数则在外部
        这种方法适用于任意形状的多边形，包括凸多边形和凹多边形，但不适用于自交多边形（例如8字形）
        """
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
    def _bbox_overlaps_polygon(cls, bbox, polygon): # cls 是类方法的约定参数，表示当前类对象，可以通过 cls.方法名() 来调用其他类方法，例如 cls._point_in_polygon() 和 cls._bbox_bottom_center()，这样可以保持代码的组织性和可读性
        """
        判断一个边界框是否与一个多边形重叠，重叠的定义是边界框的底部中心点或者中心点在多边形内
        这种方法简单且效率较高，适用于检测车辆是否进入了斑马线区域或是否越过了停止线，虽然可能存在一些误判（例如车辆部分进入斑马线但底部中心点未进入），但在实际应用中通常已经足够使用
        """
        x1, y1, x2, y2 = bbox
        return any(cls._point_in_polygon(p, polygon) for p in [
            cls._bbox_bottom_center(bbox),
            ((x1 + x2) // 2, (y1 + y2) // 2),
        ])

    @staticmethod
    def _stop_line_endpoints(sl):
        """
        获取停止线的两个端点坐标，返回一个 (x1, y1, x2, y2) 元组，其中 (x1, y1) 是停止线的第一个端点坐标，(x2, y2) 是停止线的第二个端点坐标
        本函数只接受 pts（多边形顶点列表）格式；如果不存在 pts 则返回 None。
        """
        pts = sl.get('pts')
        if pts and len(pts) >= 2:
            return pts[0][0], pts[0][1], pts[1][0], pts[1][1]
        return None

    @classmethod
    def _bbox_crosses_stop_line(cls, bbox, sl):
        """
        bbox： 机车检测框
        sl: 停止线区域，包含 type='stop_line' 和对应的几何信息，例如 pts（多边形顶点列表）
        闯红灯判断：
            判断一个边界框是否越过了一个停止线，越过的定义是边界框的底部中心点在停止线的水平范围内，并且在停止线的下方（假设停止线是水平的，且车辆从上方进入交叉口）
        """
        bx1, by1, bx2, by2 = bbox
        endpoints = cls._stop_line_endpoints(sl)
        if endpoints is None:
            return False
        lx1, ly1, lx2, ly2 = endpoints
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
        """
        det is 机车
        更新车辆轨迹信息，返回该检测结果的 track_id（如果有的话），并在内部维护一个车辆轨迹字典 _vehicle_tracks，键是 track_id，值是一个包含 positions（位置列表）和 last_seen（最后一次看到该车辆的时间戳）的字典
        每当检测到一个车辆结果时，都会调用该方法来更新其轨迹信息：
            - 首先获取该检测结果的 track_id，如果没有则返回 None
            - 计算该检测结果的底部中心点坐标 (cx, cy)
            - 在 _vehicle_tracks 中查找该 track_id 的轨迹信息，如果没有则创建一个新的条目，初始位置列表为空，last_seen 设置为当前时间
            - 将当前的底部中心点坐标和时间戳添加到位置列表中，如果位置列表长度超过 yield_track_len 则移除最旧的位置
            - 更新 last_seen 为当前时间
            - 返回 track_id
        """
        track_id = getattr(det, 'track_id', None) # getattr() 函数用于获取对象的属性值，第一个参数是对象，第二个参数是属性名称字符串，如果该属性不存在则返回 None（或者可以指定一个默认值），这样可以避免直接访问 det.track_id 时可能出现的 AttributeError 异常
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
        ts = datetime.now().strftime('%H:%M:%S') # 可以改成更精确的时间格式，例如 '%Y-%m-%d %H:%M:%S.%f'，以包含日期和毫秒
        self._cleanup(now)

        stop_lines = [z for z in self.zones if z.get('type') == 'stop_line'] # list.get('type') 可以避免 KeyError，如果该键不存在则返回 None，不会抛出异常，这样可以更健壮地处理输入数据
        crosswalks = [z for z in self.zones if z.get('type') == 'crosswalk']

        ped_on_cw = {}
        """
        代表在每个斑马线区域内的行人检测结果，键是斑马线索引，值是行人检测结果列表，
        例如 {0: [ped1, ped2], 1: [ped3]}，其中 ped1、ped2、ped3 是检测结果对象，YOLODetector类型
        """
        for i, cw in enumerate(crosswalks): 
            """
            对于每个斑马线区域，找出所有在该区域内的行人检测结果，存储在 ped_on_cw 字典中，键是斑马线索引，值是行人检测结果列表 
            enumerate(crosswalks) 会返回一个包含索引和值的迭代器，例如 [(0, cw1), (1, cw2), ...]，其中 cw1、cw2 是斑马线区域的字典，索引值 i 可以用来在 ped_on_cw 中存储对应的行人列表
            cw.get('pts', []) 可以获取斑马线区域的 **多边形顶点列表**，如果该键不存在则返回空列表，这样可以避免 KeyError 异常
            """ 
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
