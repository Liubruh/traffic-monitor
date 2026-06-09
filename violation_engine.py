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
        self.track_ttl = 2.5 # 违法跟踪时间，单位秒，如果一个检测结果在该时间内持续存在且满足条件则视为同一次违法事件，可以避免重复报警
        self.crosswalk_static_secs = 1.5 # 占用斑马线的静止时间阈值，单位秒，如果一个车辆在斑马线上静止超过该时间则认为是占用状态，可以避免由于检测噪声导致的误判
        self.static_move_tol_px = 8.0 # 静止移动容忍度，单位像素，如果一个车辆在斑马线上移动的距离小于该值则认为是静止状态，可以避免由于检测噪声导致的误判
        self.yield_track_len = 10 # 最长跟踪位置个数
        self.yield_threat_ratio = 3.0
        self._last_alert = {} # 上次报警时间字典，键是违法类型加上 track_id 或离散化的 bbox 键，例如 '闯红灯_track_1' 或 '占用斑马线_bbox_10_20_50_100'，值是上次报警的时间戳，可以用于实现冷却时间机制，避免同一辆车在短时间内多次触发同一类型的违法事件
        self._tracked_vio = {} # 违法跟踪字典，键是 track_id，值是一个包含 type（违法类型）和 last_seen（最后一次看到该车辆的时间戳）的字典，例如 {1: {'type': '闯红灯', 'last_seen': t1}, ...}
        self._crosswalk_wait = {} # 斑马线里的车辆序列，键是检测结果的唯一键（例如 track_id 或离散化的 bbox 坐标），值是一个包含 last_center（最后一次看到该检测结果的底部中心点坐标）、last_move（最后一次移动时间戳）和 last_seen（最后一次看到该检测结果的时间戳）的字典，例如 {'track_1': {'last_center': (x, y), 'last_move': t1, 'last_seen': t1}, ...}
        self._vehicle_tracks = {} # 车辆轨迹字典，键是 track_id，值是一个包含 positions（位置列表）和 last_seen（最后一次看到该车辆的时间戳）的字典，例如 {1: {'positions': [(x1, y1, t1), (x2, y2, t2)], 'last_seen': t2}, ...}

    # ── 几何工具（静态方法）──────────────────────────────────────────────

    @staticmethod
    def _bbox_bottom_center(bbox):
        """
        计算边界框的底部中心点坐标，返回一个 (x, y) 元组，其中 x 是边界框左右边界的平均值，y 是边界框的下边界坐标
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)

    @staticmethod
    def _point_in_polygon(pt, polygon):
        """
        判断一个点是否在一个多边形内，使用射线法（ray casting algorithm），通过计算从该点向右水平发出的一条射线与多边形边界的交点数量来判断，如果交点数量是奇数则在内部，偶数则在外部
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
        if lx2 != lx1:# 避免除以零，如果停止线是垂直的则直接比较 y 坐标
            t = (mid_x - lx1) / (lx2 - lx1)
            line_y = ly1 + t * (ly2 - ly1)
        else:
            line_y = ly1
        return by1 <= line_y <= by2

    # ── 基础设施 ──────────────────────────────────────────────────────

    def set_zones(self, zones):
        self.zones = zones or []

    def _det_key(self, det):
        """
        相当于在斑马线停留的时候，给车辆一个临时身份证
        获取一个检测结果的唯一键，用于在内部字典中跟踪该检测结果，优先使用 track_id，如果没有则使用边界框坐标的离散化值
        """
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
        """
            判断一个车辆是否在移动，移动的定义是该车辆在其轨迹中至少有三个位置，并且从第一个位置到最后一个位置的总位移超过静止移动容忍度（static_move_tol_px），这样可以避免由于检测噪声导致的误判
        """
        entry = self._vehicle_tracks.get(track_id)
        if entry is None or len(entry['positions']) < 3:
            return False
        pos = entry['positions']
        """
        计算从第一个位置到最后一个位置的总位移，使用欧几里得距离（hypot）来计算两点之间的距离
        pos[-1][0] - pos[0][0] 是最后一个位置的 x 坐标减去第一个位置的 x 坐标，pos[-1][1] - pos[0][1] 是最后一个位置的 y 坐标减去第一个位置的 y 坐标，np.hypot() 函数会返回这两个差值的欧几里得距离，即 sqrt(dx^2 + dy^2)，表示车辆在轨迹中的总位移
        """

        total_disp = float(np.hypot(pos[-1][0] - pos[0][0], pos[-1][1] - pos[0][1]))
        return total_disp > self.static_move_tol_px

    def _vehicle_direction(self, track_id):
        """
        计算一个车辆的运动方向向量，方向向量是一个单位向量，表示从轨迹的第一个位置指向最后一个位置的方向，如果该车辆没有足够的轨迹信息或者没有移动则返回 (0.0, 0.0)，否则返回 (dx / mag, dy / mag)，其中 dx 和 dy 是最后一个位置与第一个位置的坐标差值，mag 是 dx 和 dy 的欧几里得距离（即向量的模长），加上一个小常数 1e-9 来避免除以零的情况
        """
        entry = self._vehicle_tracks.get(track_id)
        if entry is None or len(entry['positions']) < 2:
            return 0.0, 0.0
        pos = entry['positions']
        dx = pos[-1][0] - pos[0][0]
        dy = pos[-1][1] - pos[0][1]
        mag = float(np.hypot(dx, dy)) + 1e-9
        return dx / mag, dy / mag # sinx cosx

    def _ped_in_vehicle_path(self, track_id, ped_bbox):
        """
        判断一个行人是否在一个车辆的运动路径上，路径的定义是从车辆轨迹的第一个位置到最后一个位置的方向向量所指示的半平面，如果该行人在该半平面内则认为在路径上，否则不在路径上
        """
        vx, vy = self._vehicle_direction(track_id)
        if vx == 0.0 and vy == 0.0:
            return False
        entry = self._vehicle_tracks.get(track_id)
        if entry is None:
            return False
        v_cx, v_cy, _ = entry['positions'][-1] # 获取车辆轨迹中最后一个位置的底部中心点坐标，作为当前车辆的位置
        p_cx, p_cy = self._bbox_bottom_center(ped_bbox)
        to_ped_x = p_cx - v_cx
        to_ped_y = p_cy - v_cy
        return (to_ped_x * vx + to_ped_y * vy) > 0 # 点积大于零表示行人在车辆前方的半平面内

    def _ped_within_threat(self, det_bbox, ped_bbox):
        """
        判断一个行人是否在一个车辆的威胁范围内，威胁范围的定义是以车辆为中心，参考高度（取车辆和行人高度的最大值）乘以 yield_threat_ratio 作为半径的圆形区域，如果行人在该区域内则认为在威胁范围内，否则不在威胁范围内
        """
        v_cx, v_cy = self._bbox_bottom_center(det_bbox)
        p_cx, p_cy = self._bbox_bottom_center(ped_bbox)
        dist = float(np.hypot(p_cx - v_cx, p_cy - v_cy))
        # 第四个参数是边界框的下边界坐标，减去上边界坐标得到边界框的高度，这里取车辆和行人高度的最大值作为参考高度，以适应不同类型的车辆和行人，避免过于严格或宽松的威胁范围判断
        _, _, _, vehicle_h = det_bbox
        _, _, _, ped_h = ped_bbox
        ref_h = max(vehicle_h, ped_h, 1)
        return dist < ref_h * self.yield_threat_ratio

    def _emit(self, det, vio_type, now, ts, violations):
        """
        触发一个违法事件，参数包括：
            det: 机车检测结果对象，包含 class_name、confidence、bbox 等属性
            vio_type: 违法类型字符串，例如 '闯红灯'、'占用斑马线'、'未礼让行人'
            now: 当前时间戳，用于更新内部状态和判断冷却时间
            ts: 当前时间的字符串表示，用于记录违法事件的发生时间
            violations: 违法事件列表，将新的违法事件以字典形式添加到该列表中，字典包含 type（违法类型）、class_name（车辆类别）、confidence（检测置信度）、time（违法发生时间字符串）和 bbox（违法车辆的边界框坐标）

        """
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
        """
        应用粘滞效果，如果一个检测结果在之前的违法跟踪字典 _tracked_vio 中有对应的 track_id，并且该违法事件在 track_ttl 时间内持续存在，
        则将该检测结果的 violation 属性设置为之前的违法类型，这样可以保持同一辆车在短时间内持续触发同一类型的违法事件，而不需要每帧都重新判断是否满足条件
        """
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
            cw是斑马线区域的字典，包含 type='crosswalk' 和对应的几何信息，例如 pts（多边形顶点列表），可以通过 cw.get('pts', []) 来获取斑马线区域的多边形顶点列表，如果该键不存在则返回空列表，这样可以避免 KeyError 异常
            对于每个斑马线区域，找出所有在该区域内的行人检测结果，存储在 ped_on_cw 字典中，键是斑马线索引，值是行人检测结果列表
            enumerate(crosswalks) 会返回一个包含索引和值的迭代器，例如 [(0, cw1), (1, cw2), ...]，其中 cw1、cw2 是斑马线区域的字典，索引值 i 可以用来在 ped_on_cw 中存储对应的行人列表
            cw.get('pts', []) 可以获取斑马线区域的多边形顶点列表，如果该键不存在则返回空列表，这样可以避免 KeyError 异常
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
                # any() 函数用于判断一个可迭代对象中是否至少有一个元素满足条件，如果满足条件则返回 True，否则返回 False，这里用来判断当前检测结果是否在任何一个斑马线区域内
                in_cw = any(self._bbox_overlaps_polygon(det.bbox, cw.get('pts', []))
                            for cw in crosswalks)
                det_key = self._det_key(det)

                if in_cw:
                    cx, cy = self._bbox_bottom_center(det.bbox)
                    state = self._crosswalk_wait.get(det_key)
                    if state is None:
                        self._crosswalk_wait[det_key] = {
                            'last_center': (cx, cy),
                            'last_move': now, # 记录最后一次移动的时间戳，初始值为当前时间，这样可以让占用斑马线的状态在连续帧中持续显示，提升用户体验，同时也可以避免由于检测结果消失导致的误判
                            'last_seen': now, # 记录最后一次看到该检测结果的时间戳，用于后续清理过期数据，这样可以避免由于检测结果消失导致的误判，同时也可以让占用斑马线的状态在连续帧中持续显示，提升用户体验
                        }
                    else:
                        px, py = state['last_center']
                        if float(np.hypot(cx - px, cy - py)) > self.static_move_tol_px: # 如果当前底部中心点与上次记录的底部中心点之间的距离超过静止移动容忍度，则认为车辆发生了移动，更新 last_move 时间戳和 last_center 坐标
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
                self._emit(det, vio_type, now, ts, violations) # 触发违法事件并添加到 violations 列表中
            else: # 如果当前帧没有检测到新的违法事件，则尝试应用粘滞效果来保持之前的违法状态，这样可以避免同一辆车在短时间内多次触发同一类型的违法事件，同时也可以让违法状态在连续帧中持续显示，提升用户体验
                self._apply_sticky(det, now)

        return violations
