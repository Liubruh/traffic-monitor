import cv2
import time


class VideoSource:
    def __init__(self, source_type, source_value):
        self.source_type = source_type  # 'camera', 'file', 'rtsp'
        self.source_value = source_value
        self.cap = None
        self.fps = 30

    def open(self):
        try:
            if self.source_type == 'camera':
                idx = int(self.source_value) if str(self.source_value).isdigit() else 0
                self.cap = cv2.VideoCapture(idx)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            elif self.source_type in ('file', 'rtsp'):
                self.cap = cv2.VideoCapture(self.source_value)
                if self.source_type == 'file':
                    self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            else:
                return False

            return self.cap.isOpened()
        except Exception as e:
            print(f'[VideoSource] 打开失败: {e}')
            return False

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def reset(self):
        """循环播放视频文件"""
        if self.cap and self.source_type == 'file':
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_info(self):
        if self.cap is None:
            return {}
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'source_type': self.source_type,
        }
