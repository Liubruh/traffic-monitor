import cv2
import time


class VideoSource:
    def __init__(self, source_type, source_value):
        self.source_type = source_type  # 'camera', 'file', 'rtsp'
        self.source_value = source_value
        self.cap = None
        self.fps = 30

    def open(self):
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

    def read(self):
        """
        cap.read() 返回两个值，第一个是布尔值，表示是否成功读取到帧；第二个是读取到的帧图像（如果成功的话）
        frame格式例如：
            frame.shape = (720, 1280, 3) # 高、宽、通道数
            frame.dtype = uint8 # 数据类型为无符号8位整数
            frame = np.array([
                [[B, G, R], [B, G, R], ..., [B, G, R]],  # 第1行像素的BGR值
                [[B, G, R], [B, G, R], ..., [B, G, R]],  # 第2行像素的BGR值
                ...,
                [[B, G, R], [B, G, R], ..., [B, G, R]],  # 第720行像素的BGR值
            ])本质三位数组，每个元素是一个包含蓝绿红三个通道值的列表，范围0~255，表示该像素的颜色信息。
        """
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


