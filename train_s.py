from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    workers=4,
    patience=40,
    project='runs/traffic',
    name='yolov8s_baseline',
    amp=False,
    cache=False,
    cos_lr=True,
    close_mosaic=10,
)