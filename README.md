# Traffic Monitor — YOLOv8

Real-time traffic violation detection with a web dashboard. Detects red light running, lane crossing, crosswalk obstruction, and failure to yield to pedestrians.

## Features

- **4 violation types**: red light running, lane crossing, crosswalk obstruction, failure to yield
- **Multi-source video**: webcam, video file, or RTSP stream
- **Multi-backend inference**: auto-selects CUDA → MPS → OpenVINO → CPU
- **Object tracking**: ByteTrack for persistent vehicle/pedestrian IDs across frames
- **Traffic light color recognition**: HSV-based red/green/yellow detection with temporal smoothing
- **Web dashboard**: live MJPEG feed, zone configuration, alert log, real-time metrics

## Model

| Metric | Value |
|--------|-------|
| Architecture | YOLOv8s |
| mAP50 | 81.5% |
| mAP50-95 | 64.0% |
| Classes | 7 custom traffic violation classes |

## Installation

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
flask>=2.3.0
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
```

Optional for GPU acceleration:
- NVIDIA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- Intel: `pip install openvino`

## Usage

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/video_feed` | GET | MJPEG video stream |
| `/api/start` | POST | Start detection |
| `/api/stop` | POST | Stop detection |
| `/api/set_zones` | POST | Configure detection zones (JSON) |
| `/api/status` | GET | Metrics, FPS, alert log |

### Zone Configuration

Send a JSON body to `/api/set_zones` to define:
- `stop_line` — line segment for red-light detection
- `lane_lines` — list of segments for lane-crossing detection
- `crosswalk` — polygon for crosswalk/yield detection
- `traffic_light_roi` — bounding box for the traffic light region

## Project Structure

```
yolo_project/
├── app.py                  # Flask server & detection loop
├── detector.py             # YOLOv8 inference + ByteTrack
├── violation_engine.py     # Violation geometry logic
├── pedestrian_yield_engine.py  # Pedestrian yield detection
├── video_source.py         # Camera / file / RTSP input
├── metrics.py              # FPS, counts, alert logging
├── train_s.py              # Fine-tuning script
├── data.yaml               # Dataset config
├── yolov8s.pt              # Pre-trained base model
├── templates/index.html    # Web UI
└── runs/                   # Training outputs & best weights
```

## Training

To fine-tune on a custom dataset:

```bash
python train_s.py
```

Edit `data.yaml` to point to your dataset. Output weights are saved to `runs/detect/runs/traffic/yolov8s_baseline/weights/best.pt`.
