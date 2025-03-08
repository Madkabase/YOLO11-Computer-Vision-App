# Weights Directory

This directory is for storing the YOLO11 model weights used by the application.

Due to file size limitations, the model weights are not included in this repository. You need to download them separately.

## Required Model Weights

The application requires the following model weights:
- `yolo11n.pt` (Detection model)
- `yolo11n-seg.pt` (Segmentation model)
- `yolo11n-pose.pt` (Pose Estimation model)

## How to Download

You can download the model weights from the [Ultralytics YOLO11 releases](https://github.com/ultralytics/ultralytics/releases) and place them in this directory.

Alternatively, you can use the following commands to download the models:

```bash
# Create weights directory (if it doesn't exist)
mkdir -p weights

# Download models
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').save('weights/yolo11n.pt')"
python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt').save('weights/yolo11n-seg.pt')"
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt').save('weights/yolo11n-pose.pt')"
```