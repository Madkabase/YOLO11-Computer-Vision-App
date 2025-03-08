# YOLO11 Computer Vision Web App

This is a Streamlit web application that uses YOLO11 (You Only Look Once) models for computer vision tasks including:
- Object Detection
- Image Segmentation
- Pose Estimation

## Features

- Upload and process custom images
- Process sample videos
- Adjust confidence threshold
- View detection results

## Requirements

- Python 3.8+
- Ultralytics YOLO11
- Streamlit
- OpenCV
- PIL (Pillow)

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Model Weights

This application requires YOLO11 model weights. Due to file size limitations, the model weights are not included in this repository. You need to download them separately:

1. Create a `weights` directory in the project root
2. Download the following model weights from the [Ultralytics YOLO11 releases](https://github.com/ultralytics/ultralytics/releases) and place them in the `weights` directory:
   - `yolo11n.pt` (Detection model)
   - `yolo11n-seg.pt` (Segmentation model)
   - `yolo11n-pose.pt` (Pose Estimation model)

Alternatively, you can use the following commands to download the models:

```bash
# Create weights directory
mkdir -p weights

# Download models
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').save('weights/yolo11n.pt')"
python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt').save('weights/yolo11n-seg.pt')"
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt').save('weights/yolo11n-pose.pt')"
```

## Usage

Run the Streamlit app with:

```bash
cd YOLO11_Streamlit
streamlit run main.py
```

Or using the Python module approach:

```bash
python -m streamlit run main.py
```

The web interface will open in your default browser.

## Project Structure

- `main.py`: The main Streamlit application
- `images/`: Sample images and detection results
- `videos/`: Sample videos for processing
- `weights/`: Pre-trained YOLO11 model weights (you need to download these separately)

## Troubleshooting

If you encounter the error "streamlit is not recognized as a command", use the Python module approach:

```bash
python -m streamlit run main.py
```

If you see deprecation warnings about `use_column_width`, these have been fixed in the latest version by using `use_container_width` instead.