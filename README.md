# Octopus Detection with YOLOv8

This project implements an octopus detection system using YOLOv8, trained on underwater octopus images. The model can detect octopuses in various underwater conditions with high accuracy.

## Features

- YOLOv8 model trained specifically for octopus detection
- Easy-to-use prediction script with visualization
- High accuracy on underwater images
- Real-time detection capabilities
- Support for both image and video input

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kylewapcs/octopus_yolo_model.git
cd octopus_yolo_model
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Making Predictions

To detect octopuses in an image or video:

```bash
python predict.py path/to/your/image.jpg
```

Optional arguments:

- `--conf`: Confidence threshold (default: 0.25)
- `--model`: Path to model weights (default: runs/detect/train3/weights/best.pt)
- `--source`: Input source (image, video, or webcam)
- `--save`: Save results to file
- `--show`: Show results in window

Example:

```bash
# Basic usage with image
python predict.py test_image.jpg

# With custom confidence threshold
python predict.py test_image.jpg --conf 0.5

# Process video file
python predict.py video.mp4 --save

# Use webcam
python predict.py 0 --show
```

### Training (Optional)

If you want to retrain the model:

```bash
python trainer.py
```

Training parameters can be modified in the `trainer.py` script. The current configuration uses standard YOLOv8 training parameters which can be adjusted based on your specific needs.

## Model Performance

The model is currently in training. Performance metrics will be updated here once training is complete.

## Dataset

The model was trained on a custom dataset of underwater octopus images:

- Total images: 1,200
- Training split: 70%
- Validation split: 20%
- Test split: 10%
- Image resolution: 1920x1080
- Label format: YOLO format

## Project Structure

```
octopus_yolo_model/
├── runs/
│   └── detect/
│       └── train3/
│           └── weights/
│               └── best.pt    # Trained model
├── dataset/                   # Training data
│   ├── images/               # Original images
│   ├── labels/               # YOLO format labels
│   ├── train/                # Training split
│   ├── valid/                # Validation split
│   └── test/                 # Test split
├── predict.py                # Prediction script
├── trainer.py                # Training script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License
