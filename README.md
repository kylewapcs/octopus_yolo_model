# Octopus YOLOv8 Segmentation Model

This repository contains code for training a YOLOv8 segmentation model to detect and segment octopuses in underwater images.

## Dataset

The dataset contains 2000 underwater images:

- ~780 images with octopus annotations (segmentation masks)
- The remaining images are negative examples (no octopus)
- Split into training (80%), validation (10%), and test (10%) sets

## Model

We use YOLOv8m-seg (medium-sized model) for instance segmentation:

- Input size: 640x640
- Backbone: CSPDarknet
- Neck: PANet
- Head: Segmentation head with mask prediction
- Training: 100 epochs with AdamW optimizer

## Project Structure

```
train_yolo_hacker/
├── datasets/
│   └── octopus_segmentation/
│       ├── train/
│       │   ├── images/      # Training images (1600 images)
│       │   └── labels/      # Training labels (segmentation masks)
│       ├── valid/
│       │   ├── images/      # Validation images (200 images)
│       │   └── labels/      # Validation labels
│       ├── test/
│       │   ├── images/      # Test images (200 images)
│       │   └── labels/      # Test labels
│       └── data.yaml        # Dataset configuration
├── train_segmentation.py    # Training script
├── predict_segmentation.py  # Prediction & visualization script
└── requirements.txt        # Dependencies
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/kylewapcs/octopus_yolo_model.git
cd octopus_yolo_model
```

2. Install dependencies:

```bash
pip install ultralytics torch opencv-python
```

3. Configure dataset paths in `datasets/octopus_segmentation/data.yaml`:

```yaml
names:
  - octopus
nc: 1
test: C:/Users/klipk/train_yolo_hacker/datasets/octopus_segmentation/test/images
train: C:/Users/klipk/train_yolo_hacker/datasets/octopus_segmentation/train/images
val: C:/Users/klipk/train_yolo_hacker/datasets/octopus_segmentation/valid/images
```

## Training

To train the model:

```bash
python train_segmentation.py --model-size m --epochs 100 --batch-size 16
```

### Training Parameters

- `--model-size`: YOLOv8 model size (n, s, m, l, x)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--imgsz`: Input image size (default: 640)
- `--device`: Device to train on (cuda device or cpu)

## Making Predictions

To run predictions and visualize results:

```bash
python predict_segmentation.py --source path/to/image.jpg --model runs/segment/train/weights/best.pt --output predictions
```

Parameters:

- `--source`: Path to image or directory of images
- `--model`: Path to trained model weights
- `--conf`: Confidence threshold (default: 0.25)
- `--output`: Output directory for visualizations

The script will:

1. Run prediction on the input image(s)
2. Draw segmentation masks in green
3. Save visualizations to the output directory

## Results

Training results and model checkpoints are saved in `runs/segment/train/`:

- `weights/best.pt`: Best model weights
- `weights/last.pt`: Last epoch weights
- Training plots and metrics in the same directory

## Model Performance

The model achieves:

- Training set: [metrics to be added after training]
- Validation set: [metrics to be added after training]
- Test set: [metrics to be added after training]

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
