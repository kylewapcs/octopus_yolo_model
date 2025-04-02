# Octopus YOLOv8 Segmentation Model

This repository contains code for training a YOLOv8 segmentation model to detect and segment octopuses in images.

## Project Structure

```
train_yolo_hacker/
├── datasets/
│   └── octopus_segmentation/
│       ├── train/
│       │   ├── images/      # Training images
│       │   └── labels/      # Training labels
│       ├── valid/
│       │   ├── images/      # Validation images
│       │   └── labels/      # Validation labels
│       ├── test/
│       │   ├── images/      # Test images
│       │   └── labels/      # Test labels
│       └── data.yaml        # Dataset configuration
└── train_segmentation.py    # Training script
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/kylewapcs/octopus_yolo_model.git
cd octopus_yolo_model
```

2. Install dependencies:

```bash
pip install ultralytics torch
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

## Dataset

The dataset contains approximately 2000 images of octopuses with corresponding segmentation masks. The images are split into training, validation, and test sets.

## Model

We use YOLOv8, specifically the medium-sized model (YOLOv8m) for instance segmentation. The model is trained to detect and segment octopuses in images.

## Results

Training results and model checkpoints will be saved in the `runs/segment/` directory.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
