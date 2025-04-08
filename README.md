# Octopus YOLOv8 Segmentation Model

This repository contains code for training a YOLOv8 segmentation model to detect and segment octopuses in underwater images.

## Quick Start Guide

1. **Dataset Setup**

   - Place all dataset zip files in the `octopus_data` folder
   - Run the data preparation script:

   ```bash
   python prepare_octopus_data.py
   ```

   - This will create the required YOLO format dataset in `datasets/octopus_dataset`

2. **Environment Setup**

   ```bash
   pip install -r requirements.txt
   ```

3. **Training**

   ```bash
   python trainer.py
   ```

   Note: Training requires a GPU for reasonable performance

4. **Using the Trained Model**
   After training, the model will be saved in `runs/segment/train/weights/`:

   - `best.pt`: Best model weights (recommended for inference)
   - `last.pt`: Latest model weights

   To use the trained model for predictions:

   ```bash
   python predict_segmentation.py --source path/to/image.jpg --model runs/segment/train/weights/best.pt --output predictions
   ```

## Project Overview

This project uses YOLOv8m-seg (medium-sized model) for octopus instance segmentation:

- Input size: 640x640
- Backbone: CSPDarknet
- Neck: PANet
- Head: Segmentation head with mask prediction

## Project Structure

```
octopus_yolo_model/
├── octopus_data/           # Place dataset zip files here
├── datasets/
│   └── octopus_dataset/    # Processed dataset (created after running prepare_octopus_data.py)
│       ├── train/          # Training images and labels
│       ├── valid/          # Validation images and labels
│       └── test/           # Test images and labels
├── runs/
│   └── segment/
│       └── train/
│           └── weights/    # Trained model weights (created after training)
│               ├── best.pt # Best model weights
│               └── last.pt # Latest model weights
├── prepare_octopus_data.py # Script to prepare and format the dataset
├── trainer.py              # Main training script
├── predict_segmentation.py # Prediction & visualization script
├── visualize_results_of_training_run.py  # Script to visualize training results
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Dataset

The dataset contains underwater images:

- Training set: 80% of images
- Validation set: 10% of images
- Test set: 10% of images

## Training

The training script (`trainer.py`) includes:

- YOLOv8m-seg model configuration
- Training parameters optimization
- Data augmentation
- Model checkpointing
- Training progress visualization

To visualize training results:

```bash
python visualize_results_of_training_run.py
```

## Making Predictions

After training, you can run predictions:

```bash
python predict_segmentation.py --source path/to/image.jpg --model runs/segment/train/weights/best.pt --output predictions
```

Parameters:

- `--source`: Path to image or directory of images
- `--model`: Path to trained model weights (default: runs/segment/train/weights/best.pt)
- `--conf`: Confidence threshold (default: 0.25)
- `--output`: Output directory for visualizations

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (required for training)
- Ultralytics
- OpenCV

All dependencies are listed in `requirements.txt`.

## Notes

- Training requires a GPU for reasonable performance
- The dataset consists of underwater images split into train/valid/test sets
- Training progress and results will be saved in the `runs` directory
- Model checkpoints are saved during training in `runs/segment/train/weights/`
- Use `best.pt` for inference as it contains the best performing model weights

## License

This project is licensed under the MIT License.
