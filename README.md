# Octopus Detection Model Training

This repository contains the code for training a YOLO-based octopus detection model.

## Quick Start Guide

1. **Dataset Setup**

   - You will receive 8 zip files containing the dataset
   - Place all zip files in the `octopus_data` folder
   - Run `python prepare_octopus_data.py` to prepare the dataset
   - This will create the required YOLO format dataset in `datasets/octopus_dataset`

2. **Environment Setup**

   ```bash
   pip install -r requirements.txt
   ```

    3. **Training**
    ```bash
    python trainer.py
   ```

## Project Structure

- `prepare_octopus_data.py`: Script to prepare and format the dataset
- `trainer.py`: Main training script
- `visualize_results_of_training_run.py`: Script to visualize training results
- `datasets/octopus_dataset/`: Contains the processed dataset (will be created after running prepare_octopus_data.py)
- `octopus_data/`: Place the 8 zip files here

## Requirements

- Python 3.8+
- PyTorch
- YOLOv5
- Other dependencies listed in requirements.txt

## Notes

- Training requires a GPU for reasonable performance
- The dataset consists of 2000 images split into train/valid/test sets
- Training progress and results will be saved in the `runs` directory
