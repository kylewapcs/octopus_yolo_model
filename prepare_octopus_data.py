import os
import shutil
import zipfile
import random
from pathlib import Path
import yaml
import logging
from logging.handlers import RotatingFileHandler
import sys
import xml.etree.ElementTree as ET
import numpy as np

def setup_logging(log_file: str = "prepare_octopus_data.log", log_level: int = logging.INFO):
    """Sets up logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def create_directory_structure(base_path: str):
    """Creates the required directory structure for the dataset."""
    dirs = [
        'train/images', 'train/labels',
        'valid/images', 'valid/labels',
        'test/images', 'test/labels'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        logging.info(f"Created directory: {full_path}")

def extract_zip_files(zip_dir: str, extract_dir: str):
    """Extracts all zip files from the source directory."""
    zip_files = [f for f in os.listdir(zip_dir) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError(f"No zip files found in {zip_dir}")
    
    logging.info(f"Found {len(zip_files)} zip files: {', '.join(zip_files)}")
    
    for zip_file in zip_files:
        zip_path = os.path.join(zip_dir, zip_file)
        batch_dir = os.path.join(extract_dir, f"batch{zip_file.split('.')[0][-1]}")
        os.makedirs(batch_dir, exist_ok=True)
        
        logging.info(f"Extracting {zip_file} to {batch_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(batch_dir)
            
            # Log what was extracted
            files = zip_ref.namelist()
            logging.info(f"Found {len(files)} files in {zip_file}")
            if files:
                logging.info(f"First 5 files: {files[:5]}")
        
        xml_file = os.path.join(batch_dir, 'annotations.xml')
        if os.path.exists(xml_file):
            logging.info(f"Found annotations.xml in {zip_file}")
        else:
            logging.warning(f"No annotations.xml found in {zip_file}")

def convert_polygon_to_bbox(points_str):
    """Convert polygon points to YOLO format bounding box."""
    # Parse points string into x,y coordinates
    points = []
    for point in points_str.split(';'):
        if ',' in point:
            x, y = map(float, point.split(','))
            points.append((x, y))
    
    if not points:
        return None
        
    # Convert to numpy array for easier manipulation
    points = np.array(points)
    
    # Get min/max coordinates
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    
    # Convert to YOLO format (x_center, y_center, width, height) normalized
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height

def convert_xml_to_yolo(xml_path: str, image_dir: str, output_dir: str):
    """Converts XML annotations to YOLO format."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        conversion_count = 0
        for image_ann in root.findall('.//image'):
            image_name = image_ann.get('name')
            if not image_name:
                continue
                
            image_path = os.path.join(image_dir, image_name)
            if not os.path.exists(image_path):
                logging.warning(f"Image not found: {image_path}")
                continue
                
            # Get image dimensions
            width = float(image_ann.get('width', 0))
            height = float(image_ann.get('height', 0))
            if width == 0 or height == 0:
                logging.warning(f"Invalid dimensions for image: {image_name}")
                continue
            
            # Create YOLO format label file
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(output_dir, label_name)
            
            with open(label_path, 'w') as f:
                # Process polygon annotations
                for polygon in image_ann.findall('.//polygon'):
                    # Get polygon points
                    points_str = polygon.get('points', '')
                    if not points_str:
                        continue
                        
                    # Convert polygon to YOLO format bbox
                    bbox = convert_polygon_to_bbox(points_str)
                    if bbox is None:
                        continue
                        
                    x_center, y_center, box_width, box_height = bbox
                    
                    # Normalize coordinates
                    x_center /= width
                    y_center /= height
                    box_width /= width
                    box_height /= height
                    
                    # Class id is 0 for octopus
                    f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")
            
            conversion_count += 1
            
        logging.info(f"Converted {conversion_count} annotations from {xml_path}")
        return conversion_count
    except Exception as e:
        logging.error(f"Error processing {xml_path}: {str(e)}")
        return 0

def process_batch(batch_dir: str, dataset_dir: str, split: str):
    """Processes a single batch directory."""
    images_dir = os.path.join(batch_dir, 'images')
    xml_path = os.path.join(batch_dir, 'annotations.xml')
    
    if not os.path.exists(images_dir) or not os.path.exists(xml_path):
        logging.warning(f"Missing required files in {batch_dir}")
        return 0
    
    # Convert annotations
    temp_labels_dir = os.path.join(batch_dir, 'labels')
    converted_count = convert_xml_to_yolo(xml_path, images_dir, temp_labels_dir)
    
    if converted_count == 0:
        return 0
    
    # Move files to final location
    dst_images_dir = os.path.join(dataset_dir, split, 'images')
    dst_labels_dir = os.path.join(dataset_dir, split, 'labels')
    
    for img_file in os.listdir(images_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + '.txt'
        label_path = os.path.join(temp_labels_dir, label_file)
        
        if not os.path.exists(label_path):
            continue
            
        # Copy image and label
        shutil.copy2(os.path.join(images_dir, img_file), os.path.join(dst_images_dir, img_file))
        shutil.copy2(label_path, os.path.join(dst_labels_dir, label_file))
    
    return converted_count

def distribute_data(extract_dir: str, dataset_dir: str):
    """Distributes the data across train/valid/test sets."""
    batch_dirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
    random.shuffle(batch_dirs)
    
    # Split batches across sets
    n_batches = len(batch_dirs)
    train_split = int(n_batches * 0.7)  # 70% for training
    val_split = int(n_batches * 0.15)   # 15% for validation
    
    splits = {
        'train': batch_dirs[:train_split],
        'valid': batch_dirs[train_split:train_split + val_split],
        'test': batch_dirs[train_split + val_split:]
    }
    
    total_processed = 0
    for split_name, split_batches in splits.items():
        split_count = 0
        for batch in split_batches:
            batch_dir = os.path.join(extract_dir, batch)
            processed = process_batch(batch_dir, dataset_dir, split_name)
            split_count += processed
        logging.info(f"Processed {split_count} images for {split_name} set")
        total_processed += split_count
    
    return total_processed

def create_yaml_config(dataset_dir: str):
    """Creates the data.yaml configuration file."""
    yaml_data = {
        'train': os.path.join(dataset_dir, 'train', 'images'),
        'val': os.path.join(dataset_dir, 'valid', 'images'),
        'test': os.path.join(dataset_dir, 'test', 'images'),
        'nc': 1,  # number of classes
        'names': ['octopus']  # class names
    }
    
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)
    logging.info(f"Created YAML configuration at {yaml_path}")

def main():
    # Setup logging
    setup_logging()
    logging.info("Starting octopus dataset preparation")
    
    # Define paths
    zip_dir = "octopus_data"  # Directory containing the zip files
    extract_dir = "temp_extract"  # Temporary directory for extraction
    dataset_dir = os.path.join("datasets", "octopus_dataset")  # Final dataset directory
    
    # Create necessary directories
    os.makedirs(extract_dir, exist_ok=True)
    create_directory_structure(dataset_dir)
    
    try:
        # Extract zip files
        extract_zip_files(zip_dir, extract_dir)
        
        # Process and distribute data
        total_processed = distribute_data(extract_dir, dataset_dir)
        logging.info(f"Total images processed: {total_processed}")
        
        # Create YAML config
        create_yaml_config(dataset_dir)
        
        logging.info("Dataset preparation completed successfully")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
    finally:
        # Clean up temporary directory
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
            logging.info(f"Cleaned up temporary directory: {extract_dir}")

if __name__ == "__main__":
    main() 