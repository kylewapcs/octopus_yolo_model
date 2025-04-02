import os
import shutil
import random
from pathlib import Path
import zipfile
import yaml
import cv2
import numpy as np
from bs4 import BeautifulSoup

def parse_xml_annotations(xml_path):
    """Parse XML annotation file to extract polygon points for each image"""
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    soup = BeautifulSoup(xml_content, 'xml')
    annotations = {}
    
    # Find all image annotations
    for image in soup.find_all('image'):
        filename = image.get('name', '')
        polygons = []
        
        # Find all polygon elements for this image
        for polygon in image.find_all('polygon'):
            points_str = polygon.get('points', '')
            # Convert points string to list of (x,y) tuples
            points = []
            for point in points_str.split(';'):
                if point:
                    x, y = map(float, point.split(','))
                    points.append([x, y])
            if points:
                polygons.append(points)
        
        # Store annotations even if no polygons (means no octopus in image)
        annotations[filename] = polygons
    
    return annotations

def convert_polygons_to_yolo_seg(polygons, img_width, img_height):
    """Convert polygon coordinates to YOLO segmentation format"""
    yolo_segments = []
    for polygon in polygons:
        # Normalize coordinates to [0,1]
        normalized = []
        for x, y in polygon:
            nx = x / img_width
            ny = y / img_height
            normalized.extend([nx, ny])
        yolo_segments.append(normalized)
    return yolo_segments

def setup_octopus_segmentation_dataset(zip_files_list):
    """Sets up the octopus segmentation dataset from zip files"""
    # Base paths
    base_dir = Path('datasets/octopus_segmentation')
    temp_dir = base_dir / 'temp'
    
    # Create directories
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            (base_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for extraction
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract all zip files
    print("Extracting zip files...")
    all_images = []
    for zip_path in zip_files_list:
        print(f"Processing {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                batch_dir = temp_dir / Path(zip_path).stem
                batch_dir.mkdir(exist_ok=True)
                zip_ref.extractall(batch_dir)
                
                # Look for images and annotations
                images_dir = batch_dir / 'images'
                annotations_file = batch_dir / 'annotations.xml'
                
                if images_dir.exists() and annotations_file.exists():
                    # Parse all annotations first
                    annotations = parse_xml_annotations(annotations_file)
                    
                    # Get all images
                    image_files = list(images_dir.glob('*.jpg'))
                    image_files.extend(list(images_dir.glob('*.png')))
                    
                    # Include all images, with or without annotations
                    for img_path in image_files:
                        polygons = annotations.get(img_path.name, [])  # Empty list if no annotations
                        all_images.append((img_path, polygons))
                    
                    print(f"Found {len(image_files)} images in {zip_path}")
                else:
                    print(f"Warning: Missing images folder or annotations.xml in {zip_path}")
        except Exception as e:
            print(f"Error processing {zip_path}: {str(e)}")
    
    if not all_images:
        print("Error: No images found in any of the zip files!")
        return
    
    # Shuffle all images
    print(f"\nTotal images found: {len(all_images)}")
    random.shuffle(all_images)
    
    # Split files (80% train, 10% valid, 10% test)
    train_split = int(len(all_images) * 0.8)
    val_split = int(len(all_images) * 0.9)
    
    splits = {
        'train': all_images[:train_split],
        'valid': all_images[train_split:val_split],
        'test': all_images[val_split:]
    }
    
    # Process and move files
    print("\nProcessing dataset...")
    for split_name, files in splits.items():
        for img_path, polygons in files:
            # Get image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            height, width = img.shape[:2]
            
            # Move image
            dest_img_path = base_dir / split_name / 'images' / img_path.name
            shutil.copy2(img_path, dest_img_path)
            
            # Save YOLO format segmentation labels if there are annotations
            if polygons:
                segments = convert_polygons_to_yolo_seg(polygons, width, height)
                dest_label_path = base_dir / split_name / 'labels' / (img_path.stem + '.txt')
                with open(dest_label_path, 'w') as f:
                    for segment in segments:
                        # Write class index (0 for octopus) followed by normalized polygon points
                        line = '0 ' + ' '.join(map(str, segment)) + '\n'
                        f.write(line)
    
    # Create data.yaml with absolute paths
    yaml_content = {
        'train': str(base_dir / 'train' / 'images'),
        'val': str(base_dir / 'valid' / 'images'),
        'test': str(base_dir / 'test' / 'images'),
        'nc': 1,
        'names': ['octopus']
    }
    
    with open(base_dir / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f)
    
    # Cleanup
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    # Print summary
    print("\nDataset setup complete!")
    print(f"Training images: {len(splits['train'])}")
    print(f"Validation images: {len(splits['valid'])}")
    print(f"Test images: {len(splits['test'])}")

if __name__ == "__main__":
    # List all your zip files here
    zip_files = [
        r"C:\Users\klipk\Downloads\batch1.zip",
        r"C:\Users\klipk\Downloads\batch2.zip",
        r"C:\Users\klipk\Downloads\batch3.zip",
        r"C:\Users\klipk\Downloads\batch4.zip",
        r"C:\Users\klipk\Downloads\batch5.zip",
        r"C:\Users\klipk\Downloads\batch6.zip",
        r"C:\Users\klipk\Downloads\batch7.zip",
        r"C:\Users\klipk\Downloads\batch8.zip"
    ]
    
    setup_octopus_segmentation_dataset(zip_files) 