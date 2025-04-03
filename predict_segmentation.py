import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Run YOLOv8 segmentation on images')
    parser.add_argument('--source', type=str, required=True, help='Path to image or directory of images')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='predictions', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load the model
    model = YOLO('models/best.pt')
    
    # Run inference
    results = model(args.source, conf=args.conf)
    
    # Process and save results
    for i, result in enumerate(results):
        # Get the original image
        img = result.orig_img.copy()
        
        # Draw segmentation masks
        if result.masks is not None:
            for mask in result.masks:
                # Convert mask to binary image
                binary_mask = mask.data[0].cpu().numpy()
                binary_mask = (binary_mask * 255).astype(np.uint8)
                
                # Create colored overlay
                overlay = img.copy()
                overlay[binary_mask > 0] = [0, 255, 0]  # Green color for octopus
                
                # Blend original image with overlay
                alpha = 0.5
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # Draw bounding boxes
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f'Octopus {conf:.2f}', (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the result
        output_path = os.path.join(args.output, f'pred_{i}.jpg')
        cv2.imwrite(output_path, img)
        print(f'Saved prediction to {output_path}')

if __name__ == '__main__':
    main() 