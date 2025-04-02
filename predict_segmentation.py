from ultralytics import YOLO
import cv2
import numpy as np
import argparse
from pathlib import Path

def visualize_prediction(model, image_path, output_dir, conf_threshold=0.25):
    """
    Run prediction and visualize segmentation masks
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run prediction
    results = model.predict(image_path, conf=conf_threshold, save=True, save_txt=True)
    
    # Process each result
    for i, result in enumerate(results):
        # Get the original image
        orig_img = result.orig_img
        
        # Draw masks on image
        masks = result.masks
        if masks is not None:
            annotated_img = orig_img.copy()
            for j, mask in enumerate(masks):
                # Get mask pixels
                mask_pixels = mask.data.cpu().numpy().squeeze()
                
                # Create color overlay
                color = np.array([0, 255, 0], dtype=np.uint8)  # Green color for mask
                mask_overlay = np.zeros_like(orig_img, dtype=np.uint8)
                mask_overlay[mask_pixels > 0] = color
                
                # Blend with original image
                alpha = 0.5
                annotated_img = cv2.addWeighted(annotated_img, 1, mask_overlay, alpha, 0)
                
                # Draw contours
                mask_uint8 = (mask_pixels * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_img, contours, -1, (0, 255, 0), 2)
            
            # Save annotated image
            output_path = output_dir / f'pred_{Path(image_path).stem}.jpg'
            cv2.imwrite(str(output_path), annotated_img)
            print(f"Saved prediction visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run YOLOv8 segmentation prediction with visualization')
    parser.add_argument('--model', type=str, default='runs/segment/train/weights/best.pt',
                      help='path to model weights')
    parser.add_argument('--source', type=str, required=True,
                      help='path to image or directory of images')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='confidence threshold')
    parser.add_argument('--output', type=str, default='predictions',
                      help='output directory for visualizations')
    
    args = parser.parse_args()
    
    # Load model
    model = YOLO(args.model)
    
    # Run prediction and visualization
    visualize_prediction(model, args.source, args.output, args.conf)

if __name__ == '__main__':
    main() 