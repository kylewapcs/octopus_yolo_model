from ultralytics import YOLO
import cv2
import numpy as np
import argparse

def predict_octopus_segmentation(source, model_path='runs/segment/train/weights/best.pt', conf=0.25):
    # Load the trained model
    model = YOLO(model_path)
    
    # Run prediction
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        save_txt=True,
        show=True,
        boxes=False,  # Don't show bounding boxes
        show_labels=True,
        show_conf=True
    )
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run octopus segmentation prediction')
    parser.add_argument('source', type=str, help='Path to image or video')
    parser.add_argument('--model', type=str, default='runs/segment/train/weights/best.pt', help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    predict_octopus_segmentation(args.source, args.model, args.conf) 