from ultralytics import YOLO
import torch
import argparse

def train_segmentation_model(data_yaml, 
                           model_size='m',
                           epochs=100,
                           imgsz=640,
                           batch_size=16,
                           device=None):
    """
    Train YOLOv8 segmentation model
    
    Args:
        data_yaml: Path to data.yaml file
        model_size: YOLO model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size
        device: Device to train on (cuda device, i.e. 0 or cpu)
    """
    # Select device
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Load a model
    model = YOLO(f'yolov8{model_size}-seg.pt')  # load a pretrained model
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        patience=50,  # Early stopping patience
        save=True,  # Save checkpoints
        save_period=10,  # Save every 10 epochs
        plots=True,  # Generate training plots
        verbose=True
    )
    
    print("\nTraining completed!")
    print(f"Results saved to {results}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 segmentation model')
    parser.add_argument('--data', type=str, default='datasets/octopus_segmentation/data.yaml',
                      help='path to data.yaml')
    parser.add_argument('--model-size', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
                      help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100,
                      help='number of epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                      help='input image size')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='batch size')
    parser.add_argument('--device', type=str, default=None,
                      help='device to train on (cuda device, i.e. 0 or cpu)')
    
    args = parser.parse_args()
    
    train_segmentation_model(
        data_yaml=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device
    ) 