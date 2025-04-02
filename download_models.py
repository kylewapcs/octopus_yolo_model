from ultralytics import YOLO

def download_models():
    """Download pretrained YOLOv8 segmentation models"""
    # Download models of different sizes
    models = ['n', 's', 'm', 'l', 'x']
    
    print("Downloading YOLOv8 segmentation models...")
    for size in models:
        print(f"\nDownloading yolov8{size}-seg.pt...")
        try:
            YOLO(f'yolov8{size}-seg.pt')
            print(f"Successfully downloaded yolov8{size}-seg.pt")
        except Exception as e:
            print(f"Error downloading yolov8{size}-seg.pt: {str(e)}")

if __name__ == "__main__":
    download_models() 