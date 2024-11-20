import sys
import time
from ultralytics import YOLO

def main(images_path):
    # space separated list of image paths
    images = images_path[1:]
    print(f"Images to process: {images}")

    # Load the YOLO model once before the loop
    model = YOLO('model.pt')  # Ensure 'model.pt' is your model file

    while True:
        for image_path in images:
            # Record the start time
            start_time = time.time()

            # Run inference
            results = model(image_path)

            # Process results
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs

                # Check if any boxes were detected
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Extract class ID, confidence, and bounding box coordinates
                        cls_id = int(box.cls.cpu().numpy())
                        class_name = result.names[cls_id]
                        conf = float(box.conf.cpu().numpy())
                        xyxy = box.xyxy.cpu().numpy().astype(int)[0]  # [x1, y1, x2, y2]

                        # Print detection results
                        print(f"Detected '{class_name}' with confidence {conf:.2f} at {xyxy}")
                else:
                    print("No objects detected.")

            # Record the end time
            end_time = time.time()

            # Calculate and print the processing time
            elapsed_time = end_time - start_time
            print(f"Processing time: {elapsed_time:.2f} seconds")

            # Sleep for 1 second before the next detection
            time.sleep(0.5)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detector.py <image_path>")
        sys.exit(1)
    main(sys.argv)
