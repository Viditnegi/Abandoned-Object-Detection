import cv2
from ultralytics import YOLO
import os
import re

model = YOLO("yolo_models/yolov8m_cctv_training_1.pt") 
images_path = 'training_data/new_new_images'
labels_path = 'training_data/new_new_labels'
os.makedirs(labels_path, exist_ok=True)
allowed_formats = ['.jpg', '.png', '.jpeg', '.webp']

# Rename files with numerical format
counter = 1
for filename in os.listdir(images_path):
    file_path = os.path.join(images_path, filename)
    if any(filename.endswith(ext) for ext in allowed_formats):
        ext = os.path.splitext(filename)[1]
        new_filename = f"{counter:05d}{ext}"
        new_file_path = os.path.join(images_path, new_filename)
        os.rename(file_path, new_file_path)
        counter += 1

# Process renamed files
for filename in os.listdir(images_path):
    if not any(filename.endswith(ext) for ext in allowed_formats):
        print(f"Removing invalid file: {filename}")
        os.remove(os.path.join(images_path, filename))
        continue

    image_path = os.path.join(images_path, filename)
    try:
        image = cv2.imread(image_path)
    except cv2.error as e:
        print(f"Error reading {filename}: {e}")
        continue

    if image is None:
        print(f"Removing corrupted image: {filename}")
        os.remove(image_path)
        continue

    results = model(image_path)
    base_filename = os.path.splitext(filename)[0]
    annotations_path = os.path.join(labels_path, f"{base_filename}.txt")

    with open(annotations_path, 'w') as f:
        for result in results:
            for box in result.boxes.xyxy.tolist():
                x1, y1, x2, y2 = box
                width, height = image.shape[1], image.shape[0]
                class_id = int(result.boxes.cls[result.boxes.xyxy.tolist().index(box)])
                x_center = (x1 + x2) / (2 * width)
                y_center = (y1 + y2) / (2 * height)
                width = (x2 - x1) / width
                height = (y2 - y1) / height
                annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                f.write(annotation)

    print(f"Processed image: {filename}")

print("Done!")