import os
import cv2
import albumentations as A
import numpy as np

source_folder = r'D:\vidit\Abandoned-Object-Detection\training_data\new_common'
destination_folder = r'D:\vidit\Abandoned-Object-Detection\training_data\augmented_new_common'
num_augmentations = 3

os.makedirs(destination_folder, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.6),
    A.VerticalFlip(p=0.6),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=.2, scale_limit=0, rotate_limit=30, p=0.8, border_mode=cv2.BORDER_CONSTANT),
], bbox_params=A.BboxParams(format='yolo',min_visibility=0.6))


def read_image_and_annotations(image_path, annotation_path):
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    
    for line in lines:
        class_id, x, y, box_width, box_height = map(float, line.strip().split())
        
        # Convert relative coordinates to absolute coordinates
        x_abs = x 
        y_abs = y 
        box_width_abs = box_width 
        box_height_abs = box_height 

        bboxes.append([x_abs, y_abs, box_width_abs, box_height_abs, int(class_id)])
    
    return image, bboxes


def save_augmented_image_and_annotations(image, bboxes, output_image_path, output_annotation_path):
    transformed_images = []
    transformed_bboxes_list = []

    for _ in range(num_augmentations):  # Four augmentations per image
        transformed = transform(image=image, bboxes=bboxes)
        transformed_images.append(transformed['image'])
        transformed_bboxes_list.append(transformed['bboxes'])
    
    for idx, transformed_image in enumerate(transformed_images):
        transformed_bboxes = transformed_bboxes_list[idx]

        coco_bboxes = []
        for bbox in transformed_bboxes:
            x_min, y_min, width, height, class_id = bbox
            coco_bboxes.append([class_id, x_min, y_min, width, height])

        augmented_image_path = output_image_path.replace('.jpg', f'aug_{idx}.jpg').replace('.png', f'aug_{idx}.png')
        augmented_annotation_path = output_annotation_path.replace('.txt', f'aug_{idx}.txt')

        cv2.imwrite(augmented_image_path, transformed_image)

        with open(augmented_annotation_path, 'w') as f:
            for bbox in coco_bboxes:
                f.write(' '.join(map(str, bbox)) + '\n')


for filename in os.listdir(source_folder):
    try:
        if filename.endswith('.jpg') or filename.endswith('.png'): 
            image_path = os.path.join(source_folder, filename)
            annotation_path = os.path.join(source_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            image, bboxes = read_image_and_annotations(image_path, annotation_path)
            
            # Define output paths for augmented image and annotation
            output_image_path = os.path.join(destination_folder, filename)
            output_annotation_path = os.path.join(destination_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            save_augmented_image_and_annotations(image, bboxes, output_image_path, output_annotation_path)
    except Exception as e: 
        print(f"Error in file {filename}: {e}")
        print("Skipping file.")
        continue
