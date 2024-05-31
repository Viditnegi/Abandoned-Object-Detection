import os
import cv2

def delete_corrupted_images_and_annotations(folder_path):
    """
    Delete corrupted images and corresponding annotations in the specified folder.
    Also removes lines from annotation files with class IDs above 79 or less than 5 numbers.
    
    Args:
    - folder_path (str): Path to the folder containing images and annotations.
    """
    files = os.listdir(folder_path)
    deleted_files = []

    for file_name in files:
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
            image_path = os.path.join(folder_path, file_name)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    deleted_files.append(file_name)
            except Exception as e:
                print(f"Error reading {image_path}: {e}")
                deleted_files.append(file_name)

    for file_name in deleted_files:
        image_path = os.path.join(folder_path, file_name)
        annotation_path = os.path.splitext(image_path)[0] + ".txt"  
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(annotation_path):
            os.remove(annotation_path)
        print(f"Deleted: {image_path} and {annotation_path}")

    for file_name in files:
        if file_name.endswith('.txt'):
            annotation_path = os.path.join(folder_path, file_name)
            with open(annotation_path, 'r') as file:
                lines = file.readlines()
            updated_lines = []
            for line in lines:
                line_parts = line.strip().split()
                if len(line_parts) == 5:
                    class_id = int(line_parts[0])
                    if class_id <= 79:
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)
            if len(updated_lines) < len(lines):
                with open(annotation_path, 'w') as file:
                    file.write(''.join(updated_lines))
                print(f"Updated annotation file: {annotation_path}")
    print("Done.")

# Example usage:
folder_path = r"D:\vidit\Abandoned-Object-Detection\training_data\people_cctv\people_cctv"
delete_corrupted_images_and_annotations(folder_path)
