import os
import shutil
from sklearn.model_selection import train_test_split

# Define the source directory and the target directory structure
source_dir = 'training_data/collected_all'
target_dir = 'data'

# Create the target directory structure if it doesn't exist
os.makedirs(os.path.join(target_dir, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'labels', 'val'), exist_ok=True)

# List all files in the source directory
all_files = os.listdir(source_dir)

# Separate images and labels
images = [f for f in all_files if f.endswith(('.jpg', '.png', '.jpeg','.webp'))]
labels = [f for f in all_files if f.endswith('.txt')]

# Filter out images without corresponding labels
filtered_images = [img for img in images if os.path.splitext(img)[0] + '.txt' in labels]

# Split the data into training and validation sets
train_images, val_images = train_test_split(filtered_images, test_size=0.2, random_state=1000)
train_labels = [os.path.splitext(img)[0] + '.txt' for img in train_images]
val_labels = [os.path.splitext(img)[0] + '.txt' for img in val_images]


print(len(train_images))
print(len(val_images))


# Function to copy files
def copy_files(file_list, target_subdir, file_type):
    for file in file_list:
        if file_type == 'image':
            shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, 'images', target_subdir, file))
        elif file_type == 'label':
            shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, 'labels', target_subdir, file))

# Copy the files to their corresponding directories
copy_files(train_images, 'train', 'image')
copy_files(val_images, 'val', 'image')
copy_files(train_labels, 'train', 'label')
copy_files(val_labels, 'val', 'label')

print("Files have been successfully copied.")