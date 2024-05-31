import os
import glob
import uuid

def rename_files(directory):
    # Get all image files in the directory
    image_files = glob.glob(os.path.join(directory, '*.jpg')) + \
                  glob.glob(os.path.join(directory, '*.jpeg')) + \
                  glob.glob(os.path.join(directory, '*.png')) + \
                  glob.glob(os.path.join(directory, '*.webp'))

    print(len(image_files))
    for image_file in image_files:
        # Get the base name of the file (without extension)
        base_name = image_file[:image_file.rfind('.')]

        # Generate a new unique name
        new_name = "people_cctv__2" + str(uuid.uuid4()) 

        # Rename the image file
        os.rename(image_file, os.path.join(directory, new_name + image_file[image_file.rfind('.'):]))

        # If a corresponding .txt file exists, rename it as well
        if os.path.isfile(base_name + '.txt'):
            os.rename(base_name + '.txt', os.path.join(directory, new_name + '.txt'))

    print("Renaming completed.")

    # Delete any image or annotation file that doesn't have a corresponding pair
    all_files = glob.glob(os.path.join(directory, '*'))
    for file in all_files:
        base_name = file[:file.rfind('.')]
        extension = file[file.rfind('.'):]
        if extension in ['.jpg', '.jpeg', '.png', '.webp']:
            if not os.path.isfile(base_name + '.txt'):
                os.remove(file)
        elif extension == '.txt':
            if not any(os.path.isfile(base_name + ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                os.remove(file)

    print("Deletion of unpaired files completed.")

# Call the function with the path to your directory
rename_files(r'D:\vidit\Abandoned-Object-Detection\training_data\people_cctv2\people_cctv2')
