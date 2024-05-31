import os
import json

# Define the mapping for category IDs
# category_mapping = {
#     '0': '0',
#     '24': '1',
#     '26': '2',
#     '28': '3',
#     '68': '4',
#     '69': '5'
# }

category_mapping = {
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '3',
    '5': '2'
}
# category_mapping = {
#     '0': '0',
#     '1': '24',
#     '2': '26',
#     '3': '28',
#     '4': '68',
#     '5': '69'
# }

# Specify the directory containing the text files
directory = r'D:\vidit\Abandoned-Object-Detection\training_data\dataset2_dataset3'

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename == 'classes.txt':
        continue
    if filename.endswith('.txt'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Read the contents of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Open the file for writing
        with open(file_path, 'w') as file:
            for line in lines:
                # Split the line into components
                components = line.strip().split()
                
                # Check if the category ID needs to be remapped
                category_id = (components[0])
                if category_id in category_mapping:
                    components[0] = str(category_mapping[category_id])
                
                # Write the modified line to the file
                file.write(' '.join(components) + '\n')
        
        print(f'File {filename} has been modified.')