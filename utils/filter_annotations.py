import os

coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']

class_names = {name: idx for idx, name in enumerate(coco_classes)}


def filter_annotations(path, classes_list):
  
    annotation_files = [f for f in os.listdir(path) if f.endswith('.txt')]

    for filename in annotation_files:
        file_path = os.path.join(path, filename)
        filtered_lines = []

        with open(file_path, 'r') as f:
            for line in f:
              
                components = line.strip().split()

                class_id = int(components[0])
                class_name = coco_classes[class_id]
                if class_name in classes_list:
                    filtered_lines.append(line)

        with open(file_path, 'w') as f:
            f.writelines(filtered_lines)
            print(f'written for {filename}')
    


class_list = ['person','backpack','handbag','suitcase']

path = os.path.join('training_data','labels')

filter_annotations(path,class_list)