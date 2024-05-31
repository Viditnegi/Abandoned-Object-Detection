
import cv2
from ultralytics import YOLO
import os
import math

model = YOLO("yolo_models/yolov8m.pt") 

class_dict = model.names

# stationary_classes = ['person', 'backpack', 'handbag', 'suitcase', 'umbrella', 'bottle' , 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear']
stationary_classes = ['person', 'bag', 'suitcase','cart','soft_toy','box','microwave','oven','polybag']
# stationary_indices = [list(class_dict.values()).index(class_name) for class_name in stationary_classes]

stationary_indices=[0, 1 ,2,3,4,5, 24, 26, 28,68]
print(stationary_indices)

def filter_bboxes(frame,results):
    filtered_bboxes = []
    filtered_class_names = []
    for bbox, cls in zip(results[0].boxes.data, results[0].boxes.cls):
        if cls in stationary_indices:
            filtered_bboxes.append(bbox)
            filtered_class_names.append(class_dict[int(cls)])

    filtered_frame = plot_bboxes(frame.copy(),filtered_bboxes,class_dict)
    filtered_frame = plot_midpoints(filtered_frame,filtered_bboxes,filtered_class_names)
    annotated_frame = results[0].plot()
    
    return filtered_frame,annotated_frame
    
    
    
def plot_bboxes(frame, bboxes, class_dict):
    for bbox in bboxes:
        x1, y1, x2, y2, confidence, class_id = [int(x) for x in bbox]
        class_name = class_dict[class_id]
        
        # Define colors for different classes
        if class_name == 'person':
            color = (0, 255, 255)  # Yellow color for person bounding box
        else:
            color = (0, 255, 0)  # Green color for other bounding boxes

        # Draw the bounding box rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw the class name, confidence, and bounding box coordinates
        label = f"{class_name}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def plot_midpoints(frame, bboxes,filtered_class_names):
    for bbox in bboxes:
        xmin, ymin, xmax, ymax,conf, class_id = [int(x) for x in bbox]
    
        # Calculate the midpoint of the bounding box
        midpoint_x = (xmin + xmax) // 2
        midpoint_y = (ymin + ymax) // 2
        midpoint = (midpoint_x, midpoint_y)
        
        # Draw a circle at the midpoint
        cv2.circle(frame, midpoint, 5, (255, 234, 0), -1)  # Red circle
        
    
    return frame


def plot_midpoint(frame, bbox):
    xmin, ymin, xmax, ymax = [int(x) for x in bbox]
    
    # Calculate the midpoint of the bounding box
    midpoint_x = (xmin + xmax) // 2
    midpoint_y = (ymin + ymax) // 2
    midpoint = (midpoint_x, midpoint_y)
    bbox_height = ymax - ymin
    # Draw a circle at the midpoint
    cv2.circle(frame, midpoint, 5, (255, 234, 0), -1)  # Red circle
        
    return frame,midpoint,bbox_height


def calculate_distance(midpoint1, midpoint2):
    x1, y1 = midpoint1
    x2, y2 = midpoint2
    
    # Calculate the Euclidean distance between the two midpoints
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return distance




def person_object_distance(bboxes,classes):
    pass
