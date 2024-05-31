import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt
import numpy as np
import os

from utils.utils import filter_bboxes
from utils.utils import plot_bboxes
from utils.utils import plot_midpoints
from utils.utils import person_object_distance



# Load the YOLOv8 model
# model = YOLO("yolo_models/yolov8m.pt") 
model = YOLO(r"D:\vidit\Abandoned-Object-Detection\yolo_models\yolov8m_all_data_all_classes.pt")
image_path = r'D:\vidit\Abandoned-Object-Detection\test_data\suitcases.png'

# Open the video capture
video_path =  os.path.join('.', 'ABODA','video9.avi')
# video_path = os.path.join('.', 'people.mp4')

cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames / 3))



# print("done")

# Process each frame
while True:
    # ret, frame = cap.read()
    # if not ret:
    #     break

    # Run YOLOv8 detection
    frame = cv2.imread(image_path)
    # if not frame:
    #     print("lol")
    #     break
    results = model(frame)

    # Filter out bounding boxes and classes
    filtered_frame,annotated_frame = filter_bboxes(frame,results)
    # print(filtered_class_names)
    # distances = person_object_distance(filtered_bboxes,filtered_class_names)

  
    cv2.imshow('YOLOv8 Detection', filtered_frame)
    cv2.imshow('All classes',annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
# out.release()
cv2.destroyAllWindows()