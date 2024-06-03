import os
import random
import cv2
from ultralytics import YOLO

from utils import Tracker
from utils.utils import stationary_indices
from utils.utils import plot_midpoints
from utils.utils import plot_midpoint
from utils.utils import calculate_distance

# video_path = os.path.join('.', 'ABODA','video5.avi')
# video_path = os.path.join('.', 'ABODA','new','people.mp4')
video_path = r'test_data\baggage.mp4'
# video_path = r'D:\vidit\Abandoned-Object-Detection\test_data\Left Luggage Detection.mp4'
# video_path = r'D:\vidit\Abandoned-Object-Detection\test_data\videoplayback.mp4'
# video_path = r'D:\vidit\Abandoned-Object-Detection\ABODA\new\VIRAT_S_040103_05_000729_000804.mp4'

image_path = r'test_data\suitcases.png'

# model = YOLO(r"D:\vidit\Abandoned-Object-Detection\yolo_models\yolov8m_temp.pt")
model = YOLO(r"yolo_models\yolov8m.pt")
# model = YOLO('yolov8m')
class_dict = model.names

cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames / 3))


tracker = Tracker()

colors = [(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)) for j in range(100)]

detection_threshold = 0.3

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame,(640,420))
    
    # frame = cv2.imread(image_path)
    results = model(frame)
    
    
    person_info = []
    bag_info = []
    
    for ind,result in enumerate(results):
        # if(ind!=1):
        #     continue
        print(ind)
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold and class_id in stationary_indices:
                detections.append([x1, y1, x2, y2, score,class_id])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            class_ind = track.class_name
            class_name = class_dict[class_ind]
            
            
            if class_name == 'person':
                person_info.append((x1,y1,x2,y2,class_name,track_id))
            else:
                bag_info.append((x1,y1,x2,y2,class_name,track_id))
            
            
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            # cv2.putText(frame, f'{str(track_id)}--{class_name}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # _, mid,bbox_height = plot_midpoint(frame,bbox)
            
    for x1,y1,x2,y2,class_name,track_id in person_info:                                # Plot bboxes of all people
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
        cv2.putText(frame, f'{str(track_id)}--{class_name}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    for x1,y1,x2,y2,class_name1,track_id_bag in bag_info:                           # Plot different bboxes for abandoned bags
    
        _, mid1, bbox_height_bag = plot_midpoint(frame,[x1,y1,x2,y2])
        abandoned = True
        
        for a1,b1,a2,b2,class_name2,track_id_person in person_info:
            _, mid2, bbox_height_person = plot_midpoint(frame,[a1,b1,a2,b2])
            
            distance = calculate_distance(mid1,mid2)
            if distance < bbox_height_person:
                abandoned = False
                break
            
        if not abandoned:    
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id_bag % len(colors)]), 3)
            cv2.putText(frame, f'{str(track_id_bag)}--{class_name1}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)           
        else:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 3)
            cv2.putText(frame, "ABANDONED", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                
            
    cv2.imshow('feed',frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
