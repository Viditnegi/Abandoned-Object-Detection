import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_folder, num_frames=50):

    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = total_frames // num_frames

    frame_counter = 0
    frame_index = 0

    while True:
 
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % step == 0:
   
            output_file = output_folder / f"{video_path.stem}_{frame_counter:05d}.jpg"
            cv2.imwrite(str(output_file), frame)
            frame_counter += 1
            
            if frame_counter >= num_frames:
                break

     
        frame_index += 1

    cap.release()


video_folder = Path("ABODA/new")


output_folder = Path("ABODA/new/extracted_images")
output_folder.mkdir(parents=True, exist_ok=True)


for video_file in video_folder.glob("*.mp4"):
   
    extract_frames(video_file, output_folder)
    print(f"Frames extracted from {video_file.name}")

print("Frame extraction completed.")