
# Abandoned Object Detection 

For any query contact at viditsn@gmail.com
## Problem Statement - 

This project was built to detect all the abandoned or unattended objects (mainly bags) in crowded areas through cctv/surveillance footage.
The classes were divided into 3 categories of bags namely 1) backpack, 2) handbag, 3) suitcase and a separate class  4) person.

## Dataset -

The dataset was curated from different sources such as youtube, roboflow.com, and a public dataset named ABODA through github.
Frames were extracted from the videos and then annotated individually. Each frame had several instances of each of the 4 classes.
The original images annotated were 3,500 and after augmentation the total number became 10,000.

## Training and Algorithm - 

We trained the images on the model yolov8m (m for medium size) and the training validation accuracy reached around 70% percent.
Then we had to track the objects in the video to identify each individual object, for this we added deep-sort object tracking algorithm.

Then comes the part where we had to detect when an object is abandoned.
The conditions for this are-
The nearest person to a bag should stray away from the bag at a distance more that the height of that person.
The object must not be moving and should be stationary.

## Testing and Improvements -

We tested the project on some videos and noticed the model is performing very well on scenes that are not heavily crowded and where range of the cctv is close-medium (the footage does not cover a large area), but struggles to detect some bags in a crowded or dim lit scene
To further improve upon this, more data needs to be curated and annotated precisely to cover a variety of situations for the model to be trained on.


# Test Images
![temp](https://github.com/Viditnegi/Abandoned-Object-Detection/assets/106267998/4931c8df-a8e8-46cf-af3a-757df7beda7c) 
![temp1](https://github.com/Viditnegi/Abandoned-Object-Detection/assets/106267998/8907c3a5-edf7-4f66-87d9-679941a3d835)
![temp](https://github.com/Viditnegi/Abandoned-Object-Detection/assets/106267998/9578ac95-35b2-4df9-b965-645d6495a3c6) 
![temp2](https://github.com/Viditnegi/Abandoned-Object-Detection/assets/106267998/cc21ad97-052e-470d-a77c-466a1db96eee)


# Setup
1) Python env 3.7.16
2) Install cuda and cuDNN
3) Install pytorch and tensorflow for the corresponding cuda versions. (ref. https://pytorch.org/get-started/previous-versions/)
4) pip install -r requirements.txt
5) run src/main.py

Data link - https://drive.google.com/drive/folders/1uAXSOwTRBcxrY9soR6HGzSLmNCpKrs9D?usp=sharing
