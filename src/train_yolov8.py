# import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt
import numpy as np
import torch

import os



if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    
    model = YOLO("yolov8m") 
    
    # results = model.train(data='config.yaml', epochs=500)
    results = model.train(data='yolov8_config.yaml', epochs=500,device='cuda',batch=8,weight_decay = 0.01)
    



