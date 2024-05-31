import cv2
import torch
import matplotlib.pyplot as plt 
import numpy as np



# model_type = 'DPT_Large'
# model_type = 'DPT_Hybrid'
model_type = 'MiDaS_small'

# Download the MiDaS
midas = torch.hub.load('intel-isl/MiDaS', model_type)
midas.to('cuda')
midas.eval()
# Input transformation pipeline
midas_transformer = torch.hub.load('intel-isl/MiDaS', 'transforms')
if model_type == 'DPT_Large' or model_type == 'DPT_Hybrid':
    transform = midas_transformer.dpt_transform
else:
    transform = midas_transformer.small_transform 

# Hook into OpenCV
cap = cv2.VideoCapture(r'D:\vidit\Abandoned-Object-Detection\ABODA\video11.avi')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames / 3))

while cap.isOpened(): 
    ret, frame = cap.read()

    # Transform input for midas 
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cuda')

    # Make a prediction
    with torch.no_grad(): 
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2], 
            mode='bicubic', 
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    
    depth_map = cv2.normalize(depth_map,None,0,1,norm_type = cv2.NORM_MINMAX,dtype=cv2.CV_64F)
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map,cv2.COLORMAP_MAGMA)
    
 
    
    mixed = 0.4*depth_map + 0.8*img 
    mixed = cv2.normalize(mixed,None,0,1,norm_type = cv2.NORM_MINMAX,dtype=cv2.CV_64F)
    mixed = (mixed*255).astype(np.uint8)
    
    cv2.imshow('Midas', depth_map)
    # cv2.imshow('Original', img)
    cv2.imshow('Mixed', mixed)
    # plt.pause(0.00001)

    if cv2.waitKey(1) == ord('q'): 
        cap.release()
        cv2.destroyAllWindows()

# plt.show()