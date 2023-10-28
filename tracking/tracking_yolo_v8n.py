import os
import cv2 as cv
from ultralytics import YOLO

model = YOLO('yolo_v8n_ver2.pt')

for image_name in os.listdir('../train_dataset_dataset/video0/frames_rgb'):
    image = cv.imread(f'../train_dataset_dataset/video0/frames_rgb/{image_name}')
    results = model.track(image, persist=True)

    # print(results[0].boxes.cls)
    cv.imshow('window', results[0].plot())

    cv.waitKey(0)

cv.destroyAllWindows()
