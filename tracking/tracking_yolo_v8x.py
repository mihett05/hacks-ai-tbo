import os
import cv2 as cv
from ultralytics import YOLO

model = YOLO('yolo_v8x.pt')

for image_name in os.listdir('../train_dataset_dataset/video1/frames_rgb'):
    image = cv.imread(f'../train_dataset_dataset/video1/frames_rgb/{image_name}')
    results = model.track(image, persist=True, conf=0.1)

    # print(results[0].boxes.cls)
    cv.imshow('window', results[0].plot())

    cv.waitKey(0)

cv.destroyAllWindows()
