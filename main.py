from ultralytics import YOLO

# model = YOLO("yolov8n.pt")
#
# result = model.train(data="data.yaml", epochs=1)
# print(result)

model = YOLO("runs/detect/train/weights/last.pt")
model.predict("train_dataset_dataset/video0/frames_rgb/0000.png", save=True)
