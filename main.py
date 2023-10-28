from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import Results
from pathlib import Path
from counter.counter import count_frame, count_frames
from test.frames import test_frames

# model = YOLO("runs/detect/train/weights/best.pt")
#
# result = model.train(data="data.yaml", epochs=1)
# print(result)


# sample_name = "video0"

# count_frames(Path("train_dataset_dataset/video0"))
test_frames(Path("train_dataset_dataset/video0"))
