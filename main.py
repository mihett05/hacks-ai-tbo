from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import Results
from pathlib import Path
from test.frames import metrics
from tracking.tracking import process_sample

# model = YOLO("runs/detect/train/weights/best.pt")
#
# result = model.train(data="data.yaml", epochs=1)
# print(result)


# sample_name = "video0"

# count_frames(Path("train_dataset_dataset/video0"))
process_sample(Path(__file__).parent / "train_dataset_dataset" / "video0", future=True)
print(metrics(Path("train_dataset_dataset/video0")))
