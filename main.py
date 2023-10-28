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
# for sample in ["video0", "video1", "video2"]:
#     process_sample(
#         Path(__file__).parent / "train_dataset_dataset" / sample, future=True
#     )
#
# for sample in ["video0", "video1", "video2"]:
#     print(metrics(Path(f"train_dataset_dataset/{sample}")))

sample_name = input("Укажите название папки с видео в папке train_dataset_dataset: ")
process_sample(
    Path(__file__).parent / "train_dataset_dataset" / sample_name,
    future=True,
    output_dir="frames_output",
    output_file="output.txt",
)
