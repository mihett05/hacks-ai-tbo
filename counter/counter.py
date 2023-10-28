import os
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import Results

model = YOLO(Path(__file__).parent.parent / "best.pt")


def count_frame(path: Path) -> list[int]:
    output = [0] * 4
    results: list[Results] = model.predict(str(path))
    for result in results:
        for cls in list(map(int, result.boxes.cls.tolist())):
            output[cls] += 1
    return output


def count_frames(sample_path: Path):
    _, _, files = list(os.walk(sample_path / "frames_rgb"))[0]
    output = sample_path / "output"
    dataset = sample_path / "frames_rgb"
    os.mkdir(output)
    for file in files:
        result = count_frame(dataset / file)
        with open(output / (file.split(".")[0] + ".txt"), "w") as f:
            f.write("\n".join(map(str, result)) + "\n")


def count_video(sample_path: Path) -> list[int]:
    pass
