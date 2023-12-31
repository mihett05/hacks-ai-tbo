import os
import cv2 as cv
import numpy as np
from pathlib import Path
from ultralytics import YOLO

model = YOLO(Path.cwd() / "yolo_v8l.pt")

classes = ["wood", "glass", "plastic", "metal"]


def process_sample(
    sample_path: Path,
    future: bool = False,
    output_dir: str = "output",
    output_file="result.txt",
):
    images_path = sample_path / "frames_rgb"
    output_path = sample_path / output_dir
    items = {i: set() for i in range(len(classes))}
    if not output_path.exists():
        os.mkdir(output_path)
    for image_name in sorted(os.listdir(images_path)):
        image = cv.imread(str(images_path / image_name))
        parts = [28, 14]
        if future:
            image1 = (
                cv.imread(
                    str(images_path / f"{int(image_name.split('.')[0]) + 1:04}.png")
                )
                if (
                    images_path / f"{int(image_name.split('.')[0]) + 1:04}.png"
                ).exists()
                else np.ones(
                    (image.shape[0], sum(parts), image.shape[2]), dtype=np.uint8
                )
            )
            image2 = (
                cv.imread(
                    str(images_path / f"{int(image_name.split('.')[0]) + 2:04}.png")
                )
                if (
                    images_path / f"{int(image_name.split('.')[0]) + 2:04}.png"
                ).exists()
                else np.ones(
                    (image.shape[0], sum(parts), image.shape[2]), dtype=np.uint8
                )
            )
            image = np.hstack(
                (image2[:, parts[1] : parts[0]], image1[:, : parts[0]], image)
            )
        results = model.track(image, persist=True)
        local = [0] * 4
        for i, key in enumerate(map(int, results[0].boxes.id)):
            cls = int(results[0].boxes.cls[i])
            local[cls] += 1
            items[cls].add(key)

        with open(output_path / (image_name.split(".")[0] + ".txt"), "w") as f:
            f.write("\n".join(map(str, local)) + "\n")
    with open(sample_path / output_file, "w") as f:
        f.write("\n".join([str(len(items[key])) for key in items.keys()]) + "\n")
