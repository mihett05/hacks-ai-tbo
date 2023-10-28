import os
import cv2 as cv
import numpy as np
from pathlib import Path
from ultralytics import YOLO

model = YOLO(Path(__file__).parent.parent / "yolo_v8n.pt")

classes = ["wood", "glass", "plastic", "metal"]

items = {i: set() for i in range(len(classes))}


def process_sample(sample_path: Path, future: bool = False):
    images_path = sample_path / "frames_rgb"
    output_path = sample_path / "output"
    if not output_path.exists():
        os.mkdir(output_path)
    for image_name in sorted(os.listdir(images_path)):
        image = cv.imread(str(images_path / image_name))
        if future:
            if (images_path / f"{int(image_name.split('.')[0]) + 2:04}.png").exists():
                image1 = cv.imread(
                    str(images_path / f"{int(image_name.split('.')[0]) + 1:04}.png")
                )
                image2 = cv.imread(
                    str(images_path / f"{int(image_name.split('.')[0]) + 2:04}.png")
                )
                image = np.hstack((image2[:, 14:28], image1[:, :28], image))
            else:
                break
        results = model.track(image, persist=True)
        local = [0] * 4
        for i, key in enumerate(map(int, results[0].boxes.id)):
            cls = int(results[0].boxes.cls[i])
            local[cls] += 1
            items[cls].add(key)

        with open(output_path / (image_name.split(".")[0] + ".txt"), "w") as f:
            f.write("\n".join(map(str, local)) + "\n")
    with open(sample_path / "result.txt", "w") as f:
        f.write("\n".join([str(len(items[key])) for key in items.keys()]) + "\n")
