import io
import json
import os
import cv2
from pathlib import Path

dataset_path = Path(__file__).parent.parent / "train_dataset_dataset"


def sample_to_coco(sample_name: str, output: io.IOBase):
    result = {}
    sample_path = dataset_path / sample_name
    _, _, files = list(os.walk(sample_path / "frames_ms"))[0]
    image_h, image_w, _ = cv2.imread(str(sample_path / "frames_rgb" / "0000.png")).shape

    result["images"] = [
        {
            "id": f"{sample_name}_{i}",
            "file_name": str(
                (sample_path / "frame_ms" / file).relative_to(
                    Path(__file__).parent.parent
                )
            ),
            "width": image_w,
            "height": image_h,
        }
        for i, file in enumerate(sorted(files))
    ]

    with open(sample_path / f"{sample_name}.txt") as f:
        result["annotations"] = [
            {
                "id": i + 1,
                "image_id": f"{sample_name}_{int(image_id)}",
                "category_id": int(category_id),
                "bbox": [
                    (x - w / 2) * image_w,
                    (y - h / 2) * image_h,
                    w * image_w,
                    h * image_h,
                ],
                "iscrowd": 0,
                "area": w * image_w * h * image_h,
                "ignore": 0,
            }
            for i, (image_id, object_id, category_id, x, y, w, h) in enumerate(
                [
                    tuple(map(float, line.split(", ")))
                    for line in f.read().split("\n")
                    if line.strip()
                ]
            )
        ]

    json.dump(result, output, indent=2)


def merge_coco(files: list[io.IOBase], output: io.IOBase):
    offset = 0
    result = {
        "images": [],
        "annotations": [],
    }
    for sample in [json.load(file) for file in files]:
        result["images"] += sample["images"]
        result["annotations"] += [
            {**a, "id": offset + a["id"]} for a in sample["annotations"]
        ]
        offset += int(sample["annotations"][-1]["id"]) + 1
    json.dump(result, output)
