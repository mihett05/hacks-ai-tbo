import io
import json
import os
from pathlib import Path
from functools import reduce

dataset_path = Path(__file__).parent.parent / "train_dataset_dataset"


def sample_to_coco(sample_name: str, output: io.IOBase):
    result = {}
    sample_path = dataset_path / sample_name
    _, _, files = list(os.walk(sample_path / "frames_ms"))[0]
    with open(sample_path / f"{sample_name}.txt") as f:
        annotations = reduce(lambda curr, base: {**base, curr[0]: [
            # Доделай
        ]}, [line.split(", ") for line in f.read().split("\n")], {})
    print(annotations)
    for i, file in enumerate(sorted(files)):
        pass
    json.dump({}, output)

