import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path


def test_frames(sample_path: Path):
    _, _, files = list(os.walk(sample_path / "output"))[0]
    passed = 0

    for i, file in enumerate(files):
        with open(sample_path / "frames_output" / file) as expected_f, open(
            sample_path / "output" / file
        ) as real_f:
            expected = [int(line) for line in expected_f.read().split() if line]
            real = [int(line) for line in real_f.read().split() if line]
            if expected == real:
                passed += 1
            else:
                diff = np.array(expected) - np.array(real)
                classes = ["wood", "glass", "plastic", "metal"]
                print(file, *[f"{classes[i]}: {diff[i]}" for i in range(4) if diff[i]])
                image = cv.imread(
                    str(sample_path / "frames_rgb" / (file.split(".")[0] + ".png"))
                )
                img_h, img_w = image.shape
                with open(
                    sample_path / ".." / "new" / sample_path.name / "labels" / file
                ) as f:
                    for cls, x, y, w, h in [
                        [int(line.split()[0]), *list(map(float, line.split()))]
                        for line in f.read().split()
                    ]:
                        (x - w / 2) * img_w, (y - h / 2) * img_h,

                break
    plt.show()
    print(f"{passed}/{len(files)}")
