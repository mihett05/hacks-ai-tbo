import os
import numpy as np
from pathlib import Path


def metrics(sample_path: Path):
    _, _, files = list(os.walk(sample_path / "output"))[0]
    passed = 0
    expected = [[] for _ in range(4)]
    predicted = [[] for _ in range(4)]

    for i, file in enumerate(files):
        with open(sample_path / "frames_output" / file) as expected_f, open(
            sample_path / "output" / file
        ) as real_f:
            exp = [int(line) for line in expected_f.read().split() if line]
            real = [int(line) for line in real_f.read().split() if line]
            for i in range(4):
                expected[i].append(exp[i])
                predicted[i].append(real[i])
            if exp == real:
                passed += 1
            else:
                diff = np.array(exp) - np.array(real)
                classes = ["wood", "glass", "plastic", "metal"]
                print(file, *[f"{classes[i]}: {diff[i]}" for i in range(4) if diff[i]])
    print(f"{passed}/{len(files)}")

    m1 = sum([rmsd(predicted[i], expected[i]) for i in range(4)]) / 4
    with open(sample_path / "result.txt") as real_f, open(
        sample_path / "output.txt"
    ) as expected_f:
        exp = [int(line) for line in expected_f.read().split() if line]
        real = [int(line) for line in real_f.read().split() if line]
    m2 = sum([rmsd([real[i]], [exp[i]]) for i in range(4)]) / 4

    return m1 * 0.75 + m2 * 0.25


def rmsd(predicted: list[int], expected: list[int]) -> float:
    assert len(predicted) == len(expected)
    return np.sqrt(
        sum([(predicted[t] - expected[t]) ** 2 for t in range(len(predicted))])
        / len(predicted)
    )
