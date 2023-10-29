import os
from pathlib import Path


def output_to_csv(sample_path: Path):
    header = "frame_id;wood;glass;plastic;metal".split(";")
    output = [header]
    for frame in sorted(list(os.listdir(sample_path / "frames_output"))):
        with open(sample_path / "frames_output" / frame) as f:
            output.append(
                ["f" + str(int(frame.split(".")[0]))]
                + [d for d in f.read().split() if d]
            )
    with open(sample_path / "output.txt") as f:
        output.append(["f_all"] + [d for d in f.read().split() if d])
    with open(sample_path / "output.csv", "w") as f:
        f.write("\n".join([";".join(row) for row in output]))
