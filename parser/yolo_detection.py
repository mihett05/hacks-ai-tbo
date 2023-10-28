import os
from pathlib import Path

dataset_path = Path(__file__).parent.parent / "train_dataset_dataset"
crop_dataset_path = dataset_path / "crop_dataset"

if not os.path.exists(crop_dataset_path):
    os.makedirs(crop_dataset_path)


def sample_to_yolo_detection(sample_name: str, image_crop_index: int = None):
    annotations_path = dataset_path / sample_name / f"{sample_name}.txt"

    with open(file=annotations_path, mode="r") as file_data:
        lines = [list(map(float, line.split(", "))) for line in file_data.readlines()]

        image_index = 0
        print_buffer = []
        for line in lines:
            if image_crop_index is None or image_index <= image_crop_index:
                if int(line[0]) != image_index:
                    save_file_name = crop_dataset_path / f"{str(image_index).zfill(4)}.txt"
                    print(f"{print_buffer=}")
                    print("\n".join(print_buffer), file=open(save_file_name, "w"))

                    print_buffer = []
                    image_index += 1

                # print(*line[2:-1])
                print_buffer.append(
                    "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(int(line[2]), *line[3:])
                )
            else:
                break
