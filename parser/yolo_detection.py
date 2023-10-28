import os
from pathlib import Path
from shutil import copyfile
from random import shuffle

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
                    save_file_name = (
                        crop_dataset_path / f"{str(image_index).zfill(4)}.txt"
                    )
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


def convert_dataset_to_yolo(sample_name: str):
    sample = dataset_path / sample_name
    _, _, files = list(os.walk(sample / "frames_rgb"))[0]
    files.sort()

    result = dataset_path / "new" / f"{sample_name}"
    if not os.path.exists(dataset_path / "new"):
        os.mkdir(dataset_path / "new")
    if not os.path.exists(result):
        os.mkdir(result)

    # Копирование картинок
    if not os.path.exists(result / "images"):
        os.mkdir(result / "images")
        for file in files:
            copyfile(sample / "frames_rgb" / file, result / "images" / file)

    # Копирование аннотаций
    if not os.path.exists(result / "labels"):
        os.mkdir(result / "labels")
        with open(sample / f"{sample_name}.txt") as f:
            boxes = {}
            for image_id, *data in [
                line.split(", ") for line in f.read().split("\n") if line
            ]:
                key = int(image_id)
                if key not in boxes:
                    boxes[key] = []
                boxes[key].append(" ".join(data[1:]))
            for key in boxes:
                with open(result / "labels" / f"{key:04}.txt", "w") as f:
                    f.write("\n".join(boxes[key]))


def merge_datasets_from_new(result_sample: str):
    if not (dataset_path / "new").exists():
        raise ValueError("Nothing to merge")
    result = dataset_path / result_sample
    if not result.exists():
        os.mkdir(dataset_path / result_sample)

    _, dirs, _ = list(os.walk(dataset_path / "new"))[0]
    dirs.sort()

    images = []
    labels = []
    for d in dirs:
        _, _, files = list(os.walk(dataset_path / "new" / d / "images"))[0]
        images += [
            str(dataset_path / "new" / d / "images" / file) for file in sorted(files)
        ]

        _, _, files = list(os.walk(dataset_path / "new" / d / "labels"))[0]
        labels += [
            str(dataset_path / "new" / d / "labels" / file) for file in sorted(files)
        ]
    images = {i: path for i, path in enumerate(images)}
    labels = {i: path for i, path in enumerate(labels)}
    images_p = result / "images"
    labels_p = result / "labels"
    if not images_p.exists():
        os.mkdir(images_p)
    if not labels_p.exists():
        os.mkdir(labels_p)

    folders = {
        "train": int(0.7 * len(images)),
        "val": int(0.1 * len(images)),
        "test": int(0.2 * len(images)),
    }

    for folder in folders:
        for path in [images_p, labels_p]:
            if not (path / folder).exists():
                os.mkdir(path / folder)

    keys = list(images.keys())
    shuffle(keys)
    for key in keys:
        result_folder = [folder for folder in folders if folders[folder] > 0]
        if result_folder:
            result_folder = result_folder[0]
            folders[result_folder] -= 1
        else:
            result_folder = "train"
        copyfile(images[key], images_p / result_folder / f"{key:04}.png")
        copyfile(labels[key], labels_p / result_folder / f"{key:04}.txt")
