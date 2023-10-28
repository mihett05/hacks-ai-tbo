from parser.yolo_detection import convert_dataset_to_yolo, merge_datasets_from_new

convert_dataset_to_yolo("video0")
convert_dataset_to_yolo("video1")
convert_dataset_to_yolo("video2")

merge_datasets_from_new("all")
