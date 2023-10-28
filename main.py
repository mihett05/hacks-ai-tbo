from parser import coco
# from parser import yolo_detection

for sample in ["video0", "video1", "video2"]:
    with open(f"dataset/{sample}.json", "w") as f:
        coco.sample_to_coco(sample, f)

# yolo_detection.sample_to_yolo_detection("video0", 30)
