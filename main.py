from parser import coco

with open("output.json", "w") as f:
    coco.sample_to_coco("video0", f)
