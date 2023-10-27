from parser import coco

for sample in ["video0", "video1", "video2"]:
    with open(f"dataset/{sample}.json", "w") as f:
        coco.sample_to_coco(sample, f)
