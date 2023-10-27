from parser import coco

samples = ["video0", "video1", "video2"]
for sample in samples:
    with open(f"dataset/{sample}.json", "w") as f:
        coco.sample_to_coco(sample, f)

files = [open(f"dataset/{sample}.json") for sample in samples]
with open(f"dataset/video.json", "w") as f:
    coco.merge_coco(files, f)
for file in files:
    file.close()
