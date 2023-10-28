import os
import cv2 as cv
from ultralytics import YOLO

model = YOLO("../yolo_v8x.pt")

classes = ["wood", "glass", "plastic", "metal"]

for image_name in sorted(os.listdir("../train_dataset_dataset/video1/frames_rgb")):
    image = cv.imread(f"../train_dataset_dataset/video1/frames_rgb/{image_name}")
    results = model.track(image, persist=True)

    with open(
        f"../train_dataset_dataset/new/video1/labels/{image_name.split('.')[0]}.txt"
    ) as f:
        labels = [
            (int(line.split()[0]), *tuple(map(float, line.split()[1:])))
            for line in f.read().split("\n")
        ]
    img_h, img_w, _ = image.shape
    for cls, x, y, w, h in labels:
        pt1 = (int((x - w / 2) * img_w), int((y - h / 2) * img_h))
        cv.rectangle(
            image,
            pt1,
            (int((x + w / 2) * img_w), int((y + h / 2) * img_h)),
            color=(0, 0, 255),
            thickness=2,
        )
        cv.putText(
            image,
            classes[cls],
            (pt1[0], pt1[1] + 10),
            fontFace=cv.FONT_HERSHEY_COMPLEX,
            fontScale=0.5,
            color=0.5,
            thickness=1,
            lineType=cv.LINE_AA,
            bottomLeftOrigin=False,
        )

    # print(results[0].boxes.cls)
    cv.imshow("window", results[0].plot())

    cv.waitKey(0)

cv.destroyAllWindows()
