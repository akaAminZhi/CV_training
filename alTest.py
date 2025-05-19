import cv2
import albumentations as A
from albumentations.core.transforms_interface import DualTransform


class OccludeObject(DualTransform):
    def __init__(self, method="black", p=1.0):
        super().__init__(always_apply=False, p=p)
        self.method = method

    def apply(self, img, **params):
        for x_c, y_c, bw, bh in params["bboxes"]:
            h, w = img.shape[:2]
            x1, y1 = int((x_c - bw / 2) * w), int((y_c - bh / 2) * h)
            x2, y2 = int((x_c + bw / 2) * w), int((y_c + bh / 2) * h)
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 0, 0) if self.method == "black" else (255, 0, 255),
                -1,
            )
        return img

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def get_transform_init_args_names(self):
        return ("method",)


img = cv2.imread("detection_result0.jpg")
bboxes = [[0.5, 0.5, 0.2, 0.2]]  # 随便造个框
pipe = A.Compose(
    [A.Rotate(limit=15, p=1), OccludeObject(method="black", p=1)],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)
out = pipe(image=img, bboxes=bboxes, class_labels=[0])["image"]
cv2.imwrite("test_out.jpg", out)
