import albumentations as A
import cv2
import os
import glob

# --- Paths ---
image_input_dir = "dataset/LSB-telcom-v4yolov11/train/images"
label_input_dir = "dataset/LSB-telcom-v4yolov11/train/labels"
image_output_dir = "dataset/LSB-telcom-v4yolov11/train/images"
label_output_dir = "dataset/LSB-telcom-v4yolov11/train/labels"
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(label_output_dir, exist_ok=True)

# --- YOLO helpers ---


def load_yolo_labels(label_path):
    if not os.path.exists(label_path):
        return [], []
    labels, class_ids = [], []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # if len(parts) == 5:
            cls, x, y, w, h = map(float, parts)
            labels.append([x, y, w, h])
            class_ids.append(int(cls))
    return labels, class_ids


def save_yolo_labels(path, boxes, class_ids):
    with open(path, "w") as f:
        for cls, box in zip(class_ids, boxes):
            x, y, w, h = box
            f.write(f"{cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n")


def draw_boxes_on_image(image, boxes, class_ids, color=(0, 255, 0)):
    h, w = image.shape[:2]
    out = image.copy()
    for cls, (x, y, bw, bh) in zip(class_ids, boxes):
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out, str(cls), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    return out


# --- Define augmentation variants ---
augmentations = {
    "blur": A.Compose(
        [A.GaussianBlur(blur_limit=(3, 5), p=1.0)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    ),
    "bright": A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    ),
    "dark": A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=-0.3, contrast_limit=-0.3, p=1.0)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    ),
    "rot+15": A.Compose(
        [A.Rotate(limit=(15, 15), border_mode=cv2.BORDER_REFLECT_101, p=1.0)],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.2
        ),
    ),
    "rot-15": A.Compose(
        [A.Rotate(limit=(-15, -15), border_mode=cv2.BORDER_REFLECT_101, p=1.0)],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.2
        ),
    ),
    "scale0.9": A.Compose(
        [A.Affine(scale=(0.9, 0.9), fit_output=True)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    ),
    "scale1.1": A.Compose(
        [A.Affine(scale=(1.1, 1.1), fit_output=True)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    ),
    "safe_crop_512": A.Compose(
        [A.RandomSizedBBoxSafeCrop(width=512, height=512, p=1.0)],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.2
        ),
    ),
    # "crop_512": A.Compose(
    #     [A.RandomCrop(width=512, height=512, p=1.0)],
    #     bbox_params=A.BboxParams(
    #         format="yolo", label_fields=["class_labels"], min_visibility=0.2
    #     ),
    # ),
}

# --- Process all images ---
for img_path in glob.glob(os.path.join(image_input_dir, "*.jpg")):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_input_dir, base + ".txt")

    boxes, class_ids = load_yolo_labels(label_path)
    if not boxes:
        continue

    for aug_name, transform in augmentations.items():
        transformed = transform(image=img, bboxes=boxes, class_labels=class_ids)
        aug_img = transformed["image"]
        aug_boxes = transformed["bboxes"]
        aug_classes = transformed["class_labels"]

        # Output paths
        img_out = os.path.join(image_output_dir, f"{base}_{aug_name}.jpg")
        lbl_out = os.path.join(label_output_dir, f"{base}_{aug_name}.txt")
        dbg_out = os.path.join(image_output_dir, f"{base}_{aug_name}_debug.jpg")

        # Save image and label
        cv2.imwrite(img_out, aug_img)
        save_yolo_labels(lbl_out, aug_boxes, aug_classes)

        # Save debug image with bounding boxes
        # debug_img = draw_boxes_on_image(aug_img, aug_boxes, aug_classes)
        # cv2.imwrite(dbg_out, debug_img)

print("Augmentation complete: Blurred, Bright/Dark, Rotated ±15°, Scaled 0.9 / 1.1.")
