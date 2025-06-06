import albumentations as A
import cv2
import os
import glob
import random

# --- Paths ---
image_input_dir = "dataset/LSB_telcom_v6yolov11/train/images"
label_input_dir = "dataset/LSB_telcom_v6yolov11/train/labels"
image_output_dir = "dataset/LSB_telcom_v6yolov11/train_center_aug/images"
label_output_dir = "dataset/LSB_telcom_v6yolov11/train_center_aug/labels"
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
            if len(parts) == 5:
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


# ── 1. pool of *single* transforms ────────────────────────────────────────────
tf_pool = {
    "blur": A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    "bright": A.RandomBrightnessContrast(0, 0.15, p=1.0),
    "dark": A.RandomBrightnessContrast(-0.15, 0, p=1.0),
    "rotate": A.Rotate(
        limit=15, border_mode=cv2.BORDER_REFLECT_101, p=1.0
    ),  # picks –15 … +15°
    "scale": A.Affine(scale=(0.9, 1.1), fit_output=True, p=1.0),
    "safe_crop": A.RandomSizedBBoxSafeCrop(512, 512, p=1.0),
}


def build_random_pipeline():
    """
    Create (pipeline, combo_name) where `pipeline` is a Compose with a random
    subset of the transforms above, in random order.
    """
    # n = random.randint(1, len(tf_pool))  # 1 … 6 transforms
    n = random.randint(2, 3)  # 1 … 3 transforms

    keys = random.sample(list(tf_pool.keys()), n)  # choose which ones
    random.shuffle(keys)  # random execution order
    pipeline = A.Compose(
        [tf_pool[k] for k in keys],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,  # keep boxes that still occupy ≥20 %
        ),
    )
    return pipeline, "_".join(keys)  # e.g. "safe_crop_dark"


# ───────────────────────────────────────────────────────────────────────────────

# ── 2. generate 8 variants for every source image ─────────────────────────────
for img_path in glob.glob(os.path.join(image_input_dir, "*.jpg")):
    img = cv2.imread(img_path)
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_input_dir, base + ".txt")

    boxes, class_ids = load_yolo_labels(label_path)
    if not boxes:
        continue

    for idx in range(4):  # ← exactly eight
        pipe, combo = build_random_pipeline()

        tr = pipe(image=img, bboxes=boxes, class_labels=class_ids)
        if not tr["bboxes"]:  # all boxes lost → skip
            continue

        img_out = os.path.join(image_output_dir, f"{base}_aug{idx}_{combo}.jpg")
        lbl_out = os.path.join(label_output_dir, f"{base}_aug{idx}_{combo}.txt")

        cv2.imwrite(img_out, tr["image"])
        save_yolo_labels(lbl_out, tr["bboxes"], tr["class_labels"])

print("Finished: every source image now has 8 randomly-augmented versions!")
