# split_train_val_test.py
import pathlib
import random
import shutil

SRC_IMG = pathlib.Path("dataset/LSB_telcom_v6yolov11/train_center_aug/images")
SRC_LBL = pathlib.Path("dataset/LSB_telcom_v6yolov11/train_center_aug/labels")

DEST_ROOT = pathlib.Path("export_split")
SPLIT_RATIOS = dict(train=0.90, valid=0.10, test=0)

SEED = 42  # reproducible shuffle

# ───────────────────────────────────────────────────────────
# 1. collect stems that have BOTH image and label (.txt may be empty)
# ───────────────────────────────────────────────────────────
pairs = []
for img_path in SRC_IMG.glob("*.*"):
    lbl_path = SRC_LBL / f"{img_path.stem}.txt"
    if not lbl_path.exists():
        # if you really want to keep these images, create an empty label file:
        # lbl_path.write_text("")
        continue
    pairs.append((img_path, lbl_path))

n_total = len(pairs)
if not n_total:
    raise RuntimeError("No image-label pairs found in export_centre/")

# ───────────────────────────────────────────────────────────
# 2. shuffle once
# ───────────────────────────────────────────────────────────
random.Random(SEED).shuffle(pairs)

# ───────────────────────────────────────────────────────────
# 3. compute split counts
# ───────────────────────────────────────────────────────────
n_train = int(n_total * SPLIT_RATIOS["train"])
n_val = int(n_total * SPLIT_RATIOS["valid"])
n_test = n_total - n_train - n_val  # remainder

split_tags = ["train"] * n_train + ["valid"] * n_val + ["test"] * n_test

# ───────────────────────────────────────────────────────────
# 4. prepare destination folders
# ───────────────────────────────────────────────────────────
for tag in SPLIT_RATIOS:
    (DEST_ROOT / tag / "images").mkdir(parents=True, exist_ok=True)
    (DEST_ROOT / tag / "labels").mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────────────────
# 5. copy files into their split
# ───────────────────────────────────────────────────────────
for (img_src, lbl_src), tag in zip(pairs, split_tags):
    img_dst = DEST_ROOT / tag / "images" / img_src.name
    lbl_dst = DEST_ROOT / tag / "labels" / lbl_src.name
    shutil.copy2(img_src, img_dst)
    shutil.copy2(lbl_src, lbl_dst)

print(f"✅ Split     : {n_train} train / {n_val} val / {n_test} test")
print(f"   Output dir: {DEST_ROOT}")

# ───────────────────────────────────────────────────────────
# 6. write Ultralytics data.yaml
# ───────────────────────────────────────────────────────────
# yaml_text = f"""# auto-generated
# train: {DEST_ROOT/'train/images'}
# val:   {DEST_ROOT/'val/images'}
# test:  {DEST_ROOT/'test/images'}

# nc: 1               # ← adjust to your class count
# names: ["object"]   # ← replace with your class names
# """
# (DEST_ROOT / "data.yaml").write_text(yaml_text)
print("📝 data.yaml written.")
