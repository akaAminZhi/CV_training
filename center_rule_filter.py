# centre_rule_filter.py
import pathlib
import shutil
import cv2

IMG_DIR = pathlib.Path("dataset/LSB_telcom_v5yolov11/train/images")
LBL_DIR = pathlib.Path("dataset/LSB_telcom_v5yolov11/train//labels")

OUT_IMG = pathlib.Path("dataset/LSB_telcom_v5yolov11/train_center2/images")
OUT_LBL = pathlib.Path("dataset/LSB_telcom_v5yolov11/train_center2/labels")

TILE = 512
PAD = TILE // 8  # 64-px ignored border
MIN_AREA = 16  # px²

for d in (OUT_IMG, OUT_LBL):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

for img_path in IMG_DIR.glob("*.*"):
    lbl_path = LBL_DIR / f"{img_path.stem}.txt"
    if not lbl_path.exists():  # shouldn’t happen in Roboflow export
        continue

    # ---------- read original label file ----------
    raw_lines = [ln for ln in lbl_path.read_text().splitlines() if ln.strip()]
    orig_has_boxes = bool(raw_lines)

    if orig_has_boxes:
        h, w = cv2.imread(str(img_path)).shape[:2]
        keep = []

        for line in raw_lines:
            cls, xc, yc, bw, bh = map(float, line.split())
            xc_px, yc_px = xc * w, yc * h

            # skip if the centre lies in the 64-px border
            if xc_px < PAD or xc_px > w - PAD or yc_px < PAD or yc_px > h - PAD:
                continue
            # optional: drop if box becomes extremely small
            if bw * w * bh * h < MIN_AREA:
                continue
            keep.append(line)

        # **keep tile only if at least one box survived**
        if not keep:
            # print(img_path)
            continue

        label_out = "\n".join(keep)

    else:
        # original txt was empty → negative tile → keep as is
        label_out = ""  # write empty file

    # ---------- copy image and write new label ----------
    (OUT_IMG / img_path.name).write_bytes(img_path.read_bytes())
    (OUT_LBL / lbl_path.name).write_text(label_out)

print("✅ Filtering done → kept negatives and centre-positives only:")
print("   images:", OUT_IMG)
print("   labels:", OUT_LBL)
