# make_tiles.py  – run once before you start labelling
import cv2
import pathlib

BIG_DIR = pathlib.Path("dataset/images/lsb_telcom")  # original pages
TILE_DIR = pathlib.Path("dataset/images/lsb_telcom_tiles4")  # output tiles
TILE = 512
PAD = TILE // 8  # 64-px pad on each side  → 25 % overlap
STRIDE = TILE - 2 * PAD  # 448 px  (perfect sliding window)

TILE_DIR.mkdir(parents=True, exist_ok=True)

for img_path in BIG_DIR.glob("*.*"):
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]

    for y0 in range(0, H, STRIDE):
        for x0 in range(0, W, STRIDE):
            # padded patch  (size = 512 + left pad + right pad)
            x1 = max(x0 - PAD, 0)
            y1 = max(y0 - PAD, 0)
            x2 = min(x0 + TILE + PAD, W)
            y2 = min(y0 + TILE + PAD, H)

            patch = img[y1:y2, x1:x2]
            stem = f"{img_path.stem}_{y0}_{x0}"
            cv2.imwrite(str(TILE_DIR / f"{stem}.png"), patch)

print("✅ Tiles written to", TILE_DIR)
