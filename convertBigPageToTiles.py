import cv2
import pathlib

PAGE_DIR = pathlib.Path("dataset/images/lsb_telcom")
# YOLO will read these
TILE_DIR = pathlib.Path("dataset/images/lsb_telcom_tiles2")
SIZE, STRIDE = 512, 512  # no overlap; tweak as needed

TILE_DIR.mkdir(parents=True, exist_ok=True)
for page in PAGE_DIR.glob("*.png"):
    img = cv2.imread(str(page))
    H, W = img.shape[:2]
    for y in range(0, H - SIZE + 1, STRIDE):
        for x in range(0, W - SIZE + 1, STRIDE):
            tile = img[y : y + SIZE, x : x + SIZE]
            cv2.imwrite(str(TILE_DIR / f"{page.stem}_{y}_{x}.png"), tile)
