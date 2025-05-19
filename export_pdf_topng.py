# pdf_to_png.py
import fitz
import pathlib
import concurrent.futures
import cv2
import numpy as np

PDF_PATH = "pdf_files/Telecom Device val.pdf"
OUT_DIR = pathlib.Path("dataset/images/lsb_telcom")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render(idx):
    doc = fitz.open(PDF_PATH)  # reopen per worker
    pix = doc[idx].get_pixmap(dpi=300)  # 400 dpi keeps fine detail
    img = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(OUT_DIR / f"page_{idx+1:04d}.png"), img)


with concurrent.futures.ThreadPoolExecutor() as ex:
    ex.map(render, range(fitz.open(PDF_PATH).page_count))
