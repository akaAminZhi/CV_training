# ──────────────────────────────────────────────────────────────────────────
#  COMMON PARTS – model, helpers, PDF writer
# ──────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import fitz
from ultralytics import YOLO

# MODEL = YOLO("./lsbyolo11.pt")
MODEL = YOLO("weights/receptale_yolo11m.pt")
# MODEL = YOLO("runs/detect/LSB_receptacle_v1_from_yolov11m/weights/best.pt")

input = "TestFiles/receptacle_all_CMSC_page3.pdf"
output = "TestFiles/receptacle_all_CMSC_page3_result_with_YOLO11_m.pdf"

TILE = 512
STRIDE = 512  # = TILE for no-overlap; any ≥1 for approach B
CONF_THR = 0.6


def write_boxes_to_pdf(pdf_path, out_path, detections_pp, dpi=300):
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom).prerotate(
            page.rotation
        )  # 1️⃣ like get_pixmap  PDF->png

        imat = fitz.Matrix(mat)
        imat.invert()  # 2️⃣ invert matrix png->PDF
        for det in detections_pp[i]:
            # p1 = fitz.Point(det["x1"], det["y1"]) * imat
            # p2 = fitz.Point(det["x2"], det["y2"]) * imat
            # rect = fitz.Rect(p1, p2)

            p1 = fitz.Point(det["x1"], det["y1"]) * imat
            p2 = fitz.Point(det["x2"], det["y2"]) * imat
            rect = fitz.Rect(p1, p2).normalize()

            # skip zero-area boxes
            if rect.width == 0 or rect.height == 0:
                continue
            annot = page.add_rect_annot(rect)

            # Set the subject of the annotation
            annot.set_info(subject=f"{det['label']}")
            annot.set_colors(stroke=(1, 0, 0))  # 红色边框
            annot.set_border(width=1)  # 边框宽度
            annot.update()
    doc.save(out_path, garbage=4, deflate=True)
    doc.close()
    print("✅ Saved:", out_path)


# ──────────────────────────────────────────────────────────────────────────
#  APPROACH A – NO-OVERLAP + GLOBAL NMS
# ──────────────────────────────────────────────────────────────────────────
def iou(b1, b2):
    # b? = [x1,y1,x2,y2]
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if not inter:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def nms(dets, thr=0.5):
    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    keep = []
    while dets:
        cur = dets.pop(0)
        keep.append(cur)
        dets = [
            d
            for d in dets
            if iou(
                [cur["x1"], cur["y1"], cur["x2"], cur["y2"]],
                [d["x1"], d["y1"], d["x2"], d["y2"]],
            )
            < thr
        ]
    return keep


def detect_page_A(img):
    H, W = img.shape[:2]
    all_det = []
    for y in range(0, H - TILE + 1, STRIDE):
        for x in range(0, W - TILE + 1, STRIDE):
            tile = img[y : y + TILE, x : x + TILE]
            for r in MODEL.predict(tile, conf=CONF_THR, verbose=False):
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    all_det.append(
                        dict(
                            label=r.names[int(box.cls)],
                            confidence=box.conf[0].item(),
                            x1=x1 + x,
                            y1=y1 + y,
                            x2=x2 + x,
                            y2=y2 + y,
                        )
                    )
    return nms(all_det, thr=0.5)


# ──────────────────────────────────────────────────────────────────────────
#  APPROACH B – OVERLAP + CENTRE-CROP
# ──────────────────────────────────────────────────────────────────────────
PAD = TILE // 8  # 64 px on each side for 512-tile  ➜ 25 % extra pixels


def detect_page_B(img):
    H, W = img.shape[:2]
    dets = []
    for y0 in range(0, H, STRIDE):
        for x0 in range(0, W, STRIDE):
            # read enlarged patch (mind edges)
            x1 = max(x0 - PAD, 0)
            y1 = max(y0 - PAD, 0)
            x2 = min(x0 + TILE + PAD, W)
            y2 = min(y0 + TILE + PAD, H)
            patch = img[y1:y2, x1:x2]
            offx, offy = x1, y1
            """
            假设模型在一张图像中检测到了两个框，一个是“人”，另一个是“狗”，但它们重叠很多：

            如果 agnostic_nms=False:因为“人”和“狗”是不同类别, 两个框都会保留。

            如果 agnostic_nms=True:如果它们的重叠度超过 iou 阈值，比如 0.5, 就会只保留置信度高的那个框，另一个会被去掉。

            """
            for r in MODEL.predict(
                patch,
                conf=CONF_THR,
                verbose=False,
                iou=0.3,
                agnostic_nms=True,
            ):
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    # keep only if centre in inner TILE × TILE
                    if PAD <= cx <= PAD + TILE and PAD <= cy <= PAD + TILE:
                        dets.append(
                            dict(
                                label=r.names[int(box.cls)],
                                confidence=box.conf[0].item(),
                                x1=bx1 + offx,
                                y1=by1 + offy,
                                x2=bx2 + offx,
                                y2=by2 + offy,
                            )
                        )
    return dets  # already unique → no NMS needed


def draw_detections(img, detections):
    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            det["label"],
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return img


# ──────────────────────────────────────────────────────────────────────────
#  FULL PIPELINE DRIVER
# ──────────────────────────────────────────────────────────────────────────
def run(pdf_path, out_path, use_overlap=False):
    doc = fitz.open(pdf_path)
    det_pp = []  # list of lists (per page)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        if use_overlap:
            dets = detect_page_B(img)  # Approach B
            dets = nms(dets, thr=0.5)
        else:
            dets = detect_page_A(img)  # Approach A

        # drawback to image
        # result_img = draw_detections(img, dets)
        # cv2.imwrite("detection_result" + str(i) + ".png", result_img)

        det_pp.append(dets)
        print(f"Page {i+1}/{len(doc)}: {len(dets)} boxes")

    doc.close()  # we reopen in writer
    write_boxes_to_pdf(pdf_path, out_path, det_pp, dpi=300)


# ──────────────────────────────────────────────────────────────────────────
#  EXAMPLE CALLS
# ──────────────────────────────────────────────────────────────────────────
# 1. No-overlap tiling + global NMS
# run("Test2.pdf", "annotated_NMS.pdf", use_overlap=False)

# 2. Overlap tiling (64-px pad) + centre-crop
#    You can reduce STRIDE to e.g. 448 (= TILE-2*PAD) for true sliding window.
run(input, output, use_overlap=True)
