# ──────────────────────────────────────────────────────────────────────────
#  COMMON PARTS – model, helpers, PDF writer
# ──────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import fitz
from ultralytics import YOLO
import torch
from torchvision.ops import nms
import json
import math

MODEL = None
YOLO_Model_Path = "weights/lsb_receptacle_yolo26_large_v5.pt"
file_name = "TestFiles/lsb_recep/l_6_A"
input = f"{file_name}.pdf"
output = f"{file_name}_result.pdf"

TILE = 512
total_count = 0
PAD = TILE // 4  # 64 px on each side for 512-tile  ➜ 25 % extra pixels
STRIDE = TILE - 2 * PAD  # 448 px  (perfect sliding window)
CONF_THR = 0.6


OKABE_ITO = [
    (0.85, 0.10, 0.10),  # red
    (0.00, 0.00, 0.00),  # black
    (0.00, 0.55, 0.20),  # green
    (0.00, 0.35, 0.85),  # blue
    (0.90, 0.50, 0.00),  # orange
    (0.55, 0.20, 0.70),  # purple
    (0.55, 0.30, 0.10),  # brown
    (0.00, 0.65, 0.65),  # teal
    (0.95, 0.20, 0.55),  # pink
    (0.75, 0.75, 0.00),  # mustard
    (0.10, 0.70, 0.90),  # cyan
    (0.60, 0.00, 0.40),  # magenta wine
]


def rebuild_and_save(doc, out_path):
    newdoc = fitz.open()
    # copy all pages; this reconstructs objects
    newdoc.insert_pdf(doc)  # or insert in chunks if the PDF is huge
    newdoc.save(out_path, garbage=2, clean=True, deflate=True, incremental=False)
    newdoc.close()


def img_point_to_page_point(
    page: fitz.Page, px: float, py: float, dpi: int = 144
) -> fitz.Point:
    """
    把 get_pixmap(dpi=...) 得到的图像像素坐标，转换回 page 的原始 PDF 坐标。
    适用于 page.rotation = 0/90/180/270。
    """
    zoom = dpi / 72.0

    # 1) 像素坐标 -> 旋转后的 page 坐标
    p_rot = fitz.Point(px / zoom, py / zoom)

    # 2) 旋转后的 page 坐标 -> 原始 page 坐标
    p_pdf = p_rot * page.derotation_matrix
    return p_pdf


def img_rect_to_page_rect(
    page: fitz.Page, x1: float, y1: float, x2: float, y2: float, dpi: int = 144
) -> fitz.Rect:
    p1 = img_point_to_page_point(page, x1, y1, dpi)
    p2 = img_point_to_page_point(page, x2, y2, dpi)
    return fitz.Rect(p1, p2).normalize()


def write_boxes_to_pdf(pdf_path, out_path, detections_pp, dpi=144, palette=OKABE_ITO):
    import fitz
    import hashlib

    doc = fitz.open(pdf_path)
    objects = {}
    palette_n = len(palette)

    def get_color_for_label(label: str):
        # stable hash -> stable index (across pages & runs)
        h = hashlib.md5(label.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % palette_n
        return palette[idx]

    for i, page in enumerate(doc):

        for det in detections_pp[i]:
            label = det["label"]
            objects[label] = objects.get(label, 0) + 1

            rect = img_rect_to_page_rect(
                page, det["x1"], det["y1"], det["x2"], det["y2"], dpi=dpi
            )
            if rect.width <= 0 or rect.height <= 0:
                continue
            if rect.width == 0 or rect.height == 0:
                continue

            annot = page.add_rect_annot(rect)
            annot.set_info(
                subject=str(label), content=f"confidence: {det['confidence']:.2f}"
            )
            annot.set_colors(stroke=get_color_for_label(label))
            annot.set_border(width=1)
            annot.update()

    sorted_dict = {key: objects[key] for key in sorted(objects)}
    for key, value in sorted_dict.items():
        print(f"{key}: {value}")

    rebuild_and_save(doc, out_path)
    doc.close()
    print("✅ Saved:", out_path)


def _img_to_pdf_matrices(page, *, dpi=144):
    """Return (mat, imat) where imat converts pixel‑coords → PDF."""
    zoom = dpi / 72  # 72 pt per inch
    mat = fitz.Matrix(zoom, zoom).prerotate(page.rotation)
    imat = fitz.Matrix(mat)
    imat.invert()
    return mat, imat


def detect_page_B_with_weighted_nms(img):
    H, W = img.shape[:2]
    dets = []
    global total_count
    for y0 in range(0, H, STRIDE):
        for x0 in range(0, W, STRIDE):
            x1 = max(x0 - PAD, 0)
            y1 = max(y0 - PAD, 0)
            x2 = min(x0 + TILE + PAD, W)
            y2 = min(y0 + TILE + PAD, H)
            patch = img[y1:y2, x1:x2]
            offx, offy = x1, y1

            for r in MODEL.predict(
                patch,
                conf=CONF_THR,
                verbose=False,
                # classes=[9],
                # iou=0.3,
                # agnostic_nms=False,
            ):
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    if PAD <= cx <= PAD + TILE and PAD <= cy <= PAD + TILE:
                        conf = box.conf[0].item()
                        area = (bx2 - bx1) * (by2 - by1)
                        weighted_score = conf * area
                        dets.append(
                            dict(
                                x1=bx1 + offx,
                                y1=by1 + offy,
                                x2=bx2 + offx,
                                y2=by2 + offy,
                                confidence=conf,
                                weighted_score=weighted_score,
                                label=r.names[int(box.cls)],
                                # label=int(box.cls[0]),  # 类别
                            )
                        )
                        # if r.names[int(box.cls)] == "changed":
                        #     x1 = math.ceil(bx1 + offx)
                        #     y1 = math.ceil(by1 + offy)
                        #     x2 = math.ceil(bx2 + offx)
                        #     y2 = math.ceil(by2 + offy)
                        #     patch = img[y1:y2, x1:x2]
                        #     cv2.imwrite(
                        #         "detect_images/lsb_fa_test/lsb_fa_changed"
                        #         + str(total_count)
                        #         + ".png",
                        #         patch,
                        #     )
                        #     total_count += 1

    if not dets:
        return []

    # 转换为 Tensor
    boxes = torch.tensor([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets])
    scores = torch.tensor([d["weighted_score"] for d in dets])  # 用加权分数做排序
    # labels = [d["label"] for d in dets]  # 可用于扩展类别感知 NMS

    keep_indices = nms(boxes, scores, iou_threshold=0.3)

    # 返回保留结果
    return [
        dict(
            x1=dets[i]["x1"],
            y1=dets[i]["y1"],
            x2=dets[i]["x2"],
            y2=dets[i]["y2"],
            confidence=dets[i]["confidence"],
            weighted_score=dets[i]["weighted_score"],
            label=dets[i]["label"],
        )
        for i in keep_indices
    ]


def draw_detections(img, detections):
    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            img,
            det["label"],
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return img


def load_dets_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def check_file_rotation_and_recover(pdf_input, pdf_output):
    doc = fitz.open(pdf_input)
    for i, page in enumerate(doc):
        # 选择某一页（例如第1页，索引从0开始）
        page = doc[0]

        # 获取旋转角度
        rotation = page.rotation

        print(f"Page 1 rotation: {rotation} degrees")
        page.set_rotation(0)
    doc.save("TestFiles/lsb_cd_recep_2.pdf")
    doc.close()


# ──────────────────────────────────────────────────────────────────────────
#  FULL PIPELINE DRIVER
# ──────────────────────────────────────────────────────────────────────────
def run(pdf_path, out_path, Yolo=True):
    if Yolo:
        global MODEL
        MODEL = YOLO(YOLO_Model_Path)
    doc = fitz.open(pdf_path)
    det_pp = []  # list of lists (per page)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        if Yolo:
            dets = detect_page_B_with_weighted_nms(img)
            with open("detectResult.json", "w") as f:
                json.dump(dets, f, indent=4)
        else:
            dets = load_dets_from_json("detectResult.json")
        det_pp.append(dets)
        # drawback to image
        # result_img = img
        # result_img = draw_detections(img, dets)
        # cv2.imwrite("detect_images/lsb_fa/lsb_fa" + str(i) + ".png", result_img)

    # doc.save(out_path, garbage=4, deflate=True)

    doc.close()  # we reopen in writer
    write_boxes_to_pdf(pdf_path, out_path, det_pp, dpi=300)


run(input, output, Yolo=True)
