# ──────────────────────────────────────────────────────────────────────────
#  COMMON PARTS – model, helpers, PDF writer
# ──────────────────────────────────────────────────────────────────────────
import torch
import pickle
import cv2
import numpy as np
import fitz
from ultralytics import YOLO
from torchvision.ops import nms
import json
from paddleocr import PaddleOCR


MODEL = None
ocr_MODEL = None
YOLO_Model_Path = "weights/IAD06_mechnical_v3.pt"
input = "TestFiles/IAD06_Roof_CH.pdf"
output = "TestFiles/IAD06_Roof_CH2_result.pdf"

TILE = 2048

PAD = TILE // 4  # 64 px on each side for 512-tile  ➜ 25 % extra pixels, TILE // 8
STRIDE = TILE - 2 * PAD  # 448 px  (perfect sliding window)
CONF_THR = 0.6


def write_boxes_to_pdf(pdf_path, out_path, detections_pp, dpi=300):
    doc = fitz.open(pdf_path)
    objects = {}
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
            has_this_object = objects.get(det["label"])
            if has_this_object:
                objects[det["label"]] = objects[det["label"]] + 1
            else:
                objects[det["label"]] = 1
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
    sorted_dict = {key: objects[key] for key in sorted(objects)}
    for key, value in sorted_dict.items():
        print(f"{key}: {value}")

    print("✅ Saved:", out_path)


def _img_to_pdf_matrices(page, *, dpi=300):
    """Return (mat, imat) where imat converts pixel‑coords → PDF."""
    zoom = dpi / 72  # 72 pt per inch
    mat = fitz.Matrix(zoom, zoom).prerotate(page.rotation)
    imat = fitz.Matrix(mat)
    imat.invert()
    return mat, imat


def detect_page_B_with_weighted_nms(img):
    H, W = img.shape[:2]
    dets = []

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
                iou=0.3,
                agnostic_nms=True,
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

    if not dets:
        return []
    # drawed_img = draw_detections(img, dets)
    # cv2.imwrite("test_result" + ".png", drawed_img)
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
            str(det["weighted_score"]) + " conf: " + str(det["confidence"]),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return img


def read_text_from_image(img, detections):
    filter_result = []

    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        result = ocr_MODEL.predict(input=img[y1:y2, x1:x2])

        for res in result:
            temp = []
            for item in res["rec_texts"]:
                if len(item.strip("_")) >= 6:
                    temp.append(item.strip("_"))

        filter_result.append(temp)
    # print(filter_result)
    return filter_result


def load_dets_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ──────────────────────────────────────────────────────────────────────────
#  FULL PIPELINE DRIVER
# ──────────────────────────────────────────────────────────────────────────
def run(pdf_path, out_path, Yolo=True, OCR=True):
    if Yolo:
        global MODEL
        MODEL = YOLO(YOLO_Model_Path)
    if OCR:
        # 初始化 PaddleOCR 实例
        global ocr_MODEL
        ocr_MODEL = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            text_detection_model_dir=r"C:\Users\zhimin qin\.paddlex\official_models\PP-OCRv5_server_det",
            text_recognition_model_dir=r"C:\Users\zhimin qin\.paddlex\official_models\PP-OCRv5_server_rec",
            textline_orientation_model_dir=r"C:\Users\zhimin qin\.paddlex\official_models\PP-LCNet_x1_0_textline_ori",
        )
    doc = fitz.open(pdf_path)
    det_pp = []  # list of lists (per page)
    det_text = []
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
        if OCR:
            text = read_text_from_image(img, dets)
            det_text.append(text)
        det_pp.append(dets)
        # drawback to image
        # result_img = img
        # result_img = draw_detections(img, dets)

    # doc.save(out_path, garbage=4, deflate=True)
    with open("IAD06_Roof_CH.pkl", "wb") as f:
        pickle.dump(det_text, f)
    doc.close()  # we reopen in writer
    write_boxes_to_pdf(pdf_path, out_path, det_pp, dpi=300)


run(input, output, Yolo=True, OCR=True)
