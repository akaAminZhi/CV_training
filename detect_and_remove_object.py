# ──────────────────────────────────────────────────────────────────────────
#  COMMON PARTS – model, helpers, PDF writer
# ──────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import fitz
from ultralytics import YOLO
import easyocr
import torch
from torchvision.ops import nms
import json

reader = easyocr.Reader(
    ["en"], gpu=True
)  # this needs to run only once to load the model into memory
# MODEL = YOLO("./lsbyolo11.pt")
MODEL = YOLO("weights/lsb_fa_m_v3.pt")
# MODEL = YOLO("weights/lsb_receptacle_robotlow_train_v3.pt")

# MODEL = YOLO(
#     "runs/detect/detect_room_v10_yolov11_m_fine_tune_unfreeze_all/weights/best.pt"
# )

input = "pdf_files/LSB_room5b.pdf"

output = "pdf_files/LSB_room5b_detect.pdf"

# input = "TestFiles/LSB_5B.pdf"
# output = "TestFiles/LSB_5B_result.pdf"

TILE = 512
# STRIDE = 512  # = TILE for no-overlap; any ≥1 for approach B

PAD = TILE // 8  # 64 px on each side for 512-tile  ➜ 25 % extra pixels
STRIDE = TILE - 2 * PAD  # 448 px  (perfect sliding window)
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
                imgsz=1024,
                conf=CONF_THR,
                verbose=False,
                iou=0.5,
                agnostic_nms=True,
            ):
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    room_plate = ""
                    if r.names[int(box.cls)] == "room_plate":
                        room_plate = " ".join(
                            reader.readtext(
                                patch[int(by1) : int(by2), int(bx1) : int(bx2)],
                                detail=0,
                            )
                        )
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
                                confidence=f"{conf:.2f}",
                                weighted_score=weighted_score,
                                label=r.names[int(box.cls)],
                                room_plate=room_plate,
                                # label=int(box.cls[0]),  # 类别
                            )
                        )

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
            room_plate=dets[i]["room_plate"],
        )
        for i in keep_indices
    ]


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
                imgsz=1024,
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

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 6)
        cv2.putText(
            img,
            det["label"] + str(det["confidence"]),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
    return img


def remove_object(img, detections):
    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        if det["label"] == "room_plate":
            continue
        img[y1:y2, x1:x2] = (255, 255, 255)

    return img


# def read_text(img, detections):
#     area = []
#     for det in detections:
#         room_device = []
#         x1 = int(det["x1"])
#         y1 = int(det["y1"])
#         x2 = int(det["x2"])
#         y2 = int(det["y2"])
#         cropped_img = img[y1:y2, x1:x2]
#         for r in lighting_MODEL.predict(
#             cropped_img,
#             imgsz=640,
#             conf=0.5,
#             verbose=False,
#             iou=0.3,
#             agnostic_nms=True,
#         ):

#             for box in r.boxes:
#                 label = r.names[int(box.cls)]

#                 if label == "Room_Plate":
#                     bx1, by1, bx2, by2 = box.xyxy[0].tolist()
#                     room_number = reader.readtext(
#                         cropped_img[int(by1) : int(by2), int(bx1) : int(bx2)], detail=0
#                     )[-1]
#                     room_device.insert(0, room_number)
#                 else:
#                     room_device.append(label)
#             if len(room_device) > 1:
#                 area.append(room_device)
#     result = {item[0]: item[1:] for item in area}
#     return result


def download_dets_to_json(dets):
    with open("lsb_fa_room_plate.json", "w") as f:
        json.dump(
            [
                {
                    "x1": d["x1"],
                    "y1": d["y1"],
                    "x2": d["x2"],
                    "y2": d["y2"],
                    "confidence": d["confidence"],
                    "weighted_score": d["weighted_score"],
                    "label": d["label"],
                    "room_plate": d["room_plate"],
                }
                for d in dets  # iterate over all detections
                if d.get("label") == "room_plate"
            ],
            f,
            indent=4,
        )


# ──────────────────────────────────────────────────────────────────────────
#  FULL PIPELINE DRIVER
# ──────────────────────────────────────────────────────────────────────────
def run(pdf_path, out_path, use_overlap=False):
    doc = fitz.open(pdf_path)
    det_pp = []  # list of lists (per page)
    # all_room_info = {}
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        dets = detect_page_B_with_weighted_nms(img)
        # drawback to image
        # room_device_info = read_text(img, dets)
        # all_room_info.update(room_device_info)
        # result_img = draw_detections(img, dets)
        result_img = remove_object(img, dets)
        download_dets_to_json(dets)
        cv2.imwrite("detect_images/lsb/lsb_fa_5b" + str(i) + ".png", result_img)
        # cv2.imwrite("detect_images/lsb/lsb_recep" + str(i) + ".png", result_img)

        det_pp.append(dets)
        print(f"Page {i+1}/{len(doc)}: {len(dets)} boxes")

    doc.close()  # we reopen in writer
    # print(json.dumps(all_room_info, indent=4, ensure_ascii=False, sort_keys=True))
    # with open("output1.json", "w", encoding="utf-8") as f:
    #     json.dump(all_room_info, f, indent=4, ensure_ascii=False, sort_keys=True)
    write_boxes_to_pdf(pdf_path, out_path, det_pp, dpi=300)


run(input, output, use_overlap=True)
