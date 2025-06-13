# ──────────────────────────────────────────────────────────────────────────
#  COMMON PARTS – model, helpers, PDF writer
# ──────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import fitz
from ultralytics import YOLO
import torch
from torchvision.ops import nms
from sklearn.cluster import DBSCAN, KMeans
from collections import defaultdict
from math import hypot
import json


MODEL = None
YOLO_Model_Path = "weights/lsb_receptacle_robotlow_train_v3.pt"
input = "TestFiles/CMSC_lenth_test4_clean2.pdf"
output = "TestFiles/CMSC_lenth_test4_with_roboflow_m_fine_tune_v1.pdf"

TILE = 512
# STRIDE = 512  # = TILE for no-overlap; any ≥1 for approach B

PAD = TILE // 8  # 64 px on each side for 512-tile  ➜ 25 % extra pixels
STRIDE = TILE - 2 * PAD  # 448 px  (perfect sliding window)
CONF_THR = 0.6

# Visual parameters for annotation appearance
ANN_COLORS = {
    "DET_RECT": (1, 0, 0),  # red boxes around detections
    "JB_CIRCLE": (0, 0, 1),  # blue circles for Junction Boxes
    "DEV_PATH": (1, 0, 0),  # red polylines Device → JB
    "PANEL_PATH": (1, 0.5, 0),  # orange polylines JB → Panel
}
JB_DIAM_PT = 8  # diameter of JB circle in PDF points (≈1.4 mm)
LINE_WIDTH_PT = 0.5  # stroke width for paths in PDF points


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


def _img_to_pdf_matrices(page, *, dpi=300):
    """Return (mat, imat) where imat converts pixel‑coords → PDF."""
    zoom = dpi / 72  # 72 pt per inch
    mat = fitz.Matrix(zoom, zoom).prerotate(page.rotation)
    imat = fitz.Matrix(mat)
    imat.invert()
    return mat, imat


# def add_annotations_to_page(
#     page, dets, jb_centroids, dev_paths, panel_paths, *, dpi=300
# ):
#     """Add rectangles, circles and polylines to *page* as PDF annotations."""
#     _, imat = _img_to_pdf_matrices(page, dpi=dpi)

#     # ① Rectangles for raw detections
#     for det in dets:
#         p1 = fitz.Point(det["x1"], det["y1"]) * imat
#         p2 = fitz.Point(det["x2"], det["y2"]) * imat
#         rect = fitz.Rect(p1, p2).normalize()
#         if rect.width == 0 or rect.height == 0:
#             continue
#         ann = page.add_rect_annot(rect)
#         ann.set_info(subject=det["label"])
#         ann.set_colors(stroke=ANN_COLORS["DET_RECT"])
#         ann.set_border(width=LINE_WIDTH_PT)
#         ann.update()

#     # ② Junction‐box circles
#     for idx, (x_px, y_px) in enumerate(jb_centroids):
#         ctr = fitz.Point(x_px, y_px) * imat
#         r = JB_DIAM_PT / 2  # radius in *PDF points*
#         circ_rect = fitz.Rect(ctr.x - r, ctr.y - r, ctr.x + r, ctr.y + r)
#         ann = page.add_circle_annot(circ_rect)
#         ann.set_info(subject=f"JB{idx}")
#         ann.set_colors(stroke=ANN_COLORS["JB_CIRCLE"])
#         ann.set_border(width=LINE_WIDTH_PT)
#         ann.update()

#     # helper to add a polyline
#     def _add_polyline(pts_px, color, subj):
#         pts_pdf = [fitz.Point(x, y) * imat for (x, y) in pts_px]
#         ann = page.add_polyline_annot(pts_pdf)
#         ann.set_colors(stroke=color)
#         ann.set_info(subject=subj)
#         ann.set_border(width=LINE_WIDTH_PT)
#         ann.update()

#     # ③ Polylines: Device → JB
#     for seg in dev_paths:
#         _add_polyline(seg, ANN_COLORS["DEV_PATH"], "Device→JB")

#     # ④ Polylines: JB → Panel
#     for seg in panel_paths:
#         _add_polyline(seg, ANN_COLORS["PANEL_PATH"], "JB→Panel")

# ──────────────────────────────────────────────────────────────────────────
#  Helper: polyline length in *pixels*
# ──────────────────────────────────────────────────────────────────────────


def polyline_length_px(pts):
    """Return total *L‑path* (Manhattan) length for a poly‑line (list[(x,y)])."""
    return sum(
        abs(x2 - x1) + abs(y2 - y1) for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:])
    )


def add_annotations_to_page(
    page, dets, jb_centroids, dev_paths, panel_paths, *, dpi=300
):
    """Write rectangles, circles and polylines to *page* as PDF annotations."""
    _, imat = _img_to_pdf_matrices(page, dpi=dpi)

    # ① rectangles for detections
    for det in dets:
        p1 = fitz.Point(det["x1"], det["y1"]) * imat
        p2 = fitz.Point(det["x2"], det["y2"]) * imat
        rect = fitz.Rect(p1, p2).normalize()
        if rect.width == 0 or rect.height == 0:
            continue
        ann = page.add_rect_annot(rect)
        ann.set_colors(stroke=ANN_COLORS["DET_RECT"])
        ann.set_border(width=LINE_WIDTH_PT)
        ann.set_info(subject=det["label"])
        ann.update()

    # ② circles for JB
    for idx, (x_px, y_px) in enumerate(jb_centroids):
        ctr = fitz.Point(x_px, y_px) * imat
        r = JB_DIAM_PT / 2
        circ_rect = fitz.Rect(ctr.x - r, ctr.y - r, ctr.x + r, ctr.y + r)
        ann = page.add_circle_annot(circ_rect)
        ann.set_colors(stroke=ANN_COLORS["JB_CIRCLE"])
        ann.set_border(width=LINE_WIDTH_PT)
        ann.set_info(subject=f"JB{idx}")
        ann.update()

    # helper: add polyline + comment length(px)
    def _add_polyline(pts_px, color, subj):
        pts_pdf = [fitz.Point(x, y) * imat for (x, y) in pts_px]
        ann = page.add_polyline_annot(pts_pdf)
        ann.set_colors(stroke=color)
        ann.set_border(width=LINE_WIDTH_PT)
        len_px = int(round(polyline_length_px(pts_px)))
        ann.set_info(subject=subj, content=f"{len_px}")
        ann.update()

    # ③ device → JB paths
    for seg in dev_paths:
        _add_polyline(seg, ANN_COLORS["DEV_PATH"], "Device→JB")

    # ④ JB → Panel paths
    for seg in panel_paths:
        _add_polyline(seg, ANN_COLORS["PANEL_PATH"], "JB→Panel")


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

    # 转换为 Tensor
    boxes = torch.tensor([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets])
    scores = torch.tensor([d["weighted_score"] for d in dets])  # 用加权分数做排序
    # labels = [d["label"] for d in dets]  # 可用于扩展类别感知 NMS

    keep_indices = nms(boxes, scores, iou_threshold=0.5)

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
        # cv2.putText(
        #     img,
        #     det["label"],
        #     (x1, y1 - 5),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 0, 0),
        #     1,
        #     cv2.LINE_AA,
        # )
    return img


def draw_junction_boxes(
    img, jb_centroids, *, px2m=1.0, color_jb=(0, 0, 255), radius_px=20
):
    """
    Overlay each JB as a red circle + ID text.

    Parameters
    ----------
    jb_centroids : list[(x_m, y_m)]
        Output from `auto_place_junction_boxes` in metres.
    px2m : float   –  metre ➜ pixel scaling factor (inverse of draw_detections).
    """
    m2px = 1.0 / px2m

    for idx, (x_m, y_m) in enumerate(jb_centroids):
        x_px, y_px = int(x_m * m2px), int(y_m * m2px)
        cv2.circle(img, (x_px, y_px), radius_px, color_jb, -1)
        cv2.putText(
            img,
            f"     JB{idx}",
            (x_px + 5, y_px - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color_jb,
            1,
            cv2.LINE_AA,
        )


def auto_place_junction_boxes(
    dets,
    *,
    capacity=8,
    eps_px=400,
    grid_px=50,
    merge_eps=50,
    min_conf=0.30,
    keep_labels=None,  # ← 传 None = “除了 panel 全要”
    return_coords=False,  # 新：是否把设备坐标一并返回
):
    """
    返回:
        jb_centroids   : [(x_px, y_px), ...]
        dev2jb         : {device_idx -> jb_idx}
        groups         : {jb_idx -> [device_idx]}
        dev_coords(*)  : [(x_px, y_px), ...]  ↖ return_coords=True 时才有
    """
    # ---------- 0. 设备筛选 ---------------------------------
    if keep_labels is None:
        keep_labels = {d["label"] for d in dets if d["label"] != "PANEL"}

    centres = []
    for d in dets:
        if d["confidence"] < min_conf or d["label"] not in keep_labels:
            continue
        centres.append([(d["x1"] + d["x2"]) * 0.5, (d["y1"] + d["y2"]) * 0.5])
    centres = np.asarray(centres, float)
    if centres.size == 0:
        raise ValueError("No valid devices after filtering")

    # ---------- 1. 合并并排插座 ------------------------------
    merge_lbl = DBSCAN(eps=merge_eps, min_samples=1).fit_predict(centres)

    merged = []  # 合并后线路中心
    orig2merged = {}  # 原始设备 idx -> merged idx
    for g in np.unique(merge_lbl):
        idx = np.where(merge_lbl == g)[0]
        pt = centres[idx].mean(axis=0)
        merged_idx = len(merged)
        merged.append(pt)
        for i in idx:
            orig2merged[i] = merged_idx
    merged = np.asarray(merged, float)

    # ---------- 2. JB 聚类 + 容量切分 ------------------------
    big_lbl = DBSCAN(eps=eps_px, min_samples=1).fit_predict(merged)

    def snap(v, s=grid_px):
        return round(v / s) * s

    final_lbl = -np.ones(len(merged), int)
    jb_xy = []
    cid = 0
    for c in np.unique(big_lbl):
        idx = np.where(big_lbl == c)[0]
        need = int(np.ceil(len(idx) / capacity))
        sub = (
            np.zeros(len(idx), int)
            if need == 1
            else KMeans(n_clusters=need, n_init=10, random_state=0).fit_predict(
                merged[idx]
            )
        )
        for s in np.unique(sub):
            sub_idx = idx[sub == s]
            pt = merged[sub_idx].mean(axis=0)
            jb_xy.append((snap(pt[0]), snap(pt[1])))
            final_lbl[sub_idx] = cid
            cid += 1

    # ---------- 3. 构建完整映射 -----------------------------
    dev2jb = {}
    for orig_i, m_i in orig2merged.items():
        dev2jb[orig_i] = int(final_lbl[m_i])

    groups = defaultdict(list)
    for d, jb in dev2jb.items():
        groups[jb].append(d)

    if return_coords:
        return jb_xy, dev2jb, dict(groups), centres  # centres 按 orig_i 顺序
    return jb_xy, dev2jb, dict(groups)


# ─────────────────────────────────────────────────────────────
# ①  从 dets 中分离 panel centre
# ─────────────────────────────────────────────────────────────
def extract_panel_centres(dets, *, label_set=frozenset({"PANEL"})):
    centres = []
    for d in dets:
        if d["label"] in label_set:
            cx = (d["x1"] + d["x2"]) * 0.5
            cy = (d["y1"] + d["y2"]) * 0.5
            centres.append((cx, cy))
    return centres  # list[(x_px, y_px)]


# ─────────────────────────────────────────────────────────────
# ②  生成路径 (曼哈顿折线：水平后垂直)
# ─────────────────────────────────────────────────────────────
def build_paths(jb_centroids, dev2jb, dev_centres, panel_centres):
    """
    返回两组折线:
        dev_paths   : [( (x1,y1), (x_mid,y1), (x_mid,y2) ) …]
        panel_paths : [( (xj,yj), (xj, yp), (xp, yp) ) …]
    """
    # -------- a) 设备 → JB -----------------------------------
    dev_paths = []
    for dev_idx, jb_idx in dev2jb.items():
        x1, y1 = dev_centres[dev_idx]
        x2, y2 = jb_centroids[jb_idx]
        # L 型：先水平后垂直
        dev_paths.append(((x1, y1), (x2, y1), (x2, y2)))

    # -------- b) JB → 最近 panel -----------------------------
    panel_paths = []
    for xj, yj in jb_centroids:
        if not panel_centres:
            break
        # 找最近 panel
        xp, yp = min(panel_centres, key=lambda p: hypot(p[0] - xj, p[1] - yj))
        panel_paths.append(((xj, yj), (xj, yp), (xp, yp)))

    return dev_paths, panel_paths


# ─────────────────────────────────────────────────────────────
# ③  在图上画路径
# ─────────────────────────────────────────────────────────────
def draw_paths(
    img,
    dev_paths,
    panel_paths,
    *,
    color_dev=(0, 0, 255),
    color_panel=(0, 128, 255),
    thickness=1,
):
    def _polyline(color, pts):
        cv2.polylines(
            img, [np.int32(pts)], isClosed=False, color=color, thickness=thickness
        )

    def _polyline_panel(color, pts):
        cv2.polylines(img, [np.int32(pts)], isClosed=False, color=color, thickness=2)

    for seg in dev_paths:
        _polyline(color_dev, seg)
    for seg in panel_paths:
        _polyline_panel(color_panel, seg)


def load_dets_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


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

        # drawback to image
        # result_img = img
        result_img = draw_detections(img, dets)
        # cv2.imwrite("detection_result" + str(i) + ".png", result_img)
        px2m = 1
        jb_centroids, dev2jb, groups, dev_coords = auto_place_junction_boxes(
            dets,
            capacity=8,
            eps_px=500,
            grid_px=10,
            merge_eps=5,
            keep_labels=None,  # “除了 panel 都要”
            return_coords=True,  # ← 新
        )
        # 4) 提取 panel 中心
        panel_centres = extract_panel_centres(dets)

        # 5) 生成路径
        # 生成路径
        dev_paths, panel_paths = build_paths(
            jb_centroids, dev2jb, dev_coords, panel_centres  # ← 用同源坐标
        )
        # (5) overlay JBs
        draw_junction_boxes(result_img, jb_centroids, px2m=px2m)
        draw_paths(result_img, dev_paths, panel_paths)
        cv2.imwrite("detection_result" + str(i) + ".png", result_img)
        det_pp.append(dets)
        total_dev_px = sum(polyline_length_px(p) for p in dev_paths)
        total_panel_px = sum(polyline_length_px(p) for p in panel_paths)

        # ──────────────────────────────────────────────
        # ④ 距离换算：像素 → 米        （若你用像素坐标）
        # ──────────────────────────────────────────────
        # px2m = 0.0254 / 300  # ← 如果是 300 dpi，保持与前文一致
        px2m = 3.0 / 367.0
        total_dev_len_m = total_dev_px * px2m
        total_jb_len_m = total_panel_px * px2m
        print(f"设备 → JB  总长 : {total_dev_len_m:.2f} m {total_dev_px} px")
        print(f"JB  → Panel 总长 : {total_jb_len_m:.2f} m {total_panel_px} px")
        print(f"Page {i+1}/{len(doc)}: {len(dets)} boxes")
        add_annotations_to_page(
            page, dets, jb_centroids, dev_paths, panel_paths, dpi=300
        )
    doc.save(out_path, garbage=4, deflate=True)

    doc.close()  # we reopen in writer
    # write_boxes_to_pdf(pdf_path, out_path, det_pp, dpi=300)


run(input, output, Yolo=False)
