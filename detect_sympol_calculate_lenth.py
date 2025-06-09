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

# MODEL = YOLO("./lsbyolo11.pt")
MODEL = YOLO("weights/lsb_receptacle_robotlow_train_v2.pt")
# MODEL = YOLO(
#     "runs/detect/lsb_power_plan_receptacle_roboflow_v8_yolov11_m_fine_tune_unfreeze_all/weights/best.pt"
# )

input = "TestFiles/Test2.pdf"
output = "TestFiles/Test2_with_yolo11_m_fine_tune_v1.pdf"

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


def nms_manul(dets, thr=0.5):
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
    return nms_manul(all_det, thr=0.5)


# ──────────────────────────────────────────────────────────────────────────
#  APPROACH B – OVERLAP + CENTRE-CROP
# ──────────────────────────────────────────────────────────────────────────


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


def detect_page_B_with_torchvision_nms(img):
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
                        dets.append(
                            (
                                bx1 + offx,
                                by1 + offy,
                                bx2 + offx,
                                by2 + offy,
                                box.conf[0].item(),
                                r.names[int(box.cls)],
                                # int(box.cls[0]),  # 类别索引
                            )
                        )

    if not dets:
        return []

    # 转为 Tensor

    boxes = torch.tensor([d[:4] for d in dets])
    scores = torch.tensor([d[4] for d in dets])
    # labels = [d[5] for d in dets]  # 如果你要做按类 NMS，可用

    keep = nms(boxes, scores, iou_threshold=0.5)

    # 返回保留的结果
    return [
        dict(
            x1=dets[i][0],
            y1=dets[i][1],
            x2=dets[i][2],
            y2=dets[i][3],
            confidence=dets[i][4],
            label=dets[i][5],
        )
        for i in keep
    ]


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
                                area=area,
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
            else KMeans(n_clusters=need, n_init="auto").fit_predict(merged[idx])
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
    color_dev=(154, 26, 0),
    color_panel=(0, 128, 255),
    thickness=1,
):
    def _polyline(color, pts):
        cv2.polylines(
            img, [np.int32(pts)], isClosed=False, color=color, thickness=thickness
        )

    for seg in dev_paths:
        _polyline(color_dev, seg)
    for seg in panel_paths:
        _polyline(color_panel, seg)


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
            # dets = detect_page_B(img)  # Approach B
            # dets = nms_manul(dets, thr=0.5)
            dets = detect_page_B_with_weighted_nms(img)
        else:
            dets = detect_page_A(img)  # Approach A

        # drawback to image
        # result_img = img
        result_img = draw_detections(img, dets)
        # cv2.imwrite("detection_result" + str(i) + ".png", result_img)
        px2m = 1
        jb_centroids, dev2jb, groups, dev_coords = auto_place_junction_boxes(
            dets,
            capacity=8,
            eps_px=500,
            grid_px=50,
            merge_eps=50,
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
        dev_paths, panel_paths = [], []
        total_dev_px, total_panel_px = 0.0, 0.0  # 以像素为单位累积

        for dev_idx, jb_idx in dev2jb.items():
            x1, y1 = dev_coords[dev_idx]  # 设备中心 (px)
            x2, y2 = jb_centroids[jb_idx]  # JB 坐标  (px)

            # 水平→垂直折线
            dev_paths.append(((x1, y1), (x2, y1), (x2, y2)))
            total_dev_px += abs(x2 - x1) + abs(y2 - y1)

        # ──────────────────────────────────────────────
        # ③ JB → 最近 panel 路径 & 长度
        # ──────────────────────────────────────────────

        for xj, yj in jb_centroids:
            if not panel_centres:
                break
            xp, yp = min(panel_centres, key=lambda p: hypot(p[0] - xj, p[1] - yj))
            panel_paths.append(((xj, yj), (xj, yp), (xp, yp)))
            total_panel_px += abs(yp - yj) + abs(xp - xj)

        # ──────────────────────────────────────────────
        # ④ 距离换算：像素 → 米        （若你用像素坐标）
        # ──────────────────────────────────────────────
        px2m = 0.0254 / 300  # ← 如果是 300 dpi，保持与前文一致
        px2m = 3.0 / 367.0
        total_dev_len_m = total_dev_px * px2m
        total_jb_len_m = total_panel_px * px2m
        print(f"设备 → JB  总长 : {total_dev_len_m:.2f} m")
        print(f"JB  → Panel 总长 : {total_jb_len_m:.2f} m")
        print(f"Page {i+1}/{len(doc)}: {len(dets)} boxes")

    doc.close()  # we reopen in writer
    # write_boxes_to_pdf(pdf_path, out_path, det_pp, dpi=300)


# ──────────────────────────────────────────────────────────────────────────
#  EXAMPLE CALLS
# ──────────────────────────────────────────────────────────────────────────
# 1. No-overlap tiling + global NMS
# run("Test2.pdf", "annotated_NMS.pdf", use_overlap=False)

# 2. Overlap tiling (64-px pad) + centre-crop
#    You can reduce STRIDE to e.g. 448 (= TILE-2*PAD) for true sliding window.
run(input, output, use_overlap=True)
