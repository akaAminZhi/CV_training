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

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

MODEL = None
MODEL2 = None
YOLO_Model_Path = "weights/lsb_fa_asi_cloud_large_v1.pt"
YOLO_Model_Path2 = "weights/lsb_fa_large_v4.pt"
file_name = "TestFiles/lsb_fa/lsb_fa_asi4"
file_name2 = "TestFiles/lsb_fa/lsb_fa_asi3"

input = f"{file_name}.pdf"
output = f"{file_name}_result.pdf"

total_changed = 0
TILE = 512

PAD = TILE // 4  # 64 px on each side for 512-tile  ➜ 25 % extra pixels
STRIDE = TILE - 2 * PAD  # 448 px  (perfect sliding window)
CONF_THR = 0.6


def _pdf_escape_paren(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _upsert_pdf_key(obj: str, key: str, value_expr: str) -> str:
    pat = rf"{re.escape(key)}\s+(?:\([^)]*\)|<<.*?>>|\[.*?\]|\d+\s+\d+\s+R|/[^/\s]+)"
    if re.search(pat, obj, flags=re.S):
        return re.sub(pat, f"{key} {value_expr}", obj, flags=re.S)

    i = obj.rfind(">>")
    if i == -1:
        raise RuntimeError("Unexpected PDF object format")
    return obj[:i].rstrip() + f"\n  {key} {value_expr}\n" + obj[i:]


def _rect_to_pdf_array(page: fitz.Page, r: fitz.Rect) -> str:
    """
    Convert MuPDF rect (page coordinate system used by add_rect_annot etc.)
    into PDF /Rect (native PDF user space) using the official transform.
    """
    m = fitz.Matrix(page.transformation_matrix)  # PDF -> MuPDF
    m.invert()  # now MuPDF -> PDF

    p0 = fitz.Point(r.x0, r.y0) * m
    p1 = fitz.Point(r.x1, r.y1) * m
    rr = fitz.Rect(p0, p1).normalize()
    return f"[{rr.x0:.6f} {rr.y0:.6f} {rr.x1:.6f} {rr.y1:.6f}]"


@dataclass(frozen=True)
class StampTemplate:
    subject: str
    ap: str
    it: Optional[str]
    rect: fitz.Rect
    w: float
    h: float


class StampLibrary:
    class SubjectView:
        def __init__(self, lib: "StampLibrary", subject: str):
            self.lib = lib
            self.subject = subject

        def size(self, *, index: int = 0) -> tuple[float, float]:
            """返回模板尺寸 (w_pt, h_pt)"""
            tpl = self.lib._by_subject[self.subject][index]
            return tpl.w, tpl.h

        def clone(
            self,
            page: fitz.Page,
            rect: fitz.Rect,
            *,
            index: int = 0,
            set_subject: Optional[str] = None,
            clear_comments: bool = True,
        ) -> fitz.Annot:
            tpl = self.lib._by_subject[self.subject][index]
            write_subj = set_subject if set_subject is not None else tpl.subject
            return self.lib._clone_keep_size(
                page=page,
                tpl=tpl,
                new_rect=rect,
                subject_to_write=write_subj,
                clear_comments=clear_comments,
            )

        def clone_at(
            self,
            page: fitz.Page,
            x: float,
            y: float,
            *,
            index: int = 0,
            anchor: str = "tl",  # "tl" | "center" | "bl"
            set_subject: Optional[str] = None,
            clear_comments: bool = True,
        ) -> fitz.Annot:
            """
            只给一个点 (x,y)，自动用模板 w/h 生成 rect，并克隆。
            anchor:
            - tl: (x,y) 作为左上角
            - center: (x,y) 作为中心
            - bl: (x,y) 作为左下角
            """
            tpl = self.lib._by_subject[self.subject][index]
            w, h = tpl.w, tpl.h

            if anchor == "tl":
                rect = fitz.Rect(x, y, x + w, y + h)
            elif anchor == "center":
                rect = fitz.Rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            elif anchor == "bl":
                rect = fitz.Rect(x, y - h, x + w, y)
            else:
                raise ValueError("anchor must be one of: 'tl', 'center', 'bl'")

            return self.clone(
                page,
                rect,
                index=index,
                set_subject=set_subject,
                clear_comments=clear_comments,
            )

    def __init__(self, doc: fitz.Document):
        self.doc = doc
        self._by_subject: Dict[str, List[StampTemplate]] = {}

    def __getitem__(self, subject: str) -> "StampLibrary.SubjectView":
        if subject not in self._by_subject:
            raise KeyError(
                f"Subject not found: {subject}. Available: {list(self._by_subject.keys())}"
            )
        return StampLibrary.SubjectView(self, subject)

    def subjects(self) -> List[str]:
        return sorted(self._by_subject.keys())

    def load_from_page(
        self,
        page: fitz.Page,
        *,
        allow_no_subject=False,
        no_subject_key="__NO_SUBJECT__",
    ):
        """
        只从指定 page 里缓存 stamp（适合你“插入模板页”方案）。
        """
        for a in page.annots() or []:
            if a.type[0] != fitz.PDF_ANNOT_STAMP:
                continue

            subj = ((a.info or {}).get("subject") or "").strip()
            if not subj:
                if not allow_no_subject:
                    continue
                subj = no_subject_key

            tpl = self._make_template(a, subj)
            self._by_subject.setdefault(subj, []).append(tpl)

        return self

    def _make_template(self, annot: fitz.Annot, subject: str) -> StampTemplate:
        obj = self.doc.xref_object(annot.xref)
        ap_m = re.search(r"/AP\s+<<.*?>>", obj, flags=re.S)
        if not ap_m:
            raise RuntimeError("Stamp has no /AP (not custom appearance)")

        it = "/IT /StampSnapshot" if "/IT /StampSnapshot" in obj else None
        r = fitz.Rect(annot.rect)

        return StampTemplate(
            subject=subject,
            ap=ap_m.group(0),
            it=it,
            rect=r,
            w=float(r.width),
            h=float(r.height),
        )

    def _clone_keep_size(
        self,
        page: fitz.Page,
        tpl: StampTemplate,
        new_rect: fitz.Rect,
        subject_to_write: Optional[str],
        clear_comments: bool,
    ) -> fitz.Annot:
        a = page.add_stamp_annot(new_rect, stamp=0)
        a.update()

        xref = a.xref
        obj = self.doc.xref_object(xref)

        # inject appearance
        if tpl.it and "/IT" not in obj:
            obj = obj.replace("/Subtype /Stamp", f"/Subtype /Stamp\n  {tpl.it}")
        obj = re.sub(r"/AP\s+<<.*?>>", tpl.ap, obj, flags=re.S)

        # force rect (avoid height override)
        obj = re.sub(
            r"/Rect\s+\[.*?\]",
            f"/Rect {_rect_to_pdf_array(page, new_rect)}",
            obj,
            flags=re.S,
        )

        # remove default Approved comment
        if clear_comments:
            obj = re.sub(r"\s*/Name\s*/\S+", "", obj)
            obj = re.sub(r"\s*/Contents\s*\(.*?\)", "", obj, flags=re.S)

        # write subject
        if subject_to_write:
            obj = _upsert_pdf_key(
                obj, "/Subj", f"({_pdf_escape_paren(subject_to_write)})"
            )

        self.doc.update_object(xref, obj)
        return a


def with_template_page(
    target_doc: fitz.Document, template_pdf_path: str, *, insert_at: int = 0
):
    """
    把 template_pdf_path 的第一页插入 target_doc，返回 (template_page_index, template_page_obj)。
    用完后你负责删除该页。
    """
    tpl_doc = fitz.open(template_pdf_path)
    if tpl_doc.page_count < 1:
        raise ValueError("Template PDF has no pages.")
    target_doc.insert_pdf(tpl_doc, from_page=0, to_page=0, start_at=insert_at)
    tpl_doc.close()

    template_page_index = insert_at
    template_page = target_doc[template_page_index]
    return template_page_index, template_page


# ===== 配色（高亮对比，灰底可见，16 色循环） =====
OKABE_ITO = [
    (0.98, 0.55, 0.15),  # orange
    (0.05, 0.55, 0.95),  # bright blue
    (0.00, 0.80, 0.55),  # teal
    (0.90, 0.25, 0.75),  # magenta
    (0.98, 0.90, 0.25),  # yellow
    (0.95, 0.35, 0.25),  # coral
    (0.40, 0.75, 1.00),  # sky
    (0.35, 0.90, 0.35),  # lime
    (1.00, 0.45, 0.65),  # pink
    (0.20, 0.55, 1.00),  # cobalt
    (1.00, 0.75, 0.30),  # amber
    (0.60, 0.35, 0.90),  # violet
    (0.15, 0.80, 0.90),  # aqua
    (0.90, 0.50, 0.40),  # salmon
    (0.80, 0.80, 0.10),  # chartreuse
    (0.55, 0.60, 0.95),  # periwinkle
]


def rebuild_and_save(doc, out_path):
    newdoc = fitz.open()
    # copy all pages; this reconstructs objects
    newdoc.insert_pdf(doc)  # or insert in chunks if the PDF is huge
    newdoc.save(out_path, garbage=2, clean=True, deflate=True, incremental=False)
    newdoc.close()


def write_boxes_to_pdf(pdf_path, out_path, detections_pp, dpi=144, palette=OKABE_ITO):
    import fitz
    import hashlib

    doc = fitz.open(pdf_path)
    # 1) 插入模板页
    tpl_index, tpl_page = with_template_page(
        doc, "TestFiles/lsb_fa/lsb_fa_lable.pdf", insert_at=0
    )

    # 2) 从模板页加载 stamps 到库
    lib = StampLibrary(doc).load_from_page(tpl_page)
    objects = {}
    palette_n = len(palette)

    def get_color_for_label(label: str):
        # stable hash -> stable index (across pages & runs)
        h = hashlib.md5(label.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % palette_n
        return palette[idx]

    # template page inserted at index 0
    TEMPLATE_PAGE_INDEX = tpl_index  # usually 0
    PAGE_OFFSET = 1  # because we insert at 0

    for i in range(len(detections_pp)):
        page = doc[i + PAGE_OFFSET]  # original page i

        # pixel -> pdf matrix (respect page rotation)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom).prerotate(page.rotation)
        imat = fitz.Matrix(mat)
        imat.invert()

        for det in detections_pp[i]:
            label = str(det["label"])
            objects[label] = objects.get(label, 0) + 1

            # pixel coords -> pdf coords
            p1 = fitz.Point(det["x1"], det["y1"]) * imat  # top-left in PDF coords
            p2 = fitz.Point(det["x2"], det["y2"]) * imat  # bottom-right in PDF coords

            # normalize just in case (rotation / coordinate ordering)
            rect = fitz.Rect(p1, p2).normalize()
            if rect.width <= 0 or rect.height <= 0:
                continue

            # ✅ place stamp at detection box top-left (keep template size)
            # if label not found in lib, skip (or fallback to rectangle)
            try:
                lib[label].clone_at(page, x=rect.x0, y=rect.y0, anchor="tl")
                # lib[label].clone(page, rect)
            except KeyError:
                # fallback: keep your old rectangle behavior if no stamp template exists
                annot = page.add_rect_annot(rect)
                annot.set_info(subject=label)
                annot.set_colors(stroke=get_color_for_label(label))
                annot.set_border(width=1)
                annot.update()

    sorted_dict = {key: objects[key] for key in sorted(objects)}
    for key, value in sorted_dict.items():
        print(f"{key}: {value}")

    # 4) 删除临时模板页
    doc.delete_page(TEMPLATE_PAGE_INDEX)
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


def detect_page_B_with_weighted_nms(img, model):
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

            for r in model.predict(
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

    if not dets:
        return []

    # 转换为 Tensor
    boxes = torch.tensor([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets])
    scores = torch.tensor([d["weighted_score"] for d in dets])  # 用加权分数做排序
    # labels = [d["label"] for d in dets]  # 可用于扩展类别感知 NMS

    keep_indices = nms(boxes, scores, iou_threshold=0.3)

    # 返回保留结果
    results = []
    for i in keep_indices:
        results.append(
            dict(
                x1=dets[i]["x1"],
                y1=dets[i]["y1"],
                x2=dets[i]["x2"],
                y2=dets[i]["y2"],
                confidence=dets[i]["confidence"],
                weighted_score=dets[i]["weighted_score"],
                label=dets[i]["label"],
            )
        )
    return results


def detect_result_with_cloud(img, model, results):
    dets = []
    global total_changed
    for result in results:
        x1 = math.ceil(result["x1"])
        y1 = math.ceil(result["y1"])
        x2 = math.ceil(result["x2"])
        y2 = math.ceil(result["y2"])

        patch = img[y1:y2, x1:x2]
        cv2.imwrite(
            "detect_images/lsb_fa_test/lsb_fa_asi03_" + str(total_changed) + ".png",
            patch,
        )
        total_changed += 1
        offx, offy = x1, y1
        for r in model.predict(
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

    keep_indices = nms(boxes, scores, iou_threshold=0.3)

    # 返回保留结果
    results = []
    for i in keep_indices:
        results.append(
            dict(
                x1=dets[i]["x1"],
                y1=dets[i]["y1"],
                x2=dets[i]["x2"],
                y2=dets[i]["y2"],
                confidence=dets[i]["confidence"],
                weighted_score=dets[i]["weighted_score"],
                label=dets[i]["label"],
            )
        )
    return results


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
        global MODEL2

        MODEL = YOLO(YOLO_Model_Path)
        MODEL2 = YOLO(YOLO_Model_Path2)
    doc = fitz.open(pdf_path)
    det_pp = []  # list of lists (per page)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=144)
        img = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        if Yolo:
            dets_result = detect_page_B_with_weighted_nms(img, MODEL)
            dets = detect_result_with_cloud(img, MODEL2, dets_result)
            # dets.extend(dets_result)
            with open("detectResult.json", "w") as f:
                json.dump(dets, f, indent=4)
        else:
            dets = load_dets_from_json("detectResult.json")
        det_pp.append(dets)
        # det_pp.append(dets_result)
        # drawback to image
        # result_img = img
        # result_img = draw_detections(img, dets)
        # cv2.imwrite("detect_images/lsb_fa/lsb_fa" + str(i) + ".png", result_img)

    # doc.save(out_path, garbage=4, deflate=True)

    doc.close()  # we reopen in writer
    write_boxes_to_pdf(pdf_path, out_path, det_pp, dpi=144)


run(input, output, Yolo=True)
