# ──────────────────────────────────────────────────────────────────────────────
#  Dual-DPI YOLO PDF Detection Pipeline
#
#  144 DPI: detect revision clouds / cloud regions
#  300 DPI: detect small fire-alarm objects inside the detected cloud regions
#
#  Key idea:
#    - Each detection carries its own `dpi` field.
#    - When writing annotations back to the PDF, coordinates are converted using
#      the detection's own DPI, not a single global DPI.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import cv2
import fitz
import hashlib
import json
import math
import re
import torch
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from torchvision.ops import nms
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────────

MODEL = None
MODEL2 = None

YOLO_Model_Path = "weights/lsb_fa_asi_cloud_large_v7.pt"
YOLO_Model_Path2 = "weights/lsb_fa_yolo26_large_v3.pt"

file_name = "TestFiles/lsb_fa/new_ASI/28 - Fire Alarm_asi009"
file_name2 = "TestFiles/lsb_fa/old_ASI/28 - Fire Alarm_asi_old"

input_pdf = f"{file_name}.pdf"
input_previous_asi = f"{file_name2}.pdf"

output_pdf = f"{file_name}_result2.pdf"
output_previous_pdf = f"{file_name2}_result.pdf"

STAMP_TEMPLATE_PDF = "TestFiles/lsb_fa/lsb_fa_lable.pdf"

# Two resolutions
CLOUD_DPI = 144
OBJECT_DPI = 300

# Tiling for cloud detection
TILE = 1024
PAD = TILE // 4
STRIDE = TILE - 2 * PAD

# YOLO thresholds
CONF_THR = 0.8
CONF_THR_CLOUD = 0.7

# NMS thresholds
NMS_IOU_CLOUD = 0.5
NMS_IOU_OBJECT = 0.5

# Optional crop expansion around each cloud box on 300-DPI image
OBJECT_CROP_MARGIN_PX = 16

# Debug JSON path
DEBUG_JSON_PATH = "detectResult_dual_dpi.json"


# ──────────────────────────────────────────────────────────────────────────────
#  COLORS
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
#  PDF STAMP HELPERS
# ──────────────────────────────────────────────────────────────────────────────


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
    Convert MuPDF rect into PDF /Rect native user space.
    """
    m = fitz.Matrix(page.transformation_matrix)  # PDF -> MuPDF
    m.invert()  # MuPDF -> PDF

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

        def size(self, *, index: int = 0) -> Tuple[float, float]:
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
            comment: Optional[str] = None,
        ) -> fitz.Annot:
            tpl = self.lib._by_subject[self.subject][index]
            write_subj = set_subject if set_subject is not None else tpl.subject

            return self.lib._clone_keep_size(
                page=page,
                tpl=tpl,
                new_rect=rect,
                subject_to_write=write_subj,
                clear_comments=clear_comments,
                comment=comment,
            )

        def clone_at(
            self,
            page: fitz.Page,
            x: float,
            y: float,
            *,
            index: int = 0,
            anchor: str = "tl",
            set_subject: Optional[str] = None,
            clear_comments: bool = True,
            comment: Optional[str] = None,
        ) -> fitz.Annot:
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
                comment=comment,
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
        allow_no_subject: bool = False,
        no_subject_key: str = "__NO_SUBJECT__",
    ):
        """
        Load stamp templates from the inserted template page.
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
            raise RuntimeError("Stamp has no /AP; custom appearance is missing.")

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
        comment: Optional[str] = None,
    ) -> fitz.Annot:
        a = page.add_stamp_annot(new_rect, stamp=0)
        a.update()

        xref = a.xref
        obj = self.doc.xref_object(xref)

        # Inject appearance.
        if tpl.it and "/IT" not in obj:
            obj = obj.replace("/Subtype /Stamp", f"/Subtype /Stamp\n  {tpl.it}")

        obj = re.sub(r"/AP\s+<<.*?>>", tpl.ap, obj, flags=re.S)

        # Force rect to keep template size.
        obj = re.sub(
            r"/Rect\s+\[.*?\]",
            f"/Rect {_rect_to_pdf_array(page, new_rect)}",
            obj,
            flags=re.S,
        )

        # Remove default Approved comment.
        if clear_comments:
            obj = re.sub(r"\s*/Name\s*/\S+", "", obj)
            obj = re.sub(r"\s*/Contents\s*\(.*?\)", "", obj, flags=re.S)

        # Write subject.
        if subject_to_write:
            obj = _upsert_pdf_key(
                obj,
                "/Subj",
                f"({_pdf_escape_paren(subject_to_write)})",
            )

        # Write comment.
        if comment is not None:
            obj = _upsert_pdf_key(
                obj,
                "/Contents",
                f"({_pdf_escape_paren(comment)})",
            )

        self.doc.update_object(xref, obj)
        return a


def with_template_page(
    target_doc: fitz.Document,
    template_pdf_path: str,
    *,
    insert_at: int = 0,
):
    """
    Insert the first page of template_pdf_path into target_doc.
    Return (template_page_index, template_page_obj).
    Caller should delete the inserted page after use.
    """
    tpl_doc = fitz.open(template_pdf_path)

    if tpl_doc.page_count < 1:
        tpl_doc.close()
        raise ValueError("Template PDF has no pages.")

    target_doc.insert_pdf(tpl_doc, from_page=0, to_page=0, start_at=insert_at)
    tpl_doc.close()

    template_page_index = insert_at
    template_page = target_doc[template_page_index]

    return template_page_index, template_page


# ──────────────────────────────────────────────────────────────────────────────
#  PDF / IMAGE COORDINATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def rebuild_and_save(doc: fitz.Document, out_path: str):
    """
    Rebuild PDF objects and save cleanly.
    """
    newdoc = fitz.open()
    newdoc.insert_pdf(doc)
    newdoc.save(
        out_path,
        garbage=2,
        clean=True,
        deflate=True,
        incremental=False,
    )
    newdoc.close()


def render_page_to_bgr(page: fitz.Page, dpi: int) -> np.ndarray:
    """
    Render a PDF page to an OpenCV BGR image.

    This avoids pix.tobytes() + cv2.imdecode(), so it is usually faster.
    """
    pix = page.get_pixmap(dpi=dpi, alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height,
        pix.width,
        pix.n,
    )

    if pix.n == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    raise ValueError(f"Unsupported pixmap channel count: {pix.n}")


def img_point_to_page_point(
    page: fitz.Page,
    px: float,
    py: float,
    dpi: int,
) -> fitz.Point:
    """
    Convert a pixel point from page.get_pixmap(dpi=...) image coordinates
    back to the original PDF page coordinate system.

    Works with page rotation 0/90/180/270.
    """
    zoom = dpi / 72.0

    # Pixel coordinate -> rotated page coordinate.
    p_rot = fitz.Point(px / zoom, py / zoom)

    # Rotated page coordinate -> original page coordinate.
    p_pdf = p_rot * page.derotation_matrix

    return p_pdf


def img_rect_to_page_rect(
    page: fitz.Page,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    dpi: int,
) -> fitz.Rect:
    p1 = img_point_to_page_point(page, x1, y1, dpi)
    p2 = img_point_to_page_point(page, x2, y2, dpi)
    return fitz.Rect(p1, p2).normalize()


def page_rect_to_img_rect(
    page: fitz.Page,
    rect: fitz.Rect,
    dpi: int,
) -> Tuple[float, float, float, float]:
    """
    Convert original PDF page rect to image pixel rect.

    This is useful if later you want to map detections through PDF coordinates
    between current and previous ASI pages.
    """
    zoom = dpi / 72.0
    mat = page.rotation_matrix * fitz.Matrix(zoom, zoom)

    p1 = fitz.Point(rect.x0, rect.y0) * mat
    p2 = fitz.Point(rect.x1, rect.y1) * mat
    r = fitz.Rect(p1, p2).normalize()

    return r.x0, r.y0, r.x1, r.y1


# ──────────────────────────────────────────────────────────────────────────────
#  DETECTION HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def make_detection(
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float,
    label: str,
    cls_id: int,
    dpi: int,
) -> dict:
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "confidence": float(confidence),
        "weighted_score": float(confidence * area),
        "label": str(label),
        "cls_id": int(cls_id),
        "dpi": int(dpi),
    }


def scale_det_to_dpi(det: dict, dst_dpi: int) -> dict:
    """
    Scale detection coordinates from det['dpi'] to dst_dpi.
    """
    src_dpi = det.get("dpi")
    if src_dpi is None:
        raise ValueError("Detection is missing the 'dpi' field.")

    scale = dst_dpi / src_dpi

    new_det = dict(det)
    new_det["x1"] = float(det["x1"] * scale)
    new_det["y1"] = float(det["y1"] * scale)
    new_det["x2"] = float(det["x2"] * scale)
    new_det["y2"] = float(det["y2"] * scale)
    new_det["dpi"] = int(dst_dpi)

    return new_det


def clip_box_to_image(
    det: dict,
    *,
    width: int,
    height: int,
    margin_px: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    x1 = math.floor(min(det["x1"], det["x2"])) - margin_px
    y1 = math.floor(min(det["y1"], det["y2"])) - margin_px
    x2 = math.ceil(max(det["x1"], det["x2"])) + margin_px
    y2 = math.ceil(max(det["y1"], det["y2"])) + margin_px

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def apply_class_aware_nms(
    detections: List[dict],
    *,
    score_key: str = "confidence",
    iou_threshold: float = 0.3,
) -> List[dict]:
    """
    Class-aware NMS.

    Different classes will not suppress each other.
    """
    if not detections:
        return []

    boxes = torch.tensor(
        [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections],
        dtype=torch.float32,
    )

    scores = torch.tensor(
        [d.get(score_key, d.get("confidence", 0.0)) for d in detections],
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [d.get("cls_id", 0) for d in detections],
        dtype=torch.int64,
    )

    keep = nms(boxes, scores, iou_threshold)

    return [detections[int(i)] for i in keep]


def iter_padded_tiles(
    height: int,
    width: int,
    *,
    tile: int = TILE,
    pad: int = PAD,
    stride: int = STRIDE,
):
    """
    Generate padded tile regions.

    The model runs on the padded patch, but detections are accepted only when
    their centers fall inside the non-padded core region.
    """
    if stride <= 0:
        raise ValueError("stride must be positive")

    for core_y1 in range(0, height, stride):
        for core_x1 in range(0, width, stride):
            core_x2 = min(core_x1 + tile, width)
            core_y2 = min(core_y1 + tile, height)

            patch_x1 = max(core_x1 - pad, 0)
            patch_y1 = max(core_y1 - pad, 0)
            patch_x2 = min(core_x2 + pad, width)
            patch_y2 = min(core_y2 + pad, height)

            yield (
                patch_x1,
                patch_y1,
                patch_x2,
                patch_y2,
            ), (
                core_x1,
                core_y1,
                core_x2,
                core_y2,
            )


# ──────────────────────────────────────────────────────────────────────────────
#  CLOUD DETECTION - 144 DPI
# ──────────────────────────────────────────────────────────────────────────────


def detect_clouds_144dpi(img_144: np.ndarray, model: YOLO) -> List[dict]:
    """
    Detect cloud regions on a 144-DPI image.
    """
    height, width = img_144.shape[:2]
    detections: List[dict] = []

    with torch.inference_mode():
        for patch_rect, core_rect in iter_padded_tiles(height, width):
            px1, py1, px2, py2 = patch_rect
            cx1, cy1, cx2, cy2 = core_rect

            patch = img_144[py1:py2, px1:px2]
            if patch.size == 0:
                continue

            results = model.predict(
                patch,
                conf=CONF_THR_CLOUD,
                verbose=False,
            )

            for r in results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()

                    gx1 = bx1 + px1
                    gy1 = by1 + py1
                    gx2 = bx2 + px1
                    gy2 = by2 + py1

                    center_x = (gx1 + gx2) / 2.0
                    center_y = (gy1 + gy2) / 2.0

                    # Keep only detections whose centers are in the core region.
                    if not (cx1 <= center_x < cx2 and cy1 <= center_y < cy2):
                        continue

                    cls_id = int(box.cls[0].item())
                    label = r.names[cls_id]
                    conf = float(box.conf[0].item())

                    detections.append(
                        make_detection(
                            x1=gx1,
                            y1=gy1,
                            x2=gx2,
                            y2=gy2,
                            confidence=conf,
                            label=label,
                            cls_id=cls_id,
                            dpi=CLOUD_DPI,
                        )
                    )

    return apply_class_aware_nms(
        detections,
        score_key="weighted_score",
        iou_threshold=NMS_IOU_CLOUD,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  SMALL OBJECT DETECTION - 300 DPI
# ──────────────────────────────────────────────────────────────────────────────


def predict_objects_in_patch(
    patch: np.ndarray,
    model: YOLO,
    *,
    offset_x: int,
    offset_y: int,
    dpi: int,
    conf: float,
    iou: Optional[float] = None,
) -> List[dict]:
    """
    Run small-object YOLO on one crop patch and return global image coordinates.
    """
    if patch.size == 0:
        return []

    kwargs = {
        "conf": conf,
        "verbose": False,
    }

    if iou is not None:
        kwargs["iou"] = iou

    detections: List[dict] = []

    with torch.inference_mode():
        results = model.predict(patch, **kwargs)

        for r in results:
            for box in r.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()

                gx1 = bx1 + offset_x
                gy1 = by1 + offset_y
                gx2 = bx2 + offset_x
                gy2 = by2 + offset_y

                cls_id = int(box.cls[0].item())
                label = r.names[cls_id]
                conf_value = float(box.conf[0].item())

                detections.append(
                    make_detection(
                        x1=gx1,
                        y1=gy1,
                        x2=gx2,
                        y2=gy2,
                        confidence=conf_value,
                        label=label,
                        cls_id=cls_id,
                        dpi=dpi,
                    )
                )

    return detections


def detect_objects_inside_clouds_300dpi(
    img_300: np.ndarray,
    model: YOLO,
    cloud_detections_144: List[dict],
    *,
    crop_margin_px: int = OBJECT_CROP_MARGIN_PX,
) -> List[dict]:
    """
    Detect small objects from a 300-DPI image, using cloud boxes detected at 144 DPI.
    """
    height, width = img_300.shape[:2]
    detections: List[dict] = []

    for cloud_det_144 in cloud_detections_144:
        cloud_det_300 = scale_det_to_dpi(cloud_det_144, OBJECT_DPI)

        clipped = clip_box_to_image(
            cloud_det_300,
            width=width,
            height=height,
            margin_px=crop_margin_px,
        )

        if clipped is None:
            continue

        x1, y1, x2, y2 = clipped
        patch = img_300[y1:y2, x1:x2]

        patch_dets = predict_objects_in_patch(
            patch,
            model,
            offset_x=x1,
            offset_y=y1,
            dpi=OBJECT_DPI,
            conf=CONF_THR,
        )

        # Keep your original fallback behavior:
        # if no result at normal confidence, try a very low confidence.
        if not patch_dets:
            patch_dets = predict_objects_in_patch(
                patch,
                model,
                offset_x=x1,
                offset_y=y1,
                dpi=OBJECT_DPI,
                conf=0.6,
                iou=0.9,
            )

        detections.extend(patch_dets)

    return apply_class_aware_nms(
        detections,
        score_key="confidence",
        iou_threshold=NMS_IOU_OBJECT,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  PDF WRITING HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def get_color_for_label(label: str, palette=OKABE_ITO):
    """
    Stable hash -> stable color index.
    """
    h = hashlib.md5(label.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(palette)
    return palette[idx]


def add_detection_to_pdf_page(
    *,
    page: fitz.Page,
    lib: StampLibrary,
    det: dict,
    label_suffix: str = "",
    palette=OKABE_ITO,
):
    """
    Add one detection to a PDF page.

    Important:
      The detection's own `dpi` field is used for coordinate conversion.
    """
    base_label = str(det["label"])
    write_label = base_label + label_suffix

    det_dpi = int(det.get("dpi", OBJECT_DPI))

    rect = img_rect_to_page_rect(
        page,
        det["x1"],
        det["y1"],
        det["x2"],
        det["y2"],
        dpi=det_dpi,
    )

    if rect.width <= 0 or rect.height <= 0:
        return

    comment = f"confidence={det.get('confidence', 0):.3f}, dpi={det_dpi}"

    try:
        lib[write_label].clone_at(
            page,
            x=rect.x0,
            y=rect.y0,
            anchor="tl",
            comment=comment,
        )
    except KeyError:
        annot = page.add_rect_annot(rect)
        annot.set_info(
            subject=write_label,
            content=comment,
        )
        annot.set_colors(stroke=get_color_for_label(write_label, palette=palette))
        annot.set_border(width=1)
        annot.update()


def write_both_boxes_to_pdf(
    pdf_path: str,
    out_path: str,
    detections_pp: List[List[dict]],
    detections_pp_previous: List[List[dict]],
    *,
    palette=OKABE_ITO,
):
    """
    Write current detections and previous-ASI detections onto the current PDF.

    Current detections use their original label.
    Previous ASI detections use label + " old".
    """
    doc = fitz.open(pdf_path)

    # Insert stamp template page.
    tpl_index, tpl_page = with_template_page(
        doc,
        STAMP_TEMPLATE_PDF,
        insert_at=0,
    )

    # Load stamp templates.
    lib = StampLibrary(doc).load_from_page(tpl_page)

    objects: Dict[str, int] = {}

    template_page_index = tpl_index
    page_offset = 1

    # Current ASI detections.
    for i, page_dets in enumerate(detections_pp):
        page = doc[i + page_offset]

        for det in page_dets:
            label = str(det["label"])
            objects[label] = objects.get(label, 0) + 1

            add_detection_to_pdf_page(
                page=page,
                lib=lib,
                det=det,
                label_suffix="",
                palette=palette,
            )

    # Previous ASI detections.
    for i, page_dets in enumerate(detections_pp_previous):
        page = doc[i + page_offset]

        for det in page_dets:
            label = str(det["label"]) + " old"
            objects[label] = objects.get(label, 0) + 1

            add_detection_to_pdf_page(
                page=page,
                lib=lib,
                det=det,
                label_suffix=" old",
                palette=palette,
            )

    sorted_dict = {key: objects[key] for key in sorted(objects)}
    for key, value in sorted_dict.items():
        print(f"{key}: {value}")

    # Delete temporary template page.
    doc.delete_page(template_page_index)

    rebuild_and_save(doc, out_path)
    doc.close()

    print("✅ Saved:", out_path)


# ──────────────────────────────────────────────────────────────────────────────
#  OPTIONAL DEBUG HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def draw_detections(img: np.ndarray, detections: List[dict]) -> np.ndarray:
    out = img.copy()

    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])

        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            out,
            str(det["label"]),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return out


def load_dets_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_file_rotation_and_recover(pdf_input: str, pdf_output: str):
    """
    Reset page rotation to 0 and save a copy.
    """
    doc = fitz.open(pdf_input)

    for i, page in enumerate(doc):
        rotation = page.rotation
        print(f"Page {i + 1} rotation: {rotation} degrees")
        page.set_rotation(0)

    doc.save(pdf_output, garbage=4, deflate=True)
    doc.close()


# ──────────────────────────────────────────────────────────────────────────────
#  FULL PIPELINE DRIVER
# ──────────────────────────────────────────────────────────────────────────────


def run_dual_dpi(
    pdf_path: str,
    input_previous_asi_path: str,
    out_path: str,
    *,
    yolo: bool = True,
):
    """
    Full dual-DPI pipeline.

    Steps per page:
      1. Render current ASI page at 144 DPI.
      2. Detect clouds at 144 DPI.
      3. Render current ASI page at 300 DPI.
      4. Render previous ASI page at 300 DPI.
      5. Detect small objects inside scaled cloud boxes at 300 DPI.
      6. Write current and previous detections onto current PDF.
    """
    global MODEL
    global MODEL2

    if yolo:
        if MODEL is None:
            MODEL = YOLO(YOLO_Model_Path)

        if MODEL2 is None:
            MODEL2 = YOLO(YOLO_Model_Path2)

    doc = fitz.open(pdf_path)
    doc_previous_asi = fitz.open(input_previous_asi_path)

    if doc.page_count != doc_previous_asi.page_count:
        doc.close()
        doc_previous_asi.close()
        raise ValueError(
            f"Page count mismatch: current={doc.page_count}, "
            f"previous={doc_previous_asi.page_count}"
        )

    detections_pp: List[List[dict]] = []
    detections_pp_previous: List[List[dict]] = []

    debug_data = {
        "current_pdf": pdf_path,
        "previous_pdf": input_previous_asi_path,
        "cloud_dpi": CLOUD_DPI,
        "object_dpi": OBJECT_DPI,
        "pages": [],
    }

    try:
        for i in range(doc.page_count):
            page = doc[i]
            page_previous = doc_previous_asi[i]

            if yolo:
                # 1. Low-res render for cloud detection.
                img_cloud_144 = render_page_to_bgr(page, CLOUD_DPI)

                # 2. Cloud detection at 144 DPI.
                cloud_dets = detect_clouds_144dpi(img_cloud_144, MODEL)

                # 3. High-res renders for small-object detection.
                img_current_300 = render_page_to_bgr(page, OBJECT_DPI)
                img_previous_300 = render_page_to_bgr(page_previous, OBJECT_DPI)

                # 4. Current ASI small objects inside current clouds.
                current_object_dets = detect_objects_inside_clouds_300dpi(
                    img_current_300,
                    MODEL2,
                    cloud_dets,
                    crop_margin_px=OBJECT_CROP_MARGIN_PX,
                )

                # 5. Previous ASI small objects inside the same cloud regions.
                #    This assumes current and previous ASI pages are aligned.
                previous_object_dets = detect_objects_inside_clouds_300dpi(
                    img_previous_300,
                    MODEL2,
                    cloud_dets,
                    crop_margin_px=OBJECT_CROP_MARGIN_PX,
                )

                # Current PDF: current object detections.
                current_dets_for_pdf = current_object_dets

                # Previous ASI comparison layer:
                # Preserve your original behavior by also adding cloud boxes as "old".
                previous_dets_for_pdf = previous_object_dets + [
                    dict(d) for d in cloud_dets
                ]

            else:
                cloud_dets = []
                current_dets_for_pdf = load_dets_from_json("detectResult.json")
                previous_dets_for_pdf = []

            detections_pp.append(current_dets_for_pdf)
            detections_pp_previous.append(previous_dets_for_pdf)

            debug_data["pages"].append(
                {
                    "page_index": i,
                    "page_number": i + 1,
                    "clouds": cloud_dets,
                    "current_detections": current_dets_for_pdf,
                    "previous_detections": previous_dets_for_pdf,
                }
            )

            print(
                f"Page {i + 1}: "
                f"clouds={len(cloud_dets)}, "
                f"current_objects={len(current_dets_for_pdf)}, "
                f"previous_items={len(previous_dets_for_pdf)}"
            )

    finally:
        doc.close()
        doc_previous_asi.close()

    with open(DEBUG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=4)

    write_both_boxes_to_pdf(
        pdf_path,
        out_path,
        detections_pp,
        detections_pp_previous,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_dual_dpi(
        input_pdf,
        input_previous_asi,
        out_path=output_pdf,
        yolo=True,
    )
