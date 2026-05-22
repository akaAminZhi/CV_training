# ──────────────────────────────────────────────────────────────────────────────
#  Red-Cloud-Polyline + 300-DPI YOLO PDF Detection Pipeline
#
#  New idea:
#    - Do NOT use YOLO to detect revision clouds.
#    - Extract red cloud regions directly using OpenCV color thresholding.
#    - Use red cloud polyline / polygon to crop 300-DPI patches.
#    - Run small-object YOLO only inside those cropped cloud patches.
#    - Use polygon mask to filter out detections outside the red cloud.
#
#  Coordinate rule:
#    - Red cloud extraction runs directly on 300-DPI image.
#    - Small-object detections are also in 300-DPI image coordinates.
#    - When writing annotations back to PDF, each detection keeps dpi=300.
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torchvision.ops import nms
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────────

MODEL2 = None

# Only object detection model is needed now.
YOLO_Model_Path2 = "weights/lsb_fa_yolo26_xlarge_v1.pt"

file_name = "TestFiles/lsb_fa/new_ASI/28 - Fire Alarm_asi009"
file_name2 = "TestFiles/lsb_fa/old_ASI/28 - Fire Alarm_asi_old"

input_pdf = f"{file_name}.pdf"
input_previous_asi = f"{file_name2}.pdf"

output_pdf = f"{file_name}_result.pdf"
output_previous_pdf = f"{file_name2}_result.pdf"

STAMP_TEMPLATE_PDF = "TestFiles/lsb_fa/lsb_fa_lable.pdf"

# Detection resolution
OBJECT_DPI = 300

# YOLO thresholds
CONF_THR = 0.8
CONF_THR_FALLBACK = 0.6

# NMS threshold
NMS_IOU_OBJECT = 0.5

# Crop expansion around each red cloud polygon bbox
OBJECT_CROP_MARGIN_PX = 16

# Red cloud extraction parameters
RED_MIN_AREA = 300
RED_EPSILON_RATIO = 0.002

RED_CLOSE_KERNEL_SIZE = 5
RED_CLOSE_ITERATIONS = 2

RED_DILATE_KERNEL_SIZE = 3
RED_DILATE_ITERATIONS = 0

# If True, pixels outside the red cloud polygon are set to white before YOLO.
MASK_OUTSIDE_CLOUD_FOR_OBJECT_DETECTION = False

# Filter object detections by polygon mask overlap.
# 0.10 means at least 10% of the detection box area must be inside the cloud mask.
DET_MASK_OVERLAP_RATIO_THR = 0.10

# Debug outputs
DEBUG_JSON_PATH = "detectResult_red_cloud_polyline_300dpi.json"
DEBUG_DIR = "debug_red_cloud_pipeline"

# If True, write cloud bbox itself back to PDF.
# Usually keep False because you only want object detections.
WRITE_CLOUD_BBOX_TO_PDF = True


# Save each cropped cloud patch before feeding it into YOLO.
SAVE_CLOUD_CROPS_BEFORE_MODEL = True

# Save raw crop, masked crop, and mask.
SAVE_RAW_CLOUD_CROP = True
SAVE_MASKED_CLOUD_CROP = True
SAVE_CLOUD_CROP_MASK = True

# Cloud crop debug folder
CLOUD_CROP_DEBUG_DIR = "debug_red_cloud_pipeline/cloud_crops"
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


def is_triangle_like_contour(
    contour,
    *,
    coarse_epsilon_ratio=0.04,
    detail_epsilon_ratio=0.01,
    solidity_threshold=0.78,
    max_detail_vertices=12,
):
    """
    Return True if the contour looks like a red triangle symbol.

    This is used to reject non-cloud red symbols, such as revision triangle marks.

    Logic:
      - Coarse polygon approximation has 3 vertices, or
      - Convex hull approximation has 3 vertices
      - Shape is mostly convex / solid
      - Detailed approximation still has limited vertices

    Cloud lines usually have many irregular vertices and lower solidity.
    """
    area = cv2.contourArea(contour)

    if area <= 0:
        return False

    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 0:
        return False

    # Coarse approximation: triangle should become 3 points.
    coarse = cv2.approxPolyDP(
        contour,
        coarse_epsilon_ratio * perimeter,
        True,
    )

    # More detailed approximation: triangle should still be simple.
    detail = cv2.approxPolyDP(
        contour,
        detail_epsilon_ratio * perimeter,
        True,
    )

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area <= 0:
        return False

    solidity = area / hull_area

    hull_perimeter = cv2.arcLength(hull, True)

    if hull_perimeter <= 0:
        return False

    hull_coarse = cv2.approxPolyDP(
        hull,
        coarse_epsilon_ratio * hull_perimeter,
        True,
    )

    coarse_vertices = len(coarse)
    detail_vertices = len(detail)
    hull_vertices = len(hull_coarse)

    is_triangle_by_contour = coarse_vertices == 3 and cv2.isContourConvex(coarse)

    is_triangle_by_hull = hull_vertices == 3

    is_simple_shape = detail_vertices <= max_detail_vertices
    is_solid_enough = solidity >= solidity_threshold

    return (
        (is_triangle_by_contour or is_triangle_by_hull)
        and is_simple_shape
        and is_solid_enough
    )


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
    m = fitz.Matrix(page.transformation_matrix)
    m.invert()

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
                f"Subject not found: {subject}. "
                f"Available: {list(self._by_subject.keys())}"
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

        if tpl.it and "/IT" not in obj:
            obj = obj.replace("/Subtype /Stamp", f"/Subtype /Stamp\n  {tpl.it}")

        obj = re.sub(r"/AP\s+<<.*?>>", tpl.ap, obj, flags=re.S)

        obj = re.sub(
            r"/Rect\s+\[.*?\]",
            f"/Rect {_rect_to_pdf_array(page, new_rect)}",
            obj,
            flags=re.S,
        )

        if clear_comments:
            obj = re.sub(r"\s*/Name\s*/\S+", "", obj)
            obj = re.sub(r"\s*/Contents\s*\(.*?\)", "", obj, flags=re.S)

        if subject_to_write:
            obj = _upsert_pdf_key(
                obj,
                "/Subj",
                f"({_pdf_escape_paren(subject_to_write)})",
            )

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

    p_rot = fitz.Point(px / zoom, py / zoom)
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


def cloud_region_to_bbox_detection(region: dict) -> dict:
    """
    Convert a red cloud polygon region to a bbox-style detection.
    This is only for optional debug/PDF annotation.
    """
    return make_detection(
        x1=region["x1"],
        y1=region["y1"],
        x2=region["x2"],
        y2=region["y2"],
        confidence=1.0,
        label="red_cloud",
        cls_id=-1,
        dpi=region.get("dpi", OBJECT_DPI),
    )


def clip_box_to_image(
    det_or_region: dict,
    *,
    width: int,
    height: int,
    margin_px: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    x1 = math.floor(min(det_or_region["x1"], det_or_region["x2"])) - margin_px
    y1 = math.floor(min(det_or_region["y1"], det_or_region["y2"])) - margin_px
    x2 = math.ceil(max(det_or_region["x1"], det_or_region["x2"])) + margin_px
    y2 = math.ceil(max(det_or_region["y1"], det_or_region["y2"])) + margin_px

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

    final: List[dict] = []

    class_ids = sorted(set(int(d.get("cls_id", 0)) for d in detections))

    for cls_id in class_ids:
        cls_dets = [d for d in detections if int(d.get("cls_id", 0)) == cls_id]

        boxes = torch.tensor(
            [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in cls_dets],
            dtype=torch.float32,
        )

        scores = torch.tensor(
            [d.get(score_key, d.get("confidence", 0.0)) for d in cls_dets],
            dtype=torch.float32,
        )

        keep = nms(boxes, scores, iou_threshold)

        final.extend([cls_dets[int(i)] for i in keep])

    final.sort(key=lambda d: d.get(score_key, d.get("confidence", 0.0)), reverse=True)

    return final


# ──────────────────────────────────────────────────────────────────────────────
#  RED CLOUD POLYLINE EXTRACTION - 300 DPI
# ──────────────────────────────────────────────────────────────────────────────


def extract_red_cloud_regions_from_image(
    img_bgr: np.ndarray,
    *,
    dpi: int,
    min_area: float = RED_MIN_AREA,
    epsilon_ratio: float = RED_EPSILON_RATIO,
    close_kernel_size: int = RED_CLOSE_KERNEL_SIZE,
    close_iterations: int = RED_CLOSE_ITERATIONS,
    dilate_kernel_size: int = RED_DILATE_KERNEL_SIZE,
    dilate_iterations: int = RED_DILATE_ITERATIONS,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Extract red revision cloud regions from a rendered PDF page image.

    This version rejects triangle-like red symbols.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Red wraps around HSV hue range.
    lower_red_1 = np.array([0, 70, 50])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 70, 50])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    if close_kernel_size > 0 and close_iterations > 0:
        close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=close_iterations,
        )

    if dilate_kernel_size > 0 and dilate_iterations > 0:
        dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        red_mask = cv2.dilate(
            red_mask,
            dilate_kernel,
            iterations=dilate_iterations,
        )

    contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    cloud_regions: List[dict] = []

    rejected_triangles = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        # ------------------------------------------------------------
        # Reject red triangle symbols before treating them as clouds.
        # ------------------------------------------------------------
        if is_triangle_like_contour(contour):
            rejected_triangles += 1
            continue

        epsilon = epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if approx.shape[0] < 3:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Filter tiny false positives, such as small red text fragments.
        if w < 10 or h < 10:
            continue

        points = approx.reshape(-1, 2).astype(int).tolist()

        cloud_regions.append(
            {
                "region_index": 0,
                "image_polyline": points,
                "x1": float(x),
                "y1": float(y),
                "x2": float(x + w),
                "y2": float(y + h),
                "width": float(w),
                "height": float(h),
                "area": float(area),
                "perimeter": float(perimeter),
                "closed": True,
                "dpi": int(dpi),
                "label": "red_cloud",
                "confidence": 1.0,
                "cls_id": -1,
                "rejected_triangle_count_on_page": rejected_triangles,
            }
        )

    cloud_regions.sort(key=lambda r: r["area"], reverse=True)

    for idx, region in enumerate(cloud_regions, start=1):
        region["region_index"] = idx

    return red_mask, cloud_regions


def make_cloud_patch_mask(
    *,
    patch_shape: Tuple[int, int, int],
    polygon_points: List[List[int]],
    offset_x: int,
    offset_y: int,
    expand_px: int = 0,
) -> np.ndarray:
    """
    Create a filled polygon mask for a cloud crop patch.

    polygon_points are in full-page image coordinates.
    The returned mask is in patch-local coordinates.
    """
    mask = np.zeros(patch_shape[:2], dtype=np.uint8)

    if len(polygon_points) < 3:
        return mask

    local_points = []

    for x, y in polygon_points:
        local_points.append([int(round(x - offset_x)), int(round(y - offset_y))])

    pts = np.array(local_points, dtype=np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(mask, [pts], 255)

    if expand_px > 0:
        kernel = np.ones((expand_px, expand_px), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def apply_white_background_outside_mask(
    patch_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Set pixels outside the cloud mask to white.

    This helps YOLO focus on objects inside the red cloud region.
    """
    output = patch_bgr.copy()
    output[mask == 0] = (255, 255, 255)

    return output


def detection_mask_overlap_ratio(
    det: dict,
    patch_mask: np.ndarray,
    *,
    offset_x: int,
    offset_y: int,
) -> float:
    """
    Calculate how much of a detection box is inside the cloud polygon mask.
    """
    h, w = patch_mask.shape[:2]

    x1 = math.floor(min(det["x1"], det["x2"]) - offset_x)
    y1 = math.floor(min(det["y1"], det["y2"]) - offset_y)
    x2 = math.ceil(max(det["x1"], det["x2"]) - offset_x)
    y2 = math.ceil(max(det["y1"], det["y2"]) - offset_y)

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = patch_mask[y1:y2, x1:x2]

    if roi.size == 0:
        return 0.0

    return float(cv2.countNonZero(roi)) / float(roi.size)


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


def save_cloud_crop_before_model(
    *,
    patch_raw: np.ndarray,
    patch_for_model: np.ndarray,
    patch_mask: np.ndarray,
    output_dir: str,
    source_tag: str,
    page_number: int,
    cloud_region_index: int,
    save_raw: bool = True,
    save_model_input: bool = True,
    save_mask: bool = True,
) -> dict:
    """
    Save each cloud crop before feeding it into YOLO.

    source_tag:
      - "current"
      - "previous"

    Saved files:
      - raw crop: original crop from PDF image
      - model input crop: masked crop actually passed to YOLO
      - mask: cloud polygon mask
    """
    base_dir = (
        Path(output_dir)
        / source_tag
        / f"page_{page_number:03d}"
        / f"cloud_{cloud_region_index:03d}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    saved = {
        "source": source_tag,
        "page_number": page_number,
        "cloud_region_index": cloud_region_index,
        "folder": str(base_dir),
        "raw_crop": None,
        "model_input_crop": None,
        "mask": None,
    }

    if save_raw:
        raw_path = base_dir / "raw_crop.png"
        cv2.imwrite(str(raw_path), patch_raw)
        saved["raw_crop"] = str(raw_path)

    if save_model_input:
        model_input_path = base_dir / "model_input_crop.png"
        cv2.imwrite(str(model_input_path), patch_for_model)
        saved["model_input_crop"] = str(model_input_path)

    if save_mask:
        mask_path = base_dir / "cloud_mask.png"
        cv2.imwrite(str(mask_path), patch_mask)
        saved["mask"] = str(mask_path)

    return saved


def detect_objects_inside_clouds_300dpi(
    img_300: np.ndarray,
    model: YOLO,
    cloud_regions_300: List[dict],
    *,
    crop_margin_px: int = OBJECT_CROP_MARGIN_PX,
    mask_outside_cloud: bool = MASK_OUTSIDE_CLOUD_FOR_OBJECT_DETECTION,
    mask_overlap_ratio_thr: float = DET_MASK_OVERLAP_RATIO_THR,
    save_cloud_crops_before_model: bool = SAVE_CLOUD_CROPS_BEFORE_MODEL,
    cloud_crop_output_dir: str = CLOUD_CROP_DEBUG_DIR,
    source_tag: str = "current",
    page_number: int = 1,
    saved_cloud_crops: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Detect small objects from a 300-DPI image using red cloud polygon regions.

    New behavior:
      - Before feeding each cloud crop into YOLO, optionally save:
          1. raw crop
          2. masked model-input crop
          3. cloud mask

    source_tag:
      - "current" for current ASI PDF
      - "previous" for previous ASI PDF
    """
    height, width = img_300.shape[:2]

    detections: List[dict] = []

    for cloud_region in cloud_regions_300:
        cloud_region_index = int(cloud_region["region_index"])

        clipped = clip_box_to_image(
            cloud_region,
            width=width,
            height=height,
            margin_px=crop_margin_px,
        )

        if clipped is None:
            continue

        x1, y1, x2, y2 = clipped

        patch = img_300[y1:y2, x1:x2]

        if patch.size == 0:
            continue

        patch_mask = make_cloud_patch_mask(
            patch_shape=patch.shape,
            polygon_points=cloud_region["image_polyline"],
            offset_x=x1,
            offset_y=y1,
            expand_px=3,
        )

        if mask_outside_cloud:
            patch_for_model = apply_white_background_outside_mask(patch, patch_mask)
        else:
            patch_for_model = patch.copy()

        # ------------------------------------------------------------
        # Save cloud crop before feeding it into YOLO.
        # ------------------------------------------------------------
        if save_cloud_crops_before_model:
            saved_info = save_cloud_crop_before_model(
                patch_raw=patch,
                patch_for_model=patch_for_model,
                patch_mask=patch_mask,
                output_dir=cloud_crop_output_dir,
                source_tag=source_tag,
                page_number=page_number,
                cloud_region_index=cloud_region_index,
                save_raw=SAVE_RAW_CLOUD_CROP,
                save_model_input=SAVE_MASKED_CLOUD_CROP,
                save_mask=SAVE_CLOUD_CROP_MASK,
            )

            saved_info["crop_bbox_300dpi"] = {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "dpi": OBJECT_DPI,
            }

            saved_info["cloud_bbox_300dpi"] = {
                "x1": cloud_region["x1"],
                "y1": cloud_region["y1"],
                "x2": cloud_region["x2"],
                "y2": cloud_region["y2"],
                "dpi": cloud_region["dpi"],
            }

            if saved_cloud_crops is not None:
                saved_cloud_crops.append(saved_info)

        # ------------------------------------------------------------
        # Feed saved model-input crop equivalent into YOLO.
        # ------------------------------------------------------------
        patch_dets = predict_objects_in_patch(
            patch_for_model,
            model,
            offset_x=x1,
            offset_y=y1,
            dpi=OBJECT_DPI,
            conf=CONF_THR,
        )

        # Fallback: if no result at normal confidence, try lower confidence.
        if not patch_dets:
            patch_dets = predict_objects_in_patch(
                patch_for_model,
                model,
                offset_x=x1,
                offset_y=y1,
                dpi=OBJECT_DPI,
                conf=CONF_THR_FALLBACK,
                iou=0.9,
            )

        for det in patch_dets:
            overlap_ratio = detection_mask_overlap_ratio(
                det,
                patch_mask,
                offset_x=x1,
                offset_y=y1,
            )

            if overlap_ratio < mask_overlap_ratio_thr:
                continue

            det["cloud_region_index"] = cloud_region_index
            det["cloud_mask_overlap_ratio"] = float(overlap_ratio)
            det["source_tag"] = source_tag
            det["source_cloud_bbox"] = {
                "x1": cloud_region["x1"],
                "y1": cloud_region["y1"],
                "x2": cloud_region["x2"],
                "y2": cloud_region["y2"],
                "dpi": cloud_region["dpi"],
            }
            det["crop_bbox_300dpi"] = {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "dpi": OBJECT_DPI,
            }

            detections.append(det)

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
      The detection's own dpi field is used for coordinate conversion.
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

    if "cloud_region_index" in det:
        comment += f", cloud_region={det['cloud_region_index']}"

    if "cloud_mask_overlap_ratio" in det:
        comment += f", cloud_overlap={det['cloud_mask_overlap_ratio']:.3f}"

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

    Current detections use original label.
    Previous ASI detections use label + " old".
    """
    doc = fitz.open(pdf_path)

    tpl_index, tpl_page = with_template_page(
        doc,
        STAMP_TEMPLATE_PDF,
        insert_at=0,
    )

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

    doc.delete_page(template_page_index)

    rebuild_and_save(doc, out_path)
    doc.close()

    print("✅ Saved:", out_path)


# ──────────────────────────────────────────────────────────────────────────────
#  DEBUG HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def draw_detections(img: np.ndarray, detections: List[dict]) -> np.ndarray:
    out = img.copy()

    for det in detections:
        x1 = int(round(det["x1"]))
        y1 = int(round(det["y1"]))
        x2 = int(round(det["x2"]))
        y2 = int(round(det["y2"]))

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


def draw_cloud_regions(img: np.ndarray, cloud_regions: List[dict]) -> np.ndarray:
    out = img.copy()

    for region in cloud_regions:
        points = region["image_polyline"]

        if len(points) < 3:
            continue

        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            out,
            [pts],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        x1 = int(round(region["x1"]))
        y1 = int(round(region["y1"]))

        cv2.putText(
            out,
            f"cloud {region['region_index']}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 128, 0),
            2,
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


def run_red_cloud_polyline_pipeline(
    pdf_path: str,
    input_previous_asi_path: str,
    out_path: str,
    *,
    yolo: bool = True,
):
    """
    Full pipeline.

    Steps per page:
      1. Render current ASI page at 300 DPI.
      2. Extract red cloud polygon regions using OpenCV.
      3. Render previous ASI page at 300 DPI.
      4. Detect small objects inside red cloud polygon regions on current page.
      5. Detect small objects inside the same red cloud polygon regions on previous page.
      6. Write current and previous detections onto current PDF.
    """
    global MODEL2

    if yolo:
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

    debug_dir = Path(DEBUG_DIR)
    debug_dir.mkdir(parents=True, exist_ok=True)

    detections_pp: List[List[dict]] = []
    detections_pp_previous: List[List[dict]] = []

    debug_data = {
        "current_pdf": pdf_path,
        "previous_pdf": input_previous_asi_path,
        "object_dpi": OBJECT_DPI,
        "red_cloud_extraction": {
            "red_min_area": RED_MIN_AREA,
            "red_epsilon_ratio": RED_EPSILON_RATIO,
            "red_close_kernel_size": RED_CLOSE_KERNEL_SIZE,
            "red_close_iterations": RED_CLOSE_ITERATIONS,
            "red_dilate_kernel_size": RED_DILATE_KERNEL_SIZE,
            "red_dilate_iterations": RED_DILATE_ITERATIONS,
        },
        "pages": [],
    }

    try:
        for i in range(doc.page_count):
            page_number = i + 1

            page = doc[i]
            page_previous = doc_previous_asi[i]

            if yolo:
                # 1. Render current and previous ASI pages at 300 DPI.
                img_current_300 = render_page_to_bgr(page, OBJECT_DPI)
                img_previous_300 = render_page_to_bgr(page_previous, OBJECT_DPI)

                # 2. Extract red cloud polygon regions from current ASI page.
                red_mask, cloud_regions = extract_red_cloud_regions_from_image(
                    img_current_300,
                    dpi=OBJECT_DPI,
                    min_area=RED_MIN_AREA,
                    epsilon_ratio=RED_EPSILON_RATIO,
                    close_kernel_size=RED_CLOSE_KERNEL_SIZE,
                    close_iterations=RED_CLOSE_ITERATIONS,
                    dilate_kernel_size=RED_DILATE_KERNEL_SIZE,
                    dilate_iterations=RED_DILATE_ITERATIONS,
                )

                # 3. Detect current ASI small objects inside current red clouds.
                saved_cloud_crops_for_page = []

                current_object_dets = detect_objects_inside_clouds_300dpi(
                    img_current_300,
                    MODEL2,
                    cloud_regions,
                    crop_margin_px=OBJECT_CROP_MARGIN_PX,
                    mask_outside_cloud=MASK_OUTSIDE_CLOUD_FOR_OBJECT_DETECTION,
                    mask_overlap_ratio_thr=DET_MASK_OVERLAP_RATIO_THR,
                    save_cloud_crops_before_model=SAVE_CLOUD_CROPS_BEFORE_MODEL,
                    cloud_crop_output_dir=CLOUD_CROP_DEBUG_DIR,
                    source_tag="current",
                    page_number=page_number,
                    saved_cloud_crops=saved_cloud_crops_for_page,
                )

                # 4. Detect previous ASI small objects inside same cloud regions.
                #    This assumes current and previous ASI pages are aligned.
                previous_object_dets = detect_objects_inside_clouds_300dpi(
                    img_previous_300,
                    MODEL2,
                    cloud_regions,
                    crop_margin_px=OBJECT_CROP_MARGIN_PX,
                    mask_outside_cloud=MASK_OUTSIDE_CLOUD_FOR_OBJECT_DETECTION,
                    mask_overlap_ratio_thr=DET_MASK_OVERLAP_RATIO_THR,
                    save_cloud_crops_before_model=SAVE_CLOUD_CROPS_BEFORE_MODEL,
                    cloud_crop_output_dir=CLOUD_CROP_DEBUG_DIR,
                    source_tag="previous",
                    page_number=page_number,
                    saved_cloud_crops=saved_cloud_crops_for_page,
                )

                current_dets_for_pdf = current_object_dets
                previous_dets_for_pdf = previous_object_dets

                if WRITE_CLOUD_BBOX_TO_PDF:
                    previous_dets_for_pdf = previous_dets_for_pdf + [
                        cloud_region_to_bbox_detection(region)
                        for region in cloud_regions
                    ]

                # 5. Save debug images.
                cloud_overlay = draw_cloud_regions(img_current_300, cloud_regions)
                current_overlay = draw_detections(cloud_overlay, current_object_dets)

                previous_cloud_overlay = draw_cloud_regions(
                    img_previous_300,
                    cloud_regions,
                )
                previous_overlay = draw_detections(
                    previous_cloud_overlay,
                    previous_object_dets,
                )

                cv2.imwrite(
                    str(debug_dir / f"page_{page_number}_red_mask.png"),
                    red_mask,
                )
                cv2.imwrite(
                    str(debug_dir / f"page_{page_number}_clouds_300dpi.png"),
                    cloud_overlay,
                )
                cv2.imwrite(
                    str(debug_dir / f"page_{page_number}_current_objects.png"),
                    current_overlay,
                )
                cv2.imwrite(
                    str(debug_dir / f"page_{page_number}_previous_objects.png"),
                    previous_overlay,
                )

            else:
                cloud_regions = []
                current_dets_for_pdf = load_dets_from_json("detectResult.json")
                previous_dets_for_pdf = []

            detections_pp.append(current_dets_for_pdf)
            detections_pp_previous.append(previous_dets_for_pdf)

            debug_data["pages"].append(
                {
                    "page_index": i,
                    "page_number": page_number,
                    "cloud_regions": cloud_regions,
                    "saved_cloud_crops": saved_cloud_crops_for_page,
                    "current_detections": current_dets_for_pdf,
                    "previous_detections": previous_dets_for_pdf,
                }
            )

            print(
                f"Page {page_number}: "
                f"red_clouds={len(cloud_regions)}, "
                f"current_objects={len(current_dets_for_pdf)}, "
                f"previous_objects={len(previous_dets_for_pdf)}"
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
    run_red_cloud_polyline_pipeline(
        input_pdf,
        input_previous_asi,
        out_path=output_pdf,
        yolo=True,
    )
