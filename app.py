import io
import os
import json
import math
import zipfile
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# =========================================================
# App setup
# =========================================================

st.set_page_config(
    page_title="GT7 / Forza-style Layered SVG Vectorizer",
    layout="wide"
)


# =========================================================
# Data models
# =========================================================

@dataclass
class AppConfig:
    mode_name: str
    max_dim: int
    num_colors: int
    min_area: int
    min_area_priority: int
    overlap_px: int
    contour_epsilon: float
    contour_epsilon_priority: float
    max_points_per_path: int
    morph_open: int
    morph_close: int
    canny_low: int
    canny_high: int
    outline_dilate_px: int
    outline_min_area: int
    max_fill_parts: int
    max_outline_parts: int
    svg_limit_kb: float
    enable_outline: bool
    use_face_priority: bool
    outline_color: str


@dataclass
class VectorPart:
    part_id: str
    category: str
    layer: int
    fill: Optional[str]
    stroke: Optional[str]
    stroke_width: float
    closed: bool
    points: List[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]
    area: float
    priority: bool


@dataclass
class ProcessResult:
    original_rgb: np.ndarray
    processed_rgb: np.ndarray
    quantized_rgb: np.ndarray
    priority_mask: np.ndarray
    vector_preview_rgb: np.ndarray
    parts: List[VectorPart]
    svg_files: Dict[str, str]
    stats: Dict
    logs: List[str]


# =========================================================
# Presets
# =========================================================

PRESETS = {
    "Safe": AppConfig(
        mode_name="Safe",
        max_dim=1280,
        num_colors=10,
        min_area=120,
        min_area_priority=42,
        overlap_px=2,
        contour_epsilon=0.010,
        contour_epsilon_priority=0.006,
        max_points_per_path=120,
        morph_open=3,
        morph_close=5,
        canny_low=60,
        canny_high=150,
        outline_dilate_px=1,
        outline_min_area=14,
        max_fill_parts=180,
        max_outline_parts=70,
        svg_limit_kb=14.5,
        enable_outline=True,
        use_face_priority=True,
        outline_color="#202020",
    ),
    "High Quality Safe": AppConfig(
        mode_name="High Quality Safe",
        max_dim=1600,
        num_colors=14,
        min_area=80,
        min_area_priority=28,
        overlap_px=2,
        contour_epsilon=0.008,
        contour_epsilon_priority=0.0045,
        max_points_per_path=150,
        morph_open=3,
        morph_close=5,
        canny_low=50,
        canny_high=140,
        outline_dilate_px=1,
        outline_min_area=12,
        max_fill_parts=280,
        max_outline_parts=120,
        svg_limit_kb=14.5,
        enable_outline=True,
        use_face_priority=True,
        outline_color="#1E1E1E",
    ),
    "Detail Heavy": AppConfig(
        mode_name="Detail Heavy",
        max_dim=1920,
        num_colors=18,
        min_area=50,
        min_area_priority=20,
        overlap_px=2,
        contour_epsilon=0.006,
        contour_epsilon_priority=0.0035,
        max_points_per_path=180,
        morph_open=3,
        morph_close=5,
        canny_low=40,
        canny_high=130,
        outline_dilate_px=1,
        outline_min_area=10,
        max_fill_parts=380,
        max_outline_parts=160,
        svg_limit_kb=14.5,
        enable_outline=True,
        use_face_priority=True,
        outline_color="#181818",
    ),
}


# =========================================================
# Utility functions
# =========================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        return (0, 0, 0)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def pil_to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def open_uploaded_image(uploaded_file) -> np.ndarray:
    img = Image.open(uploaded_file).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    comp = Image.alpha_composite(bg, img).convert("RGB")
    return np.array(comp)


def resize_to_max_dim(img_bgr: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_bgr, scale


def find_contours_compat(mask: np.ndarray, mode, method):
    result = cv2.findContours(mask, mode, method)
    if len(result) == 2:
        contours, hierarchy = result
    else:
        _, contours, hierarchy = result
    return contours, hierarchy


def contour_to_points(contour: np.ndarray) -> List[Tuple[int, int]]:
    pts = contour.reshape(-1, 2)
    return [(int(p[0]), int(p[1])) for p in pts]


def simplify_contour(contour: np.ndarray, epsilon_ratio: float, max_points: int, closed: bool = True) -> np.ndarray:
    peri = cv2.arcLength(contour, closed)
    eps = max(0.5, epsilon_ratio * peri)
    approx = cv2.approxPolyDP(contour, eps, closed)

    tries = 0
    while len(approx) > max_points and tries < 8:
        eps *= 1.35
        approx = cv2.approxPolyDP(contour, eps, closed)
        tries += 1

    return approx


def luminance_from_rgb(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)


def png_bytes_from_rgb(rgb: np.ndarray) -> bytes:
    img = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def safe_filename_base(name: str) -> str:
    base = os.path.splitext(name)[0]
    keep = []
    for ch in base:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    result = "".join(keep).strip("_")
    return result or "vectorized_image"


# =========================================================
# Image preprocessing
# =========================================================

def preprocess_image(img_bgr: np.ndarray, cfg: AppConfig, logs: List[str]) -> np.ndarray:
    logs.append("Preprocess: resize / denoise / contrast adjust")

    img_bgr, scale = resize_to_max_dim(img_bgr, cfg.max_dim)
    if scale < 1.0:
        logs.append(f"Image resized to max {cfg.max_dim}px (scale={scale:.3f})")

    # Gentle denoise
    img_bgr = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=35, sigmaSpace=35)

    # LAB contrast adjust
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img_bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    return img_bgr


# =========================================================
# Face / priority mask
# =========================================================

def build_priority_mask(img_bgr: np.ndarray, logs: List[str], enabled: bool = True) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not enabled:
        return mask

    logs.append("Building priority mask (center + skin + optional face detect)")

    # Center ellipse
    center_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        center_mask,
        center=(w // 2, h // 2),
        axes=(max(1, int(w * 0.28)), max(1, int(h * 0.33))),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=120,
        thickness=-1,
    )
    mask = np.maximum(mask, center_mask)

    # Skin mask (heuristic)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )
    if np.count_nonzero(skin) > 0:
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = np.maximum(mask, (skin > 0).astype(np.uint8) * 160)

    # Optional face detection
    face_found = False
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(40, 40),
            )
            for (x, y, fw, fh) in faces:
                pad_x = int(fw * 0.25)
                pad_y = int(fh * 0.35)
                x1 = clamp(x - pad_x, 0, w - 1)
                y1 = clamp(y - pad_y, 0, h - 1)
                x2 = clamp(x + fw + pad_x, 0, w - 1)
                y2 = clamp(y + fh + pad_y, 0, h - 1)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                face_found = True
    except Exception as e:
        logs.append(f"Face detection skipped: {e}")

    if face_found:
        logs.append("Face detector found at least one face region")
    else:
        logs.append("Face detector fallback: using center/skin priority only")

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=12, sigmaY=12)
    _, mask = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)
    return mask


# =========================================================
# Quantization
# =========================================================

def quantize_image(img_bgr: np.ndarray, k: int, logs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logs.append(f"Quantization: {k} colors")

    data = img_bgr.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        24,
        0.8,
    )

    compactness, labels, centers = cv2.kmeans(
        data,
        K=k,
        bestLabels=None,
        criteria=criteria,
        attempts=4,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    centers = np.uint8(centers)
    labels_2d = labels.reshape(img_bgr.shape[:2])
    quant = centers[labels.flatten()].reshape(img_bgr.shape)

    logs.append(f"KMeans compactness: {compactness:.2f}")
    return quant, labels_2d, centers


# =========================================================
# Fill extraction
# =========================================================

def classify_fill_category(color_rgb: Tuple[int, int, int], area: float) -> str:
    lum = luminance_from_rgb(color_rgb)
    if lum < 55:
        return "shadow"
    if area < 120:
        return "detail"
    return "fill"


def extract_fill_parts(
    labels: np.ndarray,
    centers: np.ndarray,
    priority_mask: np.ndarray,
    cfg: AppConfig,
    logs: List[str]
) -> List[VectorPart]:
    h, w = labels.shape[:2]
    fill_parts: List[VectorPart] = []

    kernel_open = np.ones((cfg.morph_open, cfg.morph_open), np.uint8)
    kernel_close = np.ones((cfg.morph_close, cfg.morph_close), np.uint8)
    overlap_kernel = np.ones((max(1, cfg.overlap_px), max(1, cfg.overlap_px)), np.uint8)

    part_counter = 0

    # Process colors by pixel count descending (larger bases first)
    color_counts = []
    for idx in range(len(centers)):
        count = int(np.count_nonzero(labels == idx))
        color_counts.append((idx, count))
    color_counts.sort(key=lambda x: x[1], reverse=True)

    logs.append("Extracting fill parts")

    for color_idx, count in color_counts:
        raw_mask = (labels == color_idx).astype(np.uint8) * 255
        if count == 0:
            continue

        # Cleanup
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # Slight overlap expansion to reduce gaps
        if cfg.overlap_px > 0:
            mask = cv2.dilate(mask, overlap_kernel, iterations=1)

        contours, _ = find_contours_compat(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        color_bgr = centers[color_idx]
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        fill_hex = rgb_to_hex(color_rgb)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            roi_priority = priority_mask[y:y+bh, x:x+bw]
            is_priority = bool(np.mean(roi_priority) > 5)

            min_area = cfg.min_area_priority if is_priority else cfg.min_area
            if area < min_area:
                continue

            eps_ratio = cfg.contour_epsilon_priority if is_priority else cfg.contour_epsilon
            approx = simplify_contour(
                contour,
                epsilon_ratio=eps_ratio,
                max_points=cfg.max_points_per_path,
                closed=True
            )
            if len(approx) < 3:
                continue

            pts = contour_to_points(approx)
            category = classify_fill_category(color_rgb, area)

            fill_parts.append(VectorPart(
                part_id=f"fill_{part_counter:05d}",
                category=category,
                layer=0,  # assigned later
                fill=fill_hex,
                stroke=None,
                stroke_width=0.0,
                closed=True,
                points=pts,
                bbox=(x, y, x + bw, y + bh),
                area=area,
                priority=is_priority
            ))
            part_counter += 1

    # Sort: large base shapes first, smaller details later
    fill_parts.sort(key=lambda p: (-p.area, p.priority))

    if len(fill_parts) > cfg.max_fill_parts:
        logs.append(f"Fill parts capped: {len(fill_parts)} -> {cfg.max_fill_parts}")
        fill_parts = fill_parts[:cfg.max_fill_parts]

    for i, p in enumerate(fill_parts):
        p.layer = 100 + i

    logs.append(f"Fill parts extracted: {len(fill_parts)}")
    return fill_parts


# =========================================================
# Outline extraction
# =========================================================

def extract_outline_parts(
    img_bgr: np.ndarray,
    priority_mask: np.ndarray,
    cfg: AppConfig,
    logs: List[str]
) -> List[VectorPart]:
    if not cfg.enable_outline:
        return []

    logs.append("Extracting outline parts")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)

    if cfg.outline_dilate_px > 0:
        k = max(1, cfg.outline_dilate_px)
        edges = cv2.dilate(edges, np.ones((k, k), np.uint8), iterations=1)

    contours, _ = find_contours_compat(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline_parts: List[VectorPart] = []
    part_counter = 0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < cfg.outline_min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        roi_priority = priority_mask[y:y+bh, x:x+bw]
        is_priority = bool(np.mean(roi_priority) > 5)

        eps_ratio = cfg.contour_epsilon_priority if is_priority else cfg.contour_epsilon
        approx = simplify_contour(
            contour,
            epsilon_ratio=eps_ratio,
            max_points=max(40, cfg.max_points_per_path),
            closed=True
        )
        if len(approx) < 3:
            continue

        pts = contour_to_points(approx)
        outline_parts.append(VectorPart(
            part_id=f"outline_{part_counter:05d}",
            category="outline",
            layer=0,
            fill=cfg.outline_color,   # filled edge blob for stability
            stroke=None,
            stroke_width=0.0,
            closed=True,
            points=pts,
            bbox=(x, y, x + bw, y + bh),
            area=area,
            priority=is_priority
        ))
        part_counter += 1

    # Smaller / priority contours later on top
    outline_parts.sort(key=lambda p: (p.priority, -p.area))

    if len(outline_parts) > cfg.max_outline_parts:
        logs.append(f"Outline parts capped: {len(outline_parts)} -> {cfg.max_outline_parts}")
        outline_parts = outline_parts[:cfg.max_outline_parts]

    for i, p in enumerate(outline_parts):
        p.layer = 10000 + i

    logs.append(f"Outline parts extracted: {len(outline_parts)}")
    return outline_parts


# =========================================================
# Rendering previews
# =========================================================

def render_vector_preview(shape_hw: Tuple[int, int], parts: List[VectorPart]) -> np.ndarray:
    h, w = shape_hw
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    parts_sorted = sorted(parts, key=lambda p: p.layer)

    for part in parts_sorted:
        pts = np.array(part.points, dtype=np.int32).reshape((-1, 1, 2))
        if len(pts) < 3:
            continue

        if part.fill:
            color = hex_to_bgr(part.fill)
            cv2.fillPoly(canvas, [pts], color)

        if part.stroke and part.stroke_width > 0:
            stroke_color = hex_to_bgr(part.stroke)
            cv2.polylines(
                canvas,
                [pts],
                isClosed=part.closed,
                color=stroke_color,
                thickness=max(1, int(round(part.stroke_width))),
                lineType=cv2.LINE_AA
            )

    return bgr_to_rgb(canvas)


# =========================================================
# SVG export
# =========================================================

def points_to_svg_path(points: List[Tuple[int, int]], closed: bool = True) -> str:
    if not points:
        return ""
    cmds = [f"M {points[0][0]} {points[0][1]}"]
    for x, y in points[1:]:
        cmds.append(f"L {x} {y}")
    if closed:
        cmds.append("Z")
    return " ".join(cmds)


def part_to_svg_element(part: VectorPart) -> str:
    d = points_to_svg_path(part.points, closed=part.closed)
    if not d:
        return ""

    attrs = [f'd="{d}"']
    if part.fill:
        attrs.append(f'fill="{part.fill}"')
    else:
        attrs.append('fill="none"')

    if part.stroke:
        attrs.append(f'stroke="{part.stroke}"')
        attrs.append(f'stroke-width="{part.stroke_width:.2f}"')
        attrs.append('stroke-linecap="round"')
        attrs.append('stroke-linejoin="round"')

    attrs.append(f'data-part-id="{part.part_id}"')
    attrs.append(f'data-category="{part.category}"')
    return "<path " + " ".join(attrs) + " />"


def build_svg_document(parts: List[VectorPart], width: int, height: int, title: str) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{title}</title>",
    ]

    for part in sorted(parts, key=lambda p: p.layer):
        el = part_to_svg_element(part)
        if el:
            lines.append(el)

    lines.append("</svg>")
    return "\n".join(lines)


def split_parts_by_svg_size(
    parts: List[VectorPart],
    width: int,
    height: int,
    limit_bytes: int,
    prefix: str
) -> Dict[str, str]:
    files = {}

    if not parts:
        return files

    ordered = sorted(parts, key=lambda p: p.layer)
    chunks: List[List[VectorPart]] = []
    current_chunk: List[VectorPart] = []

    for part in ordered:
        candidate = current_chunk + [part]
        candidate_svg = build_svg_document(candidate, width, height, f"{prefix}_chunk")
        candidate_size = len(candidate_svg.encode("utf-8"))

        if current_chunk and candidate_size > limit_bytes:
            chunks.append(current_chunk)
            current_chunk = [part]
        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk)

    for i, chunk in enumerate(chunks, start=1):
        name = f"{prefix}_{i:02d}.svg"
        files[name] = build_svg_document(chunk, width, height, name)

    return files


def generate_svg_files(
    parts: List[VectorPart],
    shape_hw: Tuple[int, int],
    cfg: AppConfig,
    logs: List[str]
) -> Dict[str, str]:
    h, w = shape_hw
    files: Dict[str, str] = {}

    fill_parts = [p for p in parts if p.category != "outline"]
    outline_parts = [p for p in parts if p.category == "outline"]

    # Full SVG
    files["00_full_combined.svg"] = build_svg_document(parts, w, h, "full_combined")

    # Logical layers
    if fill_parts:
        files["01_fills.svg"] = build_svg_document(fill_parts, w, h, "fills")
    if outline_parts:
        files["02_outlines.svg"] = build_svg_document(outline_parts, w, h, "outlines")

    # GT7 multipart split
    limit_bytes = int(cfg.svg_limit_kb * 1024)
    gt7_chunks = split_parts_by_svg_size(parts, w, h, limit_bytes, "gt7_part")
    files.update(gt7_chunks)

    logs.append(f"SVG files generated: {len(files)}")
    return files


# =========================================================
# ZIP bundle
# =========================================================

def parts_to_debug_json(parts: List[VectorPart]) -> List[Dict]:
    arr = []
    for p in parts:
        arr.append({
            "part_id": p.part_id,
            "category": p.category,
            "layer": p.layer,
            "fill": p.fill,
            "stroke": p.stroke,
            "stroke_width": p.stroke_width,
            "closed": p.closed,
            "bbox": p.bbox,
            "area": p.area,
            "priority": p.priority,
            "point_count": len(p.points),
        })
    return arr


def build_zip_bundle(
    result: ProcessResult,
    cfg: AppConfig,
    source_base_name: str
) -> bytes:
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # SVG files
        for filename, content in result.svg_files.items():
            zf.writestr(filename, content)

        # Preview PNGs
        zf.writestr("preview_original.png", png_bytes_from_rgb(result.original_rgb))
        zf.writestr("preview_processed.png", png_bytes_from_rgb(result.processed_rgb))
        zf.writestr("preview_quantized.png", png_bytes_from_rgb(result.quantized_rgb))
        zf.writestr("preview_vector.png", png_bytes_from_rgb(result.vector_preview_rgb))
        zf.writestr("priority_mask.png", png_bytes_from_rgb(np.stack([result.priority_mask]*3, axis=-1)))

        # Debug summary
        debug = {
            "source": source_base_name,
            "config": asdict(cfg),
            "stats": result.stats,
            "logs": result.logs,
            "parts": parts_to_debug_json(result.parts),
        }
        zf.writestr("debug_summary.json", json.dumps(debug, ensure_ascii=False, indent=2))

    buf.seek(0)
    return buf.getvalue()


# =========================================================
# Main pipeline
# =========================================================

def process_image_pipeline(img_rgb: np.ndarray, cfg: AppConfig) -> ProcessResult:
    logs: List[str] = []
    original_rgb = img_rgb.copy()
    img_bgr = rgb_to_bgr(img_rgb)

    processed_bgr = preprocess_image(img_bgr, cfg, logs)
    priority_mask = build_priority_mask(processed_bgr, logs, enabled=cfg.use_face_priority)

    quant_bgr, labels, centers = quantize_image(processed_bgr, cfg.num_colors, logs)

    fill_parts = extract_fill_parts(labels, centers, priority_mask, cfg, logs)
    outline_parts = extract_outline_parts(processed_bgr, priority_mask, cfg, logs)

    parts = fill_parts + outline_parts
    parts.sort(key=lambda p: p.layer)

    vector_preview_rgb = render_vector_preview(processed_bgr.shape[:2], parts)
    svg_files = generate_svg_files(parts, processed_bgr.shape[:2], cfg, logs)

    point_count_total = sum(len(p.points) for p in parts)
    svg_sizes = {name: len(text.encode("utf-8")) for name, text in svg_files.items()}

    stats = {
        "image_width": int(processed_bgr.shape[1]),
        "image_height": int(processed_bgr.shape[0]),
        "fill_parts": len(fill_parts),
        "outline_parts": len(outline_parts),
        "total_parts": len(parts),
        "total_points": int(point_count_total),
        "svg_file_count": len(svg_files),
        "svg_sizes_bytes": svg_sizes,
    }

    return ProcessResult(
        original_rgb=original_rgb,
        processed_rgb=bgr_to_rgb(processed_bgr),
        quantized_rgb=bgr_to_rgb(quant_bgr),
        priority_mask=priority_mask,
        vector_preview_rgb=vector_preview_rgb,
        parts=parts,
        svg_files=svg_files,
        stats=stats,
        logs=logs,
    )


# =========================================================
# UI
# =========================================================

st.title("GT7 / Forza-style Layered SVG Vectorizer")
st.caption("落ちにくさ優先・大面積優先・顔優先・GT7向け分割付きの1ファイル版")

with st.sidebar:
    st.header("Settings")

    mode_name = st.selectbox(
        "Mode",
        list(PRESETS.keys()),
        index=1
    )
    base_cfg = PRESETS[mode_name]

    num_colors = st.slider("Number of colors", 4, 24, base_cfg.num_colors, 1)
    use_face_priority = st.checkbox("Face / center priority", value=base_cfg.use_face_priority)
    enable_outline = st.checkbox("Include outline layer", value=base_cfg.enable_outline)
    svg_limit_kb = st.slider("GT7 target part size (KB)", 8.0, 15.0, float(base_cfg.svg_limit_kb), 0.5)
    overlap_px = st.slider("Overlap / underpaint strength", 0, 4, base_cfg.overlap_px, 1)

    with st.expander("Advanced"):
        max_dim = st.slider("Max processing dimension", 800, 2400, base_cfg.max_dim, 100)
        min_area = st.slider("Min region area", 10, 300, base_cfg.min_area, 2)
        min_area_priority = st.slider("Min region area (priority)", 5, 100, base_cfg.min_area_priority, 1)
        max_fill_parts = st.slider("Max fill parts", 50, 500, base_cfg.max_fill_parts, 10)
        max_outline_parts = st.slider("Max outline parts", 0, 300, base_cfg.max_outline_parts, 10)
        outline_color = st.color_picker("Outline color", base_cfg.outline_color)
        contour_eps = st.slider("Contour simplify", 0.002, 0.03, float(base_cfg.contour_epsilon), 0.001)
        contour_eps_priority = st.slider("Contour simplify (priority)", 0.001, 0.02, float(base_cfg.contour_epsilon_priority), 0.0005)

    cfg = AppConfig(
        mode_name=mode_name,
        max_dim=max_dim,
        num_colors=num_colors,
        min_area=min_area,
        min_area_priority=min_area_priority,
        overlap_px=overlap_px,
        contour_epsilon=contour_eps,
        contour_epsilon_priority=contour_eps_priority,
        max_points_per_path=base_cfg.max_points_per_path,
        morph_open=base_cfg.morph_open,
        morph_close=base_cfg.morph_close,
        canny_low=base_cfg.canny_low,
        canny_high=base_cfg.canny_high,
        outline_dilate_px=base_cfg.outline_dilate_px,
        outline_min_area=base_cfg.outline_min_area,
        max_fill_parts=max_fill_parts,
        max_outline_parts=max_outline_parts,
        svg_limit_kb=svg_limit_kb,
        enable_outline=enable_outline,
        use_face_priority=use_face_priority,
        outline_color=outline_color,
    )

uploaded = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg", "webp"]
)

col_a, col_b = st.columns([1, 1])

if uploaded is not None:
    img_rgb = open_uploaded_image(uploaded)

    with col_a:
        st.subheader("Original")
        st.image(img_rgb, use_container_width=True)

    run = st.button("Generate SVGs", type="primary", use_container_width=True)

    if run:
        try:
            with st.spinner("Processing image..."):
                result = process_image_pipeline(img_rgb, cfg)
                st.session_state["last_result"] = result
                st.session_state["last_cfg"] = cfg
                st.session_state["source_name"] = uploaded.name
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.stop()

if "last_result" in st.session_state:
    result: ProcessResult = st.session_state["last_result"]
    last_cfg: AppConfig = st.session_state["last_cfg"]
    source_name = st.session_state.get("source_name", "image.png")
    source_base = safe_filename_base(source_name)

    tabs = st.tabs([
        "Previews",
        "SVG Files",
        "Stats",
        "Logs / Debug",
        "Download"
    ])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Processed")
            st.image(result.processed_rgb, use_container_width=True)

            st.subheader("Priority Mask")
            st.image(np.stack([result.priority_mask] * 3, axis=-1), use_container_width=True)

        with c2:
            st.subheader("Quantized")
            st.image(result.quantized_rgb, use_container_width=True)

            st.subheader("Vector Preview")
            st.image(result.vector_preview_rgb, use_container_width=True)

    with tabs[1]:
        st.subheader("Generated SVG files")
        for filename, svg_text in result.svg_files.items():
            size_bytes = len(svg_text.encode("utf-8"))
            size_kb = size_bytes / 1024.0
            with st.expander(f"{filename}  ({size_kb:.2f} KB)"):
                st.code(svg_text[:4000] + ("\n\n... (truncated)" if len(svg_text) > 4000 else ""), language="xml")
                st.download_button(
                    label=f"Download {filename}",
                    data=svg_text.encode("utf-8"),
                    file_name=filename,
                    mime="image/svg+xml",
                    use_container_width=True,
                    key=f"download_{filename}"
                )

    with tabs[2]:
        st.subheader("Stats")

        stats = result.stats
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fill parts", stats["fill_parts"])
        m2.metric("Outline parts", stats["outline_parts"])
        m3.metric("Total parts", stats["total_parts"])
        m4.metric("Total points", stats["total_points"])

        st.json(stats)

    with tabs[3]:
        st.subheader("Logs")
        for log in result.logs:
            st.write("-", log)

        st.subheader("Part Debug")
        debug_parts = parts_to_debug_json(result.parts)
        st.json(debug_parts[:80])

    with tabs[4]:
        st.subheader("Download bundle")

        zip_bytes = build_zip_bundle(result, last_cfg, source_base)
        zip_name = f"{source_base}_layered_svg_bundle.zip"

        st.download_button(
            label="Download ZIP bundle",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            use_container_width=True
        )

        st.caption("ZIP includes: full SVG, logical SVGs, GT7 split SVGs, preview PNGs, priority mask, debug JSON")

else:
    st.info("画像をアップロードして Generate SVGs を押してください。")