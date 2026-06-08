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
    page_title="GT7 / Forza-style Generic Layered SVG Vectorizer V3",
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
    outline_min_area: int
    outline_width: float
    max_fill_parts: int
    max_outline_parts: int
    svg_limit_kb: float
    use_face_priority: bool
    remove_background: bool
    protect_skin: bool
    protect_accents: bool
    line_darkness_threshold: int
    outline_color: str
    edge_strength: int


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
    foreground_mask: np.ndarray
    priority_mask: np.ndarray
    quantized_rgb: np.ndarray
    vector_preview_rgb: np.ndarray
    parts: List[VectorPart]
    svg_files: Dict[str, str]
    stats: Dict
    logs: List[str]


# =========================================================
# Presets
# =========================================================

PRESETS = {
    "Anime Clean": AppConfig(
        mode_name="Anime Clean",
        max_dim=1500,
        num_colors=14,
        min_area=40,
        min_area_priority=12,
        overlap_px=1,
        contour_epsilon=0.0045,
        contour_epsilon_priority=0.0020,
        max_points_per_path=220,
        morph_open=3,
        morph_close=5,
        outline_min_area=10,
        outline_width=1.0,
        max_fill_parts=320,
        max_outline_parts=140,
        svg_limit_kb=14.5,
        use_face_priority=True,
        remove_background=True,
        protect_skin=True,
        protect_accents=True,
        line_darkness_threshold=95,
        outline_color="#202020",
        edge_strength=100,
    ),
    "Anime Detailed": AppConfig(
        mode_name="Anime Detailed",
        max_dim=1800,
        num_colors=18,
        min_area=28,
        min_area_priority=8,
        overlap_px=1,
        contour_epsilon=0.0035,
        contour_epsilon_priority=0.0014,
        max_points_per_path=320,
        morph_open=3,
        morph_close=5,
        outline_min_area=8,
        outline_width=0.95,
        max_fill_parts=450,
        max_outline_parts=220,
        svg_limit_kb=14.5,
        use_face_priority=True,
        remove_background=True,
        protect_skin=True,
        protect_accents=True,
        line_darkness_threshold=100,
        outline_color="#1F1F1F",
        edge_strength=110,
    ),
    "Illustration Generic": AppConfig(
        mode_name="Illustration Generic",
        max_dim=1600,
        num_colors=16,
        min_area=38,
        min_area_priority=10,
        overlap_px=1,
        contour_epsilon=0.0040,
        contour_epsilon_priority=0.0018,
        max_points_per_path=260,
        morph_open=3,
        morph_close=5,
        outline_min_area=10,
        outline_width=1.0,
        max_fill_parts=360,
        max_outline_parts=160,
        svg_limit_kb=14.5,
        use_face_priority=True,
        remove_background=True,
        protect_skin=True,
        protect_accents=True,
        line_darkness_threshold=95,
        outline_color="#222222",
        edge_strength=100,
    ),
    "Safe Universal": AppConfig(
        mode_name="Safe Universal",
        max_dim=1280,
        num_colors=12,
        min_area=60,
        min_area_priority=16,
        overlap_px=1,
        contour_epsilon=0.0055,
        contour_epsilon_priority=0.0025,
        max_points_per_path=180,
        morph_open=3,
        morph_close=5,
        outline_min_area=12,
        outline_width=1.05,
        max_fill_parts=240,
        max_outline_parts=100,
        svg_limit_kb=14.5,
        use_face_priority=True,
        remove_background=True,
        protect_skin=True,
        protect_accents=True,
        line_darkness_threshold=92,
        outline_color="#242424",
        edge_strength=95,
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
    while len(approx) > max_points and tries < 10:
        eps *= 1.25
        approx = cv2.approxPolyDP(contour, eps, closed)
        tries += 1

    return approx


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


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


def png_bytes_from_rgb(rgb: np.ndarray) -> bytes:
    img = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def luminance_bgr(color_bgr: Tuple[int, int, int]) -> float:
    b, g, r = [float(x) for x in color_bgr]
    return 0.0722 * b + 0.7152 * g + 0.2126 * r


def color_sat_hsv(color_bgr: Tuple[int, int, int]) -> float:
    arr = np.uint8([[list(color_bgr)]])
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0, 0]
    return float(hsv[1])


def draw_mask_preview(mask: np.ndarray) -> np.ndarray:
    return np.stack([mask] * 3, axis=-1)


def erode_mask(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


def dilate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


# =========================================================
# Preprocess
# =========================================================

def preprocess_image(img_bgr: np.ndarray, cfg: AppConfig, logs: List[str]) -> np.ndarray:
    logs.append("Preprocess: resize / denoise / contrast normalize")
    img_bgr, scale = resize_to_max_dim(img_bgr, cfg.max_dim)
    if scale < 1.0:
        logs.append(f"Image resized to max {cfg.max_dim}px (scale={scale:.3f})")

    img_bgr = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=40, sigmaSpace=40)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img_bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return img_bgr


# =========================================================
# Foreground extraction (generic-ish)
# =========================================================

def build_foreground_mask(img_bgr: np.ndarray, cfg: AppConfig, logs: List[str]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if not cfg.remove_background:
        return np.full((h, w), 255, dtype=np.uint8)

    logs.append("Foreground mask: border-connected background removal + saliency fallback")

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Border sample
    border = np.concatenate([
        img_bgr[0, :, :],
        img_bgr[-1, :, :],
        img_bgr[:, 0, :],
        img_bgr[:, -1, :]
    ], axis=0)
    border_med = np.median(border, axis=0).astype(np.uint8)

    diff = np.sqrt(np.sum((img_bgr.astype(np.float32) - border_med.astype(np.float32)) ** 2, axis=2))
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # likely background if close to border color OR very bright & low sat
    bg_like = ((diff < 25) | ((val > 220) & (sat < 25))).astype(np.uint8) * 255

    # Keep only border-connected bg
    flood_seed_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_input = bg_like.copy()
    flood_fill_img = flood_input.copy()

    # Flood fill from corners
    for sx, sy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flood_fill_img[sy, sx] > 0:
            cv2.floodFill(flood_fill_img, flood_seed_mask, (sx, sy), 128)

    border_connected_bg = (flood_fill_img == 128).astype(np.uint8) * 255
    fg = cv2.bitwise_not(border_connected_bg)

    # Fallback: if too little foreground, use edge/saliency-based fallback
    fg_ratio = float(np.count_nonzero(fg)) / float(h * w)
    if fg_ratio < 0.04 or fg_ratio > 0.96:
        logs.append("Foreground fallback: using edge-density subject extraction")
        edges = cv2.Canny(gray, 60, 160)
        edges = dilate_mask(edges, 3)
        contours, _ = find_contours_compat(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fallback = np.zeros((h, w), dtype=np.uint8)
        contour_areas = [(c, cv2.contourArea(c)) for c in contours]
        contour_areas.sort(key=lambda x: x[1], reverse=True)
        kept = 0
        for c, area in contour_areas:
            if area < (h * w * 0.002):
                continue
            cv2.drawContours(fallback, [c], -1, 255, thickness=-1)
            kept += 1
            if kept >= 12:
                break
        if np.count_nonzero(fallback) > 0:
            fg = fallback

    # Morph cleanup
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Keep main components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8), connectivity=8)
    if num_labels > 1:
        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
        areas.sort(key=lambda x: x[1], reverse=True)
        fg2 = np.zeros_like(fg)
        kept_area = 0
        area_limit = h * w * 0.65
        for i, area in areas:
            if area < max(80, int(h * w * 0.0005)):
                continue
            fg2[labels == i] = 255
            kept_area += area
            if kept_area >= area_limit and len(areas) > 2:
                break
        if np.count_nonzero(fg2) > 0:
            fg = fg2

    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = dilate_mask(fg, 1)

    logs.append(f"Foreground ratio: {np.count_nonzero(fg) / float(h*w):.3f}")
    return fg


# =========================================================
# Priority masks
# =========================================================

def detect_face_regions(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    faces = []
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    except Exception:
        pass
    return list(faces) if len(faces) else []


def build_skin_mask(img_bgr: np.ndarray, fg_mask: np.ndarray, logs: List[str]) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    skin1 = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 180, 135], dtype=np.uint8))
    # additional hsv heuristic
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    skin2 = (((h < 25) | (h > 170)) & (s > 20) & (s < 170) & (v > 60)).astype(np.uint8) * 255

    skin = cv2.bitwise_and(skin1, skin2)
    skin = cv2.bitwise_and(skin, fg_mask)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    logs.append(f"Skin candidate pixels: {int(np.count_nonzero(skin))}")
    return skin


def build_accent_mask(img_bgr: np.ndarray, fg_mask: np.ndarray, logs: List[str]) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    accent = ((sat > 90) & (val > 40)).astype(np.uint8) * 255
    accent = cv2.bitwise_and(accent, fg_mask)
    accent = cv2.morphologyEx(accent, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # discard very huge accent masks
    total_fg = max(1, np.count_nonzero(fg_mask))
    accent_ratio = np.count_nonzero(accent) / float(total_fg)
    if accent_ratio > 0.45:
        accent = (((sat > 120) & (val > 50)).astype(np.uint8) * 255)
        accent = cv2.bitwise_and(accent, fg_mask)

    logs.append(f"Accent candidate ratio (within FG): {np.count_nonzero(accent) / float(total_fg):.3f}")
    return accent


def build_priority_mask(
    img_bgr: np.ndarray,
    fg_mask: np.ndarray,
    cfg: AppConfig,
    logs: List[str]
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    priority = np.zeros((h, w), dtype=np.uint8)

    x1, y1, x2, y2 = mask_bbox(fg_mask)
    if x2 <= x1 or y2 <= y1:
        return priority

    # Subject core: top-middle of foreground bbox
    subj = np.zeros((h, w), dtype=np.uint8)
    cx = (x1 + x2) // 2
    cy = y1 + int((y2 - y1) * 0.32)
    ax = max(20, int((x2 - x1) * 0.23))
    ay = max(20, int((y2 - y1) * 0.22))
    cv2.ellipse(subj, (cx, cy), (ax, ay), 0, 0, 360, 130, -1)
    priority = np.maximum(priority, subj)

    # Face detector boost
    faces = detect_face_regions(img_bgr) if cfg.use_face_priority else []
    if faces:
        logs.append(f"Face detector regions: {len(faces)}")
        for (fx, fy, fw, fh) in faces:
            px = int(fw * 0.3)
            py = int(fh * 0.35)
            rx1 = clamp(fx - px, 0, w - 1)
            ry1 = clamp(fy - py, 0, h - 1)
            rx2 = clamp(fx + fw + px, 0, w - 1)
            ry2 = clamp(fy + fh + py, 0, h - 1)
            cv2.rectangle(priority, (rx1, ry1), (rx2, ry2), 255, -1)
    else:
        logs.append("Face detector fallback: subject-core ellipse only")

    # Skin boost
    if cfg.protect_skin:
        skin = build_skin_mask(img_bgr, fg_mask, logs)
        priority = np.maximum(priority, (skin > 0).astype(np.uint8) * 220)
    else:
        skin = np.zeros((h, w), dtype=np.uint8)

    # Accent boost
    if cfg.protect_accents:
        accent = build_accent_mask(img_bgr, fg_mask, logs)
        priority = np.maximum(priority, (accent > 0).astype(np.uint8) * 190)
    else:
        accent = np.zeros((h, w), dtype=np.uint8)

    # Detail boost via local edges near subject
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.bitwise_and(edges, fg_mask)
    priority = np.maximum(priority, ((edges > 0) & (subj > 0)).astype(np.uint8) * 160)

    priority = cv2.GaussianBlur(priority, (0, 0), sigmaX=10, sigmaY=10)
    _, priority = cv2.threshold(priority, 40, 255, cv2.THRESH_BINARY)

    # keep within foreground
    priority = cv2.bitwise_and(priority, fg_mask)
    return priority


# =========================================================
# Base quantization
# =========================================================

def quantize_image_masked(img_bgr: np.ndarray, fg_mask: np.ndarray, k: int, logs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logs.append(f"Base quantization: {k} colors (foreground-aware)")

    h, w = img_bgr.shape[:2]
    pixels = img_bgr.reshape((-1, 3)).astype(np.float32)
    fg_flat = (fg_mask.reshape(-1) > 0)

    if np.count_nonzero(fg_flat) < 10:
        labels = np.zeros((h, w), dtype=np.int32)
        centers = np.array([[255, 255, 255]], dtype=np.uint8)
        return img_bgr.copy(), labels, centers

    fg_pixels = pixels[fg_flat]
    K = min(k, len(fg_pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.8)

    compactness, labels_fg, centers = cv2.kmeans(
        fg_pixels,
        K=K,
        bestLabels=None,
        criteria=criteria,
        attempts=4,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    centers = np.uint8(centers)

    # assign non-fg to nearest very bright background-like center
    quant = np.full_like(pixels, 245, dtype=np.uint8)
    label_map = np.full((h * w,), -1, dtype=np.int32)
    quant[fg_flat] = centers[labels_fg.flatten()]
    label_map[fg_flat] = labels_fg.flatten()

    quant = quant.reshape((h, w, 3))
    labels = label_map.reshape((h, w))

    # For preview, restore background to light gray/white
    bg = (fg_mask == 0)
    quant[bg] = np.array([245, 245, 245], dtype=np.uint8)

    logs.append(f"KMeans compactness: {compactness:.2f}")
    return quant, labels, centers


# =========================================================
# Cluster role classification
# =========================================================

def classify_cluster_role(
    color_bgr: Tuple[int, int, int],
    cluster_mask: np.ndarray,
    fg_mask: np.ndarray,
    priority_mask: np.ndarray,
    skin_mask: np.ndarray,
    accent_mask: np.ndarray,
) -> str:
    area = np.count_nonzero(cluster_mask)
    if area == 0:
        return "mid_base"

    lum = luminance_bgr(color_bgr)
    sat = color_sat_hsv(color_bgr)
    overlap_skin = np.count_nonzero(cv2.bitwise_and(cluster_mask, skin_mask)) / float(area)
    overlap_accent = np.count_nonzero(cv2.bitwise_and(cluster_mask, accent_mask)) / float(area)
    overlap_priority = np.count_nonzero(cv2.bitwise_and(cluster_mask, priority_mask)) / float(area)

    if overlap_skin > 0.18 and lum > 60:
        return "skin_base"
    if overlap_accent > 0.12 and sat > 70:
        return "accent"
    if lum > 205:
        return "light_base"
    if lum < 60:
        # if small and priority, better treated like line candidate will be separate anyway,
        # but for fill role keep dark areas as dark_base/shadow.
        if overlap_priority > 0.15:
            return "dark_shadow"
        return "dark_base"
    if lum < 110:
        return "mid_shadow"
    return "mid_base"


def layer_for_role(role: str, area: float) -> int:
    # lower number = below
    base = {
        "light_base": 100,
        "mid_base": 110,
        "dark_base": 120,
        "skin_base": 130,
        "mid_shadow": 210,
        "dark_shadow": 220,
        "accent": 300,
        "highlight": 320,
        "outline": 500,
        "detail": 520,
    }.get(role, 150)
    # smaller parts later inside same role
    area_offset = int(max(0, 10000 - min(9999, area)) / 500)
    return base + area_offset


# =========================================================
# Fill extraction
# =========================================================

def extract_fill_parts(
    labels: np.ndarray,
    centers: np.ndarray,
    fg_mask: np.ndarray,
    priority_mask: np.ndarray,
    skin_mask: np.ndarray,
    accent_mask: np.ndarray,
    cfg: AppConfig,
    logs: List[str]
) -> List[VectorPart]:
    h, w = labels.shape[:2]
    parts: List[VectorPart] = []
    part_counter = 0

    kernel_open = np.ones((cfg.morph_open, cfg.morph_open), np.uint8)
    kernel_close = np.ones((cfg.morph_close, cfg.morph_close), np.uint8)
    overlap_kernel = np.ones((max(1, cfg.overlap_px), max(1, cfg.overlap_px)), np.uint8)

    # cluster counts
    color_counts = []
    for idx in range(len(centers)):
        count = int(np.count_nonzero(labels == idx))
        color_counts.append((idx, count))
    color_counts.sort(key=lambda x: x[1], reverse=True)

    logs.append("Extracting fill parts")

    for color_idx, _count in color_counts:
        if color_idx < 0:
            continue

        raw_mask = ((labels == color_idx).astype(np.uint8) * 255)
        raw_mask = cv2.bitwise_and(raw_mask, fg_mask)
        if np.count_nonzero(raw_mask) == 0:
            continue

        color_bgr = tuple(int(v) for v in centers[color_idx])
        role = classify_cluster_role(color_bgr, raw_mask, fg_mask, priority_mask, skin_mask, accent_mask)

        # morphological cleanup
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        if role in ("light_base", "skin_base", "mid_base"):
            mask = cv2.dilate(mask, overlap_kernel, iterations=1)

        contours, _ = find_contours_compat(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            roi_p = priority_mask[y:y+bh, x:x+bw]
            is_priority = bool(np.mean(roi_p) > 5)

            min_area = cfg.min_area_priority if is_priority else cfg.min_area
            if area < min_area:
                continue

            eps_ratio = cfg.contour_epsilon_priority if is_priority else cfg.contour_epsilon
            max_pts = cfg.max_points_per_path * (2 if is_priority else 1)
            approx = simplify_contour(contour, eps_ratio, max_pts, closed=True)
            if len(approx) < 3:
                continue

            pts = contour_to_points(approx)
            color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
            fill_hex = rgb_to_hex(color_rgb)

            parts.append(VectorPart(
                part_id=f"fill_{part_counter:05d}",
                category=role,
                layer=layer_for_role(role, area),
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

    # Optional protective overlays
    if cfg.protect_skin and np.count_nonzero(skin_mask) > 0:
        parts.extend(extract_protective_parts(
            skin_mask, img_color=average_mask_color_bgr(labels, centers, skin_mask),
            category="skin_base", cfg=cfg, part_prefix="skin", force_layer=135
        ))

    if cfg.protect_accents and np.count_nonzero(accent_mask) > 0:
        # Accent protection only for smaller regions within fg
        protected = cv2.bitwise_and(accent_mask, erode_mask(fg_mask, 1))
        parts.extend(extract_protective_parts(
            protected, img_color=average_mask_color_bgr(labels, centers, protected),
            category="accent", cfg=cfg, part_prefix="accent", force_layer=310
        ))

    # dedupe/cap
    parts.sort(key=lambda p: (p.layer, -p.area))
    if len(parts) > cfg.max_fill_parts:
        logs.append(f"Fill parts capped: {len(parts)} -> {cfg.max_fill_parts}")
        parts = parts[:cfg.max_fill_parts]

    logs.append(f"Fill parts extracted: {len(parts)}")
    return parts


def average_mask_color_bgr(labels: np.ndarray, centers: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    if np.count_nonzero(mask) == 0 or len(centers) == 0:
        return (180, 180, 180)
    lbls = labels[mask > 0]
    lbls = lbls[lbls >= 0]
    if len(lbls) == 0:
        return (180, 180, 180)
    vals, counts = np.unique(lbls, return_counts=True)
    idx = vals[np.argmax(counts)]
    return tuple(int(v) for v in centers[int(idx)])


def extract_protective_parts(
    mask: np.ndarray,
    img_color: Tuple[int, int, int],
    category: str,
    cfg: AppConfig,
    part_prefix: str,
    force_layer: int
) -> List[VectorPart]:
    parts: List[VectorPart] = []
    kernel = np.ones((3, 3), np.uint8)
    proc = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel)

    contours, _ = find_contours_compat(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    counter = 0
    fill_hex = rgb_to_hex((img_color[2], img_color[1], img_color[0]))

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(12, cfg.min_area_priority):
            continue
        approx = simplify_contour(contour, cfg.contour_epsilon_priority, cfg.max_points_per_path * 2, closed=True)
        if len(approx) < 3:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        parts.append(VectorPart(
            part_id=f"{part_prefix}_{counter:04d}",
            category=category,
            layer=force_layer,
            fill=fill_hex,
            stroke=None,
            stroke_width=0.0,
            closed=True,
            points=contour_to_points(approx),
            bbox=(x, y, x + bw, y + bh),
            area=area,
            priority=True
        ))
        counter += 1

    return parts


# =========================================================
# Line extraction (separate pipeline)
# =========================================================

def extract_outline_parts(
    img_bgr: np.ndarray,
    fg_mask: np.ndarray,
    priority_mask: np.ndarray,
    cfg: AppConfig,
    logs: List[str]
) -> List[VectorPart]:
    logs.append("Extracting outline parts (separate line pipeline)")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Dark-line candidates
    dark = ((gray < cfg.line_darkness_threshold).astype(np.uint8) * 255)
    dark = cv2.bitwise_and(dark, fg_mask)

    # Edge candidates
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, max(25, cfg.edge_strength - 40), max(70, cfg.edge_strength + 40))
    edges = cv2.bitwise_and(edges, fg_mask)

    combined = cv2.bitwise_and(edges, dilate_mask(dark, 1))
    combined = dilate_mask(combined, 1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    contours, _ = find_contours_compat(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    parts: List[VectorPart] = []
    counter = 0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < cfg.outline_min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        roi_p = priority_mask[y:y+bh, x:x+bw]
        is_priority = bool(np.mean(roi_p) > 5)

        approx = simplify_contour(
            contour,
            cfg.contour_epsilon_priority if is_priority else cfg.contour_epsilon,
            cfg.max_points_per_path,
            closed=False
        )

        if len(approx) < 2:
            continue

        pts = contour_to_points(approx)

        # filter absurdly tiny polylines
        if len(pts) < 2:
            continue

        parts.append(VectorPart(
            part_id=f"outline_{counter:05d}",
            category="outline",
            layer=500 if not is_priority else 540,
            fill=None,
            stroke=cfg.outline_color,
            stroke_width=max(0.5, cfg.outline_width * (1.0 if not is_priority else 1.05)),
            closed=False,
            points=pts,
            bbox=(x, y, x + bw, y + bh),
            area=area,
            priority=is_priority
        ))
        counter += 1

    # sort: larger first, priority on top
    parts.sort(key=lambda p: (p.layer, -p.area))
    if len(parts) > cfg.max_outline_parts:
        logs.append(f"Outline parts capped: {len(parts)} -> {cfg.max_outline_parts}")
        parts = parts[:cfg.max_outline_parts]

    logs.append(f"Outline parts extracted: {len(parts)}")
    return parts


# =========================================================
# Preview rendering
# =========================================================

def render_vector_preview(shape_hw: Tuple[int, int], parts: List[VectorPart]) -> np.ndarray:
    h, w = shape_hw
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)

    for part in sorted(parts, key=lambda p: (p.layer, -p.area)):
        pts = np.array(part.points, dtype=np.int32).reshape((-1, 1, 2))
        if len(pts) < 2:
            continue

        if part.fill and len(pts) >= 3:
            cv2.fillPoly(canvas, [pts], hex_to_bgr(part.fill))

        if part.stroke and part.stroke_width > 0:
            cv2.polylines(
                canvas,
                [pts],
                isClosed=part.closed,
                color=hex_to_bgr(part.stroke),
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
    attrs.append(f'fill="{part.fill}"' if part.fill else 'fill="none"')

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
    for part in sorted(parts, key=lambda p: (p.layer, -p.area)):
        el = part_to_svg_element(part)
        if el:
            lines.append(el)
    lines.append("</svg>")
    return "\n".join(lines)


def split_parts_by_svg_size(parts: List[VectorPart], width: int, height: int, limit_bytes: int, prefix: str) -> Dict[str, str]:
    files = {}
    ordered = sorted(parts, key=lambda p: (p.layer, -p.area))
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


def generate_svg_files(parts: List[VectorPart], shape_hw: Tuple[int, int], cfg: AppConfig, logs: List[str]) -> Dict[str, str]:
    h, w = shape_hw
    files: Dict[str, str] = {}
    fill_parts = [p for p in parts if p.fill]
    outline_parts = [p for p in parts if p.stroke]

    files["00_full_combined.svg"] = build_svg_document(parts, w, h, "full_combined")
    if fill_parts:
        files["01_fills.svg"] = build_svg_document(fill_parts, w, h, "fills")
    if outline_parts:
        files["02_outlines.svg"] = build_svg_document(outline_parts, w, h, "outlines")

    limit_bytes = int(cfg.svg_limit_kb * 1024)
    files.update(split_parts_by_svg_size(parts, w, h, limit_bytes, "gt7_part"))

    logs.append(f"SVG files generated: {len(files)}")
    return files


# =========================================================
# Bundle / debug
# =========================================================

def parts_to_debug_json(parts: List[VectorPart]) -> List[Dict]:
    return [
        {
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
        }
        for p in parts
    ]


def build_zip_bundle(result: ProcessResult, cfg: AppConfig, source_base_name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, content in result.svg_files.items():
            zf.writestr(filename, content)

        zf.writestr("preview_original.png", png_bytes_from_rgb(result.original_rgb))
        zf.writestr("preview_processed.png", png_bytes_from_rgb(result.processed_rgb))
        zf.writestr("foreground_mask.png", png_bytes_from_rgb(draw_mask_preview(result.foreground_mask)))
        zf.writestr("priority_mask.png", png_bytes_from_rgb(draw_mask_preview(result.priority_mask)))
        zf.writestr("preview_quantized.png", png_bytes_from_rgb(result.quantized_rgb))
        zf.writestr("preview_vector.png", png_bytes_from_rgb(result.vector_preview_rgb))

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
    fg_mask = build_foreground_mask(processed_bgr, cfg, logs)

    priority_mask = build_priority_mask(processed_bgr, fg_mask, cfg, logs)
    skin_mask = build_skin_mask(processed_bgr, fg_mask, logs) if cfg.protect_skin else np.zeros_like(fg_mask)
    accent_mask = build_accent_mask(processed_bgr, fg_mask, logs) if cfg.protect_accents else np.zeros_like(fg_mask)

    quant_bgr, labels, centers = quantize_image_masked(processed_bgr, fg_mask, cfg.num_colors, logs)

    fill_parts = extract_fill_parts(
        labels=labels,
        centers=centers,
        fg_mask=fg_mask,
        priority_mask=priority_mask,
        skin_mask=skin_mask,
        accent_mask=accent_mask,
        cfg=cfg,
        logs=logs
    )

    outline_parts = extract_outline_parts(
        img_bgr=processed_bgr,
        fg_mask=fg_mask,
        priority_mask=priority_mask,
        cfg=cfg,
        logs=logs
    )

    parts = fill_parts + outline_parts
    parts.sort(key=lambda p: (p.layer, -p.area))

    vector_preview_rgb = render_vector_preview(processed_bgr.shape[:2], parts)
    svg_files = generate_svg_files(parts, processed_bgr.shape[:2], cfg, logs)

    stats = {
        "image_width": int(processed_bgr.shape[1]),
        "image_height": int(processed_bgr.shape[0]),
        "foreground_pixels": int(np.count_nonzero(fg_mask)),
        "priority_pixels": int(np.count_nonzero(priority_mask)),
        "fill_parts": len(fill_parts),
        "outline_parts": len(outline_parts),
        "total_parts": len(parts),
        "total_points": int(sum(len(p.points) for p in parts)),
        "svg_file_count": len(svg_files),
        "svg_sizes_bytes": {k: len(v.encode("utf-8")) for k, v in svg_files.items()},
    }

    return ProcessResult(
        original_rgb=original_rgb,
        processed_rgb=bgr_to_rgb(processed_bgr),
        foreground_mask=fg_mask,
        priority_mask=priority_mask,
        quantized_rgb=bgr_to_rgb(quant_bgr),
        vector_preview_rgb=vector_preview_rgb,
        parts=parts,
        svg_files=svg_files,
        stats=stats,
        logs=logs
    )


# =========================================================
# UI
# =========================================================

st.title("GT7 / Forza-style Generic Layered SVG Vectorizer V3")
st.caption("汎用性強化版：foreground / priority / accent / skin / line を分離して扱うV3")

with st.sidebar:
    st.header("Settings")

    mode_name = st.selectbox("Mode", list(PRESETS.keys()), index=1)
    base_cfg = PRESETS[mode_name]

    num_colors = st.slider("Number of colors", 6, 28, base_cfg.num_colors, 1)
    use_face_priority = st.checkbox("Use face / subject priority", value=base_cfg.use_face_priority)
    remove_background = st.checkbox("Remove border-connected background", value=base_cfg.remove_background)
    protect_skin = st.checkbox("Protect skin-like regions", value=base_cfg.protect_skin)
    protect_accents = st.checkbox("Protect accent colors", value=base_cfg.protect_accents)

    svg_limit_kb = st.slider("GT7 target part size (KB)", 8.0, 15.0, float(base_cfg.svg_limit_kb), 0.5)
    outline_width = st.slider("Outline width", 0.5, 2.0, float(base_cfg.outline_width), 0.05)
    overlap_px = st.slider("Overlap / underpaint", 0, 4, base_cfg.overlap_px, 1)

    with st.expander("Advanced"):
        max_dim = st.slider("Max processing dimension", 800, 2600, base_cfg.max_dim, 100)
        min_area = st.slider("Min region area", 8, 300, base_cfg.min_area, 1)
        min_area_priority = st.slider("Min region area (priority)", 4, 120, base_cfg.min_area_priority, 1)
        contour_epsilon = st.slider("Contour simplify", 0.001, 0.02, float(base_cfg.contour_epsilon), 0.0005)
        contour_epsilon_priority = st.slider(
            "Contour simplify (priority)",
            0.0005,
            0.01,
            float(base_cfg.contour_epsilon_priority),
            0.0005
        )
        max_fill_parts = st.slider("Max fill parts", 50, 700, base_cfg.max_fill_parts, 10)
        max_outline_parts = st.slider("Max outline parts", 0, 400, base_cfg.max_outline_parts, 10)
        line_darkness_threshold = st.slider("Line darkness threshold", 50, 140, base_cfg.line_darkness_threshold, 1)
        edge_strength = st.slider("Edge strength", 40, 180, base_cfg.edge_strength, 1)
        outline_color = st.color_picker("Outline color", base_cfg.outline_color)

    cfg = AppConfig(
        mode_name=mode_name,
        max_dim=max_dim,
        num_colors=num_colors,
        min_area=min_area,
        min_area_priority=min_area_priority,
        overlap_px=overlap_px,
        contour_epsilon=contour_epsilon,
        contour_epsilon_priority=contour_epsilon_priority,
        max_points_per_path=base_cfg.max_points_per_path,
        morph_open=base_cfg.morph_open,
        morph_close=base_cfg.morph_close,
        outline_min_area=base_cfg.outline_min_area,
        outline_width=outline_width,
        max_fill_parts=max_fill_parts,
        max_outline_parts=max_outline_parts,
        svg_limit_kb=svg_limit_kb,
        use_face_priority=use_face_priority,
        remove_background=remove_background,
        protect_skin=protect_skin,
        protect_accents=protect_accents,
        line_darkness_threshold=line_darkness_threshold,
        outline_color=outline_color,
        edge_strength=edge_strength,
    )

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is not None:
    img_rgb = open_uploaded_image(uploaded)

    st.subheader("Original")
    st.image(img_rgb, use_container_width=True)

    if st.button("Generate SVGs", type="primary", use_container_width=True):
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

    tabs = st.tabs(["Previews", "SVG Files", "Stats", "Logs / Debug", "Download"])

    with tabs[0]:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Processed")
            st.image(result.processed_rgb, use_container_width=True)

            st.subheader("Foreground Mask")
            st.image(draw_mask_preview(result.foreground_mask), use_container_width=True)

            st.subheader("Priority Mask")
            st.image(draw_mask_preview(result.priority_mask), use_container_width=True)

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
            with st.expander(f"{filename} ({size_kb:.2f} KB)"):
                preview = svg_text[:4000] + ("\n\n... (truncated)" if len(svg_text) > 4000 else "")
                st.code(preview, language="xml")
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
        st.json(parts_to_debug_json(result.parts[:120]))

    with tabs[4]:
        st.subheader("Download bundle")
        zip_bytes = build_zip_bundle(result, last_cfg, source_base)
        zip_name = f"{source_base}_layered_svg_bundle_v3.zip"

        st.download_button(
            label="Download ZIP bundle",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            use_container_width=True
        )

        st.caption("ZIP includes: SVG files, previews, masks, and debug summary JSON.")
else:
    st.info("画像をアップロードして Generate SVGs を押してください。")