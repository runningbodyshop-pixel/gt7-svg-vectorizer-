import io
import os
import json
import zipfile
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
from collections import deque

import cv2
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(page_title="GT7 Layered SVG Vectorizer V2", layout="wide")


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
    max_points_priority: int
    morph_open: int
    morph_close: int
    canny_low: int
    canny_high: int
    outline_min_length: int
    max_fill_parts: int
    max_outline_parts: int
    svg_limit_kb: float
    outline_mode: str
    use_face_priority: bool
    remove_white_bg: bool
    protect_cyan: bool
    protect_skin: bool
    outline_color: str
    outline_width: float


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


PRESETS = {
    "Safe": AppConfig(
        mode_name="Safe",
        max_dim=1280,
        num_colors=10,
        min_area=90,
        min_area_priority=24,
        overlap_px=0,
        contour_epsilon=0.006,
        contour_epsilon_priority=0.0028,
        max_points_per_path=180,
        max_points_priority=320,
        morph_open=2,
        morph_close=3,
        canny_low=70,
        canny_high=160,
        outline_min_length=18,
        max_fill_parts=230,
        max_outline_parts=90,
        svg_limit_kb=14.5,
        outline_mode="stroke",
        use_face_priority=True,
        remove_white_bg=True,
        protect_cyan=True,
        protect_skin=True,
        outline_color="#202020",
        outline_width=1.15,
    ),
    "High Quality Safe": AppConfig(
        mode_name="High Quality Safe",
        max_dim=1600,
        num_colors=16,
        min_area=50,
        min_area_priority=12,
        overlap_px=0,
        contour_epsilon=0.0045,
        contour_epsilon_priority=0.0018,
        max_points_per_path=240,
        max_points_priority=420,
        morph_open=2,
        morph_close=3,
        canny_low=55,
        canny_high=145,
        outline_min_length=14,
        max_fill_parts=380,
        max_outline_parts=150,
        svg_limit_kb=14.5,
        outline_mode="stroke",
        use_face_priority=True,
        remove_white_bg=True,
        protect_cyan=True,
        protect_skin=True,
        outline_color="#1E1E1E",
        outline_width=1.10,
    ),
    "Detail Heavy": AppConfig(
        mode_name="Detail Heavy",
        max_dim=1900,
        num_colors=20,
        min_area=34,
        min_area_priority=8,
        overlap_px=0,
        contour_epsilon=0.0035,
        contour_epsilon_priority=0.0012,
        max_points_per_path=300,
        max_points_priority=560,
        morph_open=1,
        morph_close=3,
        canny_low=45,
        canny_high=130,
        outline_min_length=10,
        max_fill_parts=520,
        max_outline_parts=220,
        svg_limit_kb=14.5,
        outline_mode="stroke",
        use_face_priority=True,
        remove_white_bg=True,
        protect_cyan=True,
        protect_skin=True,
        outline_color="#181818",
        outline_width=1.00,
    ),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return f"#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}"


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    s = hex_color.strip().lstrip("#")
    if len(s) != 6:
        return (0, 0, 0)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def safe_filename_base(name: str) -> str:
    base = os.path.splitext(name)[0]
    chars = []
    for ch in base:
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "vectorized_image"


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
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return img_bgr, scale


def find_contours_compat(mask: np.ndarray, mode, method):
    result = cv2.findContours(mask, mode, method)
    if len(result) == 2:
        return result[0], result[1]
    return result[1], result[2]


def contour_to_points(contour: np.ndarray) -> List[Tuple[int, int]]:
    pts = contour.reshape(-1, 2)
    return [(int(x), int(y)) for x, y in pts]


def simplify_contour(contour: np.ndarray, eps_ratio: float, max_points: int, closed: bool) -> np.ndarray:
    peri = cv2.arcLength(contour, closed)
    eps = max(0.35, peri * eps_ratio)
    approx = cv2.approxPolyDP(contour, eps, closed)
    tries = 0
    while len(approx) > max_points and tries < 10:
        eps *= 1.25
        approx = cv2.approxPolyDP(contour, eps, closed)
        tries += 1
    return approx


def png_bytes_from_rgb(rgb: np.ndarray) -> bytes:
    img = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def preprocess_image(img_bgr: np.ndarray, cfg: AppConfig, logs: List[str]) -> np.ndarray:
    logs.append("Preprocess: resize + gentle denoise")
    img_bgr, scale = resize_to_max_dim(img_bgr, cfg.max_dim)
    if scale < 1.0:
        logs.append(f"Resized: max_dim={cfg.max_dim}, scale={scale:.3f}")

    img_bgr = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=28, sigmaSpace=28)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return img_bgr


def make_connected_white_background_mask(img_bgr: np.ndarray, logs: List[str]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    near_white = ((sat < 38) & (val > 226)).astype(np.uint8) * 255
    bg = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=np.uint8)
    q = deque()

    for x in range(w):
        if near_white[0, x]:
            q.append((x, 0))
        if near_white[h - 1, x]:
            q.append((x, h - 1))
    for y in range(h):
        if near_white[y, 0]:
            q.append((0, y))
        if near_white[y, w - 1]:
            q.append((w - 1, y))

    while q:
        x, y = q.popleft()
        if visited[y, x]:
            continue
        visited[y, x] = 1
        if near_white[y, x] == 0:
            continue
        bg[y, x] = 255
        if x > 0:
            q.append((x - 1, y))
        if x < w - 1:
            q.append((x + 1, y))
        if y > 0:
            q.append((x, y - 1))
        if y < h - 1:
            q.append((x, y + 1))

    bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    logs.append(f"Connected white background pixels: {int(np.count_nonzero(bg))}")
    return bg


def build_foreground_mask(img_bgr: np.ndarray, cfg: AppConfig, logs: List[str]) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    if not cfg.remove_white_bg:
        logs.append("Foreground mask: white background removal OFF")
        return np.full((h, w), 255, dtype=np.uint8)

    bg = make_connected_white_background_mask(img_bgr, logs)
    fg = cv2.bitwise_not(bg)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    fg = cv2.bitwise_or(fg, edges)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    saturated = ((sat > 45) & (val > 60)).astype(np.uint8) * 255
    fg = cv2.bitwise_or(fg, saturated)

    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = find_contours_compat(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(fg)
    min_keep = max(60, int(h * w * 0.00006))
    kept = 0
    for c in contours:
        if cv2.contourArea(c) >= min_keep:
            cv2.drawContours(clean, [c], -1, 255, -1)
            kept += 1

    clean = cv2.dilate(clean, np.ones((3, 3), np.uint8), iterations=1)
    logs.append(f"Foreground components kept: {kept}")
    return clean


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def build_priority_mask(img_bgr: np.ndarray, foreground_mask: np.ndarray, cfg: AppConfig, logs: List[str]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not cfg.use_face_priority:
        logs.append("Priority mask OFF")
        return mask

    bbox = bbox_from_mask(foreground_mask)
    if bbox is None:
        logs.append("Priority fallback: no foreground, using image center")
        cx, cy = w // 2, h // 2
        ax, ay = int(w * 0.28), int(h * 0.32)
    else:
        x1, y1, x2, y2 = bbox
        bw = x2 - x1 + 1
        bh = y2 - y1 + 1
        cx = int(x1 + bw * 0.52)
        cy = int(y1 + bh * 0.28)
        ax = max(40, int(bw * 0.24))
        ay = max(45, int(bh * 0.22))

    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 178, 130], dtype=np.uint8))
    skin = cv2.bitwise_and(skin, foreground_mask)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.bitwise_or(mask, skin)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    cyan = cv2.inRange(hsv, np.array([78, 45, 55], dtype=np.uint8), np.array([105, 255, 255], dtype=np.uint8))
    cyan = cv2.bitwise_and(cyan, foreground_mask)
    cyan = cv2.dilate(cyan, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.bitwise_or(mask, cyan)

    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(35, 35))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda r: (r[1], -r[2] * r[3]))
                x, y, fw, fh = faces[0]
                pad_x = int(fw * 0.35)
                pad_y = int(fh * 0.45)
                cv2.rectangle(
                    mask,
                    (clamp(x - pad_x, 0, w - 1), clamp(y - pad_y, 0, h - 1)),
                    (clamp(x + fw + pad_x, 0, w - 1), clamp(y + fh + pad_y, 0, h - 1)),
                    255,
                    -1,
                )
                logs.append("Priority: face detector added a face box")
            else:
                logs.append("Priority: bbox/skin/accent priority used")
    except Exception as e:
        logs.append(f"Priority: face detector skipped: {e}")

    mask = cv2.bitwise_and(mask, cv2.dilate(foreground_mask, np.ones((9, 9), np.uint8), iterations=1))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=8, sigmaY=8)
    _, mask = cv2.threshold(mask, 42, 255, cv2.THRESH_BINARY)

    logs.append(f"Priority pixels: {int(np.count_nonzero(mask))}")
    return mask


def weighted_quantize_foreground(
    img_bgr: np.ndarray,
    foreground_mask: np.ndarray,
    priority_mask: np.ndarray,
    cfg: AppConfig,
    logs: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    fg_idx = np.where(foreground_mask > 0)
    if len(fg_idx[0]) < 10:
        logs.append("Quantization fallback: foreground too small, using full image")
        fg_idx = np.where(np.full((h, w), 255, dtype=np.uint8) > 0)

    pixels = img_bgr[fg_idx].reshape(-1, 3)
    max_sample = 240_000
    if len(pixels) > max_sample:
        rng = np.random.default_rng(1234)
        sample_idx = rng.choice(len(pixels), size=max_sample, replace=False)
        sample = pixels[sample_idx]
    else:
        sample = pixels

    extra_samples = []

    pr_pixels = img_bgr[priority_mask > 0].reshape(-1, 3)
    if len(pr_pixels) > 0:
        take = min(len(pr_pixels), max(4000, len(sample) // 4))
        rng = np.random.default_rng(5678)
        ids = rng.choice(len(pr_pixels), size=take, replace=False)
        extra_samples.append(pr_pixels[ids])

    if cfg.protect_cyan:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        cyan_mask = cv2.inRange(hsv, np.array([78, 45, 55], dtype=np.uint8), np.array([105, 255, 255], dtype=np.uint8))
        cyan_mask = cv2.bitwise_and(cyan_mask, foreground_mask)
        cyan_pixels = img_bgr[cyan_mask > 0].reshape(-1, 3)
        if len(cyan_pixels) > 0:
            take = min(len(cyan_pixels), 10000)
            rng = np.random.default_rng(9101)
            ids = rng.choice(len(cyan_pixels), size=take, replace=False)
            extra_samples.append(cyan_pixels[ids])
            logs.append(f"Protected cyan sample pixels: {take}")

    if cfg.protect_skin:
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 178, 130], dtype=np.uint8))
        skin = cv2.bitwise_and(skin, foreground_mask)
        skin_pixels = img_bgr[skin > 0].reshape(-1, 3)
        if len(skin_pixels) > 0:
            take = min(len(skin_pixels), 12000)
            rng = np.random.default_rng(1112)
            ids = rng.choice(len(skin_pixels), size=take, replace=False)
            extra_samples.append(skin_pixels[ids])
            logs.append(f"Protected skin sample pixels: {take}")

    if extra_samples:
        sample = np.vstack([sample] + extra_samples)

    sample = sample.astype(np.float32)
    k = min(cfg.num_colors, max(2, len(sample)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.55)
    compactness, _, centers = cv2.kmeans(
        sample,
        K=k,
        bestLabels=None,
        criteria=criteria,
        attempts=5,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    centers = np.uint8(centers)
    fg_pixels = img_bgr[fg_idx].astype(np.int16)
    centers_i = centers.astype(np.int16)
    all_labels = np.full((h, w), -1, dtype=np.int16)

    chunk = 80_000
    assigned_blocks = []
    for start in range(0, len(fg_pixels), chunk):
        block = fg_pixels[start:start + chunk]
        dist = np.sum((block[:, None, :] - centers_i[None, :, :]) ** 2, axis=2)
        assigned_blocks.append(np.argmin(dist, axis=1).astype(np.int16))
    assigned = np.concatenate(assigned_blocks) if assigned_blocks else np.array([], dtype=np.int16)
    all_labels[fg_idx] = assigned

    quant = np.full_like(img_bgr, 255)
    quant[fg_idx] = centers[assigned]

    logs.append(f"Quantization: k={k}, compactness={compactness:.1f}, foreground_pixels={len(fg_pixels)}")
    return quant, all_labels, centers


def classify_color_category(color_rgb: Tuple[int, int, int]) -> str:
    r, g, b = color_rgb
    lum = luminance(color_rgb)
    mx = max(r, g, b)
    mn = min(r, g, b)
    sat = mx - mn

    if g > 120 and b > 120 and g > r + 25 and b > r + 25:
        return "accent"
    if r > 150 and g > 105 and b > 85 and r > b + 25 and g > b - 5:
        return "skin"
    if lum < 58:
        return "dark_detail"
    if lum > 214 and sat < 45:
        return "light_base"
    if lum < 105:
        return "shadow"
    return "base"


def category_layer_base(category: str) -> int:
    return {
        "underpaint": 50,
        "light_base": 120,
        "base": 180,
        "skin": 240,
        "shadow": 420,
        "dark_detail": 620,
        "accent": 760,
        "highlight": 820,
        "outline": 1000,
        "face_detail": 1100,
    }.get(category, 500)


def extract_color_parts(
    labels: np.ndarray,
    centers: np.ndarray,
    foreground_mask: np.ndarray,
    priority_mask: np.ndarray,
    cfg: AppConfig,
    logs: List[str]
) -> List[VectorPart]:
    parts: List[VectorPart] = []
    part_counter = 0

    color_counts = []
    for idx in range(len(centers)):
        count = int(np.count_nonzero(labels == idx))
        if count > 0:
            color_counts.append((idx, count))
    color_counts.sort(key=lambda x: x[1], reverse=True)

    logs.append("Extracting color vector parts")

    kernel_open_default = np.ones((max(1, cfg.morph_open), max(1, cfg.morph_open)), np.uint8)
    kernel_close_default = np.ones((max(1, cfg.morph_close), max(1, cfg.morph_close)), np.uint8)

    for color_idx, _ in color_counts:
        mask = ((labels == color_idx).astype(np.uint8) * 255)
        mask = cv2.bitwise_and(mask, foreground_mask)

        bgr = centers[color_idx]
        color_rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        color_hex = rgb_to_hex(color_rgb)
        category = classify_color_category(color_rgb)

        if category in ("dark_detail", "accent"):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        else:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open_default)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close_default)

        if cfg.overlap_px > 0 and category not in ("dark_detail", "accent"):
            k = np.ones((cfg.overlap_px + 1, cfg.overlap_px + 1), np.uint8)
            mask = cv2.dilate(mask, k, iterations=1)

        contours, _ = find_contours_compat(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            roi = priority_mask[y:y + bh, x:x + bw]
            is_priority = bool(np.mean(roi) > 8)

            min_area = cfg.min_area_priority if is_priority else cfg.min_area
            if category in ("accent", "dark_detail"):
                min_area = max(4, min_area // 2)
            if area < min_area:
                continue

            max_points = cfg.max_points_priority if is_priority else cfg.max_points_per_path
            eps = cfg.contour_epsilon_priority if is_priority else cfg.contour_epsilon
            if category in ("accent", "dark_detail") and is_priority:
                eps *= 0.75
                max_points = int(max_points * 1.2)

            approx = simplify_contour(contour, eps, max_points, closed=True)
            if len(approx) < 3:
                continue

            pts = contour_to_points(approx)
            layer = category_layer_base(category) + part_counter

            parts.append(VectorPart(
                part_id=f"{category}_{part_counter:05d}",
                category=category,
                layer=layer,
                fill=color_hex,
                stroke=None,
                stroke_width=0.0,
                closed=True,
                points=pts,
                bbox=(int(x), int(y), int(x + bw), int(y + bh)),
                area=area,
                priority=is_priority,
            ))
            part_counter += 1

    parts.sort(key=lambda p: (category_layer_base(p.category), -int(p.priority), -p.area))

    if len(parts) > cfg.max_fill_parts:
        keep = sorted(
            parts,
            key=lambda p: (int(p.priority), p.category in ("accent", "skin", "dark_detail"), p.area),
            reverse=True,
        )[:cfg.max_fill_parts]
        keep_ids = {p.part_id for p in keep}
        parts = [p for p in parts if p.part_id in keep_ids]
        logs.append(f"Color parts capped to {cfg.max_fill_parts}")

    category_offsets = {}
    for p in sorted(parts, key=lambda q: (category_layer_base(q.category), -q.area)):
        n = category_offsets.get(p.category, 0)
        p.layer = category_layer_base(p.category) * 1000 + n
        category_offsets[p.category] = n + 1

    logs.append(f"Color parts extracted: {len(parts)}")
    return parts


def extract_outline_stroke_parts(
    img_bgr: np.ndarray,
    foreground_mask: np.ndarray,
    priority_mask: np.ndarray,
    cfg: AppConfig,
    logs: List[str]
) -> List[VectorPart]:
    if cfg.outline_mode == "off":
        logs.append("Outline extraction OFF")
        return []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    edges = cv2.bitwise_and(edges, cv2.dilate(foreground_mask, np.ones((5, 5), np.uint8), iterations=1))

    dark = cv2.inRange(gray, 0, 90)
    dark = cv2.bitwise_and(dark, foreground_mask)
    dark_edges = cv2.bitwise_and(edges, cv2.dilate(dark, np.ones((3, 3), np.uint8), iterations=1))
    edges = cv2.bitwise_or(edges, dark_edges)

    contours, _ = find_contours_compat(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    parts: List[VectorPart] = []
    part_counter = 0

    for contour in contours:
        length = cv2.arcLength(contour, False)
        if length < cfg.outline_min_length:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        roi = priority_mask[y:y + bh, x:x + bw]
        is_priority = bool(np.mean(roi) > 8)

        eps = cfg.contour_epsilon_priority * 0.9 if is_priority else cfg.contour_epsilon * 0.85
        max_points = cfg.max_points_priority if is_priority else cfg.max_points_per_path
        approx = simplify_contour(contour, eps, max_points, closed=False)
        if len(approx) < 2:
            continue

        pts = contour_to_points(approx)
        x0, y0 = pts[0]
        x1, y1 = pts[-1]
        closed_like = (abs(x0 - x1) + abs(y0 - y1)) <= 2

        stroke_w = cfg.outline_width
        if is_priority:
            stroke_w = max(0.85, cfg.outline_width * 0.95)

        parts.append(VectorPart(
            part_id=f"outline_{part_counter:05d}",
            category="outline",
            layer=category_layer_base("outline") * 1000 + part_counter,
            fill=None,
            stroke=cfg.outline_color,
            stroke_width=stroke_w,
            closed=closed_like,
            points=pts,
            bbox=(int(x), int(y), int(x + bw), int(y + bh)),
            area=float(length),
            priority=is_priority,
        ))
        part_counter += 1

    parts.sort(key=lambda p: (int(p.priority), p.area), reverse=True)
    if len(parts) > cfg.max_outline_parts:
        parts = parts[:cfg.max_outline_parts]
        logs.append(f"Outline stroke parts capped to {cfg.max_outline_parts}")

    for i, p in enumerate(parts):
        p.layer = category_layer_base("outline") * 1000 + i

    logs.append(f"Outline stroke parts extracted: {len(parts)}")
    return parts


def render_vector_preview(shape_hw: Tuple[int, int], parts: List[VectorPart]) -> np.ndarray:
    h, w = shape_hw
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    for part in sorted(parts, key=lambda p: p.layer):
        if len(part.points) < 2:
            continue
        pts = np.array(part.points, dtype=np.int32).reshape((-1, 1, 2))

        if part.fill and len(part.points) >= 3:
            cv2.fillPoly(canvas, [pts], hex_to_bgr(part.fill), lineType=cv2.LINE_AA)

        if part.stroke:
            cv2.polylines(
                canvas,
                [pts],
                isClosed=part.closed,
                color=hex_to_bgr(part.stroke),
                thickness=max(1, int(round(part.stroke_width))),
                lineType=cv2.LINE_AA,
            )

    return bgr_to_rgb(canvas)


def points_to_path(points: List[Tuple[int, int]], closed: bool) -> str:
    if not points:
        return ""
    chunks = [f"M{points[0][0]} {points[0][1]}"]
    for x, y in points[1:]:
        chunks.append(f"L{x} {y}")
    if closed:
        chunks.append("Z")
    return " ".join(chunks)


def svg_escape_text(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def part_to_svg_element(part: VectorPart) -> str:
    d = points_to_path(part.points, part.closed)
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
        attrs.append('vector-effect="non-scaling-stroke"')

    attrs.append(f'data-category="{part.category}"')
    return "<path " + " ".join(attrs) + "/>"


def build_svg_document(parts: List[VectorPart], width: int, height: int, title: str) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{svg_escape_text(title)}</title>",
    ]

    for part in sorted(parts, key=lambda p: p.layer):
        el = part_to_svg_element(part)
        if el:
            lines.append(el)

    lines.append("</svg>")
    return "\n".join(lines)


def split_parts_by_svg_size(parts: List[VectorPart], width: int, height: int, limit_bytes: int, prefix: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    if not parts:
        return files

    ordered = sorted(parts, key=lambda p: p.layer)
    chunks: List[List[VectorPart]] = []
    current: List[VectorPart] = []

    for part in ordered:
        candidate = current + [part]
        size = len(build_svg_document(candidate, width, height, f"{prefix}_chunk").encode("utf-8"))

        if current and size > limit_bytes:
            chunks.append(current)
            current = [part]
        else:
            current = candidate

    if current:
        chunks.append(current)

    for i, chunk in enumerate(chunks, start=1):
        name = f"{prefix}_{i:02d}.svg"
        files[name] = build_svg_document(chunk, width, height, name)

    return files


def generate_svg_files(parts: List[VectorPart], shape_hw: Tuple[int, int], cfg: AppConfig, logs: List[str]) -> Dict[str, str]:
    h, w = shape_hw
    files: Dict[str, str] = {}

    files["00_full_combined.svg"] = build_svg_document(parts, w, h, "full_combined")

    categories = [
        ("01_bases.svg", {"light_base", "base", "skin"}),
        ("02_shadows.svg", {"shadow", "dark_detail"}),
        ("03_accents.svg", {"accent", "highlight"}),
        ("04_outlines.svg", {"outline"}),
    ]

    for filename, cats in categories:
        selected = [p for p in parts if p.category in cats]
        if selected:
            files[filename] = build_svg_document(selected, w, h, filename)

    limit_bytes = int(cfg.svg_limit_kb * 1024)
    split_files = split_parts_by_svg_size(parts, w, h, limit_bytes, "gt7_part")
    files.update(split_files)

    logs.append(f"SVG files generated: {len(files)}")
    return files


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
        zf.writestr("preview_foreground_mask.png", png_bytes_from_rgb(np.stack([result.foreground_mask] * 3, axis=-1)))
        zf.writestr("preview_priority_mask.png", png_bytes_from_rgb(np.stack([result.priority_mask] * 3, axis=-1)))
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

    return buf.getvalue()


def process_image_pipeline(img_rgb: np.ndarray, cfg: AppConfig) -> ProcessResult:
    logs: List[str] = []
    original_rgb = img_rgb.copy()

    img_bgr = rgb_to_bgr(img_rgb)
    processed_bgr = preprocess_image(img_bgr, cfg, logs)
    foreground_mask = build_foreground_mask(processed_bgr, cfg, logs)
    priority_mask = build_priority_mask(processed_bgr, foreground_mask, cfg, logs)

    quant_bgr, labels, centers = weighted_quantize_foreground(processed_bgr, foreground_mask, priority_mask, cfg, logs)

    color_parts = extract_color_parts(labels, centers, foreground_mask, priority_mask, cfg, logs)
    outline_parts = extract_outline_stroke_parts(processed_bgr, foreground_mask, priority_mask, cfg, logs)

    parts = color_parts + outline_parts
    parts.sort(key=lambda p: p.layer)

    vector_preview = render_vector_preview(processed_bgr.shape[:2], parts)
    svg_files = generate_svg_files(parts, processed_bgr.shape[:2], cfg, logs)

    stats = {
        "image_width": int(processed_bgr.shape[1]),
        "image_height": int(processed_bgr.shape[0]),
        "foreground_pixels": int(np.count_nonzero(foreground_mask)),
        "priority_pixels": int(np.count_nonzero(priority_mask)),
        "color_parts": len(color_parts),
        "outline_parts": len(outline_parts),
        "total_parts": len(parts),
        "total_points": int(sum(len(p.points) for p in parts)),
        "svg_file_count": len(svg_files),
        "svg_sizes_bytes": {k: len(v.encode("utf-8")) for k, v in svg_files.items()},
        "category_counts": {cat: int(sum(1 for p in parts if p.category == cat)) for cat in sorted(set(p.category for p in parts))},
    }

    return ProcessResult(
        original_rgb=original_rgb,
        processed_rgb=bgr_to_rgb(processed_bgr),
        foreground_mask=foreground_mask,
        priority_mask=priority_mask,
        quantized_rgb=bgr_to_rgb(quant_bgr),
        vector_preview_rgb=vector_preview,
        parts=parts,
        svg_files=svg_files,
        stats=stats,
        logs=logs,
    )


st.title("GT7 Layered SVG Vectorizer V2")
st.caption("白背景除去・顔優先・保護色・stroke線画・GT7分割対応")

with st.sidebar:
    st.header("Settings")

    mode_name = st.selectbox("Mode", list(PRESETS.keys()), index=1)
    base = PRESETS[mode_name]

    num_colors = st.slider("Number of colors", 6, 28, base.num_colors, 1)
    outline_mode = st.selectbox("Outline mode", ["stroke", "off"], index=0 if base.outline_mode == "stroke" else 1)
    remove_white_bg = st.checkbox("Remove connected white background", value=base.remove_white_bg)
    use_face_priority = st.checkbox("Face / character priority", value=base.use_face_priority)
    protect_cyan = st.checkbox("Protect cyan / blue accents", value=base.protect_cyan)
    protect_skin = st.checkbox("Protect skin color", value=base.protect_skin)
    svg_limit_kb = st.slider("GT7 target part size (KB)", 8.0, 15.0, float(base.svg_limit_kb), 0.5)

    with st.expander("Advanced"):
        max_dim = st.slider("Max processing dimension", 900, 2300, base.max_dim, 100)
        min_area = st.slider("Min region area", 8, 180, base.min_area, 2)
        min_area_priority = st.slider("Min region area in priority", 4, 60, base.min_area_priority, 1)
        max_fill_parts = st.slider("Max color parts", 80, 700, base.max_fill_parts, 10)
        max_outline_parts = st.slider("Max outline strokes", 0, 350, base.max_outline_parts, 10)
        contour_eps = st.slider("Contour simplify", 0.0015, 0.015, float(base.contour_epsilon), 0.0005)
        contour_eps_priority = st.slider("Contour simplify priority", 0.0008, 0.008, float(base.contour_epsilon_priority), 0.0002)
        overlap_px = st.slider("Overlap / underpaint", 0, 3, base.overlap_px, 1)
        outline_color = st.color_picker("Outline color", base.outline_color)
        outline_width = st.slider("Outline stroke width", 0.5, 2.5, float(base.outline_width), 0.05)

    cfg = AppConfig(
        mode_name=mode_name,
        max_dim=max_dim,
        num_colors=num_colors,
        min_area=min_area,
        min_area_priority=min_area_priority,
        overlap_px=overlap_px,
        contour_epsilon=contour_eps,
        contour_epsilon_priority=contour_eps_priority,
        max_points_per_path=base.max_points_per_path,
        max_points_priority=base.max_points_priority,
        morph_open=base.morph_open,
        morph_close=base.morph_close,
        canny_low=base.canny_low,
        canny_high=base.canny_high,
        outline_min_length=base.outline_min_length,
        max_fill_parts=max_fill_parts,
        max_outline_parts=max_outline_parts,
        svg_limit_kb=svg_limit_kb,
        outline_mode=outline_mode,
        use_face_priority=use_face_priority,
        remove_white_bg=remove_white_bg,
        protect_cyan=protect_cyan,
        protect_skin=protect_skin,
        outline_color=outline_color,
        outline_width=outline_width,
    )

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is not None:
    img_rgb = open_uploaded_image(uploaded)

    st.subheader("Original")
    st.image(img_rgb, use_container_width=True)

    if st.button("Generate SVGs", type="primary", use_container_width=True):
        try:
            with st.spinner("Processing..."):
                result = process_image_pipeline(img_rgb, cfg)
            st.session_state["last_result"] = result
            st.session_state["last_cfg"] = cfg
            st.session_state["source_name"] = uploaded.name
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.exception(e)
            st.stop()

if "last_result" in st.session_state:
    result: ProcessResult = st.session_state["last_result"]
    last_cfg: AppConfig = st.session_state["last_cfg"]
    source_name = st.session_state.get("source_name", "image.png")
    source_base = safe_filename_base(source_name)

    tabs = st.tabs(["Previews", "SVG Files", "Stats", "Logs / Debug", "Download"])

    with tabs[0]:
        st.subheader("Processed")
        st.image(result.processed_rgb, use_container_width=True)

        st.subheader("Foreground Mask")
        st.image(np.stack([result.foreground_mask] * 3, axis=-1), use_container_width=True)

        st.subheader("Priority Mask")
        st.image(np.stack([result.priority_mask] * 3, axis=-1), use_container_width=True)

        st.subheader("Quantized")
        st.image(result.quantized_rgb, use_container_width=True)

        st.subheader("Vector Preview")
        st.image(result.vector_preview_rgb, use_container_width=True)

    with tabs[1]:
        st.subheader("Generated SVG files")
        for filename, svg_text in result.svg_files.items():
            size_kb = len(svg_text.encode("utf-8")) / 1024
            with st.expander(f"{filename}  ({size_kb:.2f} KB)"):
                shown = svg_text[:5000]
                if len(svg_text) > 5000:
                    shown += "\n\n... truncated ..."
                st.code(shown, language="xml")
                st.download_button(
                    label=f"Download {filename}",
                    data=svg_text.encode("utf-8"),
                    file_name=filename,
                    mime="image/svg+xml",
                    key=f"download_{filename}",
                    use_container_width=True,
                )

    with tabs[2]:
        st.subheader("Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Color parts", result.stats["color_parts"])
        c2.metric("Outline parts", result.stats["outline_parts"])
        c3.metric("Total parts", result.stats["total_parts"])
        c4.metric("Total points", result.stats["total_points"])
        st.json(result.stats)

    with tabs[3]:
        st.subheader("Logs")
        for line in result.logs:
            st.write("-", line)

        st.subheader("Part Debug")
        st.json(parts_to_debug_json(result.parts)[:120])

    with tabs[4]:
        st.subheader("Download ZIP")
        zip_bytes = build_zip_bundle(result, last_cfg, source_base)
        st.download_button(
            label="Download ZIP bundle",
            data=zip_bytes,
            file_name=f"{source_base}_gt7_vectorizer_v2.zip",
            mime="application/zip",
            use_container_width=True,
        )
else:
    st.info("画像をアップロードして Generate SVGs を押してください。")
