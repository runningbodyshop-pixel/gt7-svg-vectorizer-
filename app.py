import base64
import html
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

APP_TITLE = "イラスト SVG ベクター化ツール"
DEFAULT_BG = "#ffffff"


# ============================================================
# Data model
# ============================================================

@dataclass
class Shape:
    id: int
    d: str
    fill: str
    area: float
    bbox: Tuple[int, int, int, int]
    role: str = "base"  # base / detail / shadow / highlight / line_fill / line_stroke
    gradient: Optional[Dict] = None
    stroke: Optional[str] = None
    stroke_width: float = 0.0
    opacity: float = 1.0


# ============================================================
# Basic helpers
# ============================================================

def clamp_int(v: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(v))))


def clamp_float(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    value = hex_color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        return (255, 255, 255)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Iterable[float]) -> str:
    r, g, b = [clamp_int(x, 0, 255) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def color_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return math.sqrt(sum((int(x) - int(y)) ** 2 for x, y in zip(a, b)))


def fmt_num(v: float, decimals: int = 2) -> str:
    if abs(v - round(v)) < 0.05:
        return str(int(round(v)))
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".")


# ============================================================
# Image prep
# ============================================================

def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def flatten_alpha(img: Image.Image, bg_hex: str) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGBA")
    bg_rgb = hex_to_rgb(bg_hex)
    bg = Image.new("RGBA", img.size, bg_rgb + (255,))
    return Image.alpha_composite(bg, img).convert("RGB")


def preprocess_rgb(rgb: np.ndarray, smooth: str) -> np.ndarray:
    if smooth == "なし":
        return rgb
    if smooth == "軽く":
        return cv2.bilateralFilter(rgb, d=5, sigmaColor=30, sigmaSpace=30)
    if smooth == "中":
        out = cv2.bilateralFilter(rgb, d=7, sigmaColor=45, sigmaSpace=45)
        return cv2.medianBlur(out, 3)
    if smooth == "強め":
        out = cv2.bilateralFilter(rgb, d=9, sigmaColor=60, sigmaSpace=60)
        return cv2.medianBlur(out, 5)
    return rgb


def quantize_rgb(rgb: np.ndarray, color_count: int) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    pil = Image.fromarray(rgb, "RGB")
    try:
        q = pil.quantize(colors=color_count, method=Image.Quantize.MEDIANCUT)
    except Exception:
        q = pil.quantize(colors=color_count)
    index_map = np.array(q, dtype=np.uint8)
    raw_palette = q.getpalette() or []
    palette: List[Tuple[int, int, int]] = []
    for i in range(color_count):
        j = i * 3
        if j + 2 < len(raw_palette):
            palette.append((raw_palette[j], raw_palette[j + 1], raw_palette[j + 2]))
    used = sorted(int(x) for x in np.unique(index_map))
    remap = {old: new for new, old in enumerate(used)}
    new_palette = [palette[i] if i < len(palette) else (0, 0, 0) for i in used]
    remapped = np.zeros_like(index_map, dtype=np.uint8)
    for old, new in remap.items():
        remapped[index_map == old] = new
    return remapped, new_palette


def rebuild_rgb_from_index(index_map: np.ndarray, palette: List[Tuple[int, int, int]]) -> np.ndarray:
    h, w = index_map.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, rgb in enumerate(palette):
        out[index_map == idx] = rgb
    return out


def clean_mask(mask: np.ndarray, clean_px: int) -> np.ndarray:
    if clean_px <= 1:
        return mask
    k = np.ones((clean_px, clean_px), np.uint8)
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)
    return out


# ============================================================
# Geometry helpers
# ============================================================

def contour_to_path(contour: np.ndarray, epsilon_ratio: float, reverse: bool = False) -> str:
    if contour is None or len(contour) < 3:
        return ""
    arc = cv2.arcLength(contour, True)
    epsilon = max(0.25, arc * epsilon_ratio)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        return ""
    if reverse:
        pts = pts[::-1]
    commands = [f"M{fmt_num(pts[0][0])} {fmt_num(pts[0][1])}"]
    last_x, last_y = pts[0]
    for x, y in pts[1:]:
        if int(x) == int(last_x) and int(y) == int(last_y):
            continue
        commands.append(f"L{fmt_num(x)} {fmt_num(y)}")
        last_x, last_y = x, y
    commands.append("Z")
    return "".join(commands)


def contour_to_polyline_path(contour: np.ndarray, epsilon_ratio: float) -> str:
    if contour is None or len(contour) < 2:
        return ""
    arc = cv2.arcLength(contour, False)
    epsilon = max(0.25, arc * epsilon_ratio)
    approx = cv2.approxPolyDP(contour, epsilon, False)
    pts = approx.reshape(-1, 2)
    if len(pts) < 2:
        return ""
    commands = [f"M{fmt_num(pts[0][0])} {fmt_num(pts[0][1])}"]
    last_x, last_y = pts[0]
    for x, y in pts[1:]:
        if int(x) == int(last_x) and int(y) == int(last_y):
            continue
        commands.append(f"L{fmt_num(x)} {fmt_num(y)}")
        last_x, last_y = x, y
    return "".join(commands)


def contour_complexity(contour: np.ndarray) -> float:
    area = float(abs(cv2.contourArea(contour)))
    if area <= 0:
        return 9999.0
    peri = float(cv2.arcLength(contour, True))
    return (peri * peri) / max(1.0, 4.0 * math.pi * area)


def bbox_fill_ratio(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0
    crop = mask[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0
    return float(np.count_nonzero(crop)) / float(crop.size)


def build_component_mask(shape: Tuple[int, int], contours: List[np.ndarray], outer_idx: int, hierarchy: np.ndarray) -> np.ndarray:
    comp = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(comp, contours, outer_idx, 255, thickness=-1)
    child = hierarchy[outer_idx][2]
    while child != -1:
        cv2.drawContours(comp, contours, child, 0, thickness=-1)
        child = hierarchy[child][0]
    return comp


def mean_color_under_mask(rgb: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    if np.count_nonzero(mask) == 0:
        return (255, 255, 255)
    ys, xs = np.where(mask > 0)
    vals = rgb[ys, xs]
    m = vals.mean(axis=0)
    return (int(round(m[0])), int(round(m[1])), int(round(m[2])))


# ============================================================
# Shape extraction
# ============================================================

def extract_base_shapes(
    index_map: np.ndarray,
    palette: List[Tuple[int, int, int]],
    epsilon_ratio: float,
    min_area: int,
    clean_px: int,
    complexity_limit: float,
) -> Tuple[List[Shape], str]:
    h, w = index_map.shape[:2]
    image_area = float(w * h)
    counts = np.bincount(index_map.flatten(), minlength=len(palette))
    dominant_idx = int(np.argmax(counts)) if len(counts) else 0
    bg_hex = rgb_to_hex(palette[dominant_idx]) if palette else "#ffffff"

    shapes: List[Shape] = []
    next_id = 1
    for idx, rgb in enumerate(palette):
        mask = (index_map == idx).astype(np.uint8) * 255
        mask = clean_mask(mask, clean_px)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        for i, hinfo in enumerate(hierarchy):
            if hinfo[3] != -1:
                continue
            area = float(abs(cv2.contourArea(contours[i])))
            if area < min_area:
                continue
            if idx == dominant_idx and area > image_area * 0.88:
                continue
            cx = contour_complexity(contours[i])
            if cx > complexity_limit:
                continue
            d_parts = [contour_to_path(contours[i], epsilon_ratio, reverse=False)]
            child = hinfo[2]
            while child != -1:
                child_area = float(abs(cv2.contourArea(contours[child])))
                if child_area >= max(4, min_area * 0.25):
                    child_path = contour_to_path(contours[child], epsilon_ratio, reverse=True)
                    if child_path:
                        d_parts.append(child_path)
                child = hierarchy[child][0]
            d = "".join(p for p in d_parts if p)
            if not d:
                continue
            x, y, bw, bh = cv2.boundingRect(contours[i])
            shapes.append(
                Shape(
                    id=next_id,
                    d=d,
                    fill=rgb_to_hex(rgb),
                    area=area,
                    bbox=(int(x), int(y), int(bw), int(bh)),
                    role="base",
                )
            )
            next_id += 1
    shapes.sort(key=lambda s: (s.area, -luminance(hex_to_rgb(s.fill))), reverse=True)
    for i, s in enumerate(shapes, start=1):
        s.id = i
    return shapes, bg_hex


def extract_detail_shapes_filtered(
    original_rgb: np.ndarray,
    base_rgb: np.ndarray,
    detail_index: np.ndarray,
    detail_palette: List[Tuple[int, int, int]],
    epsilon_ratio: float,
    min_area: int,
    clean_px: int,
    max_area_ratio: float,
    diff_threshold: float,
    complexity_limit: float,
    min_fill_ratio: float,
    start_id: int,
) -> List[Shape]:
    h, w = detail_index.shape[:2]
    image_area = float(w * h)
    shapes: List[Shape] = []
    sid = start_id

    counts = np.bincount(detail_index.flatten(), minlength=len(detail_palette))
    dominant_idx = int(np.argmax(counts)) if len(counts) else 0

    for idx, pal_rgb in enumerate(detail_palette):
        mask = (detail_index == idx).astype(np.uint8) * 255
        mask = clean_mask(mask, clean_px)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        for i, hinfo in enumerate(hierarchy):
            if hinfo[3] != -1:
                continue
            area = float(abs(cv2.contourArea(contours[i])))
            if area < min_area:
                continue
            if idx == dominant_idx and area > image_area * 0.45:
                continue
            if area > image_area * max_area_ratio:
                continue
            cx = contour_complexity(contours[i])
            if cx > complexity_limit:
                continue
            x, y, bw, bh = cv2.boundingRect(contours[i])
            comp_mask = build_component_mask(mask.shape, contours, i, hierarchy)
            fill_ratio = bbox_fill_ratio(comp_mask, (x, y, bw, bh))
            if fill_ratio < min_fill_ratio:
                continue
            orig_mean = mean_color_under_mask(original_rgb, comp_mask)
            base_mean = mean_color_under_mask(base_rgb, comp_mask)
            if color_distance(orig_mean, base_mean) < diff_threshold:
                continue
            fill_rgb = orig_mean if color_distance(orig_mean, pal_rgb) > 8 else pal_rgb
            d_parts = [contour_to_path(contours[i], epsilon_ratio, reverse=False)]
            child = hinfo[2]
            while child != -1:
                child_area = float(abs(cv2.contourArea(contours[child])))
                if child_area >= max(4, min_area * 0.25):
                    child_path = contour_to_path(contours[child], epsilon_ratio, reverse=True)
                    if child_path:
                        d_parts.append(child_path)
                child = hierarchy[child][0]
            d = "".join(p for p in d_parts if p)
            if not d:
                continue
            shapes.append(
                Shape(
                    id=sid,
                    d=d,
                    fill=rgb_to_hex(fill_rgb),
                    area=area,
                    bbox=(int(x), int(y), int(bw), int(bh)),
                    role="detail",
                )
            )
            sid += 1

    shapes.sort(key=lambda s: s.area, reverse=True)
    return shapes


def extract_residual_tone_shapes(
    original_rgb: np.ndarray,
    base_rgb: np.ndarray,
    epsilon_ratio: float,
    min_area: int,
    clean_px: int,
    dark_threshold: int,
    light_threshold: int,
    shadow_opacity: float,
    highlight_opacity: float,
    complexity_limit: float,
    blur_ksize: int,
    start_id: int,
) -> Tuple[List[Shape], List[Shape]]:
    diff = original_rgb.astype(np.int16) - base_rgb.astype(np.int16)
    lum_diff = (0.299 * diff[:, :, 0] + 0.587 * diff[:, :, 1] + 0.114 * diff[:, :, 2]).astype(np.float32)
    if blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        lum_diff = cv2.GaussianBlur(lum_diff, (blur_ksize, blur_ksize), 0)

    shadow_mask = (lum_diff < -float(dark_threshold)).astype(np.uint8) * 255
    highlight_mask = (lum_diff > float(light_threshold)).astype(np.uint8) * 255
    shadow_mask = clean_mask(shadow_mask, clean_px)
    highlight_mask = clean_mask(highlight_mask, clean_px)

    shadows: List[Shape] = []
    highlights: List[Shape] = []
    sid = start_id

    for role_name, mask, opacity, target in [
        ("shadow", shadow_mask, shadow_opacity, shadows),
        ("highlight", highlight_mask, highlight_opacity, highlights),
    ]:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        for i, hinfo in enumerate(hierarchy):
            if hinfo[3] != -1:
                continue
            area = float(abs(cv2.contourArea(contours[i])))
            if area < min_area:
                continue
            cx = contour_complexity(contours[i])
            if cx > complexity_limit:
                continue
            x, y, bw, bh = cv2.boundingRect(contours[i])
            comp_mask = build_component_mask(mask.shape, contours, i, hierarchy)
            mean_rgb = mean_color_under_mask(original_rgb, comp_mask)
            if role_name == "shadow":
                fill_rgb = tuple(clamp_int(v * 0.70, 0, 255) for v in mean_rgb)
            else:
                fill_rgb = tuple(clamp_int(v + (255 - v) * 0.45, 0, 255) for v in mean_rgb)
            d_parts = [contour_to_path(contours[i], epsilon_ratio, reverse=False)]
            child = hinfo[2]
            while child != -1:
                child_area = float(abs(cv2.contourArea(contours[child])))
                if child_area >= max(4, min_area * 0.25):
                    child_path = contour_to_path(contours[child], epsilon_ratio, reverse=True)
                    if child_path:
                        d_parts.append(child_path)
                child = hierarchy[child][0]
            d = "".join(p for p in d_parts if p)
            if not d:
                continue
            target.append(
                Shape(
                    id=sid,
                    d=d,
                    fill=rgb_to_hex(fill_rgb),
                    area=area,
                    bbox=(int(x), int(y), int(bw), int(bh)),
                    role=role_name,
                    opacity=opacity,
                )
            )
            sid += 1

    shadows.sort(key=lambda s: s.area, reverse=True)
    highlights.sort(key=lambda s: s.area, reverse=True)
    return shadows, highlights


def extract_dark_fill_line_shapes(
    original_rgb: np.ndarray,
    dark_threshold: int,
    epsilon_ratio: float,
    min_area: int,
    clean_px: int,
    start_id: int,
) -> List[Shape]:
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray < dark_threshold).astype(np.uint8) * 255
    mask = clean_mask(mask, clean_px)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    shapes: List[Shape] = []
    sid = start_id
    for i, hinfo in enumerate(hierarchy):
        if hinfo[3] != -1:
            continue
        area = float(abs(cv2.contourArea(contours[i])))
        if area < min_area:
            continue
        d_parts = [contour_to_path(contours[i], epsilon_ratio, reverse=False)]
        child = hinfo[2]
        while child != -1:
            child_area = float(abs(cv2.contourArea(contours[child])))
            if child_area >= max(4, min_area * 0.25):
                child_path = contour_to_path(contours[child], epsilon_ratio, reverse=True)
                if child_path:
                    d_parts.append(child_path)
            child = hierarchy[child][0]
        d = "".join(p for p in d_parts if p)
        if not d:
            continue
        x, y, bw, bh = cv2.boundingRect(contours[i])
        shapes.append(
            Shape(
                id=sid,
                d=d,
                fill="#111111",
                area=area,
                bbox=(int(x), int(y), int(bw), int(bh)),
                role="line_fill",
            )
        )
        sid += 1
    shapes.sort(key=lambda s: s.area, reverse=True)
    return shapes


def extract_edge_line_shapes(
    original_rgb: np.ndarray,
    canny_low: int,
    canny_high: int,
    epsilon_ratio: float,
    min_length: int,
    max_lines: int,
    stroke_color: str,
    stroke_width: float,
    start_id: int,
) -> List[Shape]:
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    h, w = gray.shape[:2]
    image_area = float(h * w)
    items: List[Shape] = []
    sid = start_id
    contours_sorted = sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
    for cnt in contours_sorted:
        arc = float(cv2.arcLength(cnt, False))
        if arc < min_length:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh > image_area * 0.80:
            continue
        path = contour_to_polyline_path(cnt, epsilon_ratio)
        if not path:
            continue
        items.append(
            Shape(
                id=sid,
                d=path,
                fill="none",
                area=arc,
                bbox=(int(x), int(y), int(bw), int(bh)),
                role="line_stroke",
                stroke=stroke_color,
                stroke_width=stroke_width,
            )
        )
        sid += 1
        if len(items) >= max_lines:
            break
    return items


def build_label_boundary_mask(index_map: np.ndarray) -> np.ndarray:
    h, w = index_map.shape[:2]
    b = np.zeros((h, w), dtype=np.uint8)
    # 4-neighborhood + diagonal to better follow slanted anime linework
    b[:, 1:] |= (index_map[:, 1:] != index_map[:, :-1]).astype(np.uint8) * 255
    b[1:, :] |= (index_map[1:, :] != index_map[:-1, :]).astype(np.uint8) * 255
    b[1:, 1:] |= (index_map[1:, 1:] != index_map[:-1, :-1]).astype(np.uint8) * 255
    b[1:, :-1] |= (index_map[1:, :-1] != index_map[:-1, 1:]).astype(np.uint8) * 255
    return b


def extract_boundary_line_shapes(
    original_rgb: np.ndarray,
    base_index: np.ndarray,
    detail_index: np.ndarray,
    dark_threshold: int,
    boundary_dilate_px: int,
    dark_expand_px: int,
    epsilon_ratio: float,
    min_length: int,
    max_lines: int,
    stroke_color: str,
    stroke_width: float,
    start_id: int,
) -> List[Shape]:
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    base_b = build_label_boundary_mask(base_index)
    detail_b = build_label_boundary_mask(detail_index)
    boundary = cv2.bitwise_or(base_b, detail_b)

    if boundary_dilate_px > 1:
        k = np.ones((boundary_dilate_px, boundary_dilate_px), np.uint8)
        boundary = cv2.dilate(boundary, k, iterations=1)

    dark_mask = (gray < dark_threshold).astype(np.uint8) * 255
    if dark_expand_px > 1:
        k2 = np.ones((dark_expand_px, dark_expand_px), np.uint8)
        dark_mask = cv2.dilate(dark_mask, k2, iterations=1)

    # Keep boundaries that actually correspond to dark ink or dark region transitions
    line_mask = cv2.bitwise_and(boundary, dark_mask)
    # Recover a little continuity without drifting far away from the original line
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    h, w = gray.shape[:2]
    image_area = float(h * w)
    items: List[Shape] = []
    sid = start_id
    contours_sorted = sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
    for cnt in contours_sorted:
        arc = float(cv2.arcLength(cnt, False))
        if arc < min_length:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh > image_area * 0.85:
            continue
        path = contour_to_polyline_path(cnt, epsilon_ratio)
        if not path:
            continue
        items.append(
            Shape(
                id=sid,
                d=path,
                fill='none',
                area=arc,
                bbox=(int(x), int(y), int(bw), int(bh)),
                role='line_stroke',
                stroke=stroke_color,
                stroke_width=stroke_width,
            )
        )
        sid += 1
        if len(items) >= max_lines:
            break
    return items


# ============================================================
# Gradient helpers
# ============================================================

def mean_region_color(region: np.ndarray, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if region.size == 0:
        return fallback
    flat = region.reshape(-1, 3)
    if len(flat) == 0:
        return fallback
    return tuple(int(round(x)) for x in flat.mean(axis=0))


def add_linear_gradients(
    shapes: List[Shape],
    original_rgb: np.ndarray,
    max_gradients: int,
    min_area_for_gradient: int,
    diff_threshold: int,
    direction: str,
    allowed_roles: Tuple[str, ...] = ("base", "detail"),
) -> None:
    if max_gradients <= 0:
        return
    h_img, w_img = original_rgb.shape[:2]
    count = 0
    for s in shapes:
        if s.role not in allowed_roles:
            continue
        if s.area < min_area_for_gradient:
            continue
        if count >= max_gradients:
            break
        x, y, bw, bh = s.bbox
        x1 = clamp_int(x, 0, w_img - 1)
        y1 = clamp_int(y, 0, h_img - 1)
        x2 = clamp_int(x + bw, 1, w_img)
        y2 = clamp_int(y + bh, 1, h_img)
        crop = original_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        fallback = hex_to_rgb(s.fill)
        top = mean_region_color(crop[: max(1, crop.shape[0] // 3), :, :], fallback)
        bottom = mean_region_color(crop[-max(1, crop.shape[0] // 3) :, :, :], fallback)
        left = mean_region_color(crop[:, : max(1, crop.shape[1] // 3), :], fallback)
        right = mean_region_color(crop[:, -max(1, crop.shape[1] // 3) :, :], fallback)
        vd = color_distance(top, bottom)
        hd = color_distance(left, right)
        use_horizontal = False
        if direction == "横":
            use_horizontal = True
        elif direction == "自動":
            use_horizontal = hd > vd
        c1, c2 = (left, right) if use_horizontal else (top, bottom)
        if color_distance(c1, c2) < diff_threshold:
            continue
        coords = (x1, y1, x2, y1) if use_horizontal else (x1, y1, x1, y2)
        s.gradient = {
            "id": f"g{s.id}",
            "c1": rgb_to_hex(c1),
            "c2": rgb_to_hex(c2),
            "coords": coords,
        }
        count += 1


# ============================================================
# SVG output
# ============================================================

def svg_defs_for_shapes(shapes: List[Shape]) -> str:
    defs = []
    for s in shapes:
        if not s.gradient:
            continue
        g = s.gradient
        x1, y1, x2, y2 = g["coords"]
        defs.append(
            f'<linearGradient id="{html.escape(g["id"])}" gradientUnits="userSpaceOnUse" '
            f'x1="{fmt_num(x1)}" y1="{fmt_num(y1)}" x2="{fmt_num(x2)}" y2="{fmt_num(y2)}">'
            f'<stop offset="0%" stop-color="{g["c1"]}"/>'
            f'<stop offset="100%" stop-color="{g["c2"]}"/>'
            f"</linearGradient>"
        )
    return "<defs>" + "".join(defs) + "</defs>" if defs else ""


def shape_to_svg_element(s: Shape, paint_stroke_width: float) -> str:
    fill = f'url(#{s.gradient["id"]})' if s.gradient else s.fill
    opacity_attr = "" if abs(s.opacity - 1.0) < 1e-6 else f' opacity="{fmt_num(s.opacity, 3)}"'
    if s.role == "line_stroke":
        sw = fmt_num(max(0.1, s.stroke_width))
        stroke = s.stroke or "#111111"
        return (
            f'<path d="{s.d}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round"{opacity_attr}/>'
        )
    if s.role == "line_fill":
        return f'<path d="{s.d}" fill="{s.fill}" fill-rule="evenodd"{opacity_attr}/>'
    if paint_stroke_width > 0 and s.role in ("base", "detail"):
        sw = fmt_num(paint_stroke_width)
        return (
            f'<path d="{s.d}" fill="{fill}" stroke="{s.fill}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round" fill-rule="evenodd"{opacity_attr}/>'
        )
    return f'<path d="{s.d}" fill="{fill}" fill-rule="evenodd"{opacity_attr}/>'


def make_svg_document(
    width: int,
    height: int,
    bg_hex: str,
    shapes: List[Shape],
    paint_stroke_width: float,
    include_background: bool = True,
    title: str = "vectorized",
) -> str:
    body = []
    if include_background:
        body.append(f'<rect width="{width}" height="{height}" fill="{bg_hex}"/>')
    body.extend(shape_to_svg_element(s, paint_stroke_width=paint_stroke_width) for s in shapes)
    defs = svg_defs_for_shapes(shapes)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" shape-rendering="geometricPrecision">'
        f'<title>{html.escape(title)}</title>{defs}{"".join(body)}</svg>'
    )


def estimate_kb(text: str) -> float:
    return len(text.encode("utf-8")) / 1024.0


def split_shapes_by_size(
    shapes: List[Shape],
    width: int,
    height: int,
    bg_hex: str,
    paint_stroke_width: float,
    target_kb: float,
    include_background_first_part: bool,
    file_prefix: str,
) -> List[Tuple[str, str, float, int]]:
    if not shapes:
        return []
    parts: List[Tuple[str, str, float, int]] = []
    current: List[Shape] = []
    part_no = 1
    target_bytes = int(max(4.0, target_kb) * 1024)

    def finalize(chunk: List[Shape], no: int) -> None:
        include_bg = include_background_first_part and no == 1
        svg = make_svg_document(
            width=width,
            height=height,
            bg_hex=bg_hex,
            shapes=chunk,
            paint_stroke_width=paint_stroke_width,
            include_background=include_bg,
            title=f"{file_prefix}_{no:03d}",
        )
        name = f"{file_prefix}_{no:03d}_{len(svg.encode('utf-8')) // 1024 + 1}KB.svg"
        parts.append((name, svg, estimate_kb(svg), len(chunk)))

    for s in shapes:
        trial = current + [s]
        include_bg = include_background_first_part and part_no == 1
        trial_svg = make_svg_document(
            width=width,
            height=height,
            bg_hex=bg_hex,
            shapes=trial,
            paint_stroke_width=paint_stroke_width,
            include_background=include_bg,
            title=f"{file_prefix}_{part_no:03d}",
        )
        if current and len(trial_svg.encode("utf-8")) > target_bytes:
            finalize(current, part_no)
            part_no += 1
            current = [s]
        else:
            current = trial
    if current:
        finalize(current, part_no)
    return parts


def make_zip(
    part_svgs: List[Tuple[str, str, float, int]],
    combined_svg: str,
    settings: Dict,
    width: int,
    height: int,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, svg, _, _ in part_svgs:
            z.writestr(name, svg)
        z.writestr("preview_combined.svg", combined_svg)
        z.writestr("settings.json", json.dumps(settings, ensure_ascii=False, indent=2))
        readme = f"""イラストSVGベクター化ツール 出力ファイル

画像サイズ: {width} x {height}px

レイヤー順:
1. part_base_*       大きい下地
2. part_detail_*     上塗りの細部
3. part_shadow_*     影補助
4. part_highlight_*  ハイライト補助
5. part_lineart_*    線画

このv5版の改善点:
- detailは baseとの差が小さいパーツを捨てる
- 複雑すぎるギザギザ形状を減らす
- 面積・密度の低いノイズ形状を減らす
- toneレイヤーに軽いぼかしを入れ、荒れを減らす
- 線画に「境界追従線画」を追加し、黒線の位置ズレを減らす

品質を上げるコツ:
- 遠目はよいが近くが荒い場合:
  detail最小面積を上げる / detail差分しきい値を上げる / detail複雑度上限を下げる / line本数を下げる
- 情報が足りない場合:
  detail色数を上げる / detail差分しきい値を下げる / detail最小面積を下げる
"""
        z.writestr("README.txt", readme)
    return buf.getvalue()


def html_preview(svg: str, height: int = 720) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"""
<div style="width:100%; background:#f8f8f8; padding:12px; border-radius:14px; box-sizing:border-box;">
  <img src="data:image/svg+xml;base64,{encoded}" style="width:100%; height:auto; max-height:{height}px; object-fit:contain; background:white; border:1px solid #ddd; border-radius:10px;" />
</div>
"""


# ============================================================
# Main conversion pipeline
# ============================================================

@st.cache_data(show_spinner=False)
def convert_cached(
    image_bytes: bytes,
    filename: str,
    bg_hex: str,
    max_side: int,
    base_smooth: str,
    detail_smooth: str,
    base_color_count: int,
    detail_color_count: int,
    base_epsilon_ratio: float,
    detail_epsilon_ratio: float,
    base_min_area: int,
    detail_min_area: int,
    clean_px: int,
    base_complexity_limit: float,
    detail_max_area_ratio: float,
    detail_diff_threshold: float,
    detail_complexity_limit: float,
    detail_min_fill_ratio: float,
    enable_tone_layers: bool,
    tone_shadow_threshold: int,
    tone_highlight_threshold: int,
    tone_min_area: int,
    tone_complexity_limit: float,
    tone_blur_ksize: int,
    shadow_opacity: float,
    highlight_opacity: float,
    lineart_mode: str,
    line_threshold: int,
    line_min_area: int,
    line_epsilon_ratio: float,
    line_canny_low: int,
    line_canny_high: int,
    line_min_length: int,
    line_max_count: int,
    line_stroke_color: str,
    line_stroke_width: float,
    boundary_dilate_px: int,
    dark_expand_px: int,
    add_gradients: bool,
    gradient_max: int,
    gradient_min_area: int,
    gradient_diff: int,
    gradient_direction: str,
    paint_stroke_width: float,
    target_kb: float,
) -> Dict:
    src = Image.open(io.BytesIO(image_bytes))
    flat = flatten_alpha(src, bg_hex)
    resized = resize_keep_aspect(flat, max_side=max_side)
    original_rgb = np.array(resized, dtype=np.uint8)

    base_processed = preprocess_rgb(original_rgb, base_smooth)
    detail_processed = preprocess_rgb(original_rgb, detail_smooth)

    base_index, base_palette = quantize_rgb(base_processed, color_count=base_color_count)
    detail_index, detail_palette = quantize_rgb(detail_processed, color_count=detail_color_count)

    base_shapes, auto_bg = extract_base_shapes(
        index_map=base_index,
        palette=base_palette,
        epsilon_ratio=base_epsilon_ratio,
        min_area=base_min_area,
        clean_px=clean_px,
        complexity_limit=base_complexity_limit,
    )
    base_rgb = rebuild_rgb_from_index(base_index, base_palette)

    next_id = len(base_shapes) + 1
    detail_shapes = extract_detail_shapes_filtered(
        original_rgb=original_rgb,
        base_rgb=base_rgb,
        detail_index=detail_index,
        detail_palette=detail_palette,
        epsilon_ratio=detail_epsilon_ratio,
        min_area=detail_min_area,
        clean_px=clean_px,
        max_area_ratio=detail_max_area_ratio,
        diff_threshold=detail_diff_threshold,
        complexity_limit=detail_complexity_limit,
        min_fill_ratio=detail_min_fill_ratio,
        start_id=next_id,
    )
    next_id += len(detail_shapes)

    shadow_shapes: List[Shape] = []
    highlight_shapes: List[Shape] = []
    if enable_tone_layers:
        shadow_shapes, highlight_shapes = extract_residual_tone_shapes(
            original_rgb=original_rgb,
            base_rgb=base_rgb,
            epsilon_ratio=detail_epsilon_ratio,
            min_area=tone_min_area,
            clean_px=clean_px,
            dark_threshold=tone_shadow_threshold,
            light_threshold=tone_highlight_threshold,
            shadow_opacity=shadow_opacity,
            highlight_opacity=highlight_opacity,
            complexity_limit=tone_complexity_limit,
            blur_ksize=tone_blur_ksize,
            start_id=next_id,
        )
        next_id += len(shadow_shapes) + len(highlight_shapes)

    line_shapes: List[Shape] = []
    if lineart_mode == "境界追従線画(おすすめ)":
        line_shapes = extract_boundary_line_shapes(
            original_rgb=original_rgb,
            base_index=base_index,
            detail_index=detail_index,
            dark_threshold=line_threshold,
            boundary_dilate_px=boundary_dilate_px,
            dark_expand_px=dark_expand_px,
            epsilon_ratio=line_epsilon_ratio,
            min_length=line_min_length,
            max_lines=line_max_count,
            stroke_color=line_stroke_color,
            stroke_width=line_stroke_width,
            start_id=next_id,
        )
    elif lineart_mode == "黒塗り線画(旧方式)":
        line_shapes = extract_dark_fill_line_shapes(
            original_rgb=original_rgb,
            dark_threshold=line_threshold,
            epsilon_ratio=line_epsilon_ratio,
            min_area=line_min_area,
            clean_px=max(1, clean_px),
            start_id=next_id,
        )
    elif lineart_mode == "Canny線画":
        line_shapes = extract_edge_line_shapes(
            original_rgb=original_rgb,
            canny_low=line_canny_low,
            canny_high=line_canny_high,
            epsilon_ratio=line_epsilon_ratio,
            min_length=line_min_length,
            max_lines=line_max_count,
            stroke_color=line_stroke_color,
            stroke_width=line_stroke_width,
            start_id=next_id,
        )

    if add_gradients:
        add_linear_gradients(
            shapes=base_shapes + detail_shapes,
            original_rgb=original_rgb,
            max_gradients=gradient_max,
            min_area_for_gradient=gradient_min_area,
            diff_threshold=gradient_diff,
            direction=gradient_direction,
            allowed_roles=("base", "detail"),
        )

    width, height = resized.size
    combined_shapes = base_shapes + detail_shapes + shadow_shapes + highlight_shapes + line_shapes
    combined_svg = make_svg_document(
        width=width,
        height=height,
        bg_hex=auto_bg,
        shapes=combined_shapes,
        paint_stroke_width=paint_stroke_width,
        include_background=True,
        title="preview_combined",
    )

    parts: List[Tuple[str, str, float, int]] = []
    parts += split_shapes_by_size(base_shapes, width, height, auto_bg, paint_stroke_width, target_kb, True, "part_base")
    parts += split_shapes_by_size(detail_shapes, width, height, auto_bg, paint_stroke_width, target_kb, False, "part_detail")
    parts += split_shapes_by_size(shadow_shapes, width, height, auto_bg, 0.0, target_kb, False, "part_shadow")
    parts += split_shapes_by_size(highlight_shapes, width, height, auto_bg, 0.0, target_kb, False, "part_highlight")
    parts += split_shapes_by_size(line_shapes, width, height, auto_bg, 0.0, target_kb, False, "part_lineart")

    settings = {
        "filename": filename,
        "input_size": src.size,
        "output_size": [width, height],
        "background_for_transparency": bg_hex,
        "auto_background_fill": auto_bg,
        "max_side": max_side,
        "base_smooth": base_smooth,
        "detail_smooth": detail_smooth,
        "base_color_count": base_color_count,
        "detail_color_count": detail_color_count,
        "base_epsilon_ratio": base_epsilon_ratio,
        "detail_epsilon_ratio": detail_epsilon_ratio,
        "base_min_area": base_min_area,
        "detail_min_area": detail_min_area,
        "clean_px": clean_px,
        "base_complexity_limit": base_complexity_limit,
        "detail_max_area_ratio": detail_max_area_ratio,
        "detail_diff_threshold": detail_diff_threshold,
        "detail_complexity_limit": detail_complexity_limit,
        "detail_min_fill_ratio": detail_min_fill_ratio,
        "enable_tone_layers": enable_tone_layers,
        "tone_shadow_threshold": tone_shadow_threshold,
        "tone_highlight_threshold": tone_highlight_threshold,
        "tone_min_area": tone_min_area,
        "tone_complexity_limit": tone_complexity_limit,
        "tone_blur_ksize": tone_blur_ksize,
        "shadow_opacity": shadow_opacity,
        "highlight_opacity": highlight_opacity,
        "lineart_mode": lineart_mode,
        "line_threshold": line_threshold,
        "line_min_area": line_min_area,
        "line_epsilon_ratio": line_epsilon_ratio,
        "line_canny_low": line_canny_low,
        "line_canny_high": line_canny_high,
        "line_min_length": line_min_length,
        "line_max_count": line_max_count,
        "line_stroke_color": line_stroke_color,
        "line_stroke_width": line_stroke_width,
        "boundary_dilate_px": boundary_dilate_px,
        "dark_expand_px": dark_expand_px,
        "add_gradients": add_gradients,
        "gradient_max": gradient_max,
        "gradient_min_area": gradient_min_area,
        "gradient_diff": gradient_diff,
        "gradient_direction": gradient_direction,
        "paint_stroke_width": paint_stroke_width,
        "target_kb": target_kb,
        "base_shape_count": len(base_shapes),
        "detail_shape_count": len(detail_shapes),
        "shadow_shape_count": len(shadow_shapes),
        "highlight_shape_count": len(highlight_shapes),
        "line_shape_count": len(line_shapes),
        "part_count": len(parts),
        "combined_kb": round(estimate_kb(combined_svg), 2),
    }
    zip_bytes = make_zip(parts, combined_svg, settings, width, height)

    preview_png = io.BytesIO()
    resized.save(preview_png, format="PNG")
    return {
        "width": width,
        "height": height,
        "source_png": preview_png.getvalue(),
        "combined_svg": combined_svg,
        "parts": parts,
        "settings": settings,
        "zip_bytes": zip_bytes,
    }


# ============================================================
# UI presets
# ============================================================

def preset_values(preset: str) -> Dict:
    if preset == "高速・軽量":
        return dict(
            max_side=800,
            base_colors=10,
            detail_colors=16,
            base_eps=0.010,
            detail_eps=0.008,
            base_min=40,
            detail_min=18,
            clean_px=2,
            target_kb=14.0,
            detail_max_ratio=0.10,
            detail_diff=20.0,
            detail_complexity=18.0,
            detail_fill=0.16,
            base_complexity=24.0,
            tone_min=16,
            tone_complexity=12.0,
            tone_blur=7,
            line_max=180,
        )
    if preset == "高品質":
        return dict(
            max_side=1400,
            base_colors=14,
            detail_colors=34,
            base_eps=0.006,
            detail_eps=0.0048,
            base_min=22,
            detail_min=8,
            clean_px=1,
            target_kb=24.0,
            detail_max_ratio=0.16,
            detail_diff=16.0,
            detail_complexity=24.0,
            detail_fill=0.12,
            base_complexity=28.0,
            tone_min=10,
            tone_complexity=14.0,
            tone_blur=5,
            line_max=280,
        )
    if preset == "15KB分割重視":
        return dict(
            max_side=1000,
            base_colors=12,
            detail_colors=22,
            base_eps=0.0085,
            detail_eps=0.0065,
            base_min=32,
            detail_min=12,
            clean_px=2,
            target_kb=14.0,
            detail_max_ratio=0.12,
            detail_diff=18.0,
            detail_complexity=18.0,
            detail_fill=0.15,
            base_complexity=24.0,
            tone_min=14,
            tone_complexity=12.0,
            tone_blur=7,
            line_max=200,
        )
    return dict(
        max_side=1200,
        base_colors=12,
        detail_colors=28,
        base_eps=0.0075,
        detail_eps=0.0055,
        base_min=28,
        detail_min=10,
        clean_px=1,
        target_kb=18.0,
        detail_max_ratio=0.14,
        detail_diff=18.0,
        detail_complexity=20.0,
        detail_fill=0.13,
        base_complexity=26.0,
        tone_min=12,
        tone_complexity=12.0,
        tone_blur=5,
        line_max=220,
    )


# ============================================================
# Streamlit app
# ============================================================

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎨", layout="wide")
    st.title("🎨 イラスト SVG ベクター化ツール")
    st.caption("大きい下地 → 差分ディテール → 影/ハイライト → 線画 の順で、近距離品質と線の追従精度を上げる v5 版です。")

    with st.expander("今回の“荒削り”と線ズレへの対策", expanded=True):
        st.markdown(
            """
遠目で良く見えても近くで荒く見える原因は、主に次の3つです。

1. **detail層に、baseとあまり変わらないパーツまで大量に入っている**
2. **ギザギザで複雑すぎる小パーツが多い**
3. **tone層や線画層が細かすぎて、近くで見るとノイズに見える**

このv4版では、以下を追加しています。
- **baseとの差分が小さいdetailを自動で捨てる**
- **複雑すぎるパーツを自動で減らす**
- **密度の低いスカスカなパーツを減らす**
- **toneレイヤーに軽いぼかしを入れて荒れを抑える**
- **線画をCannyだけでなく、色境界に沿って追従する方式で生成できる**
            """
        )

    uploaded = st.file_uploader(
        "画像をアップロードしてください",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    preset = st.selectbox("プリセット", ["標準", "高速・軽量", "高品質", "15KB分割重視"], index=0)
    pv = preset_values(preset)

    top1, top2, top3 = st.columns(3)
    with top1:
        max_side = st.slider("最大辺px", 400, 2000, pv["max_side"], 50)
        bg_hex = st.text_input("透過画像の背景色", DEFAULT_BG)
        target_kb = st.slider("1ファイル目標KB", 6.0, 80.0, pv["target_kb"], 1.0)
    with top2:
        paint_stroke_width = st.slider("塗りパーツの隙間隠しストローク", 0.0, 2.0, 0.30, 0.05)
        clean_px = st.slider("ノイズ整理px", 1, 5, pv["clean_px"], 1)
        add_gradients = st.checkbox("大きい面に直線グラデーションを試す", value=False)
    with top3:
        gradient_max = st.slider("最大グラデーション数", 0, 30, 8, 1)
        gradient_min_area = st.slider("グラデーション対象の最小面積", 200, 20000, 1800, 100)
        gradient_diff = st.slider("色差がこの値以上なら使用", 10, 120, 36, 1)
        gradient_direction = st.selectbox("グラデーション方向", ["自動", "縦", "横"], index=0)

    st.subheader("ベース / ディテール設定")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**ベースレイヤー**")
        base_smooth = st.selectbox("ベース平滑化", ["軽く", "中", "強め"], index=2)
        base_color_count = st.slider("ベース色数", 4, 32, pv["base_colors"], 1)
        base_epsilon_ratio = st.slider("ベース輪郭の単純化", 0.002, 0.020, pv["base_eps"], 0.0005, format="%.4f")
        base_min_area = st.slider("ベース最小面積", 4, 300, pv["base_min"], 1)
        base_complexity_limit = st.slider("ベース複雑度上限", 4.0, 60.0, pv["base_complexity"], 0.5, help="低いほどギザギザの形を減らします。")
    with b2:
        st.markdown("**ディテールレイヤー**")
        detail_smooth = st.selectbox("ディテール平滑化", ["なし", "軽く", "中"], index=1)
        detail_color_count = st.slider("ディテール色数", 6, 64, pv["detail_colors"], 1)
        detail_epsilon_ratio = st.slider("ディテール輪郭の単純化", 0.002, 0.020, pv["detail_eps"], 0.0005, format="%.4f")
        detail_min_area = st.slider("ディテール最小面積", 2, 120, pv["detail_min"], 1)
        detail_max_area_ratio = st.slider("ディテール最大面積比", 0.03, 0.30, pv["detail_max_ratio"], 0.01)
        detail_diff_threshold = st.slider("baseとの差分しきい値", 1.0, 60.0, pv["detail_diff"], 1.0, help="高いほど“本当に必要な差分”だけ残します。")
        detail_complexity_limit = st.slider("ディテール複雑度上限", 4.0, 80.0, pv["detail_complexity"], 0.5, help="低いほどギザギザの小片を減らします。")
        detail_min_fill_ratio = st.slider("ディテール最小充填率", 0.02, 0.80, pv["detail_fill"], 0.01, help="バウンディングボックスに対して中身がスカスカな形を減らします。")

    with st.expander("影 / ハイライト補助レイヤー", expanded=False):
        t1, t2 = st.columns(2)
        with t1:
            enable_tone_layers = st.checkbox("影・ハイライト補助を追加", value=True)
            tone_shadow_threshold = st.slider("影として拾う差", 5, 80, 18, 1)
            tone_highlight_threshold = st.slider("ハイライトとして拾う差", 5, 80, 16, 1)
            tone_min_area = st.slider("影/ハイライトの最小面積", 2, 120, pv["tone_min"], 1)
        with t2:
            tone_complexity_limit = st.slider("影/ハイライト複雑度上限", 4.0, 60.0, pv["tone_complexity"], 0.5)
            tone_blur_ksize = st.slider("toneぼかしカーネル", 1, 15, pv["tone_blur"], 2, help="大きいほど荒れを抑えます。1ならぼかしなし。")
            shadow_opacity = st.slider("影の不透明度", 0.05, 1.0, 0.35, 0.05)
            highlight_opacity = st.slider("ハイライトの不透明度", 0.05, 1.0, 0.25, 0.05)

    with st.expander("線画レイヤー設定", expanded=False):
        l1, l2 = st.columns(2)
        with l1:
            lineart_mode = st.selectbox("線画レイヤー方式", ["境界追従線画(おすすめ)", "Canny線画", "オフ", "黒塗り線画(旧方式)"], index=0)
            line_threshold = st.slider("線として扱う暗さしきい値", 20, 150, 72, 1, help="境界追従線画/旧方式で使用します。")
            line_min_area = st.slider("旧方式の最小面積", 1, 80, 4, 1)
            line_epsilon_ratio = st.slider("線画の単純化", 0.0015, 0.020, 0.0035, 0.0005, format="%.4f", help="低いほど線に忠実ですが容量は増えます。")
        with l2:
            line_canny_low = st.slider("Canny low", 10, 150, 40, 1)
            line_canny_high = st.slider("Canny high", 40, 250, 110, 1)
            line_min_length = st.slider("線の最小長さ", 4, 300, 18, 1)
            line_max_count = st.slider("線の最大本数", 20, 1200, pv["line_max"], 10)
            line_stroke_width = st.slider("線の太さ", 0.2, 3.0, 0.8, 0.1)
            line_stroke_color = st.text_input("線の色", "#111111")
            boundary_dilate_px = st.slider("境界追従: 境界の太さ", 1, 5, 2, 1, help="上げると線のつながりは増えますがズレやすくなります。")
            dark_expand_px = st.slider("境界追従: 暗部の許容幅", 1, 5, 2, 1, help="上げると拾いやすくなりますが甘くなります。")

    if uploaded is None:
        st.info("画像をアップロードすると、プレビューとZIP保存ボタンが表示されます。")
        return

    image_bytes = uploaded.getvalue()

    if st.button("SVGに変換する", type="primary", use_container_width=True):
        with st.spinner("base → detail差分 → tone → line の順で、近距離品質と線追従を重視してSVG化しています…"):
            result = convert_cached(
                image_bytes=image_bytes,
                filename=uploaded.name,
                bg_hex=bg_hex,
                max_side=max_side,
                base_smooth=base_smooth,
                detail_smooth=detail_smooth,
                base_color_count=base_color_count,
                detail_color_count=detail_color_count,
                base_epsilon_ratio=base_epsilon_ratio,
                detail_epsilon_ratio=detail_epsilon_ratio,
                base_min_area=base_min_area,
                detail_min_area=detail_min_area,
                clean_px=clean_px,
                base_complexity_limit=base_complexity_limit,
                detail_max_area_ratio=detail_max_area_ratio,
                detail_diff_threshold=detail_diff_threshold,
                detail_complexity_limit=detail_complexity_limit,
                detail_min_fill_ratio=detail_min_fill_ratio,
                enable_tone_layers=enable_tone_layers,
                tone_shadow_threshold=tone_shadow_threshold,
                tone_highlight_threshold=tone_highlight_threshold,
                tone_min_area=tone_min_area,
                tone_complexity_limit=tone_complexity_limit,
                tone_blur_ksize=tone_blur_ksize,
                shadow_opacity=shadow_opacity,
                highlight_opacity=highlight_opacity,
                lineart_mode=lineart_mode,
                line_threshold=line_threshold,
                line_min_area=line_min_area,
                line_epsilon_ratio=line_epsilon_ratio,
                line_canny_low=line_canny_low,
                line_canny_high=line_canny_high,
                line_min_length=line_min_length,
                line_max_count=line_max_count,
                line_stroke_color=line_stroke_color,
                line_stroke_width=line_stroke_width,
                boundary_dilate_px=boundary_dilate_px,
                dark_expand_px=dark_expand_px,
                add_gradients=add_gradients,
                gradient_max=gradient_max,
                gradient_min_area=gradient_min_area,
                gradient_diff=gradient_diff,
                gradient_direction=gradient_direction,
                paint_stroke_width=paint_stroke_width,
                target_kb=target_kb,
            )
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if not result:
        st.warning("設定後、上の『SVGに変換する』を押してください。")
        return

    s = result["settings"]
    st.success(
        f"変換完了: {s['output_size'][0]}×{s['output_size'][1]}px / "
        f"base {s['base_shape_count']} / detail {s['detail_shape_count']} / "
        f"shadow {s['shadow_shape_count']} / highlight {s['highlight_shape_count']} / line {s['line_shape_count']} / "
        f"分割SVG {s['part_count']}個"
    )

    st.info(
        "近くで荒い場合のおすすめ: detail最小面積を上げる / baseとの差分しきい値を上げる / detail複雑度上限を下げる / toneぼかしカーネルを上げる / 線画方式を『境界追従線画』にして線の単純化を下げる"
    )

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "ZIPを一括ダウンロード",
            data=result["zip_bytes"],
            file_name="vectorized_svg_layers_v5.zip",
            mime="application/zip",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "結合プレビューSVGをダウンロード",
            data=result["combined_svg"].encode("utf-8"),
            file_name="preview_combined.svg",
            mime="image/svg+xml",
            use_container_width=True,
        )

    p1, p2 = st.columns(2)
    with p1:
        st.subheader("元画像")
        st.image(result["source_png"], use_container_width=True)
    with p2:
        st.subheader("SVGプレビュー")
        components.html(html_preview(result["combined_svg"]), height=760, scrolling=True)

    st.subheader("分割ファイル一覧")
    rows = []
    for i, (name, _svg, kb, count) in enumerate(result["parts"], start=1):
        rows.append({"順番": i, "ファイル名": name, "容量KB": round(kb, 2), "パス数": count})
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if result["parts"]:
        names = [p[0] for p in result["parts"]]
        selected = st.selectbox("個別SVGプレビュー", names)
        selected_part = next(p for p in result["parts"] if p[0] == selected)
        st.download_button(
            "このSVGだけダウンロード",
            data=selected_part[1].encode("utf-8"),
            file_name=selected_part[0],
            mime="image/svg+xml",
            use_container_width=True,
        )
        components.html(html_preview(selected_part[1], height=520), height=580, scrolling=True)

    with st.expander("設定JSON", expanded=False):
        st.json(result["settings"])


if __name__ == "__main__":
    main()
