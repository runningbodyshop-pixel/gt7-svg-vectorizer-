import io
import os
import json
import zipfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# =========================================================
# GT7 / Forza-style Layered SVG Vectorizer V6.1
# Clothes & Underpaint Control Edition
# =========================================================

st.set_page_config(
    page_title="GT7 / Forza-style Layered SVG Vectorizer V6.1",
    layout="wide",
)


# =========================================================
# Data models
# =========================================================

@dataclass
class RoleBudget:
    underpaint: int
    base: int
    light: int
    skin: int
    dark: int
    shadow: int
    accent: int
    outline: int
    face: int


@dataclass
class AppConfig:
    mode_name: str
    max_dim: int
    num_colors: int

    min_area: int
    min_area_priority: int
    min_area_face: int

    contour_epsilon: float
    contour_epsilon_priority: float
    contour_epsilon_face: float
    max_points_per_path: int
    max_points_face: int

    morph_open: int
    morph_close: int
    overlap_px: int

    remove_background: bool
    use_subject_priority: bool
    protect_skin: bool
    protect_accents: bool
    protect_light_subject: bool
    enable_outline: bool
    enable_boundary_lines: bool
    enable_dark_lines: bool
    enable_face_lines: bool
    enable_alpha_mask: bool
    enable_line_inpaint: bool
    enable_underpaint: bool
    enable_spatial_coverage: bool
    smooth_paths: bool
    underpaint_style: str
    underpaint_expand: int
    light_mask_erode: int
    light_boundary_line_boost: bool
    bottom_spatial_boost: int

    line_inpaint_radius: int
    line_mask_dilate: int
    line_suppression_mode: str
    color_source_mode: str
    line_suppression_strength: float
    max_line_suppression_component_area: int
    spatial_min_parts_per_cell: int
    spatial_extra_cap: int

    line_darkness_threshold: int
    canny_low: int
    canny_high: int
    boundary_line_width: float
    dark_line_width: float
    face_line_width: float
    outline_color: str

    svg_limit_kb: float
    budget: RoleBudget


@dataclass
class VectorPart:
    part_id: str
    category: str
    role: str
    layer: int
    fill: Optional[str]
    stroke: Optional[str]
    stroke_width: float
    closed: bool
    points: List[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]
    area: float
    priority_score: float
    keep_score: float


@dataclass
class ProcessResult:
    original_rgb: np.ndarray
    processed_rgb: np.ndarray
    color_source_rgb: np.ndarray
    alpha_mask: np.ndarray
    line_suppression_mask: np.ndarray
    foreground_mask: np.ndarray
    subject_mask: np.ndarray
    face_detail_mask: np.ndarray
    skin_mask: np.ndarray
    accent_mask: np.ndarray
    light_subject_mask: np.ndarray
    dark_mass_mask: np.ndarray
    quantized_rgb: np.ndarray
    vector_preview_white: np.ndarray
    vector_preview_dark: np.ndarray
    vector_preview_checker: np.ndarray
    parts: List[VectorPart]
    svg_files: Dict[str, str]
    stats: Dict
    logs: List[str]


# =========================================================
# Presets
# =========================================================

PRESETS: Dict[str, AppConfig] = {
    "Quality Balanced": AppConfig(
        mode_name="Quality Balanced",
        max_dim=1700,
        num_colors=18,
        min_area=30,
        min_area_priority=10,
        min_area_face=5,
        contour_epsilon=0.0038,
        contour_epsilon_priority=0.0018,
        contour_epsilon_face=0.0009,
        max_points_per_path=260,
        max_points_face=520,
        morph_open=3,
        morph_close=5,
        overlap_px=1,
        remove_background=True,
        use_subject_priority=True,
        protect_skin=True,
        protect_accents=True,
        protect_light_subject=True,
        enable_outline=True,
        enable_boundary_lines=True,
        enable_dark_lines=True,
        enable_face_lines=True,
        enable_alpha_mask=True,
        enable_line_inpaint=True,
        enable_underpaint=True,
        enable_spatial_coverage=True,
        smooth_paths=True,
        underpaint_style="Minimal foreground",
        underpaint_expand=0,
        light_mask_erode=2,
        light_boundary_line_boost=True,
        bottom_spatial_boost=2,
        line_inpaint_radius=1,
        line_mask_dilate=0,
        line_suppression_mode="Thin dark edges only",
        color_source_mode="Soft blend",
        line_suppression_strength=0.65,
        max_line_suppression_component_area=1800,
        spatial_min_parts_per_cell=5,
        spatial_extra_cap=55,
        line_darkness_threshold=112,
        canny_low=45,
        canny_high=150,
        boundary_line_width=0.75,
        dark_line_width=0.95,
        face_line_width=0.90,
        outline_color="#1F1F1F",
        svg_limit_kb=14.5,
        budget=RoleBudget(underpaint=12, base=140, light=95, skin=80, dark=105, shadow=95, accent=80, outline=180, face=130),
    ),
    "Anime High Quality": AppConfig(
        mode_name="Anime High Quality",
        max_dim=1900,
        num_colors=22,
        min_area=22,
        min_area_priority=7,
        min_area_face=4,
        contour_epsilon=0.0030,
        contour_epsilon_priority=0.0013,
        contour_epsilon_face=0.00065,
        max_points_per_path=340,
        max_points_face=680,
        morph_open=3,
        morph_close=5,
        overlap_px=1,
        remove_background=True,
        use_subject_priority=True,
        protect_skin=True,
        protect_accents=True,
        protect_light_subject=True,
        enable_outline=True,
        enable_boundary_lines=True,
        enable_dark_lines=True,
        enable_face_lines=True,
        enable_alpha_mask=True,
        enable_line_inpaint=True,
        enable_underpaint=True,
        enable_spatial_coverage=True,
        smooth_paths=True,
        underpaint_style="Minimal foreground",
        underpaint_expand=0,
        light_mask_erode=2,
        light_boundary_line_boost=True,
        bottom_spatial_boost=2,
        line_inpaint_radius=1,
        line_mask_dilate=0,
        line_suppression_mode="Thin dark edges only",
        color_source_mode="Soft blend",
        line_suppression_strength=0.65,
        max_line_suppression_component_area=1800,
        spatial_min_parts_per_cell=5,
        spatial_extra_cap=55,
        line_darkness_threshold=118,
        canny_low=38,
        canny_high=145,
        boundary_line_width=0.70,
        dark_line_width=0.90,
        face_line_width=0.85,
        outline_color="#1D1D1D",
        svg_limit_kb=14.5,
        budget=RoleBudget(underpaint=12, base=170, light=120, skin=110, dark=135, shadow=120, accent=110, outline=240, face=190),
    ),
    "Safe Universal": AppConfig(
        mode_name="Safe Universal",
        max_dim=1350,
        num_colors=14,
        min_area=45,
        min_area_priority=14,
        min_area_face=7,
        contour_epsilon=0.0048,
        contour_epsilon_priority=0.0022,
        contour_epsilon_face=0.0011,
        max_points_per_path=210,
        max_points_face=420,
        morph_open=3,
        morph_close=5,
        overlap_px=1,
        remove_background=True,
        use_subject_priority=True,
        protect_skin=True,
        protect_accents=True,
        protect_light_subject=True,
        enable_outline=True,
        enable_boundary_lines=True,
        enable_dark_lines=True,
        enable_face_lines=True,
        enable_alpha_mask=True,
        enable_line_inpaint=True,
        enable_underpaint=True,
        enable_spatial_coverage=True,
        smooth_paths=True,
        underpaint_style="Minimal foreground",
        underpaint_expand=0,
        light_mask_erode=2,
        light_boundary_line_boost=True,
        bottom_spatial_boost=2,
        line_inpaint_radius=1,
        line_mask_dilate=0,
        line_suppression_mode="Thin dark edges only",
        color_source_mode="Soft blend",
        line_suppression_strength=0.65,
        max_line_suppression_component_area=1800,
        spatial_min_parts_per_cell=5,
        spatial_extra_cap=55,
        line_darkness_threshold=105,
        canny_low=50,
        canny_high=155,
        boundary_line_width=0.80,
        dark_line_width=1.00,
        face_line_width=0.95,
        outline_color="#222222",
        svg_limit_kb=14.5,
        budget=RoleBudget(underpaint=12, base=105, light=70, skin=60, dark=75, shadow=65, accent=65, outline=125, face=95),
    ),
    "Line Art Focus": AppConfig(
        mode_name="Line Art Focus",
        max_dim=1700,
        num_colors=16,
        min_area=34,
        min_area_priority=10,
        min_area_face=4,
        contour_epsilon=0.0042,
        contour_epsilon_priority=0.0017,
        contour_epsilon_face=0.00075,
        max_points_per_path=250,
        max_points_face=640,
        morph_open=3,
        morph_close=5,
        overlap_px=0,
        remove_background=True,
        use_subject_priority=True,
        protect_skin=True,
        protect_accents=True,
        protect_light_subject=True,
        enable_outline=True,
        enable_boundary_lines=True,
        enable_dark_lines=True,
        enable_face_lines=True,
        enable_alpha_mask=True,
        enable_line_inpaint=True,
        enable_underpaint=True,
        enable_spatial_coverage=True,
        smooth_paths=True,
        underpaint_style="Minimal foreground",
        underpaint_expand=0,
        light_mask_erode=2,
        light_boundary_line_boost=True,
        bottom_spatial_boost=2,
        line_inpaint_radius=1,
        line_mask_dilate=0,
        line_suppression_mode="Thin dark edges only",
        color_source_mode="Soft blend",
        line_suppression_strength=0.65,
        max_line_suppression_component_area=1800,
        spatial_min_parts_per_cell=5,
        spatial_extra_cap=55,
        line_darkness_threshold=125,
        canny_low=35,
        canny_high=130,
        boundary_line_width=0.75,
        dark_line_width=0.85,
        face_line_width=0.80,
        outline_color="#202020",
        svg_limit_kb=14.5,
        budget=RoleBudget(underpaint=12, base=90, light=65, skin=80, dark=85, shadow=70, accent=80, outline=260, face=220),
    ),
}


# =========================================================
# General utilities
# =========================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [int(x) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    b, g, r = [int(x) for x in bgr]
    return rgb_to_hex((r, g, b))


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    s = hex_color.strip().lstrip("#")
    if len(s) != 6:
        return (0, 0, 0)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def find_contours_compat(mask: np.ndarray, mode, method):
    result = cv2.findContours(mask, mode, method)
    if len(result) == 2:
        return result
    return result[1], result[2]


def open_uploaded_image(uploaded_file) -> Tuple[np.ndarray, np.ndarray]:
    """Return RGB preview image and the original alpha channel.

    V6 keeps alpha instead of immediately baking it away. This helps transparent
    PNGs and soft fade-outs stay stable during foreground extraction.
    """
    img = Image.open(uploaded_file).convert("RGBA")
    rgba = np.array(img)
    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3].copy()

    # For UI preview and RGB-only inputs, composite transparent pixels on white.
    if np.any(alpha < 255):
        a = (alpha.astype(np.float32) / 255.0)[:, :, None]
        rgb = (rgb.astype(np.float32) * a + 255.0 * (1.0 - a)).clip(0, 255).astype(np.uint8)
    return rgb, alpha


def resize_to_max_dim(img_bgr: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_bgr, scale


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def dilate_mask(mask: np.ndarray, k: int, iterations: int = 1) -> np.ndarray:
    if k <= 0:
        return mask
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)


def erode_mask(mask: np.ndarray, k: int, iterations: int = 1) -> np.ndarray:
    if k <= 0:
        return mask
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(mask, kernel, iterations=iterations)


def clean_mask(mask: np.ndarray, open_k: int = 3, close_k: int = 5) -> np.ndarray:
    out = mask.copy()
    if open_k > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    if close_k > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    return out


def draw_mask_preview(mask: np.ndarray) -> np.ndarray:
    return np.stack([mask, mask, mask], axis=-1).astype(np.uint8)


def resize_mask_to_shape(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape[:2] == (h, w):
        return mask.astype(np.uint8)
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)


def build_alpha_mask(alpha: np.ndarray, shape_hw: Tuple[int, int], cfg: AppConfig, logs: List[str]) -> np.ndarray:
    alpha_rs = resize_mask_to_shape(alpha, shape_hw)
    if not cfg.enable_alpha_mask:
        return np.full(shape_hw, 255, dtype=np.uint8)

    # Only treat alpha as meaningful when the input really contains transparency.
    transparent_ratio = float(np.count_nonzero(alpha_rs < 248)) / float(alpha_rs.size)
    if transparent_ratio < 0.002:
        logs.append("Alpha: no meaningful transparency detected")
        return np.full(shape_hw, 255, dtype=np.uint8)

    # Keep softly transparent areas, but use low alpha as low priority/noise.
    alpha_mask = (alpha_rs > 18).astype(np.uint8) * 255
    alpha_mask = clean_mask(alpha_mask, open_k=3, close_k=5)
    logs.append(f"Alpha: transparency detected, alpha foreground ratio={np.count_nonzero(alpha_mask)/float(alpha_mask.size):.3f}")
    return alpha_mask


def merge_alpha_foreground(fg_mask: np.ndarray, alpha_mask: np.ndarray, logs: List[str]) -> np.ndarray:
    # If alpha is meaningful, it should strongly constrain the foreground.
    if np.count_nonzero(alpha_mask == 0) < 10:
        return fg_mask
    alpha_ratio = np.count_nonzero(alpha_mask) / float(alpha_mask.size)
    if 0.01 < alpha_ratio < 0.98:
        merged = cv2.bitwise_and(dilate_mask(fg_mask, 3), dilate_mask(alpha_mask, 2))
        # Preserve opaque alpha components even when border-color removal misses them.
        merged = cv2.bitwise_or(merged, erode_mask(alpha_mask, 2))
        merged = clean_mask(merged, open_k=3, close_k=7)
        logs.append("Foreground: constrained by alpha mask")
        return merged
    return fg_mask


def filter_line_components(mask: np.ndarray, max_component_area: int, logs: Optional[List[str]] = None) -> np.ndarray:
    """Keep small/thin connected components so large dark surfaces are not inpainted.

    The previous V5 broad mask could include hair masses, dark clothing, and mascot faces.
    This filter intentionally keeps edge-like components and rejects chunky blobs.
    """
    src = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    out = np.zeros_like(mask)
    kept = 0
    rejected = 0
    max_area = max(8, int(max_component_area))

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        long_side = max(w, h)
        short_side = max(1, min(w, h))
        density = area / float(max(1, w * h))
        thickness_est = area / float(max(1, long_side))
        aspect = long_side / float(short_side)

        # Keep edge-like components: small, sparse, long/thin, or low estimated thickness.
        keep = False
        if area <= max_area and thickness_est <= 5.5:
            keep = True
        if area <= max_area * 2 and aspect >= 5.0 and density <= 0.55:
            keep = True
        if area <= 24:
            keep = True

        # Reject compact blobs even when dark; those are likely hair/clothing/cat surfaces.
        if area > max_area or (density > 0.70 and aspect < 3.0 and area > 40):
            keep = False

        if keep:
            out[labels == i] = 255
            kept += 1
        else:
            rejected += 1

    if logs is not None:
        logs.append(f"Line component filter: kept={kept}, rejected={rejected}")
    return out.astype(np.uint8)


def build_line_suppression_mask(img_bgr: np.ndarray, fg_mask: np.ndarray, cfg: AppConfig, logs: List[str]) -> np.ndarray:
    """Build a line mask for color-source cleanup, with selectable experimental modes.

    V6 defaults to a conservative thin-edge mask. The important change from V5 is that
    large dark surfaces are no longer removed from the color source by default.
    """
    if not cfg.enable_line_inpaint or cfg.line_suppression_mode == "Off":
        logs.append("Line suppression: OFF")
        return np.zeros(fg_mask.shape[:2], dtype=np.uint8)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)
    dark = ((gray < cfg.line_darkness_threshold).astype(np.uint8) * 255)
    inner_fg = erode_mask(fg_mask, 1)

    thin_dark_edges = cv2.bitwise_and(edges, dilate_mask(dark, 1))
    thin_dark_edges = cv2.bitwise_and(thin_dark_edges, inner_fg)

    mode = cfg.line_suppression_mode
    if mode == "Thin dark edges only":
        ink = thin_dark_edges
    elif mode == "Thin dark + soft boundary":
        # Add weak local contrast edges, but keep them component-filtered.
        local = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), max(20, cfg.canny_low - 12), max(70, cfg.canny_high - 22))
        local = cv2.bitwise_and(local, inner_fg)
        ink = cv2.bitwise_or(thin_dark_edges, cv2.bitwise_and(local, dilate_mask(fg_mask, 1)))
    elif mode == "Conservative face/detail":
        # Very small mask: safer for color source, less aggressive line suppression.
        ink = cv2.bitwise_and(edges, dark)
        ink = cv2.bitwise_and(ink, inner_fg)
        ink = filter_line_components(ink, max(80, cfg.max_line_suppression_component_area // 3), logs)
    elif mode == "Legacy broad dark mask":
        # Kept only for comparison; this can create V5-like artifacts.
        ink = cv2.bitwise_or(cv2.bitwise_and(dilate_mask(edges, 2), dilate_mask(dark, 2)), cv2.bitwise_and(dark, fg_mask))
        ink = cv2.bitwise_and(ink, fg_mask)
    else:
        ink = thin_dark_edges

    if mode != "Legacy broad dark mask":
        ink = filter_line_components(ink, cfg.max_line_suppression_component_area, logs)

    ink = clean_mask(ink, open_k=1, close_k=2)
    if cfg.line_mask_dilate > 0:
        ink = dilate_mask(ink, cfg.line_mask_dilate)
        if mode != "Legacy broad dark mask":
            ink = filter_line_components(ink, cfg.max_line_suppression_component_area * 2, logs)

    ink = cv2.bitwise_and(ink, fg_mask)
    ratio = np.count_nonzero(ink) / float(max(1, np.count_nonzero(fg_mask)))
    logs.append(f"Line suppression mode={mode}, pixels={int(np.count_nonzero(ink))}, ratio_in_fg={ratio:.4f}")
    return ink.astype(np.uint8)


def build_color_source(img_bgr: np.ndarray, fg_mask: np.ndarray, line_mask: np.ndarray, cfg: AppConfig, logs: List[str]) -> np.ndarray:
    """Create the fill-color source using safer alternatives to broad inpaint.

    Default V6 mode is Soft blend: it weakens line contrast without hallucinating radial
    inpaint artifacts. Telea/NS are available as comparison modes.
    """
    if not cfg.enable_line_inpaint or np.count_nonzero(line_mask) == 0 or cfg.color_source_mode == "None / original":
        logs.append("Color source: using processed image without suppression")
        return img_bgr.copy()

    mask = cv2.bitwise_and(line_mask, fg_mask).astype(np.uint8)
    radius = max(1, int(cfg.line_inpaint_radius))
    strength = float(np.clip(cfg.line_suppression_strength, 0.0, 1.0))
    mode = cfg.color_source_mode

    try:
        if mode == "Soft blend":
            k = max(3, radius * 2 + 1)
            if k % 2 == 0:
                k += 1
            # Median is stable for anime line weakening and avoids inpaint swirls.
            smooth = cv2.medianBlur(img_bgr, k)
            smooth = cv2.bilateralFilter(smooth, d=5, sigmaColor=24, sigmaSpace=24)
            alpha = (cv2.GaussianBlur(mask, (0, 0), sigmaX=max(0.6, radius * 0.55)).astype(np.float32) / 255.0)
            alpha = np.clip(alpha * strength, 0.0, 1.0)[:, :, None]
            color_source = (img_bgr.astype(np.float32) * (1.0 - alpha) + smooth.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
            logs.append(f"Color source: Soft blend radius={radius}, strength={strength:.2f}")
            return color_source

        if mode == "Median replace":
            k = max(3, radius * 2 + 1)
            if k % 2 == 0:
                k += 1
            median = cv2.medianBlur(img_bgr, k)
            color_source = img_bgr.copy()
            blend_mask = mask > 0
            color_source[blend_mask] = (img_bgr[blend_mask].astype(np.float32) * (1.0 - strength) + median[blend_mask].astype(np.float32) * strength).clip(0, 255).astype(np.uint8)
            logs.append(f"Color source: Median replace radius={radius}, strength={strength:.2f}")
            return color_source

        if mode == "Telea inpaint":
            color_source = cv2.inpaint(img_bgr, mask, radius, cv2.INPAINT_TELEA)
            color_source = cv2.bilateralFilter(color_source, d=5, sigmaColor=22, sigmaSpace=22)
            logs.append(f"Color source: Telea inpaint radius={radius}")
            return color_source

        if mode == "NS inpaint":
            color_source = cv2.inpaint(img_bgr, mask, radius, cv2.INPAINT_NS)
            color_source = cv2.bilateralFilter(color_source, d=5, sigmaColor=22, sigmaSpace=22)
            logs.append(f"Color source: NS inpaint radius={radius}")
            return color_source

        logs.append(f"Color source: unknown mode '{mode}', using original")
        return img_bgr.copy()
    except Exception as e:
        logs.append(f"Color source: suppression failed, fallback to processed image: {e}")
        return img_bgr.copy()


def chaikin_smooth_points(points: List[Tuple[int, int]], closed: bool, iterations: int = 1) -> List[Tuple[int, int]]:
    if len(points) < 4 or iterations <= 0:
        return points
    pts = [(float(x), float(y)) for x, y in points]
    for _ in range(iterations):
        new_pts = []
        n = len(pts)
        rng = range(n) if closed else range(n - 1)
        if not closed:
            new_pts.append(pts[0])
        for i in rng:
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_pts.extend([q, r])
        if not closed:
            new_pts.append(pts[-1])
        pts = new_pts
    return [(int(round(x)), int(round(y))) for x, y in pts]


def should_smooth_role(role: str, closed: bool, point_count: int) -> bool:
    if not closed:
        return False
    if point_count > 360:
        return False
    return role in {"underpaint", "base", "light", "skin", "dark", "shadow", "accent"}


def png_bytes_from_rgb(rgb: np.ndarray) -> bytes:
    img = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def safe_filename_base(name: str) -> str:
    base = os.path.splitext(name)[0]
    out = []
    for ch in base:
        out.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    s = "".join(out).strip("_")
    return s or "vectorized_image"


def luminance_bgr(color_bgr: Tuple[int, int, int]) -> float:
    b, g, r = [float(x) for x in color_bgr]
    return 0.0722 * b + 0.7152 * g + 0.2126 * r


def hsv_of_bgr(color_bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
    arr = np.uint8([[list(color_bgr)]])
    h, s, v = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0, 0]
    return float(h), float(s), float(v)


def contour_to_points(contour: np.ndarray) -> List[Tuple[int, int]]:
    pts = contour.reshape(-1, 2)
    return [(int(x), int(y)) for x, y in pts]


def simplify_contour(contour: np.ndarray, epsilon_ratio: float, max_points: int, closed: bool) -> np.ndarray:
    if len(contour) < 2:
        return contour
    peri = cv2.arcLength(contour, closed)
    eps = max(0.35, epsilon_ratio * peri)
    approx = cv2.approxPolyDP(contour, eps, closed)
    tries = 0
    while len(approx) > max_points and tries < 12:
        eps *= 1.22
        approx = cv2.approxPolyDP(contour, eps, closed)
        tries += 1
    return approx


def largest_components(mask: np.ndarray, min_area: int, max_components: int = 8) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            comps.append((i, area))
    comps.sort(key=lambda x: x[1], reverse=True)
    for i, _area in comps[:max_components]:
        out[labels == i] = 255
    return out


# =========================================================
# Preprocess and masks
# =========================================================

def preprocess_image(img_bgr: np.ndarray, cfg: AppConfig, logs: List[str]) -> np.ndarray:
    logs.append("Preprocess: resize, gentle denoise, local contrast normalize")
    img_bgr, scale = resize_to_max_dim(img_bgr, cfg.max_dim)
    if scale < 1.0:
        logs.append(f"Input resized to max_dim={cfg.max_dim}, scale={scale:.3f}")

    # Stable smoothing that keeps anime edges reasonably well.
    img_bgr = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=34, sigmaSpace=34)

    # Light contrast normalization.
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.55, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return img_bgr


def build_foreground_mask(img_bgr: np.ndarray, cfg: AppConfig, logs: List[str]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if not cfg.remove_background:
        return np.full((h, w), 255, dtype=np.uint8)

    logs.append("Foreground: border-connected background removal")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    border_pixels = np.concatenate([
        img_bgr[0, :, :], img_bgr[-1, :, :], img_bgr[:, 0, :], img_bgr[:, -1, :]
    ], axis=0)
    border_med = np.median(border_pixels, axis=0).astype(np.float32)
    diff = np.sqrt(np.sum((img_bgr.astype(np.float32) - border_med) ** 2, axis=2))

    # Generic background-like estimate: similar to border color or very low-detail light backdrop.
    bg_like = ((diff < 27) | ((sat < 24) & (val > 218))).astype(np.uint8) * 255
    bg_like = clean_mask(bg_like, open_k=3, close_k=5)

    flood = bg_like.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        sx, sy = seed
        if flood[sy, sx] > 0:
            cv2.floodFill(flood, flood_mask, seed, 128)
    border_bg = (flood == 128).astype(np.uint8) * 255
    fg = cv2.bitwise_not(border_bg)

    # If foreground is suspicious, fallback to edge expansion.
    ratio = np.count_nonzero(fg) / float(h * w)
    if ratio < 0.035 or ratio > 0.94:
        logs.append("Foreground fallback: edge-density components")
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 55, 165)
        edges = dilate_mask(edges, 7)
        fg = largest_components(edges, min_area=max(80, int(h * w * 0.001)), max_components=10)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    fg = clean_mask(fg, open_k=3, close_k=9)
    fg = largest_components(fg, min_area=max(80, int(h * w * 0.0005)), max_components=12)
    fg = dilate_mask(fg, 3)
    fg = clean_mask(fg, open_k=3, close_k=9)
    logs.append(f"Foreground ratio={np.count_nonzero(fg) / float(h*w):.3f}")
    return fg


def build_subject_mask(fg_mask: np.ndarray, logs: List[str]) -> np.ndarray:
    h, w = fg_mask.shape[:2]
    x1, y1, x2, y2 = mask_bbox(fg_mask)
    if x2 <= x1 or y2 <= y1:
        return np.zeros_like(fg_mask)

    subject = np.zeros_like(fg_mask)
    # upper-body / head priority, useful for most character art, not hard-coded to a character.
    cx = (x1 + x2) // 2
    cy = y1 + int((y2 - y1) * 0.34)
    ax = max(24, int((x2 - x1) * 0.30))
    ay = max(24, int((y2 - y1) * 0.28))
    cv2.ellipse(subject, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    # Combine with top part of foreground so non-centered heads are still included.
    top = np.zeros_like(fg_mask)
    top_y2 = y1 + int((y2 - y1) * 0.58)
    top[y1:top_y2 + 1, x1:x2 + 1] = 255
    subject = cv2.bitwise_or(cv2.bitwise_and(subject, fg_mask), cv2.bitwise_and(top, fg_mask))
    subject = clean_mask(subject, open_k=3, close_k=7)
    logs.append(f"Subject ratio within FG={np.count_nonzero(subject) / max(1, np.count_nonzero(fg_mask)):.3f}")
    return subject


def build_skin_mask(img_bgr: np.ndarray, fg_mask: np.ndarray, subject_mask: np.ndarray, logs: List[str]) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    # Broad anime-skin range. It will fail gracefully if no skin-like pixels exist.
    skin_ycc = cv2.inRange(ycrcb, np.array([0, 128, 72], np.uint8), np.array([255, 184, 142], np.uint8))
    skin_hsv = (((h_ch < 28) | (h_ch > 168)) & (s_ch > 16) & (s_ch < 178) & (v_ch > 70)).astype(np.uint8) * 255
    skin = cv2.bitwise_and(skin_ycc, skin_hsv)
    skin = cv2.bitwise_and(skin, fg_mask)

    # Give subject region more trust, but keep skin elsewhere for hands.
    subject_skin = cv2.bitwise_and(skin, dilate_mask(subject_mask, 5))
    non_subject_skin = cv2.bitwise_and(skin, cv2.bitwise_not(dilate_mask(subject_mask, 5)))
    non_subject_skin = largest_components(non_subject_skin, min_area=18, max_components=8)
    skin = cv2.bitwise_or(subject_skin, non_subject_skin)
    skin = clean_mask(skin, open_k=3, close_k=5)
    logs.append(f"Skin pixels={int(np.count_nonzero(skin))}")
    return skin


def build_accent_mask(img_bgr: np.ndarray, fg_mask: np.ndarray, logs: List[str]) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Generic accent: relatively saturated, not too dark, not huge.
    accent = ((sat > 88) & (val > 45)).astype(np.uint8) * 255
    accent = cv2.bitwise_and(accent, fg_mask)
    accent = clean_mask(accent, open_k=3, close_k=3)

    fg_count = max(1, np.count_nonzero(fg_mask))
    ratio = np.count_nonzero(accent) / float(fg_count)
    if ratio > 0.36:
        # If the whole character is colorful, reserve only very saturated parts as "accent".
        accent = ((sat > 128) & (val > 55)).astype(np.uint8) * 255
        accent = cv2.bitwise_and(accent, fg_mask)
        accent = clean_mask(accent, open_k=3, close_k=3)
        ratio = np.count_nonzero(accent) / float(fg_count)

    logs.append(f"Accent ratio within FG={ratio:.3f}")
    return accent


def build_light_subject_mask(img_bgr: np.ndarray, fg_mask: np.ndarray, cfg: AppConfig, logs: List[str]) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Protect foreground-internal whites and light grays such as white clothing / hair / accessories.
    # V6.1 makes this intentionally stricter: white clothes should survive, but not become
    # one oversized blob that covers the whole jacket silhouette.
    inner = erode_mask(fg_mask, max(1, cfg.light_mask_erode))
    light_base = ((val > 188) & (sat < 76)).astype(np.uint8) * 255
    light_mid = ((val > 164) & (sat < 50)).astype(np.uint8) * 255
    light = cv2.bitwise_or(light_base, cv2.bitwise_and(light_mid, erode_mask(fg_mask, cfg.light_mask_erode + 1)))
    light = cv2.bitwise_and(light, inner)
    # Break weak bridges so sleeves / torso / background-near pieces do not merge too aggressively.
    light = clean_mask(light, open_k=3, close_k=3)
    light = largest_components(light, min_area=12, max_components=48)
    logs.append(f"Light-subject pixels={int(np.count_nonzero(light))}, erode={cfg.light_mask_erode}")
    return light


def build_dark_mass_mask(img_bgr: np.ndarray, fg_mask: np.ndarray, logs: List[str]) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dark = ((gray < 92).astype(np.uint8) * 255)
    dark = cv2.bitwise_and(dark, fg_mask)
    dark_mass = clean_mask(dark, open_k=5, close_k=7)
    dark_mass = largest_components(dark_mass, min_area=30, max_components=24)
    logs.append(f"Dark-mass pixels={int(np.count_nonzero(dark_mass))}")
    return dark_mass


def build_face_detail_mask(img_bgr: np.ndarray, fg_mask: np.ndarray, subject_mask: np.ndarray, skin_mask: np.ndarray, accent_mask: np.ndarray, cfg: AppConfig, logs: List[str]) -> np.ndarray:
    h, w = fg_mask.shape[:2]
    face = np.zeros_like(fg_mask)

    # 1) Try Haar face, but do not depend on it.
    face_found = False
    if cfg.use_subject_priority:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(38, 38))
                for (x, y, fw, fh) in faces:
                    px = int(fw * 0.38)
                    py = int(fh * 0.45)
                    x1 = clamp(x - px, 0, w - 1)
                    y1 = clamp(y - py, 0, h - 1)
                    x2 = clamp(x + fw + px, 0, w - 1)
                    y2 = clamp(y + fh + py, 0, h - 1)
                    cv2.rectangle(face, (x1, y1), (x2, y2), 255, -1)
                    face_found = True
        except Exception as e:
            logs.append(f"Face detector skipped: {e}")

    # 2) Fallback: subject top core + skin/accent/edge density.
    x1, y1, x2, y2 = mask_bbox(fg_mask)
    if x2 > x1 and y2 > y1:
        cx = (x1 + x2) // 2
        cy = y1 + int((y2 - y1) * 0.27)
        ax = max(20, int((x2 - x1) * 0.20))
        ay = max(20, int((y2 - y1) * 0.19))
        fallback = np.zeros_like(fg_mask)
        cv2.ellipse(fallback, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        fallback = cv2.bitwise_and(fallback, dilate_mask(subject_mask, 5))
        if not face_found:
            face = cv2.bitwise_or(face, fallback)
        else:
            face = cv2.bitwise_or(face, cv2.bitwise_and(fallback, dilate_mask(face, 15)))

    # Boost skin and accents inside subject; useful for eyes/hair ornaments of arbitrary colors.
    face = cv2.bitwise_or(face, cv2.bitwise_and(dilate_mask(skin_mask, 7), subject_mask))
    face = cv2.bitwise_or(face, cv2.bitwise_and(dilate_mask(accent_mask, 5), subject_mask))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 145)
    edges = cv2.bitwise_and(dilate_mask(edges, 3), subject_mask)
    face = cv2.bitwise_or(face, edges)

    face = cv2.bitwise_and(clean_mask(face, open_k=3, close_k=7), fg_mask)
    logs.append(f"Face/detail pixels={int(np.count_nonzero(face))}, haar_found={face_found}")
    return face


# =========================================================
# Quantization and role classification
# =========================================================

def quantize_foreground(img_bgr: np.ndarray, fg_mask: np.ndarray, k: int, logs: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    pixels = img_bgr.reshape(-1, 3).astype(np.float32)
    fg_flat = fg_mask.reshape(-1) > 0

    if np.count_nonzero(fg_flat) < 20:
        labels = np.full((h, w), -1, dtype=np.int32)
        centers = np.array([[245, 245, 245]], dtype=np.uint8)
        return img_bgr.copy(), labels, centers

    fg_pixels = pixels[fg_flat]
    K = min(k, max(2, len(fg_pixels)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 28, 0.75)
    compactness, labels_fg, centers = cv2.kmeans(
        fg_pixels,
        K=K,
        bestLabels=None,
        criteria=criteria,
        attempts=4,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.astype(np.uint8)

    label_flat = np.full((h * w,), -1, dtype=np.int32)
    label_flat[fg_flat] = labels_fg.flatten()
    labels = label_flat.reshape(h, w)

    quant_flat = np.full((h * w, 3), 245, dtype=np.uint8)
    quant_flat[fg_flat] = centers[labels_fg.flatten()]
    quant = quant_flat.reshape(h, w, 3)

    logs.append(f"Quantization: K={K}, compactness={compactness:.2f}")
    return quant, labels, centers


def cluster_role(
    color_bgr: Tuple[int, int, int],
    mask: np.ndarray,
    skin_mask: np.ndarray,
    accent_mask: np.ndarray,
    light_subject_mask: np.ndarray,
    dark_mass_mask: np.ndarray,
    face_detail_mask: np.ndarray,
) -> str:
    area = max(1, int(np.count_nonzero(mask)))
    lum = luminance_bgr(color_bgr)
    _h, sat, val = hsv_of_bgr(color_bgr)

    skin_overlap = np.count_nonzero(cv2.bitwise_and(mask, skin_mask)) / float(area)
    accent_overlap = np.count_nonzero(cv2.bitwise_and(mask, accent_mask)) / float(area)
    light_overlap = np.count_nonzero(cv2.bitwise_and(mask, light_subject_mask)) / float(area)
    dark_overlap = np.count_nonzero(cv2.bitwise_and(mask, dark_mass_mask)) / float(area)
    face_overlap = np.count_nonzero(cv2.bitwise_and(mask, face_detail_mask)) / float(area)

    if face_overlap > 0.18 and skin_overlap > 0.10 and lum > 65:
        return "skin"
    if skin_overlap > 0.18 and lum > 62:
        return "skin"
    if accent_overlap > 0.12 and sat > 65:
        return "accent"
    if light_overlap > 0.18 and lum > 160:
        return "light"
    if dark_overlap > 0.12 and lum < 115:
        return "dark"
    if lum < 72:
        return "dark"
    if lum < 132:
        return "shadow"
    if lum > 205 and sat < 75:
        return "light"
    return "base"


def role_layer(role: str, priority_score: float, area: float) -> int:
    base = {
        "underpaint": 40,
        "base": 100,
        "light": 115,
        "dark": 145,
        "skin": 175,
        "shadow": 220,
        "accent": 310,
        "outline_boundary": 430,
        "outline_dark": 470,
        "outline_face": 530,
        "face_detail": 560,
    }.get(role, 180)
    # Small high-priority pieces should appear slightly later within role.
    small_bonus = int(max(0, 9000 - min(9000, area)) / 900)
    pri_bonus = int(priority_score * 18)
    return base + small_bonus + pri_bonus


def compute_priority_score(mask: np.ndarray, face_detail_mask: np.ndarray, subject_mask: np.ndarray, accent_mask: np.ndarray) -> float:
    area = max(1, int(np.count_nonzero(mask)))
    f = np.count_nonzero(cv2.bitwise_and(mask, face_detail_mask)) / float(area)
    s = np.count_nonzero(cv2.bitwise_and(mask, subject_mask)) / float(area)
    a = np.count_nonzero(cv2.bitwise_and(mask, accent_mask)) / float(area)
    return min(1.0, 0.60 * f + 0.25 * s + 0.25 * a)


def part_keep_score(role: str, area: float, priority_score: float, point_count: int) -> float:
    role_weight = {
        "underpaint": 1.3,
        "face_detail": 3.4,
        "skin": 3.0,
        "accent": 2.8,
        "dark": 2.4,
        "outline_face": 3.2,
        "outline_dark": 2.6,
        "outline_boundary": 2.1,
        "light": 2.0,
        "shadow": 1.8,
        "base": 1.5,
    }.get(role, 1.0)
    # Prefer visible areas but also keep important tiny face/accent pieces.
    area_score = np.log1p(area) / 10.0
    complexity_penalty = min(0.25, max(0, point_count - 320) / 2000.0)
    return role_weight + area_score + priority_score * 2.2 - complexity_penalty


# =========================================================
# Part extraction
# =========================================================

def extract_parts_from_mask(
    mask: np.ndarray,
    fill_hex: Optional[str],
    stroke_hex: Optional[str],
    stroke_width: float,
    role: str,
    part_prefix: str,
    closed: bool,
    cfg: AppConfig,
    subject_mask: np.ndarray,
    face_detail_mask: np.ndarray,
    accent_mask: np.ndarray,
    min_area_override: Optional[int] = None,
    epsilon_override: Optional[float] = None,
    max_points_override: Optional[int] = None,
) -> List[VectorPart]:
    parts: List[VectorPart] = []
    contours, _ = find_contours_compat(mask, cv2.RETR_EXTERNAL if closed else cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    counter = 0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not closed:
            # For open-ish line contours, use bounding area and length to avoid dropping useful thin lines.
            x0, y0, bw0, bh0 = cv2.boundingRect(contour)
            area = max(area, float(max(bw0, bh0)))

        x, y, bw, bh = cv2.boundingRect(contour)
        roi_mask = np.zeros_like(mask)
        cv2.drawContours(roi_mask, [contour], -1, 255, thickness=-1 if closed else 1)
        pri = compute_priority_score(roi_mask, face_detail_mask, subject_mask, accent_mask)

        if min_area_override is not None:
            min_area = min_area_override
        elif pri > 0.38:
            min_area = cfg.min_area_face
        elif pri > 0.10:
            min_area = cfg.min_area_priority
        else:
            min_area = cfg.min_area

        if area < min_area:
            continue

        if epsilon_override is not None:
            eps = epsilon_override
        elif pri > 0.38:
            eps = cfg.contour_epsilon_face
        elif pri > 0.10:
            eps = cfg.contour_epsilon_priority
        else:
            eps = cfg.contour_epsilon

        if max_points_override is not None:
            max_pts = max_points_override
        elif pri > 0.38:
            max_pts = cfg.max_points_face
        elif pri > 0.10:
            max_pts = int(cfg.max_points_per_path * 1.5)
        else:
            max_pts = cfg.max_points_per_path

        approx = simplify_contour(contour, eps, max_pts, closed=closed)
        if len(approx) < (3 if closed else 2):
            continue

        points = contour_to_points(approx)
        if cfg.smooth_paths and should_smooth_role(role, closed, len(points)):
            points = chaikin_smooth_points(points, closed=closed, iterations=1)
            # Re-simplify lightly if smoothing creates too many points.
            if len(points) > max_pts:
                approx2 = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                approx2 = simplify_contour(approx2, eps * 0.8, max_pts, closed=closed)
                points = contour_to_points(approx2)
        layer = role_layer(role, pri, area)
        keep = part_keep_score(role, area, pri, len(points))

        parts.append(VectorPart(
            part_id=f"{part_prefix}_{counter:05d}",
            category=role,
            role=role,
            layer=layer,
            fill=fill_hex,
            stroke=stroke_hex,
            stroke_width=stroke_width,
            closed=closed,
            points=points,
            bbox=(x, y, x + bw, y + bh),
            area=area,
            priority_score=pri,
            keep_score=keep,
        ))
        counter += 1

    return parts



def extract_underpaint_parts(
    img_bgr: np.ndarray,
    fg_mask: np.ndarray,
    subject_mask: np.ndarray,
    face_detail_mask: np.ndarray,
    accent_mask: np.ndarray,
    light_subject_mask: np.ndarray,
    cfg: AppConfig,
    logs: List[str],
) -> List[VectorPart]:
    if not cfg.enable_underpaint or np.count_nonzero(fg_mask) == 0 or cfg.underpaint_style == "Off":
        return []

    # V6.1: underpaint is now controllable. The old full-foreground underpaint
    # could make white clothes look like a giant gray blob, so the default is
    # a minimal foreground underpaint with almost no expansion.
    if cfg.underpaint_style == "Subject core only":
        mask = cv2.bitwise_and(dilate_mask(subject_mask, 3), fg_mask)
    elif cfg.underpaint_style == "Light clothes only":
        mask = cv2.bitwise_and(dilate_mask(light_subject_mask, 1), fg_mask)
    else:
        mask = erode_mask(fg_mask, 1)

    mask = clean_mask(mask, open_k=5, close_k=7)
    if cfg.underpaint_expand > 0:
        mask = dilate_mask(mask, cfg.underpaint_expand)
        mask = cv2.bitwise_and(mask, dilate_mask(fg_mask, cfg.underpaint_expand + 1))
    parts: List[VectorPart] = []
    contours, _ = find_contours_compat(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    counter = 0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < max(80, cfg.min_area * 4):
            continue
        comp = np.zeros_like(mask)
        cv2.drawContours(comp, [c], -1, 255, thickness=-1)
        color = average_color_for_component(img_bgr, comp)
        # Pull underpaint slightly toward a neutral light value so gaps are less harsh.
        b, g, r = color
        color = (int(0.65*b + 0.35*220), int(0.65*g + 0.35*220), int(0.65*r + 0.35*220))
        pri = compute_priority_score(comp, face_detail_mask, subject_mask, accent_mask)
        approx = simplify_contour(c, cfg.contour_epsilon * 1.8, max(80, cfg.max_points_per_path // 2), closed=True)
        pts = contour_to_points(approx)
        if cfg.smooth_paths:
            pts = chaikin_smooth_points(pts, closed=True, iterations=1)
        x, y, bw, bh = cv2.boundingRect(np.array(pts, dtype=np.int32).reshape((-1, 1, 2))) if pts else cv2.boundingRect(c)
        parts.append(VectorPart(
            part_id=f"underpaint_{counter:05d}",
            category="underpaint",
            role="underpaint",
            layer=40,
            fill=bgr_to_hex(color),
            stroke=None,
            stroke_width=0.0,
            closed=True,
            points=pts,
            bbox=(x, y, x + bw, y + bh),
            area=area,
            priority_score=pri,
            keep_score=part_keep_score("underpaint", area, pri, len(pts)),
        ))
        counter += 1
    logs.append(f"Underpaint parts={len(parts)}")
    return parts


def extract_fill_parts(
    img_bgr: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    fg_mask: np.ndarray,
    subject_mask: np.ndarray,
    face_detail_mask: np.ndarray,
    skin_mask: np.ndarray,
    accent_mask: np.ndarray,
    light_subject_mask: np.ndarray,
    dark_mass_mask: np.ndarray,
    cfg: AppConfig,
    logs: List[str],
) -> List[VectorPart]:
    logs.append("Fill extraction: role-based parts")
    parts: List[VectorPart] = []

    open_kernel = np.ones((cfg.morph_open, cfg.morph_open), np.uint8)
    close_kernel = np.ones((cfg.morph_close, cfg.morph_close), np.uint8)

    cluster_indices = [i for i in range(len(centers)) if np.count_nonzero(labels == i) > 0]
    cluster_indices.sort(key=lambda i: np.count_nonzero(labels == i), reverse=True)

    for idx in cluster_indices:
        raw = ((labels == idx).astype(np.uint8) * 255)
        raw = cv2.bitwise_and(raw, fg_mask)
        if np.count_nonzero(raw) == 0:
            continue

        color_bgr = tuple(int(x) for x in centers[idx])
        role = cluster_role(color_bgr, raw, skin_mask, accent_mask, light_subject_mask, dark_mass_mask, face_detail_mask)

        mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        # Underpaint/overlap only for base-like fills; not for thin accents/shadows.
        if cfg.overlap_px > 0 and role in ("base", "skin"):
            mask = dilate_mask(mask, cfg.overlap_px)
            mask = cv2.bitwise_and(mask, dilate_mask(fg_mask, cfg.overlap_px + 1))

        fill_hex = bgr_to_hex(color_bgr)
        parts.extend(extract_parts_from_mask(
            mask=mask,
            fill_hex=fill_hex,
            stroke_hex=None,
            stroke_width=0.0,
            role=role,
            part_prefix=f"{role}_{idx}",
            closed=True,
            cfg=cfg,
            subject_mask=subject_mask,
            face_detail_mask=face_detail_mask,
            accent_mask=accent_mask,
        ))

    # Protective role overlays ensure important masks are not lost by KMeans/part caps.
    if cfg.protect_light_subject and np.count_nonzero(light_subject_mask) > 0:
        parts.extend(extract_protective_color_parts(
            img_bgr, light_subject_mask, "light", "light_protect", cfg,
            subject_mask, face_detail_mask, accent_mask, force_layer=125,
        ))

    if cfg.protect_skin and np.count_nonzero(skin_mask) > 0:
        parts.extend(extract_protective_color_parts(
            img_bgr, skin_mask, "skin", "skin_protect", cfg,
            subject_mask, face_detail_mask, accent_mask, force_layer=185,
        ))

    if cfg.protect_accents and np.count_nonzero(accent_mask) > 0:
        parts.extend(extract_protective_color_parts(
            img_bgr, accent_mask, "accent", "accent_protect", cfg,
            subject_mask, face_detail_mask, accent_mask, force_layer=330,
        ))

    # Dark mass protection: restore strong dark surfaces that V3 often lost.
    if np.count_nonzero(dark_mass_mask) > 0:
        parts.extend(extract_protective_color_parts(
            img_bgr, dark_mass_mask, "dark", "dark_protect", cfg,
            subject_mask, face_detail_mask, accent_mask, force_layer=155,
        ))

    logs.append(f"Fill parts before budget={len(parts)}")
    return parts


def average_color_for_component(img_bgr: np.ndarray, component_mask: np.ndarray) -> Tuple[int, int, int]:
    pixels = img_bgr[component_mask > 0]
    if len(pixels) == 0:
        return (180, 180, 180)
    med = np.median(pixels, axis=0)
    return tuple(int(clamp(round(v), 0, 255)) for v in med)


def extract_protective_color_parts(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    role: str,
    prefix: str,
    cfg: AppConfig,
    subject_mask: np.ndarray,
    face_detail_mask: np.ndarray,
    accent_mask: np.ndarray,
    force_layer: Optional[int] = None,
) -> List[VectorPart]:
    proc = clean_mask(mask, open_k=3, close_k=5)
    proc = largest_components(proc, min_area=max(6, cfg.min_area_face), max_components=60)
    parts: List[VectorPart] = []
    contours, _ = find_contours_compat(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    counter = 0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < cfg.min_area_face:
            continue
        comp_mask = np.zeros_like(mask)
        cv2.drawContours(comp_mask, [c], -1, 255, thickness=-1)
        color = average_color_for_component(img_bgr, comp_mask)
        fill_hex = bgr_to_hex(color)

        pri = compute_priority_score(comp_mask, face_detail_mask, subject_mask, accent_mask)
        eps = cfg.contour_epsilon_face if pri > 0.35 else cfg.contour_epsilon_priority
        max_pts = cfg.max_points_face if pri > 0.35 else int(cfg.max_points_per_path * 1.5)
        approx = simplify_contour(c, eps, max_pts, closed=True)
        if len(approx) < 3:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        pts = contour_to_points(approx)
        layer = force_layer if force_layer is not None else role_layer(role, pri, area)
        parts.append(VectorPart(
            part_id=f"{prefix}_{counter:05d}",
            category=role,
            role=role,
            layer=layer + int(pri * 12),
            fill=fill_hex,
            stroke=None,
            stroke_width=0.0,
            closed=True,
            points=pts,
            bbox=(x, y, x + bw, y + bh),
            area=area,
            priority_score=pri,
            keep_score=part_keep_score(role, area, pri, len(pts)) + 0.8,
        ))
        counter += 1
    return parts


# =========================================================
# Line extraction: dark, boundary, face
# =========================================================

def label_boundary_mask(labels: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    h, w = labels.shape[:2]
    valid = labels >= 0
    boundary = np.zeros((h, w), dtype=np.uint8)
    right = np.zeros_like(boundary)
    down = np.zeros_like(boundary)
    right[:, :-1] = ((labels[:, :-1] != labels[:, 1:]) & valid[:, :-1] & valid[:, 1:]).astype(np.uint8) * 255
    down[:-1, :] = ((labels[:-1, :] != labels[1:, :]) & valid[:-1, :] & valid[1:, :]).astype(np.uint8) * 255
    boundary = cv2.bitwise_or(right, down)
    boundary = cv2.bitwise_and(boundary, fg_mask)
    return boundary


def extract_line_parts(
    img_bgr: np.ndarray,
    labels: np.ndarray,
    fg_mask: np.ndarray,
    subject_mask: np.ndarray,
    face_detail_mask: np.ndarray,
    accent_mask: np.ndarray,
    light_subject_mask: np.ndarray,
    cfg: AppConfig,
    logs: List[str],
) -> List[VectorPart]:
    if not cfg.enable_outline:
        return []

    logs.append("Line extraction: dark + boundary + face detail")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    parts: List[VectorPart] = []

    if cfg.enable_boundary_lines:
        boundary = label_boundary_mask(labels, fg_mask)
        # Keep boundary lines safely inside the foreground.
        # OpenCV bitwise_* functions require uint8 masks, not bool arrays.
        inner_fg = (erode_mask(fg_mask, 1) > 0).astype(np.uint8) * 255
        boundary = cv2.bitwise_and(boundary, inner_fg)
        boundary = dilate_mask(boundary, 2)
        boundary = clean_mask(boundary, open_k=2, close_k=2)
        parts.extend(extract_parts_from_mask(
            mask=boundary,
            fill_hex=None,
            stroke_hex="#5B5B5B",
            stroke_width=cfg.boundary_line_width,
            role="outline_boundary",
            part_prefix="boundary_line",
            closed=False,
            cfg=cfg,
            subject_mask=subject_mask,
            face_detail_mask=face_detail_mask,
            accent_mask=accent_mask,
            min_area_override=cfg.min_area_face,
            epsilon_override=cfg.contour_epsilon_priority,
            max_points_override=cfg.max_points_per_path,
        ))

        if cfg.light_boundary_line_boost and np.count_nonzero(light_subject_mask) > 0:
            light_boundary = cv2.bitwise_and(boundary, dilate_mask(light_subject_mask, 2))
            light_boundary = dilate_mask(light_boundary, 1)
            parts.extend(extract_parts_from_mask(
                mask=light_boundary,
                fill_hex=None,
                stroke_hex="#444444",
                stroke_width=max(cfg.boundary_line_width + 0.15, cfg.boundary_line_width * 1.25),
                role="outline_boundary",
                part_prefix="light_clothes_boundary",
                closed=False,
                cfg=cfg,
                subject_mask=subject_mask,
                face_detail_mask=face_detail_mask,
                accent_mask=accent_mask,
                min_area_override=max(3, cfg.min_area_face // 2),
                epsilon_override=cfg.contour_epsilon_priority,
                max_points_override=cfg.max_points_per_path,
            ))

    if cfg.enable_dark_lines:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), cfg.canny_low, cfg.canny_high)
        dark = ((gray < cfg.line_darkness_threshold).astype(np.uint8) * 255)
        dark_edge = cv2.bitwise_and(dilate_mask(edges, 2), dilate_mask(dark, 2))
        dark_edge = cv2.bitwise_and(dark_edge, fg_mask)
        dark_edge = clean_mask(dark_edge, open_k=2, close_k=2)
        parts.extend(extract_parts_from_mask(
            mask=dark_edge,
            fill_hex=None,
            stroke_hex=cfg.outline_color,
            stroke_width=cfg.dark_line_width,
            role="outline_dark",
            part_prefix="dark_line",
            closed=False,
            cfg=cfg,
            subject_mask=subject_mask,
            face_detail_mask=face_detail_mask,
            accent_mask=accent_mask,
            min_area_override=cfg.min_area_face,
            epsilon_override=cfg.contour_epsilon_priority,
            max_points_override=cfg.max_points_per_path,
        ))

    if cfg.enable_face_lines and np.count_nonzero(face_detail_mask) > 0:
        face_edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), max(25, cfg.canny_low - 12), max(80, cfg.canny_high - 18))
        face_edges = cv2.bitwise_and(dilate_mask(face_edges, 2), dilate_mask(face_detail_mask, 2))
        # Keep both dark and boundary-like face lines; face is allowed to be detailed.
        face_edges = clean_mask(face_edges, open_k=2, close_k=2)
        parts.extend(extract_parts_from_mask(
            mask=face_edges,
            fill_hex=None,
            stroke_hex=cfg.outline_color,
            stroke_width=cfg.face_line_width,
            role="outline_face",
            part_prefix="face_line",
            closed=False,
            cfg=cfg,
            subject_mask=subject_mask,
            face_detail_mask=face_detail_mask,
            accent_mask=accent_mask,
            min_area_override=max(3, cfg.min_area_face // 2),
            epsilon_override=cfg.contour_epsilon_face,
            max_points_override=cfg.max_points_face,
        ))

    logs.append(f"Line parts before budget={len(parts)}")
    return parts


# =========================================================
# Quality budget selection
# =========================================================

def budget_key_for_role(role: str) -> str:
    if role == "underpaint":
        return "underpaint"
    if role == "base":
        return "base"
    if role == "light":
        return "light"
    if role == "skin":
        return "skin"
    if role == "dark":
        return "dark"
    if role == "shadow":
        return "shadow"
    if role == "accent":
        return "accent"
    if role in ("outline_boundary", "outline_dark"):
        return "outline"
    if role in ("outline_face", "face_detail"):
        return "face"
    return "base"


def part_center_cell(p: VectorPart, shape_hw: Tuple[int, int], grid: int = 3) -> Tuple[int, int]:
    h, w = shape_hw
    x1, y1, x2, y2 = p.bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    gx = int(clamp(int(cx / max(1, w) * grid), 0, grid - 1))
    gy = int(clamp(int(cy / max(1, h) * grid), 0, grid - 1))
    return gx, gy


def apply_spatial_coverage(
    kept: List[VectorPart],
    discarded: List[VectorPart],
    shape_hw: Tuple[int, int],
    cfg: AppConfig,
    logs: List[str],
) -> List[VectorPart]:
    if not cfg.enable_spatial_coverage or not discarded:
        return kept

    kept_ids = {id(p) for p in kept}
    cell_counts: Dict[Tuple[int, int], int] = {}
    for p in kept:
        cell = part_center_cell(p, shape_hw, grid=3)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1

    added = 0
    discarded_sorted = sorted(discarded, key=lambda p: (p.keep_score, p.priority_score, p.area), reverse=True)
    for cell_y in range(3):
        for cell_x in range(3):
            cell = (cell_x, cell_y)
            base_need = cfg.spatial_min_parts_per_cell
            # V6.1: lower cells often lose legs / lower accessories, so they can receive
            # a small extra minimum without sacrificing face priority.
            if cell_y == 2:
                base_need += cfg.bottom_spatial_boost
            need = max(0, base_need - cell_counts.get(cell, 0))
            if need <= 0:
                continue
            for p in discarded_sorted:
                if added >= cfg.spatial_extra_cap:
                    break
                if id(p) in kept_ids:
                    continue
                if part_center_cell(p, shape_hw, grid=3) != cell:
                    continue
                # Avoid adding tiny non-visible underpaint/background-ish fragments.
                if p.keep_score < 2.0 and p.area < 24:
                    continue
                kept.append(p)
                kept_ids.add(id(p))
                added += 1
                cell_counts[cell] = cell_counts.get(cell, 0) + 1
                need -= 1
                if need <= 0:
                    break
    if added:
        logs.append(f"Spatial coverage added parts={added}")
    return kept


def apply_quality_budget(parts: List[VectorPart], cfg: AppConfig, logs: List[str], shape_hw: Tuple[int, int]) -> List[VectorPart]:
    groups: Dict[str, List[VectorPart]] = {}
    for p in parts:
        key = budget_key_for_role(p.role)
        groups.setdefault(key, []).append(p)

    budget_map = asdict(cfg.budget)
    kept: List[VectorPart] = []
    discarded: List[VectorPart] = []
    for key, group in groups.items():
        group.sort(key=lambda p: p.keep_score, reverse=True)
        cap = int(budget_map.get(key, 80))
        before = len(group)
        kept_group = group[:cap]
        kept.extend(kept_group)
        discarded.extend(group[cap:])
        if before > cap:
            logs.append(f"Budget cap {key}: {before} -> {cap}")

    kept = apply_spatial_coverage(kept, discarded, shape_hw, cfg, logs)

    # Final safety cap to prevent mobile crashes. Preserve priority ordering.
    max_total = sum(asdict(cfg.budget).values()) + (cfg.spatial_extra_cap if cfg.enable_spatial_coverage else 0)
    kept.sort(key=lambda p: (p.keep_score, p.priority_score, p.area), reverse=True)
    if len(kept) > max_total:
        logs.append(f"Final total cap: {len(kept)} -> {max_total}")
        kept = kept[:max_total]

    kept.sort(key=lambda p: (p.layer, -p.area))
    logs.append(f"Parts after quality budget={len(kept)}")
    return kept


# =========================================================
# Rendering previews
# =========================================================

def checker_background(h: int, w: int, tile: int = 24) -> np.ndarray:
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    c1 = np.array([235, 235, 235], dtype=np.uint8)
    c2 = np.array([205, 205, 205], dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            bg[y:y+tile, x:x+tile] = c1 if ((x // tile + y // tile) % 2 == 0) else c2
    return bg


def render_vector_preview(shape_hw: Tuple[int, int], parts: List[VectorPart], bg_mode: str = "white") -> np.ndarray:
    h, w = shape_hw
    if bg_mode == "dark":
        canvas = np.full((h, w, 3), 32, dtype=np.uint8)
    elif bg_mode == "checker":
        canvas = checker_background(h, w)
    else:
        canvas = np.full((h, w, 3), 245, dtype=np.uint8)

    for p in sorted(parts, key=lambda q: (q.layer, -q.area)):
        if not p.points:
            continue
        pts = np.array(p.points, dtype=np.int32).reshape((-1, 1, 2))
        if p.fill and len(pts) >= 3:
            cv2.fillPoly(canvas, [pts], color=hex_to_bgr(p.fill), lineType=cv2.LINE_AA)
        if p.stroke and len(pts) >= 2:
            thickness = max(1, int(round(p.stroke_width)))
            cv2.polylines(
                canvas,
                [pts],
                isClosed=p.closed,
                color=hex_to_bgr(p.stroke),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
    return bgr_to_rgb(canvas)


# =========================================================
# SVG export
# =========================================================

def points_to_svg_path(points: List[Tuple[int, int]], closed: bool) -> str:
    if not points:
        return ""
    parts = [f"M {points[0][0]} {points[0][1]}"]
    for x, y in points[1:]:
        parts.append(f"L {x} {y}")
    if closed:
        parts.append("Z")
    return " ".join(parts)


def part_to_svg_element(p: VectorPart) -> str:
    d = points_to_svg_path(p.points, p.closed)
    if not d:
        return ""
    attrs = [f'd="{d}"']
    attrs.append(f'fill="{p.fill}"' if p.fill else 'fill="none"')
    if p.stroke:
        attrs.append(f'stroke="{p.stroke}"')
        attrs.append(f'stroke-width="{p.stroke_width:.2f}"')
        attrs.append('stroke-linecap="round"')
        attrs.append('stroke-linejoin="round"')
    attrs.append(f'data-role="{p.role}"')
    attrs.append(f'data-id="{p.part_id}"')
    return "<path " + " ".join(attrs) + " />"


def build_svg_document(parts: List[VectorPart], width: int, height: int, title: str) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{title}</title>",
    ]
    for p in sorted(parts, key=lambda q: (q.layer, -q.area)):
        el = part_to_svg_element(p)
        if el:
            lines.append(el)
    lines.append("</svg>")
    return "\n".join(lines)


def split_parts_by_svg_size(parts: List[VectorPart], width: int, height: int, limit_bytes: int, prefix: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    ordered = sorted(parts, key=lambda p: (p.layer, -p.area))
    chunks: List[List[VectorPart]] = []
    current: List[VectorPart] = []

    for p in ordered:
        candidate = current + [p]
        svg = build_svg_document(candidate, width, height, f"{prefix}_chunk")
        if current and len(svg.encode("utf-8")) > limit_bytes:
            chunks.append(current)
            current = [p]
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

    by_role: Dict[str, List[VectorPart]] = {}
    for p in parts:
        by_role.setdefault(p.role, []).append(p)
    role_order = ["underpaint", "base", "light", "dark", "skin", "shadow", "accent", "outline_boundary", "outline_dark", "outline_face"]
    for role in role_order:
        role_parts = by_role.get(role, [])
        if role_parts:
            files[f"role_{role}.svg"] = build_svg_document(role_parts, w, h, f"role_{role}")

    fill_parts = [p for p in parts if p.fill]
    line_parts = [p for p in parts if p.stroke]
    if fill_parts:
        files["01_all_fills.svg"] = build_svg_document(fill_parts, w, h, "all_fills")
    if line_parts:
        files["02_all_lines.svg"] = build_svg_document(line_parts, w, h, "all_lines")

    limit_bytes = int(cfg.svg_limit_kb * 1024)
    files.update(split_parts_by_svg_size(parts, w, h, limit_bytes, "gt7_part"))
    logs.append(f"SVG files generated={len(files)}")
    return files


# =========================================================
# ZIP / debug
# =========================================================

def parts_to_debug_json(parts: List[VectorPart]) -> List[Dict]:
    return [
        {
            "part_id": p.part_id,
            "role": p.role,
            "layer": p.layer,
            "fill": p.fill,
            "stroke": p.stroke,
            "stroke_width": p.stroke_width,
            "closed": p.closed,
            "bbox": p.bbox,
            "area": round(float(p.area), 3),
            "priority_score": round(float(p.priority_score), 4),
            "keep_score": round(float(p.keep_score), 4),
            "point_count": len(p.points),
        }
        for p in parts
    ]


def build_zip_bundle(result: ProcessResult, cfg: AppConfig, source_base: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, text in result.svg_files.items():
            zf.writestr(name, text)

        zf.writestr("preview_original.png", png_bytes_from_rgb(result.original_rgb))
        zf.writestr("preview_processed.png", png_bytes_from_rgb(result.processed_rgb))
        zf.writestr("preview_color_source.png", png_bytes_from_rgb(result.color_source_rgb))
        zf.writestr("mask_alpha.png", png_bytes_from_rgb(draw_mask_preview(result.alpha_mask)))
        zf.writestr("mask_line_suppression.png", png_bytes_from_rgb(draw_mask_preview(result.line_suppression_mask)))
        zf.writestr("mask_foreground.png", png_bytes_from_rgb(draw_mask_preview(result.foreground_mask)))
        zf.writestr("mask_subject.png", png_bytes_from_rgb(draw_mask_preview(result.subject_mask)))
        zf.writestr("mask_face_detail.png", png_bytes_from_rgb(draw_mask_preview(result.face_detail_mask)))
        zf.writestr("mask_skin.png", png_bytes_from_rgb(draw_mask_preview(result.skin_mask)))
        zf.writestr("mask_accent.png", png_bytes_from_rgb(draw_mask_preview(result.accent_mask)))
        zf.writestr("mask_light_subject.png", png_bytes_from_rgb(draw_mask_preview(result.light_subject_mask)))
        zf.writestr("mask_dark_mass.png", png_bytes_from_rgb(draw_mask_preview(result.dark_mass_mask)))
        zf.writestr("preview_quantized.png", png_bytes_from_rgb(result.quantized_rgb))
        zf.writestr("preview_vector_white.png", png_bytes_from_rgb(result.vector_preview_white))
        zf.writestr("preview_vector_dark.png", png_bytes_from_rgb(result.vector_preview_dark))
        zf.writestr("preview_vector_checker.png", png_bytes_from_rgb(result.vector_preview_checker))

        debug = {
            "source": source_base,
            "config": asdict(cfg),
            "stats": result.stats,
            "logs": result.logs,
            "parts": parts_to_debug_json(result.parts),
        }
        zf.writestr("debug_summary.json", json.dumps(debug, ensure_ascii=False, indent=2))
    buf.seek(0)
    return buf.getvalue()


# =========================================================
# Pipeline
# =========================================================

def process_image_pipeline(img_rgb: np.ndarray, alpha: np.ndarray, cfg: AppConfig) -> ProcessResult:
    logs: List[str] = []
    original_rgb = img_rgb.copy()
    img_bgr = rgb_to_bgr(img_rgb)

    processed_bgr = preprocess_image(img_bgr, cfg, logs)
    shape_hw = processed_bgr.shape[:2]
    alpha_mask = build_alpha_mask(alpha, shape_hw, cfg, logs)

    fg_mask = build_foreground_mask(processed_bgr, cfg, logs)
    fg_mask = merge_alpha_foreground(fg_mask, alpha_mask, logs)

    # Separate color and line sources. Fill quantization uses color_source,
    # while line extraction keeps using processed_bgr.
    line_suppression_mask = build_line_suppression_mask(processed_bgr, fg_mask, cfg, logs)
    color_source_bgr = build_color_source(processed_bgr, fg_mask, line_suppression_mask, cfg, logs)

    subject_mask = build_subject_mask(fg_mask, logs)
    skin_mask = build_skin_mask(color_source_bgr, fg_mask, subject_mask, logs) if cfg.protect_skin else np.zeros_like(fg_mask)
    accent_mask = build_accent_mask(color_source_bgr, fg_mask, logs) if cfg.protect_accents else np.zeros_like(fg_mask)
    light_subject_mask = build_light_subject_mask(color_source_bgr, fg_mask, cfg, logs) if cfg.protect_light_subject else np.zeros_like(fg_mask)
    dark_mass_mask = build_dark_mass_mask(processed_bgr, fg_mask, logs)
    face_detail_mask = build_face_detail_mask(processed_bgr, fg_mask, subject_mask, skin_mask, accent_mask, cfg, logs)

    quant_bgr, labels, centers = quantize_foreground(color_source_bgr, fg_mask, cfg.num_colors, logs)

    underpaint_parts = extract_underpaint_parts(
        img_bgr=color_source_bgr,
        fg_mask=fg_mask,
        subject_mask=subject_mask,
        face_detail_mask=face_detail_mask,
        accent_mask=accent_mask,
        light_subject_mask=light_subject_mask,
        cfg=cfg,
        logs=logs,
    )

    fill_parts = extract_fill_parts(
        img_bgr=color_source_bgr,
        labels=labels,
        centers=centers,
        fg_mask=fg_mask,
        subject_mask=subject_mask,
        face_detail_mask=face_detail_mask,
        skin_mask=skin_mask,
        accent_mask=accent_mask,
        light_subject_mask=light_subject_mask,
        dark_mass_mask=dark_mass_mask,
        cfg=cfg,
        logs=logs,
    )
    line_parts = extract_line_parts(
        img_bgr=processed_bgr,
        labels=labels,
        fg_mask=fg_mask,
        subject_mask=subject_mask,
        face_detail_mask=face_detail_mask,
        accent_mask=accent_mask,
        light_subject_mask=light_subject_mask,
        cfg=cfg,
        logs=logs,
    )

    parts = apply_quality_budget(underpaint_parts + fill_parts + line_parts, cfg, logs, shape_hw=shape_hw)

    vector_white = render_vector_preview(shape_hw, parts, bg_mode="white")
    vector_dark = render_vector_preview(shape_hw, parts, bg_mode="dark")
    vector_checker = render_vector_preview(shape_hw, parts, bg_mode="checker")
    svg_files = generate_svg_files(parts, shape_hw, cfg, logs)

    role_counts: Dict[str, int] = {}
    for p in parts:
        role_counts[p.role] = role_counts.get(p.role, 0) + 1

    stats = {
        "image_width": int(shape_hw[1]),
        "image_height": int(shape_hw[0]),
        "foreground_pixels": int(np.count_nonzero(fg_mask)),
        "alpha_foreground_pixels": int(np.count_nonzero(alpha_mask)),
        "line_suppression_pixels": int(np.count_nonzero(line_suppression_mask)),
        "subject_pixels": int(np.count_nonzero(subject_mask)),
        "face_detail_pixels": int(np.count_nonzero(face_detail_mask)),
        "skin_pixels": int(np.count_nonzero(skin_mask)),
        "accent_pixels": int(np.count_nonzero(accent_mask)),
        "light_subject_pixels": int(np.count_nonzero(light_subject_mask)),
        "dark_mass_pixels": int(np.count_nonzero(dark_mass_mask)),
        "underpaint_parts_before_budget": len(underpaint_parts),
        "fill_parts_before_budget": len(fill_parts),
        "line_parts_before_budget": len(line_parts),
        "total_parts_after_budget": len(parts),
        "total_points": int(sum(len(p.points) for p in parts)),
        "role_counts": role_counts,
        "svg_file_count": len(svg_files),
        "svg_sizes_bytes": {name: len(text.encode("utf-8")) for name, text in svg_files.items()},
    }

    return ProcessResult(
        original_rgb=original_rgb,
        processed_rgb=bgr_to_rgb(processed_bgr),
        color_source_rgb=bgr_to_rgb(color_source_bgr),
        alpha_mask=alpha_mask,
        line_suppression_mask=line_suppression_mask,
        foreground_mask=fg_mask,
        subject_mask=subject_mask,
        face_detail_mask=face_detail_mask,
        skin_mask=skin_mask,
        accent_mask=accent_mask,
        light_subject_mask=light_subject_mask,
        dark_mass_mask=dark_mass_mask,
        quantized_rgb=bgr_to_rgb(quant_bgr),
        vector_preview_white=vector_white,
        vector_preview_dark=vector_dark,
        vector_preview_checker=vector_checker,
        parts=parts,
        svg_files=svg_files,
        stats=stats,
        logs=logs,
    )


# =========================================================
# Streamlit UI
# =========================================================

st.title("GT7 / Forza-style Layered SVG Vectorizer V6.1")
st.caption("V6.1：V6の安全な線消しを維持しつつ、白服・下地・服境界線・下側保持を調整できる検証版。")

with st.sidebar:
    st.header("Settings")
    mode_name = st.selectbox("Mode", list(PRESETS.keys()), index=1)
    base_cfg = PRESETS[mode_name]

    num_colors = st.slider("Number of colors", 8, 32, base_cfg.num_colors, 1)
    max_dim = st.slider("Max processing dimension", 900, 2600, base_cfg.max_dim, 100)

    remove_background = st.checkbox("Remove border-connected background", value=base_cfg.remove_background)
    protect_skin = st.checkbox("Protect skin-like regions", value=base_cfg.protect_skin)
    protect_accents = st.checkbox("Protect auto accent colors", value=base_cfg.protect_accents)
    protect_light_subject = st.checkbox("Protect foreground light/white regions", value=base_cfg.protect_light_subject)

    enable_outline = st.checkbox("Enable line layers", value=base_cfg.enable_outline)
    enable_boundary_lines = st.checkbox("Boundary lines from color regions", value=base_cfg.enable_boundary_lines)
    enable_dark_lines = st.checkbox("Dark ink/line detection", value=base_cfg.enable_dark_lines)
    enable_face_lines = st.checkbox("Face/detail line boost", value=base_cfg.enable_face_lines)
    enable_alpha_mask = st.checkbox("Use PNG alpha / transparency", value=base_cfg.enable_alpha_mask)
    enable_line_inpaint = st.checkbox("Line-suppressed color source", value=base_cfg.enable_line_inpaint)
    enable_underpaint = st.checkbox("Add underpaint seam-hiding layer", value=base_cfg.enable_underpaint)
    enable_spatial_coverage = st.checkbox("Protect lower/sides with spatial coverage", value=base_cfg.enable_spatial_coverage)
    smooth_paths = st.checkbox("Light path smoothing", value=base_cfg.smooth_paths)

    underpaint_style = st.selectbox(
        "Underpaint style",
        ["Off", "Minimal foreground", "Subject core only", "Light clothes only"],
        index=["Off", "Minimal foreground", "Subject core only", "Light clothes only"].index(base_cfg.underpaint_style)
    )
    underpaint_expand = st.slider("Underpaint expansion", 0, 4, base_cfg.underpaint_expand, 1)
    light_mask_erode = st.slider("Light/white mask shrink", 1, 5, base_cfg.light_mask_erode, 1)
    light_boundary_line_boost = st.checkbox("Boost light-clothes boundary lines", value=base_cfg.light_boundary_line_boost)

    svg_limit_kb = st.slider("GT7 target part size (KB)", 8.0, 15.0, float(base_cfg.svg_limit_kb), 0.5)

    with st.expander("Quality / simplification"):
        min_area = st.slider("Min region area", 6, 180, base_cfg.min_area, 1)
        min_area_priority = st.slider("Min region area priority", 3, 80, base_cfg.min_area_priority, 1)
        min_area_face = st.slider("Min region area face/detail", 2, 40, base_cfg.min_area_face, 1)
        contour_epsilon = st.slider("Contour simplify", 0.0010, 0.0120, float(base_cfg.contour_epsilon), 0.0002)
        contour_epsilon_priority = st.slider("Contour simplify priority", 0.0005, 0.0060, float(base_cfg.contour_epsilon_priority), 0.0001)
        contour_epsilon_face = st.slider("Contour simplify face/detail", 0.0003, 0.0040, float(base_cfg.contour_epsilon_face), 0.0001)
        overlap_px = st.slider("Underpaint / overlap", 0, 4, base_cfg.overlap_px, 1)

    with st.expander("Line settings"):
        line_darkness_threshold = st.slider("Line darkness threshold", 55, 155, base_cfg.line_darkness_threshold, 1)
        canny_low = st.slider("Canny low", 20, 100, base_cfg.canny_low, 1)
        canny_high = st.slider("Canny high", 80, 220, base_cfg.canny_high, 1)
        boundary_line_width = st.slider("Boundary line width", 0.4, 2.0, float(base_cfg.boundary_line_width), 0.05)
        dark_line_width = st.slider("Dark line width", 0.4, 2.0, float(base_cfg.dark_line_width), 0.05)
        face_line_width = st.slider("Face line width", 0.4, 2.0, float(base_cfg.face_line_width), 0.05)
        outline_color = st.color_picker("Outline color", base_cfg.outline_color)
        line_suppression_mode = st.selectbox(
            "Line suppression mask mode",
            ["Off", "Thin dark edges only", "Thin dark + soft boundary", "Conservative face/detail", "Legacy broad dark mask"],
            index=["Off", "Thin dark edges only", "Thin dark + soft boundary", "Conservative face/detail", "Legacy broad dark mask"].index(base_cfg.line_suppression_mode)
        )
        color_source_mode = st.selectbox(
            "Color source cleanup mode",
            ["None / original", "Soft blend", "Median replace", "Telea inpaint", "NS inpaint"],
            index=["None / original", "Soft blend", "Median replace", "Telea inpaint", "NS inpaint"].index(base_cfg.color_source_mode)
        )
        line_inpaint_radius = st.slider("Color-source cleanup radius", 1, 5, base_cfg.line_inpaint_radius, 1)
        line_mask_dilate = st.slider("Line suppression dilation", 0, 3, base_cfg.line_mask_dilate, 1)
        line_suppression_strength = st.slider("Soft/median suppression strength", 0.10, 1.00, float(base_cfg.line_suppression_strength), 0.05)
        max_line_suppression_component_area = st.slider("Max line-suppression component area", 80, 6000, base_cfg.max_line_suppression_component_area, 20)

    with st.expander("Spatial / safety"): 
        spatial_min_parts_per_cell = st.slider("Min parts per 3x3 cell", 0, 12, base_cfg.spatial_min_parts_per_cell, 1)
        spatial_extra_cap = st.slider("Spatial extra part cap", 0, 120, base_cfg.spatial_extra_cap, 5)
        bottom_spatial_boost = st.slider("Bottom-row extra min parts", 0, 8, base_cfg.bottom_spatial_boost, 1)

    with st.expander("Role budgets"):
        underpaint_budget = st.slider("Underpaint budget", 0, 40, base_cfg.budget.underpaint, 1)
        base_budget = st.slider("Base budget", 20, 260, base_cfg.budget.base, 5)
        light_budget = st.slider("Light/white budget", 20, 220, base_cfg.budget.light, 5)
        skin_budget = st.slider("Skin budget", 10, 200, base_cfg.budget.skin, 5)
        dark_budget = st.slider("Dark surface budget", 20, 240, base_cfg.budget.dark, 5)
        shadow_budget = st.slider("Shadow budget", 20, 220, base_cfg.budget.shadow, 5)
        accent_budget = st.slider("Accent budget", 10, 200, base_cfg.budget.accent, 5)
        outline_budget = st.slider("Outline budget", 20, 360, base_cfg.budget.outline, 5)
        face_budget = st.slider("Face/detail budget", 20, 300, base_cfg.budget.face, 5)

    cfg = AppConfig(
        mode_name=mode_name,
        max_dim=max_dim,
        num_colors=num_colors,
        min_area=min_area,
        min_area_priority=min_area_priority,
        min_area_face=min_area_face,
        contour_epsilon=contour_epsilon,
        contour_epsilon_priority=contour_epsilon_priority,
        contour_epsilon_face=contour_epsilon_face,
        max_points_per_path=base_cfg.max_points_per_path,
        max_points_face=base_cfg.max_points_face,
        morph_open=base_cfg.morph_open,
        morph_close=base_cfg.morph_close,
        overlap_px=overlap_px,
        remove_background=remove_background,
        use_subject_priority=base_cfg.use_subject_priority,
        protect_skin=protect_skin,
        protect_accents=protect_accents,
        protect_light_subject=protect_light_subject,
        enable_outline=enable_outline,
        enable_boundary_lines=enable_boundary_lines,
        enable_dark_lines=enable_dark_lines,
        enable_face_lines=enable_face_lines,
        enable_alpha_mask=enable_alpha_mask,
        enable_line_inpaint=enable_line_inpaint,
        enable_underpaint=enable_underpaint,
        enable_spatial_coverage=enable_spatial_coverage,
        smooth_paths=smooth_paths,
        underpaint_style=underpaint_style,
        underpaint_expand=underpaint_expand,
        light_mask_erode=light_mask_erode,
        light_boundary_line_boost=light_boundary_line_boost,
        bottom_spatial_boost=bottom_spatial_boost,
        line_inpaint_radius=line_inpaint_radius,
        line_mask_dilate=line_mask_dilate,
        line_suppression_mode=line_suppression_mode,
        color_source_mode=color_source_mode,
        line_suppression_strength=line_suppression_strength,
        max_line_suppression_component_area=max_line_suppression_component_area,
        spatial_min_parts_per_cell=spatial_min_parts_per_cell,
        spatial_extra_cap=spatial_extra_cap,
        line_darkness_threshold=line_darkness_threshold,
        canny_low=canny_low,
        canny_high=canny_high,
        boundary_line_width=boundary_line_width,
        dark_line_width=dark_line_width,
        face_line_width=face_line_width,
        outline_color=outline_color,
        svg_limit_kb=svg_limit_kb,
        budget=RoleBudget(
            underpaint=underpaint_budget,
            base=base_budget,
            light=light_budget,
            skin=skin_budget,
            dark=dark_budget,
            shadow=shadow_budget,
            accent=accent_budget,
            outline=outline_budget,
            face=face_budget,
        ),
    )

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is not None:
    img_rgb, img_alpha = open_uploaded_image(uploaded)
    st.subheader("Original")
    st.image(img_rgb, use_container_width=True)

    if st.button("Generate SVGs", type="primary", use_container_width=True):
        try:
            with st.spinner("Processing image with V6.1 clothes/underpaint pipeline..."):
                result = process_image_pipeline(img_rgb, img_alpha, cfg)
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

    tabs = st.tabs(["Previews", "Masks", "Role Previews", "SVG Files", "Stats", "Logs / Debug", "Download"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Processed")
            st.image(result.processed_rgb, use_container_width=True)
            st.subheader("Color Source (line-suppressed)")
            st.image(result.color_source_rgb, use_container_width=True)
            st.subheader("Quantized")
            st.image(result.quantized_rgb, use_container_width=True)
        with c2:
            st.subheader("Vector Preview - White")
            st.image(result.vector_preview_white, use_container_width=True)
            st.subheader("Vector Preview - Dark")
            st.image(result.vector_preview_dark, use_container_width=True)
            st.subheader("Vector Preview - Checker")
            st.image(result.vector_preview_checker, use_container_width=True)

    with tabs[1]:
        m1, m2 = st.columns(2)
        with m1:
            st.subheader("Alpha")
            st.image(draw_mask_preview(result.alpha_mask), use_container_width=True)
            st.subheader("Line Suppression")
            st.image(draw_mask_preview(result.line_suppression_mask), use_container_width=True)
            st.subheader("Foreground")
            st.image(draw_mask_preview(result.foreground_mask), use_container_width=True)
            st.subheader("Subject")
            st.image(draw_mask_preview(result.subject_mask), use_container_width=True)
            st.subheader("Face / Detail")
            st.image(draw_mask_preview(result.face_detail_mask), use_container_width=True)
            st.subheader("Skin")
            st.image(draw_mask_preview(result.skin_mask), use_container_width=True)
        with m2:
            st.subheader("Accent")
            st.image(draw_mask_preview(result.accent_mask), use_container_width=True)
            st.subheader("Light / White Subject")
            st.image(draw_mask_preview(result.light_subject_mask), use_container_width=True)
            st.subheader("Dark Mass")
            st.image(draw_mask_preview(result.dark_mass_mask), use_container_width=True)

    with tabs[2]:
        st.subheader("Role previews")
        preview_roles = ["underpaint", "light", "dark", "skin", "accent", "outline_boundary", "outline_dark", "outline_face"]
        role_cols = st.columns(2)
        for i, role in enumerate(preview_roles):
            role_parts = [p for p in result.parts if p.role == role]
            if role_parts:
                with role_cols[i % 2]:
                    st.subheader(role)
                    st.image(render_vector_preview((result.processed_rgb.shape[0], result.processed_rgb.shape[1]), role_parts, bg_mode="checker"), use_container_width=True)

    with tabs[3]:
        st.subheader("Generated SVG files")
        for filename, svg_text in result.svg_files.items():
            size_kb = len(svg_text.encode("utf-8")) / 1024.0
            with st.expander(f"{filename} ({size_kb:.2f} KB)"):
                st.code(svg_text[:4000] + ("\n\n... (truncated)" if len(svg_text) > 4000 else ""), language="xml")
                st.download_button(
                    label=f"Download {filename}",
                    data=svg_text.encode("utf-8"),
                    file_name=filename,
                    mime="image/svg+xml",
                    use_container_width=True,
                    key=f"download_{filename}",
                )

    with tabs[4]:
        st.subheader("Stats")
        s = result.stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Parts", s["total_parts_after_budget"])
        c2.metric("Points", s["total_points"])
        c3.metric("Fill before", s["fill_parts_before_budget"])
        c4.metric("Line before", s["line_parts_before_budget"])
        st.json(s)

    with tabs[5]:
        st.subheader("Logs")
        for log in result.logs:
            st.write("-", log)
        st.subheader("Part Debug")
        st.json(parts_to_debug_json(result.parts[:180]))

    with tabs[6]:
        st.subheader("Download bundle")
        zip_bytes = build_zip_bundle(result, last_cfg, source_base)
        zip_name = f"{source_base}_layered_svg_bundle_v6_1.zip"
        st.download_button(
            label="Download ZIP bundle",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            use_container_width=True,
        )
        st.caption("ZIP includes SVGs, role-layer SVGs, GT7 split parts, previews, masks, and debug JSON.")
else:
    st.info("画像をアップロードして Generate SVGs を押してください。")
