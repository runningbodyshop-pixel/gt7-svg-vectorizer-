from __future__ import annotations

import base64
import csv
import html
import io
import math
import re
import time
import zipfile
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# ============================================================
# GT7 SVG Inkscape Auto Builder
# Smartphone copy/paste edition
# ============================================================
# Goal:
# - Not just color-region tracing.
# - Build Inkscape-like layered SVGs:
#   underpaint -> clean color fills -> shadows/highlights -> lineart on top.
# - Split into many small SVG files for GT7 decal workflows.
# - Keep every path small enough for later split/optimization workflows.
# ============================================================

HARD_LIMIT_BYTES = 2 * 1024 * 1024
DEFAULT_PART_KB = 14
DEFAULT_PATH_KB = 11

PRESETS = {
    "安定 / 最初に試す": {
        "side": 1200,
        "colors": 72,
        "fill_simplify": 0.55,
        "line_simplify": 0.45,
        "min_area": 3,
        "line_min_area": 3,
        "decimals": 1,
        "tile": 256,
        "part_kb": 14,
        "path_kb": 11,
    },
    "高精度 / 1分前後": {
        "side": 1450,
        "colors": 104,
        "fill_simplify": 0.36,
        "line_simplify": 0.30,
        "min_area": 2,
        "line_min_area": 2,
        "decimals": 1,
        "tile": 224,
        "part_kb": 14,
        "path_kb": 11,
    },
    "線画優先 / くっきり": {
        "side": 1350,
        "colors": 80,
        "fill_simplify": 0.50,
        "line_simplify": 0.22,
        "min_area": 3,
        "line_min_area": 1,
        "decimals": 1,
        "tile": 224,
        "part_kb": 14,
        "path_kb": 11,
    },
    "軽量 / 落ちにくい": {
        "side": 1000,
        "colors": 48,
        "fill_simplify": 0.85,
        "line_simplify": 0.70,
        "min_area": 6,
        "line_min_area": 5,
        "decimals": 0,
        "tile": 320,
        "part_kb": 14,
        "path_kb": 11,
    },
}


@dataclass
class PathItem:
    name: str
    layer: str
    fill: Optional[str]
    stroke: Optional[str]
    stroke_width: float
    d: str
    bytes_len: int
    bbox: tuple[int, int, int, int]


@dataclass
class SvgFile:
    filename: str
    svg: str
    layer: str
    bytes_len: int
    path_count: int
    max_path_bytes: int
    note: str


# ============================================================
# Utility
# ============================================================
def clean_name(name: str) -> str:
    name = re.sub(r"\.[^.]+$", "", name)
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return name or "gt7_decal"


def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG", optimize=True)
    return b.getvalue()


def short_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = [int(max(0, min(255, x))) for x in rgb]
    s = f"{r:02x}{g:02x}{b:02x}"
    if s[0] == s[1] and s[2] == s[3] and s[4] == s[5]:
        return f"#{s[0]}{s[2]}{s[4]}"
    return f"#{s}"


def fmt_num(x: float, decimals: int) -> str:
    if decimals <= 0:
        return str(int(round(float(x))))
    out = f"{float(x):.{decimals}f}".rstrip("0").rstrip(".")
    return out.replace("-0", "0")


def safe_svg(svg: str) -> str:
    forbidden = [
        "script",
        "foreignObject",
        "image",
        "text",
        "filter",
        "mask",
        "clipPath",
        "pattern",
        "style",
        "defs",
    ]
    for tag in forbidden:
        svg = re.sub(rf"<\s*{tag}\b.*?<\s*/\s*{tag}\s*>", "", svg, flags=re.I | re.S)
        svg = re.sub(rf"<\s*{tag}\b[^>]*/\s*>", "", svg, flags=re.I | re.S)
    svg = re.sub(r"\s+", " ", svg)
    svg = svg.replace("> <", "><").strip()
    return svg


def checkerboard(size: tuple[int, int], cell: int = 14) -> Image.Image:
    w, h = size
    img = Image.new("RGBA", size, (246, 246, 246, 255))
    dr = ImageDraw.Draw(img)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            c = (225, 225, 225, 255) if ((x // cell) + (y // cell)) % 2 else (250, 250, 250, 255)
            dr.rectangle([x, y, x + cell - 1, y + cell - 1], fill=c)
    return img


def on_checker(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    bg = checkerboard(img.size)
    bg.alpha_composite(img, (0, 0))
    return bg


def image_to_data_url(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(img)).decode("ascii")


def svg_to_data_url(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def resize_for_work(img: Image.Image, side: int, enhance: bool) -> Image.Image:
    img = img.convert("RGBA")

    if enhance:
        rgb = img.convert("RGB")
        rgb = ImageOps.autocontrast(rgb)
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=115, threshold=3))
        img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))

    mx = max(img.size)
    if mx > side:
        scale = side / mx
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    return img


# ============================================================
# Image analysis
# ============================================================
def prepare_rgb_and_masks(img: Image.Image, alpha_threshold: int, white_bg: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]
    visible = alpha >= int(alpha_threshold)

    if white_bg:
        rgb = np.full((*alpha.shape, 3), 255, dtype=np.uint8)
        a = (alpha.astype(np.float32) / 255.0)[:, :, None]
        rgb = (rgba[:, :, :3].astype(np.float32) * a + rgb.astype(np.float32) * (1.0 - a)).astype(np.uint8)
        visible[:, :] = True
    else:
        rgb = rgba[:, :, :3].copy()

    return rgb, alpha, visible.astype(np.uint8)


def smooth_rgb_for_fills(rgb: np.ndarray, visible: np.ndarray, mode: str) -> np.ndarray:
    out = rgb.copy()
    if mode == "なし":
        return out
    if mode == "軽く":
        out = cv2.bilateralFilter(out, d=5, sigmaColor=30, sigmaSpace=20)
    elif mode == "強め":
        out = cv2.bilateralFilter(out, d=7, sigmaColor=45, sigmaSpace=30)
        out = cv2.medianBlur(out, 3)
    out[visible == 0] = 255
    return out


def quantize_pillow(rgb: np.ndarray, visible: np.ndarray, colors: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    # Fast and stable. Good default for Streamlit Cloud.
    img = Image.fromarray(rgb, "RGB")
    q = img.quantize(colors=int(colors), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    labels = np.array(q, dtype=np.int32)
    labels[visible == 0] = -1
    pal_raw = q.getpalette()[:768]
    used = sorted(int(x) for x in np.unique(labels) if int(x) >= 0)

    remapped = np.full_like(labels, -1)
    palette: list[tuple[int, int, int]] = []
    for new_i, old_i in enumerate(used):
        remapped[labels == old_i] = new_i
        palette.append(tuple(int(v) for v in pal_raw[old_i * 3 : old_i * 3 + 3]))
    return remapped, palette


def quantize_lab_sample(rgb: np.ndarray, visible: np.ndarray, colors: int, sample_max: int = 65000) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    # More anime-friendly color grouping than plain RGB, but slower.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    pix = lab[visible > 0].reshape(-1, 3)
    if len(pix) == 0:
        return np.full(visible.shape, -1, dtype=np.int32), []
    colors = int(max(2, min(colors, 192)))

    if len(pix) > sample_max:
        rng = np.random.default_rng(1234)
        idx = rng.choice(len(pix), size=sample_max, replace=False)
        sample = pix[idx]
    else:
        sample = pix
    sample32 = np.float32(sample)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 18, 1.0)
    _, _, centers = cv2.kmeans(sample32, colors, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.float32)

    labels = np.full(visible.shape, -1, dtype=np.int32)
    coords = np.argwhere(visible > 0)
    all_pix = lab[visible > 0].reshape(-1, 3).astype(np.float32)
    out_labels = np.empty(len(all_pix), dtype=np.int32)
    chunk = 90000
    for start in range(0, len(all_pix), chunk):
        data = all_pix[start : start + chunk]
        # squared distance to cluster centers
        d = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        out_labels[start : start + chunk] = np.argmin(d, axis=1).astype(np.int32)
    labels[coords[:, 0], coords[:, 1]] = out_labels

    # Convert centers back to RGB palette and sort by area desc.
    lab_centers = centers.reshape(-1, 1, 3).astype(np.uint8)
    rgb_centers = cv2.cvtColor(lab_centers, cv2.COLOR_LAB2RGB).reshape(-1, 3)
    counts = [(int((labels == i).sum()), i) for i in range(colors)]
    counts.sort(reverse=True)
    remapped = np.full_like(labels, -1)
    palette = []
    for new_i, (_, old_i) in enumerate(counts):
        if int((labels == old_i).sum()) == 0:
            continue
        remapped[labels == old_i] = new_i
        palette.append(tuple(int(v) for v in rgb_centers[old_i]))
    return remapped, palette


def build_quant_preview(labels: np.ndarray, palette: list[tuple[int, int, int]]) -> Image.Image:
    h, w = labels.shape
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for i, c in enumerate(palette):
        m = labels == i
        arr[m, 0] = c[0]
        arr[m, 1] = c[1]
        arr[m, 2] = c[2]
        arr[m, 3] = 255
    return Image.fromarray(arr, "RGBA")


def extract_lineart_mask(
    rgb: np.ndarray,
    visible: np.ndarray,
    method: str,
    dark_threshold: int,
    adaptive_block: int,
    canny_low: int,
    canny_high: int,
    close_px: int,
    min_area: int,
) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    visible_u8 = (visible > 0).astype(np.uint8) * 255

    if method == "暗色ベース":
        line = (gray <= int(dark_threshold)).astype(np.uint8) * 255
    elif method == "エッジベース":
        line = cv2.Canny(gray, int(canny_low), int(canny_high))
        if close_px > 0:
            k = np.ones((close_px * 2 + 1, close_px * 2 + 1), np.uint8)
            line = cv2.dilate(line, k, iterations=1)
    else:  # hybrid
        dark = (gray <= int(dark_threshold)).astype(np.uint8) * 255
        block = int(adaptive_block)
        if block % 2 == 0:
            block += 1
        block = max(9, block)
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 7)
        edges = cv2.Canny(gray, int(canny_low), int(canny_high))
        line = cv2.bitwise_or(dark, cv2.bitwise_and(adapt, cv2.dilate(edges, np.ones((3, 3), np.uint8))))

    line = cv2.bitwise_and(line, visible_u8)

    if close_px > 0:
        k = np.ones((close_px * 2 + 1, close_px * 2 + 1), np.uint8)
        line = cv2.morphologyEx(line, cv2.MORPH_CLOSE, k, iterations=1)

    line = clean_components(line, min_area=max(1, int(min_area)))
    return line


def clean_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    binary = (binary > 0).astype(np.uint8)
    if min_area <= 1:
        return binary * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros(binary.shape, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


# ============================================================
# Path generation
# ============================================================
def contour_bbox(contour: np.ndarray, offset_x: int, offset_y: int) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(contour)
    return (int(x + offset_x), int(y + offset_y), int(x + offset_x + w), int(y + offset_y + h))


def contour_to_path(
    contour: np.ndarray,
    offset_x: int,
    offset_y: int,
    simplify: float,
    min_area: int,
    decimals: int,
    max_path_bytes: int,
) -> Optional[tuple[str, tuple[int, int, int, int]]]:
    if contour is None or len(contour) < 3:
        return None
    if abs(cv2.contourArea(contour)) < float(min_area):
        return None

    eps = max(0.03, float(simplify))
    approx = contour
    path = None
    bbox = contour_bbox(contour, offset_x, offset_y)

    # Increase simplification only for oversized paths, instead of killing global quality.
    for _ in range(9):
        approx = cv2.approxPolyDP(contour, epsilon=eps, closed=True)
        if approx is None or len(approx) < 3:
            return None
        pts = approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] += offset_x
        pts[:, 1] += offset_y
        cmds = [f"M{fmt_num(pts[0,0], decimals)} {fmt_num(pts[0,1], decimals)}"]
        for x, y in pts[1:]:
            cmds.append(f"L{fmt_num(x, decimals)} {fmt_num(y, decimals)}")
        cmds.append("Z")
        path = "".join(cmds)
        if len(path.encode("utf-8")) <= max_path_bytes:
            break
        eps *= 1.35
        decimals = max(0, decimals - 1)

    if path is None or len(path) < 8:
        return None
    return path, bbox


def polyline_to_stroke_path(
    contour: np.ndarray,
    offset_x: int,
    offset_y: int,
    simplify: float,
    decimals: int,
    max_path_bytes: int,
) -> Optional[tuple[str, tuple[int, int, int, int]]]:
    if contour is None or len(contour) < 2:
        return None
    eps = max(0.05, float(simplify))
    bbox = contour_bbox(contour, offset_x, offset_y)
    out = None
    closed = bool(np.linalg.norm(contour[0, 0] - contour[-1, 0]) < 2.5)
    for _ in range(8):
        approx = cv2.approxPolyDP(contour, epsilon=eps, closed=closed)
        if approx is None or len(approx) < 2:
            return None
        pts = approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] += offset_x
        pts[:, 1] += offset_y
        cmds = [f"M{fmt_num(pts[0,0], decimals)} {fmt_num(pts[0,1], decimals)}"]
        for x, y in pts[1:]:
            cmds.append(f"L{fmt_num(x, decimals)} {fmt_num(y, decimals)}")
        if closed and len(pts) >= 3:
            cmds.append("Z")
        out = "".join(cmds)
        if len(out.encode("utf-8")) <= max_path_bytes:
            break
        eps *= 1.35
        decimals = max(0, decimals - 1)
    return (out, bbox) if out else None


def iter_tiles(width: int, height: int, tile_size: int, overlap: int) -> Iterable[tuple[int, int, int, int]]:
    tile_size = int(tile_size)
    if tile_size <= 0 or tile_size >= max(width, height):
        yield 0, 0, width, height
        return
    step = max(32, tile_size - overlap * 2)
    y = 0
    seen = set()
    while y < height:
        x = 0
        y0 = max(0, y - overlap)
        y1 = min(height, y + tile_size + overlap)
        while x < width:
            x0 = max(0, x - overlap)
            x1 = min(width, x + tile_size + overlap)
            key = (x0, y0, x1, y1)
            if key not in seen and x1 > x0 and y1 > y0:
                seen.add(key)
                yield x0, y0, x1, y1
            if x + step >= width and x != width:
                x = width
            else:
                x += step
        if y + step >= height and y != height:
            y = height
        else:
            y += step


def mask_to_fill_paths(
    mask: np.ndarray,
    layer: str,
    name_prefix: str,
    fill: str,
    simplify: float,
    min_area: int,
    decimals: int,
    tile_size: int,
    tile_overlap: int,
    max_path_bytes: int,
    max_paths: int,
) -> list[PathItem]:
    h, w = mask.shape
    out: list[PathItem] = []
    for x0, y0, x1, y1 in iter_tiles(w, h, tile_size, tile_overlap):
        crop = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
        if crop.max() == 0:
            continue
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Largest contours first, useful for stable layering.
        contours = sorted(contours, key=lambda c: abs(cv2.contourArea(c)), reverse=True)
        for c in contours:
            res = contour_to_path(
                c,
                offset_x=x0,
                offset_y=y0,
                simplify=simplify,
                min_area=min_area,
                decimals=decimals,
                max_path_bytes=max_path_bytes,
            )
            if not res:
                continue
            d, bbox = res
            out.append(PathItem(name_prefix, layer, fill, None, 0, d, len(d.encode("utf-8")), bbox))
            if len(out) >= max_paths:
                return out
    return out


def mask_to_stroke_paths(
    edge_mask: np.ndarray,
    layer: str,
    name_prefix: str,
    stroke: str,
    stroke_width: float,
    simplify: float,
    decimals: int,
    tile_size: int,
    tile_overlap: int,
    max_path_bytes: int,
    max_paths: int,
) -> list[PathItem]:
    h, w = edge_mask.shape
    out: list[PathItem] = []
    for x0, y0, x1, y1 in iter_tiles(w, h, tile_size, tile_overlap):
        crop = (edge_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
        if crop.max() == 0:
            continue
        contours, _ = cv2.findContours(crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
        for c in contours:
            if len(c) < 4:
                continue
            res = polyline_to_stroke_path(c, x0, y0, simplify, decimals, max_path_bytes)
            if not res:
                continue
            d, bbox = res
            out.append(PathItem(name_prefix, layer, None, stroke, float(stroke_width), d, len(d.encode("utf-8")), bbox))
            if len(out) >= max_paths:
                return out
    return out


# ============================================================
# Layer building
# ============================================================
def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return (mask > 0).astype(np.uint8) * 255
    k = np.ones((px * 2 + 1, px * 2 + 1), np.uint8)
    return cv2.dilate((mask > 0).astype(np.uint8) * 255, k, iterations=1)


def dominant_color(palette: list[tuple[int, int, int]], labels: np.ndarray) -> tuple[int, int, int]:
    if not palette:
        return (0, 0, 0)
    best = max(range(len(palette)), key=lambda i: int((labels == i).sum()))
    return palette[best]


def color_is_line_like(rgb: tuple[int, int, int], dark_cutoff: int) -> bool:
    r, g, b = [int(v) for v in rgb]
    return max(r, g, b) < dark_cutoff or (0.2126 * r + 0.7152 * g + 0.0722 * b) < dark_cutoff


def build_layers(
    rgb: np.ndarray,
    visible: np.ndarray,
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    cfg: dict,
) -> tuple[list[PathItem], Image.Image, np.ndarray]:
    h, w = labels.shape
    paths: list[PathItem] = []
    max_path_bytes = int(cfg["path_kb"] * 1024)
    tile = int(cfg["tile"])
    tile_overlap = int(cfg["tile_overlap"])
    max_paths_per_layer = int(cfg["max_paths_per_layer"])

    visible_mask = (visible > 0).astype(np.uint8) * 255

    # 00 Underpaint: whole silhouette, color selected from dominant fill color.
    if cfg["underpaint"]:
        under_rgb = dominant_color(palette, labels)
        under_fill = short_hex(under_rgb)
        under_mask = dilate_mask(visible_mask, int(cfg["underpaint_expand"]))
        paths.extend(
            mask_to_fill_paths(
                under_mask,
                layer="00_underpaint",
                name_prefix="underpaint",
                fill=under_fill,
                simplify=max(0.45, float(cfg["fill_simplify"]) * 1.3),
                min_area=max(2, int(cfg["min_area"])),
                decimals=int(cfg["decimals"]),
                tile_size=max(tile * 2, tile),
                tile_overlap=tile_overlap,
                max_path_bytes=max_path_bytes,
                max_paths=max_paths_per_layer,
            )
        )

    # 10 Fill layers: each color becomes its own grouped/chunked decal parts.
    counts = [(int((labels == i).sum()), i) for i in range(len(palette))]
    counts.sort(reverse=True)
    overlap_px = int(cfg["color_overlap"])
    visible_limit = dilate_mask(visible_mask, overlap_px)
    kernel = np.ones((overlap_px * 2 + 1, overlap_px * 2 + 1), np.uint8) if overlap_px > 0 else None

    for rank, (count, idx) in enumerate(counts):
        if count < int(cfg["min_area"]):
            continue
        color = palette[idx]
        if cfg["exclude_line_colors_from_fills"] and color_is_line_like(color, int(cfg["line_dark_threshold"]) + 20):
            continue
        mask = (labels == idx).astype(np.uint8) * 255
        if kernel is not None:
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.bitwise_and(mask, visible_limit)
        mask = clean_components(mask, int(cfg["min_area"]))
        fill = short_hex(color)
        layer_name = f"20_fills/color_{rank:03d}_{fill.replace('#','')}"
        paths.extend(
            mask_to_fill_paths(
                mask,
                layer=layer_name,
                name_prefix=f"fill_{rank:03d}",
                fill=fill,
                simplify=float(cfg["fill_simplify"]),
                min_area=int(cfg["min_area"]),
                decimals=int(cfg["decimals"]),
                tile_size=tile,
                tile_overlap=tile_overlap,
                max_path_bytes=max_path_bytes,
                max_paths=max_paths_per_layer,
            )
        )

    # 90 Lineart: black/dark ink fill-paths, placed on top.
    line_mask = extract_lineart_mask(
        rgb=rgb,
        visible=visible,
        method=cfg["line_method"],
        dark_threshold=int(cfg["line_dark_threshold"]),
        adaptive_block=int(cfg["adaptive_block"]),
        canny_low=int(cfg["canny_low"]),
        canny_high=int(cfg["canny_high"]),
        close_px=int(cfg["line_close"]),
        min_area=int(cfg["line_min_area"]),
    )

    if cfg["lineart_fill"]:
        line_fill = cfg["line_color"]
        paths.extend(
            mask_to_fill_paths(
                line_mask,
                layer="90_lineart_fill",
                name_prefix="lineart_fill",
                fill=line_fill,
                simplify=float(cfg["line_simplify"]),
                min_area=int(cfg["line_min_area"]),
                decimals=int(cfg["decimals"]),
                tile_size=tile,
                tile_overlap=tile_overlap,
                max_path_bytes=max_path_bytes,
                max_paths=max_paths_per_layer,
            )
        )

    # 91 Stroke reinforcement: optional. Clean strokes can sharpen thin details.
    if cfg["stroke_layer"]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, int(cfg["canny_low"]), int(cfg["canny_high"]))
        edge = cv2.bitwise_and(edge, (visible > 0).astype(np.uint8) * 255)
        edge = clean_components(edge, int(cfg["stroke_min_area"]))
        paths.extend(
            mask_to_stroke_paths(
                edge,
                layer="91_stroke_detail",
                name_prefix="stroke_detail",
                stroke=cfg["line_color"],
                stroke_width=float(cfg["stroke_width"]),
                simplify=max(0.15, float(cfg["line_simplify"]) * 1.1),
                decimals=int(cfg["decimals"]),
                tile_size=tile,
                tile_overlap=tile_overlap,
                max_path_bytes=max_path_bytes,
                max_paths=max_paths_per_layer,
            )
        )

    preview = build_quant_preview(labels, palette)
    return paths, preview, line_mask


# ============================================================
# SVG chunking and ZIP output
# ============================================================
def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def path_element(item: PathItem) -> str:
    d = html.escape(item.d, quote=True)
    if item.stroke:
        sw = fmt_num(item.stroke_width, 2)
        return f'<path fill="none" stroke="{item.stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" d="{d}"/>'
    return f'<path d="{d}"/>'


def make_svg_document(width: int, height: int, fill: Optional[str], stroke: Optional[str], items: list[PathItem]) -> str:
    parts = [svg_header(width, height)]
    if fill and not stroke:
        parts.append(f'<g fill="{fill}">')
        parts.extend(path_element(x) for x in items)
        parts.append("</g>")
    elif stroke and not fill:
        # Individual path already has stroke attributes. This keeps SVGO merge paths safer when merge is OFF.
        parts.extend(path_element(x) for x in items)
    else:
        parts.extend(path_element(x) for x in items)
    parts.append("</svg>")
    return safe_svg("".join(parts))


def chunk_paths_to_svgs(
    paths: list[PathItem],
    width: int,
    height: int,
    basename: str,
    part_kb: int,
    max_svg_count: int,
) -> list[SvgFile]:
    target = int(part_kb * 1024)
    files: list[SvgFile] = []

    # Keep layer and fill separated. This is easier to assemble in GT7 and keeps output predictable.
    groups: dict[tuple[str, Optional[str], Optional[str]], list[PathItem]] = {}
    for item in paths:
        key = (item.layer, item.fill, item.stroke)
        groups.setdefault(key, []).append(item)

    order_keys = sorted(groups.keys(), key=lambda k: k[0])
    seq = 0

    for layer, fill, stroke in order_keys:
        items = groups[(layer, fill, stroke)]
        chunk: list[PathItem] = []
        chunk_index = 1
        safe_layer = re.sub(r"[^A-Za-z0-9_/-]+", "_", layer).strip("_").replace("/", "_")
        for item in items:
            test = chunk + [item]
            svg = make_svg_document(width, height, fill, stroke, test)
            if chunk and len(svg.encode("utf-8")) > target:
                final_svg = make_svg_document(width, height, fill, stroke, chunk)
                seq += 1
                files.append(
                    SvgFile(
                        filename=f"{seq:03d}_{safe_layer}_{chunk_index:02d}.svg",
                        svg=final_svg,
                        layer=layer,
                        bytes_len=len(final_svg.encode("utf-8")),
                        path_count=len(chunk),
                        max_path_bytes=max((p.bytes_len for p in chunk), default=0),
                        note="",
                    )
                )
                chunk_index += 1
                chunk = [item]
                if len(files) >= max_svg_count:
                    return files
            else:
                chunk = test

        if chunk:
            final_svg = make_svg_document(width, height, fill, stroke, chunk)
            seq += 1
            files.append(
                SvgFile(
                    filename=f"{seq:03d}_{safe_layer}_{chunk_index:02d}.svg",
                    svg=final_svg,
                    layer=layer,
                    bytes_len=len(final_svg.encode("utf-8")),
                    path_count=len(chunk),
                    max_path_bytes=max((p.bytes_len for p in chunk), default=0),
                    note="",
                )
            )
            if len(files) >= max_svg_count:
                return files

    return files


def make_full_preview_svg(width: int, height: int, paths: list[PathItem], max_bytes: int = 900_000) -> Optional[str]:
    # Same-canvas layered preview. If it is too large, avoid browser crash.
    files = chunk_paths_to_svgs(paths, width, height, "preview", part_kb=2048, max_svg_count=9999)
    parts = [svg_header(width, height)]
    for f in files:
        body = re.sub(r"^<svg[^>]*>|</svg>$", "", f.svg)
        parts.append(body)
        if sum(len(x) for x in parts) > max_bytes:
            return None
    parts.append("</svg>")
    svg = safe_svg("".join(parts))
    if len(svg.encode("utf-8")) > max_bytes:
        return None
    return svg


def placement_guide(width: int, height: int, tile: int, overlap: int) -> Image.Image:
    img = checkerboard((width, height), cell=max(8, int(min(width, height) / 40)))
    dr = ImageDraw.Draw(img)
    # Draw a simple border and rough tile guide.
    dr.rectangle([0, 0, width - 1, height - 1], outline=(255, 0, 0, 255), width=2)
    if tile > 0 and tile < max(width, height):
        step = max(32, tile - overlap * 2)
        x = 0
        while x < width:
            dr.line([x, 0, x, height], fill=(0, 100, 255, 180), width=1)
            x += step
        y = 0
        while y < height:
            dr.line([0, y, width, y], fill=(0, 100, 255, 180), width=1)
            y += step
    dr.text((10, 10), f"canvas {width} x {height} / same viewBox for every SVG", fill=(0, 0, 0, 255))
    return img


def make_manifest_csv(files: list[SvgFile], width: int, height: int) -> str:
    s = io.StringIO()
    writer = csv.writer(s)
    writer.writerow(["filename", "layer", "bytes", "path_count", "max_path_bytes", "canvas_width", "canvas_height", "note"])
    for f in files:
        writer.writerow([f.filename, f.layer, f.bytes_len, f.path_count, f.max_path_bytes, width, height, f.note])
    return s.getvalue()


def make_readme(files: list[SvgFile], width: int, height: int, cfg: dict) -> str:
    over = [f for f in files if f.bytes_len > int(cfg["part_kb"] * 1024)]
    max_path = max((f.max_path_bytes for f in files), default=0)
    return f"""GT7 SVG Inkscape Auto Builder output

Canvas: {width} x {height}
SVG files: {len(files)}
Target per SVG: {cfg['part_kb']} KB
Target per path: {cfg['path_kb']} KB
Max path bytes: {max_path}
Files over target before SVGOMG: {len(over)}

Layer order in GT7:
1. 00_underpaint
2. 20_fills / color layers
3. 90_lineart_fill
4. 91_stroke_detail, if generated

Important SVGOMG settings:
- Prettify markup: OFF
- Remove metadata: ON
- Remove comments: ON
- Convert colors: ON
- Round/rewrite paths: ON
- Merge paths: OFF
- Remove viewBox: OFF

Every SVG uses the same width/height/viewBox. Place all parts at the same scale and position in GT7.
"""


def make_preview_html(files: list[SvgFile], width: int, height: int) -> str:
    # A local HTML preview overlaying all SVGs. This can be opened after downloading ZIP.
    imgs = []
    for f in files:
        data = base64.b64encode(f.svg.encode("utf-8")).decode("ascii")
        imgs.append(f'<img src="data:image/svg+xml;base64,{data}" title="{html.escape(f.filename)}">')
    body = "\n".join(imgs)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>GT7 SVG Preview</title>
<style>
body{{font-family:sans-serif;background:#eee;margin:16px}}
.stage{{position:relative;width:min(95vw,{width}px);aspect-ratio:{width}/{height};background:#fff;border:1px solid #999;overflow:hidden}}
.stage img{{position:absolute;left:0;top:0;width:100%;height:100%;object-fit:contain}}
</style></head><body>
<h2>GT7 SVG layered preview</h2>
<p>All SVGs are overlaid on the same canvas. Upload/place them in GT7 in manifest order.</p>
<div class="stage">{body}</div>
</body></html>"""


def build_zip(files: list[SvgFile], width: int, height: int, cfg: dict, guide: Image.Image) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.writestr(f.filename, f.svg)
        z.writestr("manifest.csv", make_manifest_csv(files, width, height))
        z.writestr("README.txt", make_readme(files, width, height, cfg))
        z.writestr("preview.html", make_preview_html(files, width, height))
        z.writestr("placement_guide.png", png_bytes(guide))
    return mem.getvalue()


# ============================================================
# Main conversion
# ============================================================
def convert_all(input_img: Image.Image, cfg: dict) -> dict:
    work = resize_for_work(input_img, int(cfg["side"]), bool(cfg["enhance"]))
    rgb, alpha, visible = prepare_rgb_and_masks(work, int(cfg["alpha"]), bool(cfg["white_bg"]))
    smooth_rgb = smooth_rgb_for_fills(rgb, visible, cfg["fill_smooth"])

    if cfg["color_method"] == "Lab k-means（高精度・遅め）":
        labels, palette = quantize_lab_sample(smooth_rgb, visible, int(cfg["colors"]))
    else:
        labels, palette = quantize_pillow(smooth_rgb, visible, int(cfg["colors"]))

    paths, quant_preview, line_mask = build_layers(smooth_rgb, visible, labels, palette, cfg)
    width, height = work.size
    files = chunk_paths_to_svgs(
        paths,
        width,
        height,
        basename="gt7",
        part_kb=int(cfg["part_kb"]),
        max_svg_count=int(cfg["max_svg_count"]),
    )
    guide = placement_guide(width, height, int(cfg["tile"]), int(cfg["tile_overlap"]))
    zip_bytes = build_zip(files, width, height, cfg, guide)
    preview_svg = make_full_preview_svg(width, height, paths, max_bytes=int(cfg["preview_max_kb"] * 1024)) if cfg["preview_svg"] else None

    return {
        "work": work,
        "rgb": rgb,
        "visible": visible,
        "labels": labels,
        "palette": palette,
        "paths": paths,
        "files": files,
        "quant_preview": quant_preview,
        "line_mask": Image.fromarray(line_mask, "L"),
        "guide": guide,
        "zip": zip_bytes,
        "preview_svg": preview_svg,
        "width": width,
        "height": height,
    }


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="GT7 SVG Inkscape Auto Builder", page_icon="🏁", layout="wide")
st.title("🏁 GT7 SVG Inkscape Auto Builder")
st.caption("手作業Inkscape風の工程に寄せた、レイヤー分解・線画上乗せ・GT7分割SVG出力版です。")

with st.expander("この版の考え方", expanded=True):
    st.markdown(
        """
この版は、単純な「色ごとトレース」ではなく、次の順でSVGを作ります。

`下塗りシルエット → 色面レイヤー → 線画塗りレイヤー → 必要ならstroke補強`

各SVGは同じキャンバス・同じviewBoxで出力されるので、GT7内で同じ位置に重ねて完成させる想定です。  
SVGOMGへ通す場合は、分割を維持するため **Merge paths はOFF** 推奨です。
"""
    )

uploaded = st.file_uploader("画像を選択", type=["png", "jpg", "jpeg", "webp"])

with st.sidebar:
    st.header("基本設定")
    preset_name = st.selectbox("品質プリセット", list(PRESETS.keys()) + ["手動設定"], index=0)
    base = PRESETS.get(preset_name, PRESETS["安定 / 最初に試す"]).copy()

    base["side"] = st.slider("処理サイズ 長辺px", 600, 1800, int(base["side"]), step=50)
    base["colors"] = st.slider("色面の色数", 16, 160, int(base["colors"]), step=4)
    base["color_method"] = st.selectbox("色解析方式", ["Pillow量子化（速い・安定）", "Lab k-means（高精度・遅め）"], index=0)
    base["fill_smooth"] = st.selectbox("塗り面のなめらか処理", ["なし", "軽く", "強め"], index=1)
    base["enhance"] = st.toggle("入力画像を軽く補正", True)
    base["white_bg"] = st.toggle("透明部分を白背景にする", False)
    base["alpha"] = st.slider("透明判定", 0, 255, 16)

    st.divider()
    st.header("Inkscape風レイヤー")
    base["underpaint"] = st.toggle("00 下塗りシルエット", True)
    base["underpaint_expand"] = st.slider("下塗りの外側拡張px", 0, 4, 1)
    base["lineart_fill"] = st.toggle("90 線画を塗りpathで作る", True)
    base["stroke_layer"] = st.toggle("91 stroke補強レイヤー（実験）", False)
    base["stroke_width"] = st.slider("stroke幅", 0.4, 3.0, 1.2, step=0.1)
    base["line_color"] = st.selectbox("線画色", ["#111", "#000", "#222", "#2b1b18"], index=0)
    base["exclude_line_colors_from_fills"] = st.toggle("黒っぽい色を塗り面から除外", True)

    st.divider()
    st.header("線画抽出")
    base["line_method"] = st.selectbox("線画抽出方式", ["ハイブリッド", "暗色ベース", "エッジベース"], index=0)
    base["line_dark_threshold"] = st.slider("黒線判定", 20, 180, 92)
    base["adaptive_block"] = st.slider("線画の局所判定サイズ", 9, 61, 29, step=2)
    base["canny_low"] = st.slider("エッジ弱", 10, 160, 55)
    base["canny_high"] = st.slider("エッジ強", 30, 260, 145)
    base["line_close"] = st.slider("線画の穴埋めpx", 0, 3, 1)
    base["line_min_area"] = st.slider("線画の小ゴミ削除", 1, 40, int(base["line_min_area"]))
    base["stroke_min_area"] = st.slider("stroke補強の小ゴミ削除", 1, 40, 5)

    st.divider()
    st.header("パス精度")
    base["fill_simplify"] = st.slider("塗りpath簡略化", 0.10, 3.0, float(base["fill_simplify"]), step=0.05)
    base["line_simplify"] = st.slider("線画path簡略化", 0.05, 2.0, float(base["line_simplify"]), step=0.05)
    base["min_area"] = st.slider("塗り面の小ゴミ削除", 1, 80, int(base["min_area"]))
    base["color_overlap"] = st.slider("色面の重ね描きpx", 0, 3, 1)
    base["decimals"] = st.slider("座標の小数桁", 0, 2, int(base["decimals"]))

    st.divider()
    st.header("GT7分割 / 安定化")
    base["part_kb"] = st.slider("各SVG目標KB", 4, 200, int(base["part_kb"]))
    base["path_kb"] = st.slider("1path最大KB", 2, 30, int(base["path_kb"]))
    base["tile"] = st.slider("内部タイル分割px", 96, 512, int(base["tile"]), step=32)
    base["tile_overlap"] = st.slider("タイル重なりpx", 0, 24, 4)
    base["max_svg_count"] = st.slider("最大SVGファイル数", 10, 240, 120, step=10)
    base["max_paths_per_layer"] = st.slider("1レイヤー最大path数", 200, 12000, 5000, step=200)

    st.divider()
    st.header("プレビュー")
    base["preview_svg"] = st.toggle("実SVGプレビューを表示", True)
    base["preview_max_kb"] = st.slider("プレビュー上限KB", 100, 1800, 750, step=50)

if uploaded is None:
    st.info("画像をアップロードしてください。")
    st.stop()

try:
    original = Image.open(uploaded).convert("RGBA")
except Exception as e:
    st.error(f"画像を開けませんでした: {e}")
    st.stop()

if st.button("Inkscape風SVGを作成", type="primary", use_container_width=True):
    started = time.time()
    with st.spinner("レイヤー解析・線画抽出・SVG分割中です…"):
        try:
            st.session_state.result = convert_all(original, base.copy())
            st.session_state.elapsed = time.time() - started
            st.session_state.filename = uploaded.name
            st.session_state.used = base.copy()
        except Exception as e:
            st.exception(e)
            st.stop()

if "result" not in st.session_state:
    st.subheader("元画像")
    st.image(on_checker(original), use_container_width=True)
    st.stop()

res = st.session_state.result
used = st.session_state.used
files: list[SvgFile] = res["files"]
paths: list[PathItem] = res["paths"]

max_file = max((f.bytes_len for f in files), default=0)
max_path = max((p.bytes_len for p in paths), default=0)
over_files = sum(1 for f in files if f.bytes_len > int(used["part_kb"] * 1024))
over_paths = sum(1 for p in paths if p.bytes_len > int(used["path_kb"] * 1024))

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("SVGファイル数", str(len(files)))
m2.metric("最大SVG", f"{max_file:,} B")
m3.metric("最大path", f"{max_path:,} B")
m4.metric("総path数", str(len(paths)))
m5.metric("ZIPサイズ", f"{len(res['zip']):,} B")
m6.metric("変換時間", f"{st.session_state.elapsed:.1f} 秒")

if over_files == 0:
    st.success("各SVGは目標サイズ内です。SVGOMG後はさらに軽くなる可能性があります。")
else:
    st.warning(f"目標KBを超えたSVGが {over_files} 個あります。各SVG目標KBを上げるか、色数/処理サイズを下げてください。")

if over_paths == 0:
    st.success("各pathは指定した最大KB以内です。")
else:
    st.warning(f"指定KBを超えたpathが {over_paths} 個あります。内部タイル分割pxを小さくするか、1path最大KBを上げてください。")

if st.session_state.elapsed > 75:
    st.warning("変換が重めです。次回は処理サイズ・色数を下げるか、Pillow量子化に戻してください。")

st.subheader("プレビュー")
left, mid, right = st.columns(3)
with left:
    st.markdown("**Original**")
    st.image(on_checker(res["work"]), use_container_width=True)
with mid:
    st.markdown("**Fill reference**")
    st.image(on_checker(res["quant_preview"]), use_container_width=True)
with right:
    st.markdown("**Line mask**")
    st.image(res["line_mask"], use_container_width=True)

if res["preview_svg"]:
    st.subheader("実SVG重ね合わせプレビュー")
    components.html(
        f"""
        <div style="background:#eee;border:1px solid #ccc;border-radius:12px;padding:12px;text-align:center;">
          <img src="{svg_to_data_url(res['preview_svg'])}" style="max-width:100%;height:auto;">
        </div>
        """,
        height=650,
        scrolling=True,
    )
else:
    st.info("実SVGプレビューはサイズ保護のため非表示です。ZIP内の preview.html で確認できます。")

st.subheader("保存")
base_name = clean_name(st.session_state.filename)
d1, d2, d3 = st.columns(3)
with d1:
    st.download_button(
        "分割SVG ZIPを保存",
        data=res["zip"],
        file_name=f"{base_name}_gt7_inkscape_auto.zip",
        mime="application/zip",
        use_container_width=True,
    )
with d2:
    st.download_button(
        "配置ガイドPNGを保存",
        data=png_bytes(res["guide"]),
        file_name=f"{base_name}_placement_guide.png",
        mime="image/png",
        use_container_width=True,
    )
with d3:
    if res["preview_svg"]:
        st.download_button(
            "重ね合わせSVGを保存",
            data=res["preview_svg"].encode("utf-8"),
            file_name=f"{base_name}_preview_all_layers.svg",
            mime="image/svg+xml",
            use_container_width=True,
        )
    else:
        st.button("重ね合わせSVGは未生成", disabled=True, use_container_width=True)

with st.expander("出力ファイル一覧"):
    rows = [
        {
            "filename": f.filename,
            "layer": f.layer,
            "bytes": f.bytes_len,
            "paths": f.path_count,
            "max_path_bytes": f.max_path_bytes,
        }
        for f in files
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

with st.expander("SVGOMG設定メモ"):
    st.markdown(
        """
おすすめ設定：

- Prettify markup：OFF
- Remove metadata：ON
- Remove comments：ON
- Convert colors：ON
- Round/rewrite paths：ON
- **Merge paths：OFF**
- Remove viewBox：OFF

`Merge paths` をONにすると、分けたpathやファイルが再結合されて、15KB制限に戻れなくなることがあります。
"""
    )
