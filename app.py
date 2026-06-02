from __future__ import annotations

import base64
import csv
import io
import math
import re
import time
import zipfile
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# ============================================================
# GT7 SVG Vectorizer - Split Decal Edition
# Smartphone copy/paste version
# ============================================================

GT7_SAFE_DEFAULT = 14 * 1024
HARD_LIMIT = 2 * 1024 * 1024

PRESETS = {
    "安定 / 30秒〜1分": {
        "colors": 72,
        "side": 1150,
        "simp": 0.55,
        "area": 2,
        "dec": 0,
        "cols": 3,
        "rows": 3,
        "overlap_pct": 4.0,
        "smooth_color": True,
        "max_parts": 40,
    },
    "高品質 / 分割前提": {
        "colors": 96,
        "side": 1350,
        "simp": 0.42,
        "area": 1,
        "dec": 1,
        "cols": 4,
        "rows": 4,
        "overlap_pct": 5.0,
        "smooth_color": True,
        "max_parts": 70,
    },
    "超高品質 / 重め": {
        "colors": 128,
        "side": 1550,
        "simp": 0.32,
        "area": 1,
        "dec": 1,
        "cols": 5,
        "rows": 5,
        "overlap_pct": 5.0,
        "smooth_color": True,
        "max_parts": 100,
    },
    "軽量 / 落ちにくい": {
        "colors": 48,
        "side": 950,
        "simp": 0.85,
        "area": 4,
        "dec": 0,
        "cols": 3,
        "rows": 3,
        "overlap_pct": 3.0,
        "smooth_color": True,
        "max_parts": 30,
    },
}


@dataclass
class Tile:
    idx: int
    x0: int
    y0: int
    x1: int
    y1: int
    level: int = 0
    kind: str = "color"


@dataclass
class SvgPart:
    filename: str
    svg: str
    size: int
    max_path_size: int
    path_count: int
    contour_count: int
    tile: Tile
    kind: str = "color"
    warning: str = ""


# -----------------------------
# basic helpers
# -----------------------------
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def short_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    s = f"{r:02x}{g:02x}{b:02x}"
    if s[0] == s[1] and s[2] == s[3] and s[4] == s[5]:
        return f"#{s[0]}{s[2]}{s[4]}"
    return f"#{s}"


def safe_svg(svg: str) -> str:
    # GT7/SVGOMG friendly: no css/image/text/filter/mask etc.
    for tag in [
        "script", "foreignObject", "image", "text", "filter", "mask",
        "clipPath", "pattern", "style", "defs", "metadata", "title", "desc",
    ]:
        svg = re.sub(rf"<\s*{tag}\b.*?<\s*/\s*{tag}\s*>", "", svg, flags=re.I | re.S)
        svg = re.sub(rf"<\s*{tag}\b[^>]*/\s*>", "", svg, flags=re.I | re.S)
    svg = re.sub(r"\s+", " ", svg)
    svg = svg.replace("> <", "><").strip()
    return svg


def fit_image(img: Image.Image, side: int, enhance: bool, smooth_color: bool) -> Image.Image:
    img = img.convert("RGBA")
    if enhance:
        rgb = ImageOps.autocontrast(img.convert("RGB"))
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=0.8, percent=110, threshold=3))
        img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))

    if max(img.size) > side:
        scale = side / max(img.size)
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.LANCZOS,
        )

    if smooth_color:
        rgba = np.array(img, dtype=np.uint8)
        rgb = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        # edge-preserving smoothing: reduces noisy tiny regions but keeps anime lines
        rgb = cv2.bilateralFilter(rgb, d=5, sigmaColor=35, sigmaSpace=35)
        rgba[:, :, :3] = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgba, "RGBA")

    return img


def checkerboard(size: tuple[int, int], cell: int = 12) -> Image.Image:
    w, h = size
    out = Image.new("RGBA", size, (245, 245, 245, 255))
    dr = ImageDraw.Draw(out)
    c1 = (252, 252, 252, 255)
    c2 = (222, 222, 222, 255)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            dr.rectangle(
                [x, y, x + cell - 1, y + cell - 1],
                fill=c1 if ((x // cell) + (y // cell)) % 2 == 0 else c2,
            )
    return out


def paste_on_checkerboard(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    bg = checkerboard(img.size)
    bg.alpha_composite(img, (0, 0))
    return bg


# -----------------------------
# color quantization
# -----------------------------
def quantize_image(
    img: Image.Image,
    n_colors: int,
    alpha_threshold: int,
    white_bg: bool,
) -> tuple[np.ndarray, list[tuple[int, int, int]], np.ndarray]:
    img = img.convert("RGBA")
    alpha = np.array(img.getchannel("A"))
    visible = alpha >= alpha_threshold

    if white_bg:
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
        visible[:] = True

    # PIL MEDIANCUT is fast and stable on Streamlit Cloud.
    q = img.convert("RGB").quantize(
        colors=int(n_colors),
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    labels = np.array(q, dtype=np.int32)
    labels[~visible] = -1

    palette_raw = q.getpalette()[:768]
    used = sorted(int(x) for x in np.unique(labels) if int(x) >= 0)

    remapped = np.full_like(labels, -1)
    palette: list[tuple[int, int, int]] = []
    for new_idx, old_idx in enumerate(used):
        remapped[labels == old_idx] = new_idx
        palette.append(tuple(palette_raw[old_idx * 3: old_idx * 3 + 3]))

    return remapped, palette, visible.astype(np.uint8)


def make_quant_preview(labels: np.ndarray, palette: list[tuple[int, int, int]]) -> Image.Image:
    arr = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.uint8)
    for i, rgb in enumerate(palette):
        m = labels == i
        arr[m, 0] = rgb[0]
        arr[m, 1] = rgb[1]
        arr[m, 2] = rgb[2]
        arr[m, 3] = 255
    return Image.fromarray(arr, "RGBA")


# -----------------------------
# path generation
# -----------------------------
def fmt_num(x: float, dec: int) -> str:
    if dec <= 0:
        return str(int(round(float(x))))
    return f"{float(x):.{dec}f}".rstrip("0").rstrip(".").replace("-0", "0")


def contour_to_path(
    contour: np.ndarray,
    simplify: float,
    min_area: int,
    decimals: int,
    offset_x: int,
    offset_y: int,
    max_path_bytes: int,
) -> Optional[str]:
    if contour is None or len(contour) < 3:
        return None

    if abs(cv2.contourArea(contour)) < min_area:
        return None

    # Try progressively stronger simplification only for this contour if it gets too big.
    eps = max(0.05, float(simplify))
    best_path = None
    for _ in range(7):
        approx = cv2.approxPolyDP(contour, epsilon=eps, closed=True)
        if approx is None or len(approx) < 3:
            return best_path
        pts = approx.reshape(-1, 2).astype(np.float32)
        cmds = [f"M{fmt_num(pts[0,0] + offset_x, decimals)} {fmt_num(pts[0,1] + offset_y, decimals)}"]
        for x, y in pts[1:]:
            cmds.append(f"L{fmt_num(x + offset_x, decimals)} {fmt_num(y + offset_y, decimals)}")
        cmds.append("Z")
        path = "".join(cmds)
        best_path = path
        if len(path.encode("utf-8")) <= max_path_bytes:
            return path
        eps *= 1.45
    return best_path


def clean_binary(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if min_area <= 1:
        return binary * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def mask_to_paths(
    binary: np.ndarray,
    simplify: float,
    min_area: int,
    decimals: int,
    offset_x: int,
    offset_y: int,
    max_path_bytes: int,
) -> tuple[list[str], int]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paths: list[str] = []
    contour_count = 0
    for c in contours:
        d = contour_to_path(
            c,
            simplify=simplify,
            min_area=min_area,
            decimals=decimals,
            offset_x=offset_x,
            offset_y=offset_y,
            max_path_bytes=max_path_bytes,
        )
        if d:
            paths.append(d)
            contour_count += 1
    return paths, contour_count


def make_tiles(w: int, h: int, cols: int, rows: int) -> list[Tile]:
    tiles: list[Tile] = []
    idx = 1
    for r in range(rows):
        y0 = int(round(h * r / rows))
        y1 = int(round(h * (r + 1) / rows))
        for c in range(cols):
            x0 = int(round(w * c / cols))
            x1 = int(round(w * (c + 1) / cols))
            tiles.append(Tile(idx=idx, x0=x0, y0=y0, x1=x1, y1=y1))
            idx += 1
    return tiles


def expand_tile(tile: Tile, w: int, h: int, overlap_px: int) -> tuple[int, int, int, int]:
    return (
        clamp(tile.x0 - overlap_px, 0, w),
        clamp(tile.y0 - overlap_px, 0, h),
        clamp(tile.x1 + overlap_px, 0, w),
        clamp(tile.y1 + overlap_px, 0, h),
    )


def split_tile(tile: Tile) -> tuple[Tile, Tile]:
    if (tile.x1 - tile.x0) >= (tile.y1 - tile.y0):
        mid = (tile.x0 + tile.x1) // 2
        a = Tile(tile.idx, tile.x0, tile.y0, mid, tile.y1, tile.level + 1, tile.kind)
        b = Tile(tile.idx, mid, tile.y0, tile.x1, tile.y1, tile.level + 1, tile.kind)
    else:
        mid = (tile.y0 + tile.y1) // 2
        a = Tile(tile.idx, tile.x0, tile.y0, tile.x1, mid, tile.level + 1, tile.kind)
        b = Tile(tile.idx, tile.x0, mid, tile.x1, tile.y1, tile.level + 1, tile.kind)
    return a, b


def dominant_color_for_mask(labels: np.ndarray, palette: list[tuple[int, int, int]], mask: np.ndarray) -> tuple[int, int, int]:
    if not palette:
        return (0, 0, 0)
    best_i = 0
    best_count = -1
    for i in range(len(palette)):
        count = int(((labels == i) & (mask > 0)).sum())
        if count > best_count:
            best_count = count
            best_i = i
    return palette[best_i]


def build_color_svg_for_tile(
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    visible_mask: np.ndarray,
    tile: Tile,
    full_w: int,
    full_h: int,
    overlap_px: int,
    coordinate_mode: str,
    simplify: float,
    min_area: int,
    decimals: int,
    max_path_bytes: int,
    add_local_underpaint: bool,
) -> SvgPart:
    ex0, ey0, ex1, ey1 = expand_tile(tile, full_w, full_h, overlap_px)
    local_labels = labels[ey0:ey1, ex0:ex1]
    local_visible = visible_mask[ey0:ey1, ex0:ex1]

    if coordinate_mode == "全パーツ同じキャンバス":
        svg_w, svg_h = full_w, full_h
        offset_x, offset_y = ex0, ey0
    else:
        svg_w, svg_h = ex1 - ex0, ey1 - ey0
        offset_x, offset_y = 0, 0

    parts: list[str] = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}">']
    path_count = 0
    contour_count = 0
    max_seen_path = 0

    if add_local_underpaint and local_visible.max() > 0:
        binary = clean_binary(local_visible, max(1, min_area))
        paths, cnt = mask_to_paths(
            binary=binary,
            simplify=max(0.2, simplify * 1.25),
            min_area=max(1, min_area),
            decimals=decimals,
            offset_x=offset_x,
            offset_y=offset_y,
            max_path_bytes=max_path_bytes,
        )
        if paths:
            fill = short_hex(dominant_color_for_mask(labels, palette, visible_mask))
            parts.append(f'<g fill="{fill}">')
            for d in paths:
                max_seen_path = max(max_seen_path, len(d.encode("utf-8")))
                parts.append(f'<path d="{d}"/>')
                path_count += 1
            parts.append('</g>')
            contour_count += cnt

    # Draw large color areas first, small/detail colors later.
    order = []
    for i in range(len(palette)):
        count = int((local_labels == i).sum())
        if count >= min_area:
            order.append((count, i))
    order.sort(reverse=True)

    for _, color_index in order:
        mask = local_labels == color_index
        if int(mask.sum()) < min_area:
            continue
        binary = clean_binary(mask, min_area=max(1, min_area))
        if binary.max() == 0:
            continue

        paths, cnt = mask_to_paths(
            binary=binary,
            simplify=simplify,
            min_area=max(1, min_area),
            decimals=decimals,
            offset_x=offset_x,
            offset_y=offset_y,
            max_path_bytes=max_path_bytes,
        )
        if not paths:
            continue

        fill = short_hex(palette[color_index])
        parts.append(f'<g fill="{fill}">')
        for d in paths:
            max_seen_path = max(max_seen_path, len(d.encode("utf-8")))
            parts.append(f'<path d="{d}"/>')
            path_count += 1
        parts.append('</g>')
        contour_count += cnt

    parts.append('</svg>')
    svg = safe_svg(''.join(parts))
    return SvgPart(
        filename="",
        svg=svg,
        size=len(svg.encode("utf-8")),
        max_path_size=max_seen_path,
        path_count=path_count,
        contour_count=contour_count,
        tile=tile,
        kind="color",
    )


def build_mask_svg_for_tile(
    mask: np.ndarray,
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    tile: Tile,
    full_w: int,
    full_h: int,
    overlap_px: int,
    coordinate_mode: str,
    simplify: float,
    min_area: int,
    decimals: int,
    max_path_bytes: int,
    fill_color: tuple[int, int, int],
    kind: str,
) -> SvgPart:
    ex0, ey0, ex1, ey1 = expand_tile(tile, full_w, full_h, overlap_px)
    local_mask = mask[ey0:ey1, ex0:ex1]

    if coordinate_mode == "全パーツ同じキャンバス":
        svg_w, svg_h = full_w, full_h
        offset_x, offset_y = ex0, ey0
    else:
        svg_w, svg_h = ex1 - ex0, ey1 - ey0
        offset_x, offset_y = 0, 0

    binary = clean_binary(local_mask, max(1, min_area))
    paths, cnt = mask_to_paths(
        binary=binary,
        simplify=simplify,
        min_area=max(1, min_area),
        decimals=decimals,
        offset_x=offset_x,
        offset_y=offset_y,
        max_path_bytes=max_path_bytes,
    )

    fill = short_hex(fill_color)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}">']
    path_count = 0
    max_seen = 0
    if paths:
        parts.append(f'<g fill="{fill}">')
        for d in paths:
            max_seen = max(max_seen, len(d.encode("utf-8")))
            parts.append(f'<path d="{d}"/>')
            path_count += 1
        parts.append('</g>')
    parts.append('</svg>')
    svg = safe_svg(''.join(parts))
    return SvgPart(
        filename="",
        svg=svg,
        size=len(svg.encode("utf-8")),
        max_path_size=max_seen,
        path_count=path_count,
        contour_count=cnt,
        tile=tile,
        kind=kind,
    )


def lineart_mask_from_image(img: Image.Image, visible: np.ndarray, low: int, high: int, dilate_px: int) -> np.ndarray:
    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Keep clear drawn lines; bilateral smoothing avoids broken noisy edges.
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)
    edges = cv2.Canny(gray, threshold1=low, threshold2=high)
    edges = cv2.bitwise_and(edges, (visible > 0).astype(np.uint8) * 255)
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        edges = cv2.dilate(edges, np.ones((k, k), np.uint8), iterations=1)
    return edges


# -----------------------------
# recursive target fitting
# -----------------------------
def fit_color_tile_recursive(
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    visible: np.ndarray,
    tile: Tile,
    full_w: int,
    full_h: int,
    overlap_px: int,
    coordinate_mode: str,
    simplify: float,
    min_area: int,
    decimals: int,
    target_bytes: int,
    max_path_bytes: int,
    add_local_underpaint: bool,
    max_parts_left: int,
    min_tile_side: int,
) -> list[SvgPart]:
    # Try a normal version, then a slightly simplified version before splitting.
    attempts = [
        (simplify, min_area, decimals),
        (simplify * 1.25, max(min_area, int(min_area * 1.2 + 1)), 0),
    ]
    best: Optional[SvgPart] = None
    for simp, area, dec in attempts:
        part = build_color_svg_for_tile(
            labels=labels,
            palette=palette,
            visible_mask=visible,
            tile=tile,
            full_w=full_w,
            full_h=full_h,
            overlap_px=overlap_px,
            coordinate_mode=coordinate_mode,
            simplify=simp,
            min_area=area,
            decimals=dec,
            max_path_bytes=max_path_bytes,
            add_local_underpaint=add_local_underpaint,
        )
        if best is None or part.size < best.size:
            best = part
        if part.size <= target_bytes and part.max_path_size <= max_path_bytes:
            return [part]

    assert best is not None
    too_small = (tile.x1 - tile.x0) <= min_tile_side or (tile.y1 - tile.y0) <= min_tile_side
    if max_parts_left <= 1 or too_small:
        best.warning = "target_over"
        return [best]

    a, b = split_tile(tile)
    left_quota = max(1, max_parts_left // 2)
    right_quota = max(1, max_parts_left - left_quota)
    return fit_color_tile_recursive(
        labels, palette, visible, a, full_w, full_h, overlap_px, coordinate_mode,
        simplify, min_area, decimals, target_bytes, max_path_bytes, add_local_underpaint,
        left_quota, min_tile_side,
    ) + fit_color_tile_recursive(
        labels, palette, visible, b, full_w, full_h, overlap_px, coordinate_mode,
        simplify, min_area, decimals, target_bytes, max_path_bytes, add_local_underpaint,
        right_quota, min_tile_side,
    )


def fit_mask_tile_recursive(
    mask: np.ndarray,
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    tile: Tile,
    full_w: int,
    full_h: int,
    overlap_px: int,
    coordinate_mode: str,
    simplify: float,
    min_area: int,
    decimals: int,
    target_bytes: int,
    max_path_bytes: int,
    fill_color: tuple[int, int, int],
    kind: str,
    max_parts_left: int,
    min_tile_side: int,
) -> list[SvgPart]:
    part = build_mask_svg_for_tile(
        mask=mask,
        labels=labels,
        palette=palette,
        tile=tile,
        full_w=full_w,
        full_h=full_h,
        overlap_px=overlap_px,
        coordinate_mode=coordinate_mode,
        simplify=simplify,
        min_area=min_area,
        decimals=decimals,
        max_path_bytes=max_path_bytes,
        fill_color=fill_color,
        kind=kind,
    )
    if part.size <= target_bytes and part.max_path_size <= max_path_bytes:
        return [part]

    too_small = (tile.x1 - tile.x0) <= min_tile_side or (tile.y1 - tile.y0) <= min_tile_side
    if max_parts_left <= 1 or too_small:
        part.warning = "target_over"
        return [part]

    a, b = split_tile(tile)
    left_quota = max(1, max_parts_left // 2)
    right_quota = max(1, max_parts_left - left_quota)
    return fit_mask_tile_recursive(
        mask, labels, palette, a, full_w, full_h, overlap_px, coordinate_mode,
        simplify, min_area, decimals, target_bytes, max_path_bytes, fill_color,
        kind, left_quota, min_tile_side,
    ) + fit_mask_tile_recursive(
        mask, labels, palette, b, full_w, full_h, overlap_px, coordinate_mode,
        simplify, min_area, decimals, target_bytes, max_path_bytes, fill_color,
        kind, right_quota, min_tile_side,
    )


# -----------------------------
# guide and zip output
# -----------------------------
def draw_placement_guide(img: Image.Image, parts: list[SvgPart], coordinate_mode: str) -> bytes:
    base = paste_on_checkerboard(img).convert("RGBA")
    # Make guide readable on smartphone.
    max_side = 1400
    scale = 1.0
    if max(base.size) > max_side:
        scale = max_side / max(base.size)
        base = base.resize((int(base.width * scale), int(base.height * scale)), Image.Resampling.LANCZOS)

    dr = ImageDraw.Draw(base)
    for n, p in enumerate(parts, start=1):
        if p.kind != "color":
            continue
        t = p.tile
        x0, y0, x1, y1 = [int(round(v * scale)) for v in (t.x0, t.y0, t.x1, t.y1)]
        dr.rectangle([x0, y0, x1, y1], outline=(255, 40, 40, 255), width=max(1, int(2 * scale)))
        dr.rectangle([x0, y0, min(x0 + 56, x1), min(y0 + 24, y1)], fill=(255, 255, 255, 220))
        dr.text((x0 + 3, y0 + 3), str(n), fill=(0, 0, 0, 255))
    return png_bytes(base)


def make_manifest_csv(parts: list[SvgPart], full_w: int, full_h: int) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "filename", "kind", "bytes", "max_path_bytes", "paths", "contours",
        "x0", "y0", "x1", "y1", "x_pct", "y_pct", "w_pct", "h_pct", "warning",
    ])
    for p in parts:
        t = p.tile
        writer.writerow([
            p.filename, p.kind, p.size, p.max_path_size, p.path_count, p.contour_count,
            t.x0, t.y0, t.x1, t.y1,
            round(t.x0 / full_w * 100, 3), round(t.y0 / full_h * 100, 3),
            round((t.x1 - t.x0) / full_w * 100, 3), round((t.y1 - t.y0) / full_h * 100, 3),
            p.warning,
        ])
    return buf.getvalue()


def svg_data_url(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def make_preview_html(parts: list[SvgPart], full_w: int, full_h: int, max_embed_bytes: int = 1200000) -> str:
    # Same-canvas SVG parts can simply be stacked in the same rectangle.
    color_parts = [p for p in parts if p.kind in ("underpaint", "color", "lineart")]
    total = sum(p.size for p in color_parts)
    if total > max_embed_bytes or len(color_parts) > 120:
        return f"""
        <html><body style="font-family:sans-serif">
        <h3>Preview skipped</h3>
        <p>Total SVG data is too large for safe smartphone preview.</p>
        <p>parts: {len(color_parts)}, total bytes: {total:,}</p>
        </body></html>
        """
    imgs = []
    # Recommended layer order: underpaint -> color -> lineart
    ordered = [p for p in color_parts if p.kind == "underpaint"] + [p for p in color_parts if p.kind == "color"] + [p for p in color_parts if p.kind == "lineart"]
    for p in ordered:
        imgs.append(
            f'<img src="{svg_data_url(p.svg)}" style="position:absolute;left:0;top:0;width:100%;height:100%;">'
        )
    return f"""
    <html><body style="margin:0;background:#eee;font-family:sans-serif;">
    <div style="padding:12px;">
    <h3>GT7 split decal preview</h3>
    <p>Layer order: underpaint → color parts → lineart.</p>
    <div style="position:relative;width:min(100%,900px);aspect-ratio:{full_w}/{full_h};background:
      linear-gradient(45deg,#ddd 25%,transparent 25%),linear-gradient(-45deg,#ddd 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ddd 75%),linear-gradient(-45deg,transparent 75%,#ddd 75%);
      background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0;border:1px solid #aaa;overflow:hidden;">
      {''.join(imgs)}
    </div>
    </div></body></html>
    """


def make_readme(settings: dict, parts: list[SvgPart], full_w: int, full_h: int) -> str:
    largest = max((p.size for p in parts), default=0)
    largest_path = max((p.max_path_size for p in parts), default=0)
    over = [p.filename for p in parts if p.warning]
    return f"""
GT7 SVG Vectorizer - Split Decal Edition

このZIPは、1枚の画像を複数のSVGデカールに分割して出力したものです。

基本方針:
- 1枚の巨大SVGではなく、複数SVGをGT7内で重ねて完成させます。
- 隣り合うパーツは少し重ねています。
- SVGOMGに通す前提で、できるだけシンプルな path / g fill だけで出力しています。
- SVGOMGで Merge paths をONにすると、分けたpathが再結合されることがあるためOFF推奨です。

おすすめの重ね順:
1. underpaint がある場合は最初に配置
2. color_part_*.svg を番号順に配置
3. lineart がある場合は最後に配置

配置方式:
- {settings.get('coordinate_mode')}
- 画像全体サイズ: {full_w} x {full_h}

出力結果:
- SVG数: {len(parts)}
- 最大SVGサイズ: {largest:,} bytes
- 最大pathサイズ: {largest_path:,} bytes
- 目標SVGサイズ: {settings.get('target_bytes'):,} bytes
- 目標pathサイズ: {settings.get('max_path_bytes'):,} bytes

注意:
- 分割サイトやSVGOMGを使う場合、Merge paths はOFF推奨です。
- Round/rewrite paths、Convert colors、Remove metadata、Remove comments はON推奨です。
- Remove viewBox はOFF推奨です。

目標超過が残ったファイル:
{chr(10).join(over) if over else 'なし'}
""".strip() + "\n"


def make_zip(parts: list[SvgPart], guide_png: bytes, preview_html: str, manifest_csv: str, readme: str, quant_png: Optional[bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in parts:
            folder = "svg"
            if p.kind == "underpaint":
                folder = "svg/00_underpaint"
            elif p.kind == "lineart":
                folder = "svg/99_lineart"
            elif p.kind == "color":
                folder = "svg/color_parts"
            z.writestr(f"{folder}/{p.filename}", p.svg)
        z.writestr("placement_guide.png", guide_png)
        z.writestr("preview.html", preview_html)
        z.writestr("manifest.csv", manifest_csv)
        z.writestr("README.txt", readme)
        if quant_png is not None:
            z.writestr("quantized_reference.png", quant_png)
    return buf.getvalue()


# -----------------------------
# main conversion
# -----------------------------
def run_conversion(img: Image.Image, cfg: dict, progress=None) -> dict:
    t0 = time.time()
    work = fit_image(img, side=int(cfg["side"]), enhance=cfg["enhance"], smooth_color=cfg["smooth_color"])
    full_w, full_h = work.size

    if progress:
        progress.progress(0.08, text="色を整理しています…")
    labels, palette, visible = quantize_image(
        work,
        n_colors=int(cfg["colors"]),
        alpha_threshold=int(cfg["alpha"]),
        white_bg=bool(cfg["white_bg"]),
    )

    tiles = make_tiles(full_w, full_h, int(cfg["cols"]), int(cfg["rows"]))
    overlap_px = int(round(max(full_w, full_h) * float(cfg["overlap_pct"]) / 100.0))
    target_bytes = int(cfg["target_bytes"])
    max_path_bytes = int(cfg["max_path_bytes"])
    max_parts = int(cfg["max_parts"])

    all_parts: list[SvgPart] = []

    # Underpaint layer, separate from color parts, prevents overlap conflicts.
    if cfg["make_underpaint"]:
        if progress:
            progress.progress(0.14, text="下塗りレイヤーを作成しています…")
        main_tile = Tile(1, 0, 0, full_w, full_h, kind="underpaint")
        fill = dominant_color_for_mask(labels, palette, visible)
        under_parts = fit_mask_tile_recursive(
            mask=visible,
            labels=labels,
            palette=palette,
            tile=main_tile,
            full_w=full_w,
            full_h=full_h,
            overlap_px=0,
            coordinate_mode=cfg["coordinate_mode"],
            simplify=max(float(cfg["simp"]) * 1.2, 0.35),
            min_area=max(int(cfg["area"]), 2),
            decimals=int(cfg["dec"]),
            target_bytes=target_bytes,
            max_path_bytes=max_path_bytes,
            fill_color=fill,
            kind="underpaint",
            max_parts_left=min(12, max_parts),
            min_tile_side=160,
        )
        for i, p in enumerate(under_parts, start=1):
            p.kind = "underpaint"
            p.filename = f"underpaint_{i:02d}.svg"
        all_parts.extend(under_parts)

    # Color parts with recursive split.
    if progress:
        progress.progress(0.22, text="色パーツを分割トレースしています…")

    color_parts: list[SvgPart] = []
    quota_per_tile = max(1, max_parts // max(1, len(tiles)))
    for i, tile in enumerate(tiles, start=1):
        if len(color_parts) >= max_parts:
            break
        current_quota = max(1, min(quota_per_tile, max_parts - len(color_parts)))
        parts = fit_color_tile_recursive(
            labels=labels,
            palette=palette,
            visible=visible,
            tile=tile,
            full_w=full_w,
            full_h=full_h,
            overlap_px=overlap_px,
            coordinate_mode=cfg["coordinate_mode"],
            simplify=float(cfg["simp"]),
            min_area=int(cfg["area"]),
            decimals=int(cfg["dec"]),
            target_bytes=target_bytes,
            max_path_bytes=max_path_bytes,
            add_local_underpaint=False,
            max_parts_left=current_quota,
            min_tile_side=96,
        )
        color_parts.extend(parts)
        if progress:
            frac = 0.22 + 0.56 * (i / max(1, len(tiles)))
            progress.progress(min(frac, 0.78), text=f"色パーツ {i}/{len(tiles)} を処理中…")

    for n, p in enumerate(color_parts, start=1):
        p.kind = "color"
        p.filename = f"color_part_{n:03d}.svg"
    all_parts.extend(color_parts)

    # Optional lineart layer, split separately.
    if cfg["make_lineart"]:
        if progress:
            progress.progress(0.82, text="線画レイヤーを作成しています…")
        line_mask = lineart_mask_from_image(
            work,
            visible=visible,
            low=int(cfg["canny_low"]),
            high=int(cfg["canny_high"]),
            dilate_px=int(cfg["line_width"]),
        )
        line_tile = Tile(1, 0, 0, full_w, full_h, kind="lineart")
        line_parts = fit_mask_tile_recursive(
            mask=line_mask,
            labels=labels,
            palette=palette,
            tile=line_tile,
            full_w=full_w,
            full_h=full_h,
            overlap_px=0,
            coordinate_mode=cfg["coordinate_mode"],
            simplify=max(float(cfg["simp"]) * 0.85, 0.25),
            min_area=max(int(cfg["area"]), 2),
            decimals=int(cfg["dec"]),
            target_bytes=target_bytes,
            max_path_bytes=max_path_bytes,
            fill_color=(8, 9, 13),
            kind="lineart",
            max_parts_left=min(20, max_parts),
            min_tile_side=128,
        )
        for i, p in enumerate(line_parts, start=1):
            p.kind = "lineart"
            p.filename = f"lineart_{i:02d}.svg"
        all_parts.extend(line_parts)

    if progress:
        progress.progress(0.91, text="ガイドとZIPを作成しています…")

    guide_png = draw_placement_guide(work, color_parts, cfg["coordinate_mode"])
    quant_png = None
    if cfg["make_quant_png"]:
        quant_png = png_bytes(paste_on_checkerboard(make_quant_preview(labels, palette)))

    manifest = make_manifest_csv(all_parts, full_w, full_h)
    settings_for_readme = cfg.copy()
    settings_for_readme["target_bytes"] = target_bytes
    settings_for_readme["max_path_bytes"] = max_path_bytes
    readme = make_readme(settings_for_readme, all_parts, full_w, full_h)
    preview_html = make_preview_html(all_parts, full_w, full_h)
    zip_bytes = make_zip(all_parts, guide_png, preview_html, manifest, readme, quant_png)

    if progress:
        progress.progress(1.0, text="完了")

    return {
        "parts": all_parts,
        "zip": zip_bytes,
        "guide_png": guide_png,
        "preview_html": preview_html,
        "manifest": manifest,
        "readme": readme,
        "work": work,
        "elapsed": time.time() - t0,
        "full_w": full_w,
        "full_h": full_h,
    }


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="GT7 SVG Split Decal Edition", page_icon="🏁", layout="wide")
st.title("🏁 GT7 SVG Vectorizer - Split Decal Edition")
st.caption("1枚SVGではなく、複数SVGデカールに分割してGT7内で重ねる方式です。")

with st.expander("この方式について", expanded=True):
    st.markdown(
        """
この版は、画像を **複数のSVGパーツ** に分割して出力します。  
GT7内では、出力されたSVGを順番に重ねて完成させます。

重要ポイント:
- 各SVGを **14KB前後** に収める方向で作ります。
- 隣り合うパーツは少し重ねて、境界の隙間を減らします。
- `underpaint` がある場合は最初、`lineart` がある場合は最後に重ねてください。
- SVGOMGでは **Merge paths はOFF推奨** です。
"""
    )

uploaded = st.file_uploader("画像を選択", type=["png", "jpg", "jpeg", "webp"])

with st.sidebar:
    st.header("設定")
    preset_name = st.selectbox("品質プリセット", list(PRESETS.keys()) + ["手動設定"], index=0)
    base = PRESETS.get(preset_name, PRESETS["安定 / 30秒〜1分"]).copy()

    st.subheader("GT7分割")
    target_kb = st.slider("各SVGの目標KB", 8, 64, 14)
    max_path_kb = st.slider("1path最大KB", 4, 15, 12)
    coordinate_mode = st.selectbox(
        "配置方式",
        ["全パーツ同じキャンバス", "タイルごとのローカル座標"],
        index=0,
    )

    cols = st.slider("横分割数", 1, 8, int(base["cols"]))
    rows = st.slider("縦分割数", 1, 8, int(base["rows"]))
    overlap_pct = st.slider("重なり幅 %", 0.0, 10.0, float(base["overlap_pct"]), step=0.5)
    max_parts = st.slider("最大SVG数", 8, 140, int(base["max_parts"]))

    st.divider()
    st.subheader("トレース品質")
    colors = st.slider("色数", 8, 192, int(base["colors"]))
    side = st.slider("処理サイズ 長辺px", 512, 1800, int(base["side"]), step=16)
    simp = st.slider("パス簡略化（小さいほど高精度）", 0.15, 3.0, float(base["simp"]), step=0.05)
    area = st.slider("小さい形状の削除", 1, 30, int(base["area"]))
    dec = st.slider("座標の小数桁", 0, 2, int(base["dec"]))
    alpha = st.slider("透明判定", 0, 255, 16)

    enhance = st.toggle("低画質を軽く補正", True)
    smooth_color = st.toggle("色面をなめらかにしてからトレース", bool(base["smooth_color"]))
    white_bg = st.toggle("透明部分を白背景にする", False)

    st.divider()
    st.subheader("追加レイヤー")
    make_underpaint = st.toggle("下塗りシルエットを別SVGで出力", True)
    make_lineart = st.toggle("線画レイヤーを別SVGで出力", False)
    line_width = st.slider("線画の太さ", 0, 2, 1)
    canny_low = st.slider("線画検出 low", 20, 180, 60)
    canny_high = st.slider("線画検出 high", 60, 260, 150)

    st.divider()
    st.subheader("スマホ保護")
    make_quant_png = st.toggle("量子化参考PNGもZIPに入れる", False)
    show_preview = st.toggle("アプリ内プレビューを表示", True)

    cfg = {
        "target_bytes": int(target_kb * 1024),
        "max_path_bytes": int(max_path_kb * 1024),
        "coordinate_mode": coordinate_mode,
        "cols": cols,
        "rows": rows,
        "overlap_pct": overlap_pct,
        "max_parts": max_parts,
        "colors": colors,
        "side": side,
        "simp": simp,
        "area": area,
        "dec": dec,
        "alpha": alpha,
        "enhance": enhance,
        "smooth_color": smooth_color,
        "white_bg": white_bg,
        "make_underpaint": make_underpaint,
        "make_lineart": make_lineart,
        "line_width": line_width,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "make_quant_png": make_quant_png,
    }

if uploaded is None:
    st.info("画像をアップロードしてください。")
    st.stop()

try:
    original_img = Image.open(uploaded).convert("RGBA")
except Exception as e:
    st.error(f"画像を開けませんでした: {e}")
    st.stop()

st.subheader("元画像")
preview_original = paste_on_checkerboard(original_img)
if max(preview_original.size) > 1100:
    scale = 1100 / max(preview_original.size)
    preview_original = preview_original.resize((int(preview_original.width * scale), int(preview_original.height * scale)), Image.Resampling.LANCZOS)
st.image(preview_original, use_container_width=True)

if st.button("分割SVGを作成", type="primary", use_container_width=True):
    progress = st.progress(0, text="開始します…")
    try:
        st.session_state.result = run_conversion(original_img, cfg, progress=progress)
        st.session_state.filename = uploaded.name
        st.session_state.used_cfg = cfg.copy()
    except Exception as e:
        st.error(f"変換中にエラーが出ました: {e}")
        st.stop()

if "result" not in st.session_state:
    st.stop()

result = st.session_state.result
parts: list[SvgPart] = result["parts"]
used_cfg = st.session_state.used_cfg

total_svg_bytes = sum(p.size for p in parts)
largest_svg = max((p.size for p in parts), default=0)
largest_path = max((p.max_path_size for p in parts), default=0)
over_count = sum(1 for p in parts if p.warning or p.size > used_cfg["target_bytes"] or p.max_path_size > used_cfg["max_path_bytes"])

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("SVG数", str(len(parts)))
m2.metric("合計SVG", f"{total_svg_bytes:,} B")
m3.metric("最大SVG", f"{largest_svg:,} B")
m4.metric("最大path", f"{largest_path:,} B")
m5.metric("目標超過", str(over_count))
m6.metric("時間", f"{result['elapsed']:.1f} 秒")

if over_count == 0:
    st.success("各SVG / 各path が目標内に収まっています。")
else:
    st.warning("一部のSVGまたはpathが目標を超えています。分割数や最大SVG数を増やす、処理サイズや色数を下げる、簡略化を少し上げてください。")

if result["elapsed"] > 75:
    st.warning("処理時間が長めです。スマホで落ちる場合は、色数・処理サイズ・最大SVG数を下げてください。")

base_name = re.sub(r"\.[^.]+$", "", st.session_state.filename)
base_name = re.sub(r"[^A-Za-z0-9_-]+", "_", base_name).strip("_") or "gt7_split"

st.download_button(
    "ZIPを保存",
    data=result["zip"],
    file_name=f"{base_name}_gt7_split_decals.zip",
    mime="application/zip",
    use_container_width=True,
)

st.subheader("配置ガイド")
st.image(result["guide_png"], use_container_width=True)

if show_preview:
    st.subheader("重ね合わせプレビュー")
    if used_cfg["coordinate_mode"] == "全パーツ同じキャンバス":
        components.html(result["preview_html"], height=720, scrolling=True)
    else:
        st.info("タイルごとのローカル座標モードでは、ZIP内の placement_guide.png と manifest.csv を見ながら配置してください。")

with st.expander("出力ファイル一覧"):
    rows = []
    for p in parts:
        rows.append({
            "file": p.filename,
            "kind": p.kind,
            "bytes": p.size,
            "max_path": p.max_path_size,
            "paths": p.path_count,
            "warning": p.warning,
        })
    st.dataframe(rows, use_container_width=True)

with st.expander("SVGOMGおすすめ設定"):
    st.markdown(
        """
SVGOMGに通す時は、次がおすすめです。

- **Prettify markup**：OFF
- **Remove metadata**：ON
- **Remove comments**：ON
- **Convert colors**：ON
- **Round/rewrite paths**：ON
- **Merge paths**：OFF
- **Remove viewBox**：OFF

特に **Merge paths はOFF推奨** です。ONにすると、分けたpathが再結合されて15KBを超える可能性があります。
"""
    )
