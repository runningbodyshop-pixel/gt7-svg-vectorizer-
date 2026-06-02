from __future__ import annotations

import base64
import csv
import html
import io
import math
import os
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
# GT7 SVG Inkscape Auto Builder v2
# Smartphone / Streamlit / copy-paste edition
#
# v2 focus:
# - Do NOT use dark-color mask as lineart. Use edge-only lineart.
# - Avoid giant black silhouette overlay.
# - Keep layers GT7/SVGOMG-friendly: svg + g fill + path only.
# - Split output into many SVGs with the same viewBox for GT7 stacking.
# ============================================================

HARD_LIMIT_BYTES = 2 * 1024 * 1024
DEFAULT_TARGET_KB = 14
DEFAULT_PATH_KB = 11

PRESETS = {
    "安定 / まずこれ": dict(side=1150, colors=80, fill_simp=0.55, line_simp=0.35, min_area=2, tile=256, target_kb=14, path_kb=11),
    "高品質 / 線重視": dict(side=1350, colors=112, fill_simp=0.38, line_simp=0.25, min_area=1, tile=224, target_kb=14, path_kb=11),
    "高速 / 落ちにくい": dict(side=950, colors=56, fill_simp=0.75, line_simp=0.50, min_area=3, tile=320, target_kb=14, path_kb=11),
    "細部重視 / 重め": dict(side=1500, colors=136, fill_simp=0.30, line_simp=0.20, min_area=1, tile=192, target_kb=14, path_kb=11),
}


@dataclass
class SvgItem:
    name: str
    layer: str
    order: int
    svg: str
    bytes_size: int
    path_count: int
    max_path_bytes: int
    note: str = ""


# -------------------------
# basic helpers
# -------------------------
def short_hex(r: int, g: int, b: int) -> str:
    s = f"{int(r):02x}{int(g):02x}{int(b):02x}"
    if s[0] == s[1] and s[2] == s[3] and s[4] == s[5]:
        return f"#{s[0]}{s[2]}{s[4]}"
    return f"#{s}"


def safe_name(s: str) -> str:
    s = re.sub(r"\.[^.]+$", "", s)
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s or "image"


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def image_to_data_url(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(img)).decode("ascii")


def svg_to_data_url(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def compact_svg(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = s.replace("> <", "><")
    # keep viewBox. remove unsupported or unwanted tags if they accidentally appear
    for tag in ["script", "foreignObject", "image", "text", "filter", "mask", "clipPath", "pattern", "style", "defs"]:
        s = re.sub(rf"<\s*{tag}\b.*?<\s*/\s*{tag}\s*>", "", s, flags=re.I | re.S)
        s = re.sub(rf"<\s*{tag}\b[^>]*/\s*>", "", s, flags=re.I | re.S)
    return s.strip()


def svg_header(w: int, h: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'


def wrap_svg(w: int, h: int, body: str) -> str:
    return compact_svg(svg_header(w, h) + body + "</svg>")


def fmt_num(x: float, decimals: int) -> str:
    if decimals <= 0:
        return str(int(round(float(x))))
    return f"{float(x):.{decimals}f}".rstrip("0").rstrip(".").replace("-0", "0")


def checkerboard(size: tuple[int, int], cell: int = 14) -> Image.Image:
    w, h = size
    bg = Image.new("RGBA", size, (240, 240, 240, 255))
    dr = ImageDraw.Draw(bg)
    a, b = (248, 248, 248, 255), (220, 220, 220, 255)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            dr.rectangle([x, y, x + cell - 1, y + cell - 1], fill=a if ((x // cell) + (y // cell)) % 2 == 0 else b)
    return bg


def paste_on_checkerboard(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    bg = checkerboard(img.size)
    bg.alpha_composite(img)
    return bg


# -------------------------
# preprocessing
# -------------------------
def fit_image(img: Image.Image, side: int, enhance: bool) -> Image.Image:
    img = img.convert("RGBA")
    if enhance:
        rgb = img.convert("RGB")
        rgb = ImageOps.autocontrast(rgb)
        # very mild sharpening only. too much creates noisy lineart.
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=0.8, percent=105, threshold=4))
        img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))
    if max(img.size) > side:
        scale = side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    return img


def rgba_arrays(img: Image.Image, alpha_threshold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    visible = alpha >= int(alpha_threshold)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgba, rgb, gray, visible


# -------------------------
# SVG path conversion
# -------------------------
def contour_to_path(contour: np.ndarray, simplify: float, min_area: int, decimals: int, max_path_bytes: int) -> Optional[str]:
    if contour is None or len(contour) < 3:
        return None
    if abs(cv2.contourArea(contour)) < min_area:
        return None

    eps = max(0.03, float(simplify))
    # Retry with more simplification if one path is too large.
    for _ in range(10):
        approx = cv2.approxPolyDP(contour, epsilon=eps, closed=True)
        if approx is None or len(approx) < 3:
            return None
        pts = approx.reshape(-1, 2)
        d = [f"M{fmt_num(pts[0][0], decimals)} {fmt_num(pts[0][1], decimals)}"]
        for x, y in pts[1:]:
            d.append(f"L{fmt_num(x, decimals)} {fmt_num(y, decimals)}")
        d.append("Z")
        out = "".join(d)
        if len(out.encode("utf-8")) <= max_path_bytes:
            return out
        eps *= 1.45
    return out if len(out) > 8 else None


def mask_to_paths_tiled(
    mask: np.ndarray,
    simplify: float,
    min_area: int,
    decimals: int,
    tile_size: int,
    max_path_bytes: int,
    mode: str = "external",
) -> list[str]:
    """Convert binary mask to paths. Tiling keeps each path small and avoids mobile crashes."""
    if mask is None or mask.size == 0:
        return []
    h, w = mask.shape[:2]
    tile_size = max(64, int(tile_size))
    bin_full = (mask > 0).astype(np.uint8) * 255
    paths: list[str] = []
    retr = cv2.RETR_EXTERNAL if mode == "external" else cv2.RETR_CCOMP

    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            tile = bin_full[y0 : min(h, y0 + tile_size), x0 : min(w, x0 + tile_size)]
            if int(tile.max()) == 0:
                continue
            contours, _ = cv2.findContours(tile, retr, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            for c in contours:
                if c is None or len(c) < 3:
                    continue
                # shift tile contour into full canvas coordinates
                c2 = c.copy()
                c2[:, 0, 0] += x0
                c2[:, 0, 1] += y0
                d = contour_to_path(c2, simplify=simplify, min_area=min_area, decimals=decimals, max_path_bytes=max_path_bytes)
                if d:
                    paths.append(d)
    return paths


def clean_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    bin_mask = (mask > 0).astype(np.uint8)
    if min_area <= 1:
        return bin_mask * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    out = np.zeros_like(bin_mask, dtype=np.uint8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == i] = 255
    return out


def pack_paths_to_svgs(
    paths: list[str],
    fill: str,
    layer: str,
    order_start: int,
    w: int,
    h: int,
    target_bytes: int,
    prefix: str,
    note: str = "",
) -> list[SvgItem]:
    items: list[SvgItem] = []
    if not paths:
        return items
    group_open = f'<g fill="{fill}">'
    group_close = "</g>"
    current: list[str] = []
    file_idx = 1

    def build(ps: list[str]) -> str:
        body = group_open + "".join(f'<path d="{html.escape(p, quote=True)}"/>' for p in ps) + group_close
        return wrap_svg(w, h, body)

    def flush() -> None:
        nonlocal file_idx, current
        if not current:
            return
        svg = build(current)
        maxpb = max(len(p.encode("utf-8")) for p in current)
        items.append(
            SvgItem(
                name=f"{prefix}_{file_idx:02d}.svg",
                layer=layer,
                order=order_start + file_idx,
                svg=svg,
                bytes_size=len(svg.encode("utf-8")),
                path_count=len(current),
                max_path_bytes=maxpb,
                note=note,
            )
        )
        file_idx += 1
        current = []

    # Keep each SVG below target. If one single path makes it exceed, keep it alone anyway;
    # path itself has already been simplified/tiled.
    for p in paths:
        trial = current + [p]
        if current and len(build(trial).encode("utf-8")) > target_bytes:
            flush()
        current.append(p)
    flush()
    return items


# -------------------------
# layer generation
# -------------------------
def quantize_pillow(rgb: np.ndarray, visible: np.ndarray, colors: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    img = Image.fromarray(rgb, "RGB")
    # MEDIANCUT is fast and stable on Streamlit Cloud.
    q = img.quantize(colors=max(2, int(colors)), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    labels = np.array(q, dtype=np.int32)
    labels[~visible] = -1
    pal_raw = q.getpalette()[:768]
    used = sorted(int(x) for x in np.unique(labels) if int(x) >= 0)
    remap = np.full_like(labels, -1, dtype=np.int32)
    palette: list[tuple[int, int, int]] = []
    for new_i, old_i in enumerate(used):
        remap[labels == old_i] = new_i
        palette.append(tuple(int(v) for v in pal_raw[old_i * 3 : old_i * 3 + 3]))
    return remap, palette


def color_preview(labels: np.ndarray, palette: list[tuple[int, int, int]]) -> Image.Image:
    h, w = labels.shape[:2]
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for i, c in enumerate(palette):
        m = labels == i
        arr[m, 0], arr[m, 1], arr[m, 2], arr[m, 3] = c[0], c[1], c[2], 255
    return Image.fromarray(arr, "RGBA")


def make_underpaint_paths(visible: np.ndarray, fill_simp: float, min_area: int, tile: int, max_path_bytes: int) -> list[str]:
    # Slightly close holes; use underpaint only when user wants it, and don't make it black.
    mask = (visible > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_to_paths_tiled(mask, simplify=max(0.5, fill_simp * 1.5), min_area=max(8, min_area), decimals=0, tile_size=tile, max_path_bytes=max_path_bytes)


def make_fill_layers(
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    visible: np.ndarray,
    w: int,
    h: int,
    fill_simp: float,
    min_area: int,
    tile: int,
    max_path_bytes: int,
    target_bytes: int,
    color_limit: int,
) -> list[SvgItem]:
    items: list[SvgItem] = []
    if not palette:
        return items

    # Large/base colors first, small/details later.
    counts = [(int((labels == i).sum()), i) for i in range(len(palette))]
    counts.sort(reverse=True)
    counts = counts[: max(2, int(color_limit))]

    order_base = 2000
    for rank, (_, idx) in enumerate(counts):
        mask = labels == idx
        if int(mask.sum()) < min_area:
            continue
        # clean only tiny noise; do not over-smooth because it removes details.
        binary = clean_components(mask, min_area=max(1, min_area))
        # Clip to visible area to avoid transparent background paths.
        binary[(visible == 0)] = 0
        if int(binary.max()) == 0:
            continue
        paths = mask_to_paths_tiled(
            binary,
            simplify=fill_simp,
            min_area=max(1, min_area),
            decimals=1,
            tile_size=tile,
            max_path_bytes=max_path_bytes,
        )
        if not paths:
            continue
        r, g, b = palette[idx]
        fill = short_hex(r, g, b)
        prefix = f"20_fill_{rank:03d}_{fill.replace('#','') }"
        items.extend(pack_paths_to_svgs(paths, fill, "20_fills", order_base + rank * 100, w, h, target_bytes, prefix, note="quantized fill layer"))
    return items


def auto_canny(gray: np.ndarray, visible: np.ndarray, strength: float) -> np.ndarray:
    vals = gray[visible]
    if vals.size == 0:
        med = float(np.median(gray))
    else:
        med = float(np.median(vals))
    # Lower strength means more edges. Use conservative defaults to avoid noise.
    sigma = max(0.10, min(0.70, strength))
    low = int(max(5, (1.0 - sigma) * med))
    high = int(min(255, (1.0 + sigma) * med))
    if high <= low + 10:
        high = min(255, low + 40)
    return cv2.Canny(gray, low, high, L2gradient=True)


def make_lineart_mask(
    rgb: np.ndarray,
    gray: np.ndarray,
    visible: np.ndarray,
    line_strength: float,
    line_width: int,
    edge_close: int,
    keep_dark_edges_only: bool,
) -> np.ndarray:
    # Key correction: lineart is edge-only. Do NOT threshold all dark pixels as black fill.
    # This prevents giant black silhouettes.
    blur = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
    edges = auto_canny(blur, visible, strength=line_strength)

    if keep_dark_edges_only:
        # Optional filter: retain edges near high local contrast / dark line zones, but never fill areas.
        dark = (gray < 145).astype(np.uint8) * 255
        grad = cv2.morphologyEx(dark, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        edges = cv2.bitwise_or(edges, cv2.bitwise_and(edges, grad))

    edges[visible == 0] = 0
    # Remove very isolated dots.
    edges = clean_components(edges > 0, min_area=2)

    if edge_close > 0:
        k = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=int(edge_close))

    if line_width > 1:
        ksize = max(2, int(line_width))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        edges = cv2.dilate(edges, kernel, iterations=1)

    edges[visible == 0] = 0
    return edges


def make_lineart_layer(
    rgb: np.ndarray,
    gray: np.ndarray,
    visible: np.ndarray,
    w: int,
    h: int,
    line_simp: float,
    min_area: int,
    tile: int,
    max_path_bytes: int,
    target_bytes: int,
    line_strength: float,
    line_width: int,
    edge_close: int,
    black: str,
    keep_dark_edges_only: bool,
) -> tuple[list[SvgItem], Image.Image]:
    mask = make_lineart_mask(rgb, gray, visible, line_strength, line_width, edge_close, keep_dark_edges_only)
    paths = mask_to_paths_tiled(
        mask,
        simplify=line_simp,
        min_area=max(1, min_area),
        decimals=1,
        tile_size=tile,
        max_path_bytes=max_path_bytes,
    )
    items = pack_paths_to_svgs(paths, black, "90_lineart_edge_only", 9000, w, h, target_bytes, "90_lineart", note="edge-only lineart; not dark-region fill")
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask > 0] = (0, 0, 0, 255)
    return items, Image.fromarray(rgba, "RGBA")


# -------------------------
# preview / zip
# -------------------------
def make_placement_guide(img: Image.Image, cols: int = 3, rows: int = 3) -> Image.Image:
    base = paste_on_checkerboard(img)
    dr = ImageDraw.Draw(base)
    w, h = base.size
    for i in range(1, cols):
        x = int(w * i / cols)
        dr.line([(x, 0), (x, h)], fill=(255, 0, 0, 170), width=1)
    for j in range(1, rows):
        y = int(h * j / rows)
        dr.line([(0, y), (w, y)], fill=(255, 0, 0, 170), width=1)
    dr.rectangle([0, 0, w - 1, h - 1], outline=(255, 0, 0, 200), width=2)
    return base


def build_preview_html(items: list[SvgItem], w: int, h: int, original: Optional[Image.Image] = None) -> str:
    # Inline SVGs in z-order. No external files needed.
    layers = []
    for it in sorted(items, key=lambda x: x.order):
        layers.append(
            f'<img alt="{html.escape(it.name)}" title="{html.escape(it.name)}" '
            f'src="{svg_to_data_url(it.svg)}" style="position:absolute;left:0;top:0;width:100%;height:100%;"/>'
        )
    original_html = ""
    if original is not None:
        original_html = f'<h2>Original reference</h2><img src="{image_to_data_url(paste_on_checkerboard(original))}" style="max-width:100%;border:1px solid #ccc;"/>'
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GT7 SVG layered preview</title>
<style>body{{font-family:Arial,sans-serif;margin:16px;background:#eee}}.stage{{position:relative;width:min(96vw,{w}px);aspect-ratio:{w}/{h};background:#fff;border:1px solid #aaa;overflow:hidden}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 6px;font-size:12px}}</style>
</head><body><h1>GT7 SVG layered preview</h1><p>All SVGs are overlaid on the same canvas. Upload/place them in manifest order.</p>
<div class="stage">{''.join(layers)}</div>{original_html}<h2>Layer list</h2><table><tr><th>order</th><th>file</th><th>bytes</th><th>paths</th><th>max path bytes</th><th>note</th></tr>
{''.join(f'<tr><td>{it.order}</td><td>{html.escape(it.name)}</td><td>{it.bytes_size}</td><td>{it.path_count}</td><td>{it.max_path_bytes}</td><td>{html.escape(it.note)}</td></tr>' for it in sorted(items, key=lambda x: x.order))}</table></body></html>"""


def zip_outputs(items: list[SvgItem], w: int, h: int, original: Image.Image, quant: Image.Image, line: Image.Image) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for it in sorted(items, key=lambda x: x.order):
            z.writestr(it.name, it.svg.encode("utf-8"))
        # manifest
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["order", "file", "layer", "bytes", "path_count", "max_path_bytes", "note"])
        for it in sorted(items, key=lambda x: x.order):
            writer.writerow([it.order, it.name, it.layer, it.bytes_size, it.path_count, it.max_path_bytes, it.note])
        z.writestr("manifest.csv", csv_buf.getvalue())
        z.writestr("preview.html", build_preview_html(items, w, h, original=original))
        z.writestr("placement_guide.png", png_bytes(make_placement_guide(original)))
        z.writestr("reference_quantized.png", png_bytes(paste_on_checkerboard(quant)))
        z.writestr("reference_lineart_mask.png", png_bytes(paste_on_checkerboard(line)))
        readme = """GT7 SVG Inkscape Auto Builder v2

Recommended GT7 stacking order: follow manifest.csv from top to bottom.
All SVG files share the same canvas/viewBox, so place them at the same position and size in GT7.

Important SVGOMG setting:
- Merge paths: OFF
- Remove viewBox: OFF
- Prettify markup: OFF
- Remove metadata/comments: ON
- Convert colors: ON
- Round/rewrite paths: ON

v2 note:
Lineart is generated from edge-only Canny contours. It does not use a dark-region fill mask, so it should not create a giant black silhouette over the image.
"""
        z.writestr("README.txt", readme)
    return buf.getvalue()


# -------------------------
# main conversion
# -------------------------
def convert(img: Image.Image, cfg: dict) -> dict:
    start = time.time()
    work = fit_image(img, int(cfg["side"]), bool(cfg["enhance"]))
    rgba, rgb, gray, visible = rgba_arrays(work, int(cfg["alpha"]))
    h, w = visible.shape[:2]

    labels, palette = quantize_pillow(rgb, visible, int(cfg["colors"]))
    quant = color_preview(labels, palette)

    target_bytes = int(cfg["target_kb"]) * 1024
    max_path_bytes = int(cfg["path_kb"]) * 1024
    tile = int(cfg["tile"])
    min_area = int(cfg["min_area"])

    items: list[SvgItem] = []

    if cfg.get("underpaint", False):
        # Pale neutral underpaint, not black. This prevents the old silhouette problem.
        paths = make_underpaint_paths(visible, float(cfg["fill_simp"]), min_area, tile, max_path_bytes)
        # Use a neutral color close to the average visible color, but clamp away from pure black.
        pix = rgb[visible]
        if pix.size:
            avg = pix.mean(axis=0)
            avg = np.clip(avg * 0.82 + 35, 45, 230).astype(int)
            fill = short_hex(int(avg[0]), int(avg[1]), int(avg[2]))
        else:
            fill = "#ddd"
        items.extend(pack_paths_to_svgs(paths, fill, "00_underpaint", 0, w, h, target_bytes, "00_underpaint", note="optional pale underpaint; not black"))

    # Fills under lineart.
    items.extend(
        make_fill_layers(
            labels=labels,
            palette=palette,
            visible=visible,
            w=w,
            h=h,
            fill_simp=float(cfg["fill_simp"]),
            min_area=min_area,
            tile=tile,
            max_path_bytes=max_path_bytes,
            target_bytes=target_bytes,
            color_limit=int(cfg["colors"]),
        )
    )

    line_items: list[SvgItem] = []
    line_img = Image.new("RGBA", work.size, (0, 0, 0, 0))
    if cfg.get("lineart", True):
        line_items, line_img = make_lineart_layer(
            rgb=rgb,
            gray=gray,
            visible=visible,
            w=w,
            h=h,
            line_simp=float(cfg["line_simp"]),
            min_area=max(1, min_area),
            tile=tile,
            max_path_bytes=max_path_bytes,
            target_bytes=target_bytes,
            line_strength=float(cfg["line_strength"]),
            line_width=int(cfg["line_width"]),
            edge_close=int(cfg["edge_close"]),
            black=cfg.get("line_color", "#111"),
            keep_dark_edges_only=bool(cfg.get("dark_edges_only", False)),
        )
        items.extend(line_items)

    # Safety: optional cap total SVG count to avoid mobile crash.
    max_files = int(cfg["max_files"])
    if len(items) > max_files:
        items = sorted(items, key=lambda x: (0 if x.layer == "00_underpaint" else 1 if x.layer == "20_fills" else 2, x.bytes_size), reverse=False)[:max_files]
        # Re-sort final order.
        items.sort(key=lambda x: x.order)

    zip_bytes = zip_outputs(items, w, h, work, quant, line_img)
    total_svg_bytes = sum(it.bytes_size for it in items)

    return {
        "items": items,
        "zip": zip_bytes,
        "work": work,
        "quant": quant,
        "line": line_img,
        "w": w,
        "h": h,
        "elapsed": time.time() - start,
        "total_svg_bytes": total_svg_bytes,
        "max_path_bytes": max([it.max_path_bytes for it in items], default=0),
    }


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="GT7 SVG Inkscape Auto Builder v2", page_icon="🏁", layout="wide")
st.title("🏁 GT7 SVG Inkscape Auto Builder v2")
st.caption("黒いシルエット化を防ぐため、線画を“暗い面”ではなく“エッジだけ”から作る修正版です。")

with st.expander("今回の修正点", expanded=True):
    st.markdown(
        """
前の版は、線画レイヤーが **暗い色の領域全体** を拾ってしまい、髪・服・傘が巨大な黒ベタとして上に乗っていました。  
このv2では、線画を **Cannyエッジのみ** から作るため、黒いシルエットになりにくくしています。

まずは **下塗りOFF / 線画ON / 線幅1** で試してください。
"""
    )

uploaded = st.file_uploader("画像を選択", type=["png", "jpg", "jpeg", "webp"])

with st.sidebar:
    st.header("設定")
    preset_name = st.selectbox("品質プリセット", list(PRESETS.keys()) + ["手動設定"], index=0)
    base = PRESETS.get(preset_name, PRESETS["安定 / まずこれ"]).copy()

    st.subheader("基本")
    base["side"] = st.slider("処理サイズ 長辺px", 512, 1700, int(base["side"]), step=16)
    base["colors"] = st.slider("色面の色数", 16, 160, int(base["colors"]), step=4)
    base["target_kb"] = st.slider("各SVG目標KB", 8, 64, int(base["target_kb"]))
    base["path_kb"] = st.slider("1path最大KB", 6, 15, int(base["path_kb"]))
    base["tile"] = st.slider("内部タイル分割px", 128, 384, int(base["tile"]), step=32)
    base["max_files"] = st.slider("最大SVGファイル数", 20, 180, 90, step=10)

    st.subheader("塗り")
    base["fill_simp"] = st.slider("塗りpath簡略化", 0.10, 2.00, float(base["fill_simp"]), step=0.05)
    base["min_area"] = st.slider("小さい塗りの削除", 1, 20, int(base["min_area"]))
    underpaint = st.toggle("下塗りシルエットを入れる（淡色・任意）", False)

    st.subheader("線画")
    lineart = st.toggle("線画レイヤーを作る", True)
    base["line_simp"] = st.slider("線画path簡略化", 0.05, 1.50, float(base["line_simp"]), step=0.05)
    line_strength = st.slider("線画の検出量（小さいほど線が増える）", 0.10, 0.70, 0.38, step=0.02)
    line_width = st.slider("線画の太さ", 1, 4, 1)
    edge_close = st.slider("線の切れ目を少し繋ぐ", 0, 2, 0)
    dark_edges_only = st.toggle("暗い線付近を優先（実験）", False)
    line_color = st.selectbox("線色", ["#111", "#000", "#222", "#1b1f2a"], index=0)

    st.subheader("その他")
    enhance = st.toggle("低画質画像を軽く補正", True)
    alpha = st.slider("透明判定", 0, 255, 16)
    show_preview = st.toggle("ブラウザ内プレビューを表示", True)

    base.update(
        dict(
            underpaint=underpaint,
            lineart=lineart,
            line_strength=line_strength,
            line_width=line_width,
            edge_close=edge_close,
            dark_edges_only=dark_edges_only,
            line_color=line_color,
            enhance=enhance,
            alpha=alpha,
        )
    )

if uploaded is None:
    st.info("画像をアップロードしてください。")
    st.stop()

try:
    original = Image.open(uploaded).convert("RGBA")
except Exception as e:
    st.error(f"画像を開けませんでした: {e}")
    st.stop()

if st.button("SVGレイヤーを作成", type="primary", use_container_width=True):
    with st.spinner("変換中です…"):
        st.session_state.result = convert(original, base)
        st.session_state.filename = uploaded.name
        st.session_state.used = base.copy()

if "result" not in st.session_state:
    st.subheader("元画像")
    st.image(paste_on_checkerboard(original), use_container_width=True)
    st.stop()

res = st.session_state.result
items: list[SvgItem] = res["items"]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("SVG数", len(items))
c2.metric("総SVG容量", f"{res['total_svg_bytes']:,} B")
c3.metric("最大path", f"{res['max_path_bytes']:,} B")
c4.metric("ZIP容量", f"{len(res['zip']):,} B")
c5.metric("画像サイズ", f"{res['w']}×{res['h']}")
c6.metric("変換時間", f"{res['elapsed']:.1f}s")

if res["max_path_bytes"] > int(st.session_state.used["path_kb"]) * 1024:
    st.warning("最大pathが目標を少し超えています。内部タイル分割pxを小さくするか、path簡略化を上げてください。")
else:
    st.success("1pathサイズは目標内です。SVGOMGでは Merge paths をOFFにしてください。")

if len(items) >= int(st.session_state.used["max_files"]):
    st.warning("最大SVGファイル数に達しました。細部が省略されている可能性があります。最大SVG数を増やすか、色数を下げてください。")

st.subheader("確認")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.caption("Original")
    st.image(paste_on_checkerboard(res["work"]), use_container_width=True)
with col_b:
    st.caption("Fill reference")
    st.image(paste_on_checkerboard(res["quant"]), use_container_width=True)
with col_c:
    st.caption("Lineart mask（黒ベタではなくエッジだけ）")
    st.image(paste_on_checkerboard(res["line"]), use_container_width=True)

if show_preview:
    total_inline = sum(len(it.svg) for it in items)
    if total_inline < 1_300_000:
        st.subheader("実SVGレイヤープレビュー")
        html_preview = build_preview_html(items, res["w"], res["h"], original=None)
        components.html(html_preview, height=720, scrolling=True)
    else:
        st.info("SVG量が多いため、スマホ保護のためブラウザ内プレビューは省略しました。ZIP内の preview.html で確認してください。")

base_name = safe_name(st.session_state.filename)
st.download_button(
    "ZIPを保存",
    data=res["zip"],
    file_name=f"{base_name}_gt7_inkscape_auto_v2.zip",
    mime="application/zip",
    use_container_width=True,
)

with st.expander("SVG一覧"):
    rows = [
        dict(order=it.order, file=it.name, layer=it.layer, bytes=it.bytes_size, paths=it.path_count, max_path=it.max_path_bytes, note=it.note)
        for it in sorted(items, key=lambda x: x.order)
    ]
    st.dataframe(rows, use_container_width=True)

with st.expander("SVGOMGおすすめ設定"):
    st.markdown(
        """
- **Merge paths：OFF**
- **Remove viewBox：OFF**
- **Prettify markup：OFF**
- Remove metadata / comments：ON
- Convert colors：ON
- Round/rewrite paths：ON

`Merge paths`をONにすると、分割したpathが再結合され、15KBを超える可能性があります。
"""
    )
