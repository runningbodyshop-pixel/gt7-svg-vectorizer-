import base64
import io
import math
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st


# ============================================================
# Inkscape / GT7 oriented anime vectorizer
# ------------------------------------------------------------
# Goal:
# - Create SVG that is easier to edit in Inkscape than normal bitmap tracing.
# - Split output into semantic layers: underpaint, flats, shadows, highlights, lineart.
# - Keep simple SVG primitives only: path, rect, image. No filters.
# ============================================================


@dataclass
class SvgLayer:
    layer_id: str
    label: str
    elements: List[str]


# -----------------------------
# Basic helpers
# -----------------------------

def pil_to_rgba(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGBA")
    return img


def composite_on_white(rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    bg.alpha_composite(rgba)
    return bg.convert("RGB")


def resize_for_processing(img: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img.copy(), 1.0
    scale = max_side / float(longest)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS), scale


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = [v / 255.0 for v in rgb]
    return 255.0 * (0.2126 * r + 0.7152 * g + 0.0722 * b)


def fmt(v: float, precision: int) -> str:
    s = f"{v:.{precision}f}"
    s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def clean_svg_text(svg: str) -> str:
    # Light cleanup for readable/smaller SVG.
    svg = re.sub(r"\n\s+", "\n", svg)
    svg = re.sub(r"\s{2,}", " ", svg)
    return svg.strip() + "\n"


def png_data_uri(img: Image.Image, max_embed_side: int = 1400) -> str:
    embed = img.copy()
    w, h = embed.size
    longest = max(w, h)
    if longest > max_embed_side:
        scale = max_embed_side / float(longest)
        embed = embed.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    bio = io.BytesIO()
    embed.save(bio, format="PNG", optimize=True)
    encoded = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# -----------------------------
# Geometry helpers
# -----------------------------

def simplify_contour(contour: np.ndarray, epsilon: float, closed: bool = True) -> np.ndarray:
    if contour is None or len(contour) < 2:
        return np.empty((0, 2), dtype=np.float32)
    approx = cv2.approxPolyDP(contour, epsilon, closed)
    pts = approx.reshape(-1, 2).astype(np.float32)
    # Remove duplicate adjacent points.
    if len(pts) > 1:
        keep = [0]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) > 0.01:
                keep.append(i)
        pts = pts[keep]
    return pts


def catmull_rom_closed_path(pts: np.ndarray, precision: int, tension: float = 0.85) -> str:
    n = len(pts)
    if n < 3:
        return ""
    d = [f"M {fmt(pts[0][0], precision)} {fmt(pts[0][1], precision)}"]
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        c1 = p1 + (p2 - p0) * (tension / 6.0)
        c2 = p2 - (p3 - p1) * (tension / 6.0)
        d.append(
            "C "
            f"{fmt(c1[0], precision)} {fmt(c1[1], precision)} "
            f"{fmt(c2[0], precision)} {fmt(c2[1], precision)} "
            f"{fmt(p2[0], precision)} {fmt(p2[1], precision)}"
        )
    d.append("Z")
    return " ".join(d)


def poly_closed_path(pts: np.ndarray, precision: int) -> str:
    if len(pts) < 3:
        return ""
    parts = [f"M {fmt(pts[0][0], precision)} {fmt(pts[0][1], precision)}"]
    for p in pts[1:]:
        parts.append(f"L {fmt(p[0], precision)} {fmt(p[1], precision)}")
    parts.append("Z")
    return " ".join(parts)


def contour_to_path(contour: np.ndarray, epsilon: float, precision: int, smooth: bool) -> str:
    pts = simplify_contour(contour, epsilon, closed=True)
    if len(pts) < 3:
        return ""
    if smooth and len(pts) >= 5:
        return catmull_rom_closed_path(pts, precision=precision)
    return poly_closed_path(pts, precision=precision)


def catmull_rom_open_path(points_xy: np.ndarray, precision: int, tension: float = 0.75) -> str:
    n = len(points_xy)
    if n < 2:
        return ""
    if n < 4:
        parts = [f"M {fmt(points_xy[0][0], precision)} {fmt(points_xy[0][1], precision)}"]
        for p in points_xy[1:]:
            parts.append(f"L {fmt(p[0], precision)} {fmt(p[1], precision)}")
        return " ".join(parts)

    pts = points_xy.astype(np.float32)
    d = [f"M {fmt(pts[0][0], precision)} {fmt(pts[0][1], precision)}"]
    for i in range(n - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(i + 2, n - 1)]
        c1 = p1 + (p2 - p0) * (tension / 6.0)
        c2 = p2 - (p3 - p1) * (tension / 6.0)
        d.append(
            "C "
            f"{fmt(c1[0], precision)} {fmt(c1[1], precision)} "
            f"{fmt(c2[0], precision)} {fmt(c2[1], precision)} "
            f"{fmt(p2[0], precision)} {fmt(p2[1], precision)}"
        )
    return " ".join(d)


# -----------------------------
# Image analysis
# -----------------------------

def quantize_rgb(rgb: Image.Image, color_count: int) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    # PIL adaptive quantization is stable and lightweight on Streamlit Cloud.
    q = rgb.quantize(colors=int(color_count), method=Image.Quantize.MEDIANCUT)
    palette_raw = q.getpalette()[: color_count * 3]
    palette = []
    for i in range(color_count):
        base = i * 3
        palette.append(tuple(int(v) for v in palette_raw[base: base + 3]))
    labels = np.array(q, dtype=np.uint8)
    return labels, palette


def dominant_label(labels: np.ndarray) -> int:
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[np.argmax(counts)])


def clean_mask(mask: np.ndarray, min_area: int, close_px: int, expand_px: int = 0) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if expand_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1))
        m = cv2.dilate(m, k)
    if min_area <= 1:
        return m
    num, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[cc == i] = 255
    return out


def alpha_silhouette_mask(rgba: Image.Image, threshold: int, expand_px: int, close_px: int) -> np.ndarray:
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    m = (alpha >= threshold).astype(np.uint8) * 255
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if expand_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1))
        m = cv2.dilate(m, k)
    return m


def mask_to_filled_paths(
    mask: np.ndarray,
    fill_hex: str,
    epsilon: float,
    precision: int,
    smooth: bool,
    min_area: int,
    opacity: float = 1.0,
    extra_attrs: str = "",
) -> List[str]:
    elements: List[str] = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return elements
    hierarchy = hierarchy[0]
    for idx, cnt in enumerate(contours):
        parent = hierarchy[idx][3]
        if parent != -1:
            continue
        area = abs(cv2.contourArea(cnt))
        if area < min_area:
            continue
        subpaths = []
        p = contour_to_path(cnt, epsilon=epsilon, precision=precision, smooth=smooth)
        if p:
            subpaths.append(p)
        child = hierarchy[idx][2]
        while child != -1:
            child_cnt = contours[child]
            if abs(cv2.contourArea(child_cnt)) >= max(4, min_area * 0.05):
                cp = contour_to_path(child_cnt, epsilon=epsilon, precision=precision, smooth=smooth)
                if cp:
                    subpaths.append(cp)
            child = hierarchy[child][0]
        if subpaths:
            opacity_attr = "" if opacity >= 0.999 else f' opacity="{fmt(opacity, 3)}"'
            elements.append(
                f'<path d="{" ".join(subpaths)}" fill="{fill_hex}" fill-rule="evenodd"{opacity_attr} {extra_attrs}/>'
            )
    return elements


def classify_color_layer(rgb: Tuple[int, int, int], median_luma: float, area_ratio: float) -> str:
    lum = luminance(rgb)
    # Large bright areas are often base/flats, not highlights.
    if lum < median_luma - 32:
        return "shadows"
    if lum > median_luma + 42 and area_ratio < 0.18:
        return "highlights"
    return "flats"


# -----------------------------
# Line art extraction
# -----------------------------

def extract_edge_map(
    rgb_img: Image.Image,
    canny_low: int,
    canny_high: int,
    blur_size: int,
    dilate_px: int,
    include_dark: bool,
    dark_threshold: int,
) -> np.ndarray:
    arr = np.array(rgb_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if blur_size > 0:
        k = blur_size * 2 + 1
        gray = cv2.bilateralFilter(gray, k, 35, 35)
    edges = cv2.Canny(gray, int(canny_low), int(canny_high), L2gradient=True)
    if include_dark:
        # Add thin dark candidates, then use morphological gradient to avoid filling large dark hair/clothes.
        dark = (gray < int(dark_threshold)).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dark_edge = cv2.morphologyEx(dark, cv2.MORPH_GRADIENT, k)
        edges = cv2.bitwise_or(edges, dark_edge)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        edges = cv2.dilate(edges, k)
    return edges


def neighbors8(p: Tuple[int, int], pixels: set) -> List[Tuple[int, int]]:
    y, x = p
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            q = (y + dy, x + dx)
            if q in pixels:
                out.append(q)
    return out


def edge_key(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def trace_binary_lines(binary: np.ndarray, min_points: int, max_paths: int) -> List[np.ndarray]:
    # Graph trace for 1px-ish edge maps. Returns arrays of XY points.
    ys, xs = np.nonzero(binary > 0)
    if len(xs) == 0:
        return []
    pixels = set(zip(ys.tolist(), xs.tolist()))
    deg: Dict[Tuple[int, int], int] = {}
    for p in pixels:
        deg[p] = len(neighbors8(p, pixels))

    nodes = [p for p, d in deg.items() if d != 2]
    visited_edges = set()
    paths: List[np.ndarray] = []

    def add_path(path_yx: List[Tuple[int, int]]):
        if len(path_yx) >= min_points:
            xy = np.array([[x, y] for y, x in path_yx], dtype=np.float32)
            paths.append(xy)

    for start in nodes:
        for nb in neighbors8(start, pixels):
            ek = edge_key(start, nb)
            if ek in visited_edges:
                continue
            visited_edges.add(ek)
            path = [start, nb]
            prev, cur = start, nb
            safety = 0
            while deg.get(cur, 0) == 2 and safety < 20000:
                nbs = neighbors8(cur, pixels)
                if len(nbs) < 2:
                    break
                nxt = nbs[0] if nbs[1] == prev else nbs[1]
                ek2 = edge_key(cur, nxt)
                if ek2 in visited_edges:
                    break
                visited_edges.add(ek2)
                path.append(nxt)
                prev, cur = cur, nxt
                safety += 1
            add_path(path)
            if len(paths) >= max_paths:
                return paths

    # Remaining cycles where every point has degree 2.
    for start in list(pixels):
        for nb in neighbors8(start, pixels):
            ek = edge_key(start, nb)
            if ek in visited_edges:
                continue
            visited_edges.add(ek)
            path = [start, nb]
            prev, cur = start, nb
            safety = 0
            while safety < 20000:
                nbs = neighbors8(cur, pixels)
                candidates = [q for q in nbs if q != prev]
                if not candidates:
                    break
                nxt = candidates[0]
                ek2 = edge_key(cur, nxt)
                if ek2 in visited_edges:
                    break
                visited_edges.add(ek2)
                path.append(nxt)
                prev, cur = cur, nxt
                safety += 1
                if cur == start:
                    break
            add_path(path)
            if len(paths) >= max_paths:
                return paths
    return paths


def line_paths_to_svg(
    paths_xy: List[np.ndarray],
    stroke_hex: str,
    stroke_width: float,
    epsilon: float,
    precision: int,
    smooth: bool,
    opacity: float = 1.0,
) -> List[str]:
    elements: List[str] = []
    for pts in paths_xy:
        if len(pts) < 2:
            continue
        cnt = pts.reshape(-1, 1, 2).astype(np.float32)
        simp = simplify_contour(cnt, epsilon=epsilon, closed=False)
        if len(simp) < 2:
            continue
        if smooth:
            d = catmull_rom_open_path(simp, precision=precision)
        else:
            parts = [f"M {fmt(simp[0][0], precision)} {fmt(simp[0][1], precision)}"]
            for p in simp[1:]:
                parts.append(f"L {fmt(p[0], precision)} {fmt(p[1], precision)}")
            d = " ".join(parts)
        elements.append(
            f'<path d="{d}" fill="none" stroke="{stroke_hex}" stroke-width="{fmt(stroke_width, 2)}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="{fmt(opacity, 3)}"/>'
        )
    return elements


# -----------------------------
# SVG builders
# -----------------------------

def build_svg(width: int, height: int, layers: List[SvgLayer], include_metadata: bool = True) -> str:
    header = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" version="1.1">'
    ]
    if include_metadata:
        header.append(
            '<metadata>Generated by Anime Inkscape Layer Vectorizer. '
            'Layers are intended for manual editing in Inkscape and GT7 decal preparation.</metadata>'
        )
    body = []
    for layer in layers:
        if not layer.elements:
            continue
        body.append(
            f'<g id="{layer.layer_id}" inkscape:groupmode="layer" inkscape:label="{layer.label}">'
        )
        body.extend(layer.elements)
        body.append('</g>')
    body.append('</svg>')
    return clean_svg_text("\n".join(header + body))


def build_layer_svg(width: int, height: int, layer: SvgLayer) -> str:
    return build_svg(width, height, [layer], include_metadata=False)


def vectorize(
    rgba_src: Image.Image,
    max_side: int,
    color_count: int,
    min_area: int,
    close_px: int,
    underpaint_expand: int,
    color_expand: int,
    smooth_paths: bool,
    precision: int,
    color_epsilon: float,
    include_background: bool,
    include_reference: bool,
    ref_opacity: float,
    line_mode: str,
    canny_low: int,
    canny_high: int,
    line_blur: int,
    line_dilate: int,
    include_dark_lines: bool,
    dark_threshold: int,
    line_min_points: int,
    line_max_paths: int,
    line_epsilon: float,
    stroke_width: float,
    stroke_color: str,
    filled_lineart: bool,
    filled_line_threshold: int,
) -> Tuple[str, List[SvgLayer], Dict[str, str]]:
    rgba_proc, scale = resize_for_processing(rgba_src, max_side=max_side)
    rgb_proc = composite_on_white(rgba_proc)
    width, height = rgb_proc.size

    layers: List[SvgLayer] = []

    # 00 reference image
    if include_reference:
        uri = png_data_uri(rgb_proc)
        layers.append(
            SvgLayer(
                "00_reference",
                "00_reference_original_locked",
                [f'<image href="{uri}" x="0" y="0" width="{width}" height="{height}" opacity="{fmt(ref_opacity, 2)}"/>'],
            )
        )

    labels, palette = quantize_rgb(rgb_proc, color_count=color_count)
    bg_label = dominant_label(labels)
    bg_color = palette[bg_label]

    flat_elements: List[str] = []
    shadow_elements: List[str] = []
    highlight_elements: List[str] = []
    underpaint_elements: List[str] = []

    # Underpaint: transparent PNG gets silhouette. Non-transparent image gets dominant color rectangle.
    arr_alpha = np.array(rgba_proc)[:, :, 3]
    has_transparency = bool(np.min(arr_alpha) < 250)
    if has_transparency:
        sil = alpha_silhouette_mask(rgba_proc, threshold=10, expand_px=underpaint_expand, close_px=max(1, close_px))
        fill = rgb_to_hex(bg_color)
        # Use a neutral average of non-transparent pixels if possible.
        arr_rgb = np.array(rgb_proc)
        inside = arr_alpha > 10
        if np.any(inside):
            mean_rgb = tuple(np.clip(np.mean(arr_rgb[inside], axis=0), 0, 255).astype(np.uint8).tolist())
            fill = rgb_to_hex(mean_rgb)
        underpaint_elements.extend(
            mask_to_filled_paths(sil, fill, epsilon=max(1.0, color_epsilon * 1.5), precision=precision, smooth=smooth_paths, min_area=max(8, min_area // 2))
        )
    else:
        if include_background:
            underpaint_elements.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{rgb_to_hex(bg_color)}"/>')

    total_px = width * height
    lumas = []
    areas = []
    for lab in range(len(palette)):
        area = int(np.sum(labels == lab))
        if area > 0:
            lumas.append(luminance(palette[lab]))
            areas.append(area)
    if areas:
        # weighted-ish median by simple expanded representative list avoidance.
        median_luma = float(np.median(lumas))
    else:
        median_luma = 128.0

    # Color layers. Larger components are drawn first; smaller detail after.
    color_items = []
    for lab, rgb in enumerate(palette):
        raw_area = int(np.sum(labels == lab))
        if raw_area <= 0:
            continue
        if (not include_background) and lab == bg_label and not has_transparency:
            continue
        color_items.append((raw_area, lab, rgb))
    color_items.sort(reverse=True, key=lambda x: x[0])

    for raw_area, lab, rgb in color_items:
        mask = (labels == lab).astype(np.uint8) * 255
        # Underpaint and adjacent-color gap protection.
        m = clean_mask(mask, min_area=min_area, close_px=close_px, expand_px=color_expand)
        if np.count_nonzero(m) < min_area:
            continue
        layer_kind = classify_color_layer(rgb, median_luma=median_luma, area_ratio=raw_area / total_px)
        elems = mask_to_filled_paths(
            m,
            fill_hex=rgb_to_hex(rgb),
            epsilon=color_epsilon,
            precision=precision,
            smooth=smooth_paths,
            min_area=min_area,
        )
        if layer_kind == "shadows":
            shadow_elements.extend(elems)
        elif layer_kind == "highlights":
            highlight_elements.extend(elems)
        else:
            flat_elements.extend(elems)

    layers.append(SvgLayer("01_underpaint", "01_underpaint_gap_guard", underpaint_elements))
    layers.append(SvgLayer("02_flats", "02_flats_large_color_shapes", flat_elements))
    layers.append(SvgLayer("03_shadows", "03_shadows", shadow_elements))
    layers.append(SvgLayer("04_highlights", "04_highlights", highlight_elements))

    line_elements: List[str] = []
    if line_mode != "なし":
        edge = extract_edge_map(
            rgb_proc,
            canny_low=canny_low,
            canny_high=canny_high,
            blur_size=line_blur,
            dilate_px=line_dilate,
            include_dark=include_dark_lines,
            dark_threshold=dark_threshold,
        )
        if line_mode == "中心線stroke":
            paths_xy = trace_binary_lines(edge, min_points=line_min_points, max_paths=line_max_paths)
            line_elements.extend(
                line_paths_to_svg(
                    paths_xy,
                    stroke_hex=stroke_color,
                    stroke_width=stroke_width,
                    epsilon=line_epsilon,
                    precision=precision,
                    smooth=smooth_paths,
                    opacity=1.0,
                )
            )
        elif line_mode == "エッジ塗り形状":
            m = clean_mask(edge, min_area=max(4, line_min_points), close_px=0, expand_px=max(1, line_dilate))
            line_elements.extend(
                mask_to_filled_paths(
                    m,
                    fill_hex=stroke_color,
                    epsilon=max(0.4, line_epsilon),
                    precision=precision,
                    smooth=smooth_paths,
                    min_area=max(4, line_min_points),
                )
            )

    if filled_lineart:
        gray = cv2.cvtColor(np.array(rgb_proc), cv2.COLOR_RGB2GRAY)
        dark = (gray < filled_line_threshold).astype(np.uint8) * 255
        dark = clean_mask(dark, min_area=max(6, min_area // 5), close_px=1, expand_px=0)
        line_elements.extend(
            mask_to_filled_paths(
                dark,
                fill_hex=stroke_color,
                epsilon=max(0.6, line_epsilon),
                precision=precision,
                smooth=smooth_paths,
                min_area=max(6, min_area // 5),
                opacity=0.96,
                extra_attrs='data-note="filled-dark-lineart"',
            )
        )

    layers.append(SvgLayer("05_lineart", "05_lineart_top_editable", line_elements))

    full_svg = build_svg(width, height, layers)
    file_map = {
        "full_editable.svg": full_svg,
        "lineart_only.svg": build_svg(width, height, [layers[-1]], include_metadata=False),
        "colors_only.svg": build_svg(width, height, layers[1:-1], include_metadata=False),
    }
    for layer in layers:
        if layer.elements:
            safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", layer.layer_id)
            file_map[f"layers/{safe_name}.svg"] = build_layer_svg(width, height, layer)

    palette_lines = ["# palette / quantized colors"]
    for i, rgb in enumerate(palette):
        palette_lines.append(f"{i:02d}\t{rgb_to_hex(rgb)}\tRGB{rgb}\tarea={int(np.sum(labels == i))}")
    file_map["palette.txt"] = "\n".join(palette_lines) + "\n"
    return full_svg, layers, file_map


def make_zip(file_map: Dict[str, str]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in file_map.items():
            zf.writestr(name, data.encode("utf-8"))
    return bio.getvalue()


def byte_size_text(s: str) -> str:
    n = len(s.encode("utf-8"))
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Anime Inkscape Layer Vectorizer", page_icon="✒️", layout="wide")

st.title("✒️ Anime Inkscape Layer Vectorizer")
st.caption("Inkscapeで後編集しやすい、レイヤー分けSVGを作る実験版。GT7デカール分割の前段階にも使えます。")

with st.expander("このアプリの狙い", expanded=False):
    st.markdown(
        """
- 通常の画像トレースより、**Inkscapeで編集しやすいSVG**を作ることを優先します。
- 色面を `underpaint / flats / shadows / highlights` に分けます。
- 線画は `stroke` または `塗り形状` として上に重ねます。
- GT7向けに使う場合は、まず `full_editable.svg` をInkscapeで調整し、必要に応じてレイヤー別SVGを軽量化してください。
        """
    )

uploaded = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg", "webp"])

with st.sidebar:
    st.header("プリセット")
    preset = st.selectbox(
        "用途",
        [
            "Inkscape高品質",
            "線画優先",
            "GT7軽量寄り",
            "細部多め",
        ],
        index=0,
    )

    if preset == "Inkscape高品質":
        default_max_side = 1000
        default_colors = 18
        default_min_area = 28
        default_epsilon = 1.15
        default_line_eps = 1.25
        default_precision = 2
        default_smooth = True
        default_line_mode = "中心線stroke"
    elif preset == "線画優先":
        default_max_side = 1100
        default_colors = 12
        default_min_area = 22
        default_epsilon = 1.45
        default_line_eps = 0.8
        default_precision = 2
        default_smooth = True
        default_line_mode = "中心線stroke"
    elif preset == "GT7軽量寄り":
        default_max_side = 800
        default_colors = 10
        default_min_area = 70
        default_epsilon = 2.2
        default_line_eps = 2.0
        default_precision = 1
        default_smooth = False
        default_line_mode = "エッジ塗り形状"
    else:
        default_max_side = 1200
        default_colors = 28
        default_min_area = 12
        default_epsilon = 0.75
        default_line_eps = 0.65
        default_precision = 2
        default_smooth = True
        default_line_mode = "中心線stroke"

    st.header("基本")
    max_side = st.slider("処理サイズ 最大辺", 400, 1800, default_max_side, 50)
    color_count = st.slider("色数", 4, 48, default_colors, 1)
    min_area = st.slider("小さいゴミ除去", 1, 250, default_min_area, 1)
    close_px = st.slider("色面の穴埋め/接続", 0, 5, 1, 1)
    underpaint_expand = st.slider("下塗りの外側拡張", 0, 8, 2, 1)
    color_expand = st.slider("色面の重なり拡張", 0, 3, 0, 1)
    color_epsilon = st.slider("色面ノード削減", 0.2, 5.0, float(default_epsilon), 0.05)
    precision = st.slider("座標小数桁", 0, 3, default_precision, 1)
    smooth_paths = st.checkbox("ベジェ曲線で滑らかにする", value=default_smooth)
    include_background = st.checkbox("背景色も含める", value=False)
    include_reference = st.checkbox("元画像を薄く埋め込む", value=True)
    ref_opacity = st.slider("元画像レイヤー濃度", 0.05, 0.8, 0.22, 0.01)

    st.header("線画")
    line_mode = st.selectbox("線画方式", ["中心線stroke", "エッジ塗り形状", "なし"], index=["中心線stroke", "エッジ塗り形状", "なし"].index(default_line_mode))
    canny_low = st.slider("エッジ弱", 10, 180, 55, 1)
    canny_high = st.slider("エッジ強", 30, 260, 145, 1)
    line_blur = st.slider("線画のノイズ抑制", 0, 5, 1, 1)
    line_dilate = st.slider("線画の太らせ", 0, 3, 0, 1)
    include_dark_lines = st.checkbox("暗い線候補を追加", value=True)
    dark_threshold = st.slider("暗い線しきい値", 20, 170, 95, 1)
    line_min_points = st.slider("短い線を削除", 2, 80, 8, 1)
    line_max_paths = st.slider("線パス最大数", 100, 8000, 2500, 100)
    line_epsilon = st.slider("線画ノード削減", 0.1, 5.0, float(default_line_eps), 0.05)
    stroke_width = st.slider("stroke線幅", 0.2, 8.0, 1.4, 0.1)
    stroke_color = st.color_picker("線色", "#1b1414")
    filled_lineart = st.checkbox("黒線を塗り形状でも追加", value=False)
    filled_line_threshold = st.slider("黒線塗り形状しきい値", 20, 160, 70, 1)

if uploaded is None:
    st.info("PNG/JPG/WebP画像をアップロードしてください。")
    st.stop()

src = Image.open(uploaded)
rgba = pil_to_rgba(src)

left, right = st.columns([1, 1])
with left:
    st.subheader("元画像")
    st.image(rgba, use_container_width=True)

run = st.button("SVGを生成", type="primary", use_container_width=True)

if not run:
    st.stop()

with st.spinner("レイヤー分けSVGを生成しています…"):
    full_svg, layers, file_map = vectorize(
        rgba_src=rgba,
        max_side=max_side,
        color_count=color_count,
        min_area=min_area,
        close_px=close_px,
        underpaint_expand=underpaint_expand,
        color_expand=color_expand,
        smooth_paths=smooth_paths,
        precision=precision,
        color_epsilon=color_epsilon,
        include_background=include_background,
        include_reference=include_reference,
        ref_opacity=ref_opacity,
        line_mode=line_mode,
        canny_low=canny_low,
        canny_high=canny_high,
        line_blur=line_blur,
        line_dilate=line_dilate,
        include_dark_lines=include_dark_lines,
        dark_threshold=dark_threshold,
        line_min_points=line_min_points,
        line_max_paths=line_max_paths,
        line_epsilon=line_epsilon,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        filled_lineart=filled_lineart,
        filled_line_threshold=filled_line_threshold,
    )

zip_bytes = make_zip(file_map)

with right:
    st.subheader("生成結果")
    st.write(f"full_editable.svg: **{byte_size_text(full_svg)}**")
    st.write(f"ZIP: **{len(zip_bytes) / 1024:.1f} KB**")
    layer_rows = []
    for layer in layers:
        svg = build_layer_svg(
            int(re.search(r'width="(\d+)"', full_svg).group(1)),
            int(re.search(r'height="(\d+)"', full_svg).group(1)),
            layer,
        )
        layer_rows.append({"layer": layer.label, "elements": len(layer.elements), "size": byte_size_text(svg)})
    st.dataframe(layer_rows, use_container_width=True, hide_index=True)

st.download_button(
    "full_editable.svg をダウンロード",
    data=full_svg.encode("utf-8"),
    file_name="full_editable.svg",
    mime="image/svg+xml",
    use_container_width=True,
)
st.download_button(
    "レイヤー別SVG ZIPをダウンロード",
    data=zip_bytes,
    file_name="inkscape_layer_vectorized.zip",
    mime="application/zip",
    use_container_width=True,
)

st.subheader("SVGコード確認")
st.code(full_svg[:20000] + ("\n<!-- 省略: SVGが長いため先頭のみ表示 -->" if len(full_svg) > 20000 else ""), language="xml")

st.subheader("調整のコツ")
st.markdown(
    """
- 形がガタガタする場合：`ベジェ曲線で滑らかにする` をON、`色面ノード削減` を少し上げる。
- 顔や目が潰れる場合：`色数` を増やし、`小さいゴミ除去` と `線画ノード削減` を下げる。
- SVGが重い場合：`処理サイズ 最大辺`、`色数`、`線パス最大数` を下げる。
- GT7用に近づける場合：`GT7軽量寄り` プリセット、`座標小数桁=1`、`背景色OFF`、`元画像レイヤーOFF` が基本。
- Inkscapeで開いたら、`05_lineart_top_editable` を最上段にし、不要な色面を消して調整すると扱いやすいです。
    """
)
