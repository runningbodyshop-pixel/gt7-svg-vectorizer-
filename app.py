from __future__ import annotations

import base64
import csv
import io
import math
import os
import re
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageOps, ImageFilter

APP_TITLE = "GT7 SVG Paint Stack Edition"
HARD_LIMIT = 2 * 1024 * 1024
GT7_TARGET = 14 * 1024

PRESETS = {
    "安定 / まずこれ": {
        "side": 1200,
        "target_kb": 14,
        "line_target_kb": 14,
        "max_path_kb": 11,
        "colors": 48,
        "simplify": 1.6,
        "min_area": 18,
        "line_simplify": 1.2,
        "line_width": 1,
        "line_low": 70,
        "line_high": 150,
        "hair_shadow_strength": 0.42,
        "hair_highlight_strength": 0.58,
        "cloth_shadow_strength": 0.38,
        "cloth_highlight_strength": 0.64,
        "gradient": True,
    },
    "高品質 / 仕上げ用": {
        "side": 1450,
        "target_kb": 14,
        "line_target_kb": 14,
        "max_path_kb": 11,
        "colors": 64,
        "simplify": 1.1,
        "min_area": 12,
        "line_simplify": 0.9,
        "line_width": 1,
        "line_low": 60,
        "line_high": 135,
        "hair_shadow_strength": 0.45,
        "hair_highlight_strength": 0.60,
        "cloth_shadow_strength": 0.40,
        "cloth_highlight_strength": 0.66,
        "gradient": True,
    },
    "軽量 / 速い": {
        "side": 1000,
        "target_kb": 12,
        "line_target_kb": 12,
        "max_path_kb": 10,
        "colors": 40,
        "simplify": 2.0,
        "min_area": 24,
        "line_simplify": 1.5,
        "line_width": 1,
        "line_low": 90,
        "line_high": 180,
        "hair_shadow_strength": 0.40,
        "hair_highlight_strength": 0.56,
        "cloth_shadow_strength": 0.36,
        "cloth_highlight_strength": 0.62,
        "gradient": False,
    },
}


@dataclass
class LayerShape:
    path_d: str
    bbox: Tuple[int, int, int, int]


@dataclass
class LayerSpec:
    name: str
    order: int
    fill: str
    shapes: List[LayerShape]
    gradient: Optional[dict] = None
    stroke: Optional[dict] = None


def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG", optimize=True)
    return b.getvalue()


def fit_image(img: Image.Image, side: int, enhance: bool) -> Image.Image:
    img = img.convert("RGBA")
    if enhance:
        rgb = ImageOps.autocontrast(img.convert("RGB"))
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=3))
        img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))
    if max(img.size) > side:
        scale = side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    return img


def short_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [int(max(0, min(255, c))) for c in rgb]
    s = f"{r:02x}{g:02x}{b:02x}"
    if s[0] == s[1] and s[2] == s[3] and s[4] == s[5]:
        return f"#{s[0]}{s[2]}{s[4]}"
    return f"#{s}"


def blend_color(rgb: Tuple[int, int, int], factor: float, toward_white: bool) -> Tuple[int, int, int]:
    arr = np.array(rgb, dtype=np.float32)
    tgt = np.array([255, 255, 255] if toward_white else [0, 0, 0], dtype=np.float32)
    out = np.clip(arr * (1.0 - factor) + tgt * factor, 0, 255)
    return tuple(int(round(x)) for x in out)


def average_color(img_rgba: np.ndarray, mask: np.ndarray, fallback=(128, 128, 128)) -> Tuple[int, int, int]:
    pts = img_rgba[mask > 0]
    if len(pts) == 0:
        return fallback
    rgb = pts[:, :3].mean(axis=0)
    return tuple(int(round(x)) for x in rgb)


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def component_filter(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.max() == 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out


def simplify_contour(cnt: np.ndarray, eps: float) -> np.ndarray:
    return cv2.approxPolyDP(cnt, max(0.2, float(eps)), True)


def contour_to_path(cnt: np.ndarray) -> Optional[str]:
    if cnt is None or len(cnt) < 3:
        return None
    pts = cnt.reshape(-1, 2)
    if len(pts) < 3:
        return None
    cmds = [f"M{int(pts[0,0])} {int(pts[0,1])}"]
    for x, y in pts[1:]:
        cmds.append(f"L{int(x)} {int(y)}")
    cmds.append("Z")
    return "".join(cmds)


def mask_to_shapes(mask: np.ndarray, simplify: float, min_area: int) -> List[LayerShape]:
    mask = component_filter(mask, min_area)
    if mask.max() == 0:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes: List[LayerShape] = []
    for cnt in contours:
        if abs(cv2.contourArea(cnt)) < min_area:
            continue
        cnt2 = simplify_contour(cnt, simplify)
        d = contour_to_path(cnt2)
        if not d:
            continue
        x, y, w, h = cv2.boundingRect(cnt2)
        shapes.append(LayerShape(d, (x, y, x + w, y + h)))
    shapes.sort(key=lambda s: (s.bbox[1], s.bbox[0]))
    return shapes


def dominant_subject_mask(alpha: np.ndarray) -> np.ndarray:
    vis = (alpha > 15).astype(np.uint8) * 255
    vis = cv2.morphologyEx(vis, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    vis = component_filter(vis, 80)
    return vis


def create_region_guides(mask: np.ndarray):
    h, w = mask.shape
    bb = mask_bbox(mask)
    if not bb:
        return None
    x0, y0, x1, y1 = bb
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    face = np.zeros_like(mask, dtype=np.uint8)
    cx = int(x0 + bw * 0.50)
    cy = int(y0 + bh * 0.23)
    axes = (max(8, int(bw * 0.12)), max(8, int(bh * 0.12)))
    cv2.ellipse(face, (cx, cy), axes, 0, 0, 360, 255, -1)

    head = np.zeros_like(mask, dtype=np.uint8)
    cv2.ellipse(head, (cx, int(y0 + bh * 0.28)), (max(10, int(bw * 0.28)), max(10, int(bh * 0.26))), 0, 0, 360, 255, -1)

    upper = np.zeros_like(mask, dtype=np.uint8)
    upper[y0:int(y0 + bh * 0.55), x0:x1] = 255

    lower = np.zeros_like(mask, dtype=np.uint8)
    lower[int(y0 + bh * 0.45):y1, x0:x1] = 255

    body = np.zeros_like(mask, dtype=np.uint8)
    body[int(y0 + bh * 0.25):y1, x0:x1] = 255

    return {"bbox": bb, "face": face, "head": head, "upper": upper, "lower": lower, "body": body}


def detect_skin(bgr: np.ndarray, visible: np.ndarray, guides: dict) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    skin = ((cr >= 133) & (cr <= 178) & (cb >= 77) & (cb <= 135) & (y > 70)).astype(np.uint8) * 255
    skin = cv2.bitwise_and(skin, visible)
    skin = cv2.medianBlur(skin, 5)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    # hands/face prefer upper-middle, but keep lower small components too
    face = guides["face"]
    preferred = cv2.bitwise_or(face, cv2.dilate(face, np.ones((29, 29), np.uint8), iterations=1))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((skin > 0).astype(np.uint8), 8)
    out = np.zeros_like(skin)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        comp = (labels == i).astype(np.uint8) * 255
        hit_pref = cv2.bitwise_and(comp, preferred).max() > 0
        if area >= 20 and (hit_pref or area >= 120):
            out[labels == i] = 255
    return out


def detect_hair(bgr: np.ndarray, visible: np.ndarray, skin: np.ndarray, guides: dict) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    upper = guides["upper"]
    head = guides["head"]
    not_skin = cv2.bitwise_and(visible, cv2.bitwise_not(skin))
    darkish = (((v < 180) & (s > 10)) | (v < 115)).astype(np.uint8) * 255
    cand = cv2.bitwise_and(darkish, not_skin)
    cand = cv2.bitwise_and(cand, cv2.bitwise_or(upper, cv2.dilate(head, np.ones((61, 61), np.uint8), iterations=1)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    cand = component_filter(cand, 70)

    # Keep components near head or large flowing components
    n, labels, stats, _ = cv2.connectedComponentsWithStats((cand > 0).astype(np.uint8), 8)
    out = np.zeros_like(cand)
    head_big = cv2.dilate(head, np.ones((81, 81), np.uint8), iterations=1)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        comp = (labels == i).astype(np.uint8) * 255
        touches_head = cv2.bitwise_and(comp, head_big).max() > 0
        x, y, w, h2, _ = stats[i]
        tall = h2 > visible.shape[0] * 0.15
        if area >= 80 and (touches_head or tall):
            out[labels == i] = 255
    return out


def detect_clothes_and_props(visible: np.ndarray, skin: np.ndarray, hair: np.ndarray, guides: dict):
    remaining = cv2.bitwise_and(visible, cv2.bitwise_not(cv2.bitwise_or(skin, hair)))
    lower = guides["lower"]
    upper = guides["upper"]
    clothes = cv2.bitwise_and(remaining, lower)
    clothes = cv2.morphologyEx(clothes, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    clothes = component_filter(clothes, 50)

    props = cv2.bitwise_and(remaining, upper)
    props = component_filter(props, 60)

    # accessories: leftovers not in clothes/props but visible
    used = cv2.bitwise_or(cv2.bitwise_or(skin, hair), cv2.bitwise_or(clothes, props))
    accessories = cv2.bitwise_and(visible, cv2.bitwise_not(used))
    accessories = component_filter(accessories, 40)
    return clothes, props, accessories


def extract_shadow_highlight(src_rgba: np.ndarray, base_mask: np.ndarray, dark_q: float, light_q: float):
    if base_mask.max() == 0:
        empty = np.zeros(base_mask.shape, dtype=np.uint8)
        return empty, empty
    rgb = src_rgba[:, :, :3].astype(np.float32)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    vals = lum[base_mask > 0]
    if len(vals) == 0:
        empty = np.zeros(base_mask.shape, dtype=np.uint8)
        return empty, empty
    dark_t = np.quantile(vals, dark_q)
    light_t = np.quantile(vals, light_q)
    shadow = ((lum <= dark_t) & (base_mask > 0)).astype(np.uint8) * 255
    highlight = ((lum >= light_t) & (base_mask > 0)).astype(np.uint8) * 255
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    shadow = component_filter(shadow, 20)
    highlight = cv2.morphologyEx(highlight, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    highlight = cv2.morphologyEx(highlight, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    highlight = component_filter(highlight, 20)
    return shadow, highlight


def detect_face_details(bgr: np.ndarray, skin: np.ndarray, hair: np.ndarray, guides: dict) -> np.ndarray:
    bb = mask_bbox(cv2.bitwise_or(skin, guides["face"])) or guides["bbox"]
    h, w = skin.shape
    x0, y0, x1, y1 = bb
    x0 = max(0, x0 - 8); y0 = max(0, y0 - 8); x1 = min(w, x1 + 8); y1 = min(h, y1 + 8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return np.zeros_like(skin)
    edges = cv2.Canny(roi, 50, 130)
    dark = (roi < np.percentile(roi, 30)).astype(np.uint8) * 255
    face_mask = guides["face"][y0:y1, x0:x1]
    hair_roi = hair[y0:y1, x0:x1]
    det = cv2.bitwise_or(edges, dark)
    det = cv2.bitwise_and(det, cv2.dilate(face_mask, np.ones((7, 7), np.uint8), iterations=1))
    det = cv2.bitwise_and(det, cv2.bitwise_not(cv2.dilate(hair_roi, np.ones((5, 5), np.uint8), iterations=1)))
    det = cv2.dilate(det, np.ones((2, 2), np.uint8), iterations=1)
    det = component_filter(det, 6)
    out = np.zeros_like(skin)
    out[y0:y1, x0:x1] = det
    return out


def detect_lineart(bgr: np.ndarray, visible: np.ndarray, low: int, high: int, width: int) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges = cv2.bitwise_and(edges, visible)
    if width > 1:
        edges = cv2.dilate(edges, np.ones((width, width), np.uint8), iterations=1)
    else:
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges = component_filter(edges, 8)
    return edges


def shapes_bbox(shapes: List[LayerShape]) -> Optional[Tuple[int, int, int, int]]:
    if not shapes:
        return None
    xs0 = [s.bbox[0] for s in shapes]
    ys0 = [s.bbox[1] for s in shapes]
    xs1 = [s.bbox[2] for s in shapes]
    ys1 = [s.bbox[3] for s in shapes]
    return min(xs0), min(ys0), max(xs1), max(ys1)


def gradient_def(grad_id: str, bbox: Tuple[int, int, int, int], c0: str, c1: str, mode: str) -> str:
    x0, y0, x1, y1 = bbox
    if mode == "vertical":
        return (
            f'<defs><linearGradient id="{grad_id}" gradientUnits="userSpaceOnUse" '
            f'x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}">'
            f'<stop offset="0%" stop-color="{c0}"/><stop offset="100%" stop-color="{c1}"/>'
            f'</linearGradient></defs>'
        )
    return (
        f'<defs><linearGradient id="{grad_id}" gradientUnits="userSpaceOnUse" '
        f'x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}">'
        f'<stop offset="0%" stop-color="{c0}"/><stop offset="100%" stop-color="{c1}"/>'
        f'</linearGradient></defs>'
    )


def split_shapes_by_size(shapes: List[LayerShape], limit_bytes: int, overhead: int = 400) -> List[List[LayerShape]]:
    if not shapes:
        return []
    groups: List[List[LayerShape]] = []
    cur: List[LayerShape] = []
    cur_len = overhead
    for s in shapes:
        est = len(s.path_d.encode("utf-8")) + 40
        if cur and cur_len + est > limit_bytes:
            groups.append(cur)
            cur = [s]
            cur_len = overhead + est
        else:
            cur.append(s)
            cur_len += est
    if cur:
        groups.append(cur)
    return groups


def make_svg_part(width: int, height: int, layer_name: str, part_index: int, total_parts: int, shapes: List[LayerShape], fill: str, gradient: Optional[dict], stroke: Optional[dict]) -> str:
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">']
    fill_ref = fill
    if gradient and shapes:
        bb = shapes_bbox(shapes)
        if bb:
            gid = f"g_{re.sub(r'[^A-Za-z0-9]+', '_', layer_name)}_{part_index}"
            parts.append(gradient_def(gid, bb, gradient["c0"], gradient["c1"], gradient.get("mode", "vertical")))
            fill_ref = f"url(#{gid})"
    if stroke:
        parts.append(
            f'<g fill="none" stroke="{stroke["color"]}" stroke-width="{stroke["width"]}" '
            f'stroke-linecap="round" stroke-linejoin="round">'
        )
        for shp in shapes:
            parts.append(f'<path d="{shp.path_d}"/>')
        parts.append('</g>')
    else:
        parts.append(f'<g fill="{fill_ref}">')
        for shp in shapes:
            parts.append(f'<path d="{shp.path_d}"/>')
        parts.append('</g>')
    parts.append('</svg>')
    return ''.join(parts)


def layer_to_files(width: int, height: int, spec: LayerSpec, target_kb: int, max_path_kb: int) -> List[Tuple[str, str]]:
    if not spec.shapes:
        return []
    # each path already separate; split groups by target size
    groups = split_shapes_by_size(spec.shapes, target_kb * 1024, overhead=500)
    out: List[Tuple[str, str]] = []
    total = len(groups)
    for i, g in enumerate(groups, start=1):
        name = f"{spec.order:03d}_{spec.name}_{i:02d}.svg"
        svg = make_svg_part(width, height, spec.name, i, total, g, spec.fill, spec.gradient, spec.stroke)
        out.append((name, svg))
    return out


def html_preview(files: List[Tuple[str, str]], width: int, height: int) -> str:
    layers_html = []
    for name, svg in files:
        encoded = base64.b64encode(svg.encode('utf-8')).decode('ascii')
        layers_html.append(
            f'<img alt="{name}" src="data:image/svg+xml;base64,{encoded}" '
            f'style="position:absolute;left:0;top:0;width:100%;height:100%;image-rendering:auto;"/>'
        )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>preview</title></head>
<body style='font-family:sans-serif;margin:20px'>
<h2>GT7 SVG Paint Stack Preview</h2>
<p>All SVGs share the same canvas. Place them in GT7 in manifest order.</p>
<div style='position:relative;width:min(90vw,{width}px);aspect-ratio:{width}/{height};border:1px solid #ccc;background:#eee'>
{''.join(layers_html)}
</div>
</body></html>"""


def placement_guide(width: int, height: int, layer_names: List[str]) -> bytes:
    img = Image.new("RGBA", (min(1400, max(600, width)), int(min(1400, max(600, width)) * (height / max(width,1))) + 140), (245,245,245,255))
    dr = ImageDraw.Draw(img)
    dr.text((20, 18), "GT7 Paint Stack Layer Order", fill=(0,0,0,255))
    y = 52
    for i, name in enumerate(layer_names, start=1):
        dr.text((20, y), f"{i:02d}. {name}", fill=(20,20,20,255))
        y += 22
    return png_bytes(img)


def build_zip(package: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, svg in package['files']:
            z.writestr(name, svg)
        z.writestr('preview.html', package['preview_html'])
        z.writestr('manifest.csv', package['manifest_csv'])
        z.writestr('README.txt', package['readme'])
        z.writestr('placement_guide.png', package['guide_png'])
    return buf.getvalue()


def csv_manifest(files: List[Tuple[str, str]]) -> str:
    out = io.StringIO()
    wr = csv.writer(out)
    wr.writerow(["order", "filename", "bytes"])
    for i, (name, svg) in enumerate(files, start=1):
        wr.writerow([i, name, len(svg.encode('utf-8'))])
    return out.getvalue()


def safe_name(s: str) -> str:
    s = re.sub(r'\.[^.]+$', '', s)
    s = re.sub(r'[^A-Za-z0-9_-]+', '_', s).strip('_')
    return s or 'output'


def render_side_by_side(orig: Image.Image, files: List[Tuple[str, str]], width: int, height: int):
    orig_data = base64.b64encode(png_bytes(orig)).decode('ascii')
    layers_html = []
    for name, svg in files:
        encoded = base64.b64encode(svg.encode('utf-8')).decode('ascii')
        layers_html.append(f'<img src="data:image/svg+xml;base64,{encoded}" style="position:absolute;left:0;top:0;width:100%;height:100%">')
    html = f"""
    <div style='display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap'>
      <div style='flex:1;min-width:260px'>
        <div style='font-weight:bold;margin-bottom:8px'>Original</div>
        <div style='border:1px solid #ddd;border-radius:12px;padding:12px;background:#eee;text-align:center'>
          <img src='data:image/png;base64,{orig_data}' style='max-width:100%;height:auto'>
        </div>
      </div>
      <div style='flex:1;min-width:260px'>
        <div style='font-weight:bold;margin-bottom:8px'>Layered Preview</div>
        <div style='position:relative;border:1px solid #ddd;border-radius:12px;padding:12px;background:#eee'>
          <div style='position:relative;width:100%;aspect-ratio:{width}/{height}'>
            {''.join(layers_html)}
          </div>
        </div>
      </div>
    </div>
    """
    components.html(html, height=640, scrolling=True)


def create_package(img: Image.Image, cfg: dict):
    work = fit_image(img, cfg['side'], cfg['enhance'])
    src = np.array(work)
    alpha = src[:, :, 3]
    visible = dominant_subject_mask(alpha)
    bgr = cv2.cvtColor(src[:, :, :3], cv2.COLOR_RGB2BGR)
    guides = create_region_guides(visible)
    if not guides:
        raise ValueError("被写体を検出できませんでした。背景が透明なPNGか、主体がはっきりした画像を使ってください。")

    skin = detect_skin(bgr, visible, guides)
    hair = detect_hair(bgr, visible, skin, guides)
    clothes, props, accessories = detect_clothes_and_props(visible, skin, hair, guides)

    hair_shadow, hair_highlight = extract_shadow_highlight(src, hair, cfg['hair_shadow_strength'], cfg['hair_highlight_strength'])
    clothes_shadow, clothes_highlight = extract_shadow_highlight(src, clothes, cfg['cloth_shadow_strength'], cfg['cloth_highlight_strength'])
    face_details = detect_face_details(bgr, skin, hair, guides)
    lineart = detect_lineart(bgr, visible, cfg['line_low'], cfg['line_high'], cfg['line_width']) if cfg['lineart'] else np.zeros_like(visible)

    # Optional underpaint base of entire subject
    underpaint = np.zeros_like(visible)
    if cfg['underpaint']:
        underpaint = component_filter(visible, cfg['min_area'])

    layers: List[LayerSpec] = []
    img_arr = src

    def add_flat_layer(name: str, order: int, mask: np.ndarray, color: Tuple[int, int, int], simplify: float, min_area: int):
        shapes = mask_to_shapes(mask, simplify, min_area)
        if shapes:
            layers.append(LayerSpec(name=name, order=order, fill=short_hex(color), shapes=shapes))

    def add_gradient_layer(name: str, order: int, mask: np.ndarray, c0: Tuple[int, int, int], c1: Tuple[int, int, int], simplify: float, min_area: int, mode: str = 'vertical'):
        shapes = mask_to_shapes(mask, simplify, min_area)
        if shapes:
            layers.append(LayerSpec(name=name, order=order, fill=short_hex(c0), shapes=shapes, gradient={"c0": short_hex(c0), "c1": short_hex(c1), "mode": mode}))

    if underpaint.max() > 0:
        rem = cv2.bitwise_and(visible, cv2.bitwise_not(cv2.bitwise_or(skin, hair)))
        under_c = average_color(src, rem if rem.max() > 0 else visible, fallback=(80, 80, 80))
        add_flat_layer('underpaint', 1, underpaint, under_c, cfg['simplify'] * 1.2, max(20, cfg['min_area']))

    if skin.max() > 0:
        add_flat_layer('skin_base', 10, skin, average_color(src, skin, fallback=(235, 220, 210)), cfg['simplify'], cfg['min_area'])
    if hair.max() > 0:
        hair_c = average_color(src, hair, fallback=(60, 60, 70))
        add_flat_layer('hair_base', 20, hair, hair_c, cfg['simplify'], cfg['min_area'])
        if hair_shadow.max() > 0:
            if cfg['gradient']:
                add_gradient_layer('hair_shadow', 30, hair_shadow, blend_color(hair_c, 0.10, True), blend_color(hair_c, 0.42, False), cfg['simplify'], cfg['min_area'])
            else:
                add_flat_layer('hair_shadow', 30, hair_shadow, blend_color(hair_c, 0.28, False), cfg['simplify'], cfg['min_area'])
        if hair_highlight.max() > 0:
            if cfg['gradient']:
                add_gradient_layer('hair_highlight', 40, hair_highlight, blend_color(hair_c, 0.28, True), blend_color(hair_c, 0.08, True), cfg['simplify'], cfg['min_area'])
            else:
                add_flat_layer('hair_highlight', 40, hair_highlight, blend_color(hair_c, 0.22, True), cfg['simplify'], cfg['min_area'])
    if cfg['face_detail'] and face_details.max() > 0:
        add_flat_layer('eyes_face_detail', 50, face_details, (25, 25, 25), cfg['line_simplify'], max(8, cfg['min_area'] // 2))
    if clothes.max() > 0:
        clothes_c = average_color(src, clothes, fallback=(55, 55, 70))
        add_flat_layer('clothes_base', 60, clothes, clothes_c, cfg['simplify'], cfg['min_area'])
        if clothes_shadow.max() > 0:
            if cfg['gradient']:
                add_gradient_layer('clothes_shadow', 70, clothes_shadow, blend_color(clothes_c, 0.08, True), blend_color(clothes_c, 0.34, False), cfg['simplify'], cfg['min_area'])
            else:
                add_flat_layer('clothes_shadow', 70, clothes_shadow, blend_color(clothes_c, 0.26, False), cfg['simplify'], cfg['min_area'])
        if cfg['cloth_highlight'] and clothes_highlight.max() > 0:
            add_gradient_layer('clothes_highlight', 75, clothes_highlight, blend_color(clothes_c, 0.22, True), blend_color(clothes_c, 0.06, True), cfg['simplify'], cfg['min_area'])
    if props.max() > 0 or accessories.max() > 0:
        combo = cv2.bitwise_or(props, accessories)
        add_flat_layer('prop_accessories', 80, combo, average_color(src, combo, fallback=(90, 90, 100)), cfg['simplify'], cfg['min_area'])
    if cfg['lineart'] and lineart.max() > 0:
        line_shapes = mask_to_shapes(lineart, cfg['line_simplify'], max(8, cfg['min_area'] // 2))
        if line_shapes:
            layers.append(LayerSpec(name='main_lineart', order=90, fill='#111', shapes=line_shapes))

    # Build files
    all_files: List[Tuple[str, str]] = []
    line_names = []
    for layer in sorted(layers, key=lambda x: x.order):
        t_kb = cfg['line_target_kb'] if 'lineart' in layer.name else cfg['target_kb']
        files = layer_to_files(work.width, work.height, layer, t_kb, cfg['max_path_kb'])
        all_files.extend(files)
        line_names.extend([f[0] for f in files])

    manifest = csv_manifest(all_files)
    preview = html_preview(all_files, work.width, work.height)
    guide = placement_guide(work.width, work.height, line_names)

    readme = f"""GT7 SVG Paint Stack Edition\n\nThis package was generated as a layered paint-stack workflow.\nRecommended order: manifest.csv top to bottom.\nRecommended SVGOMG: Merge paths OFF, Round/rewrite paths ON, Remove metadata/comments ON.\nCanvas: {work.width} x {work.height}\nLayer count: {len(all_files)} SVG files\n"""

    package = {
        'files': all_files,
        'manifest_csv': manifest,
        'preview_html': preview,
        'guide_png': guide,
        'readme': readme,
        'work_image': work,
        'visible': visible,
        'skin': skin,
        'hair': hair,
        'clothes': clothes,
        'props': props,
        'accessories': accessories,
        'face_details': face_details,
        'lineart_mask': lineart,
    }
    return package


# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, page_icon="🏁", layout="wide")
st.title("🏁 GT7 SVG Paint Stack Edition")
st.caption("軽いベースを先に作り、あとから細部を積み上げる方式の初版です。色トレース大量path方式ではなく、少数の意味レイヤーを作る方向に切り替えています。")

with st.expander("この版の考え方", expanded=True):
    st.markdown(
        """
- まず **skin / hair / clothes / prop** などの大きいベースを作る
- その上に **hair shadow / hair highlight / face detail / lineart** を追加する
- 1色ごとの大量pathではなく、**少数の大きなレイヤー** を目指す
- GT7向けに、最終的には **6〜10レイヤー前後** へ近づける方向の初版です
        """
    )

uploaded = st.file_uploader("画像を選択", type=["png", "jpg", "jpeg", "webp"])

with st.sidebar:
    st.header("設定")
    preset_name = st.selectbox("品質プリセット", list(PRESETS.keys()) + ["手動設定"], index=0)
    base = PRESETS.get(preset_name, PRESETS["安定 / まずこれ"]).copy()

    base['side'] = st.slider("処理サイズ（長辺px）", 700, 1800, int(base['side']), step=10)
    base['target_kb'] = st.slider("塗りレイヤー目標KB", 8, 25, int(base['target_kb']))
    base['line_target_kb'] = st.slider("線画レイヤー目標KB", 8, 25, int(base['line_target_kb']))
    base['max_path_kb'] = st.slider("1path最大KB（目安）", 4, 15, int(base['max_path_kb']))
    base['simplify'] = st.slider("塗りpath簡略化", 0.8, 3.0, float(base['simplify']), step=0.1)
    base['line_simplify'] = st.slider("線画path簡略化", 0.5, 2.4, float(base['line_simplify']), step=0.1)
    base['min_area'] = st.slider("小さい形状の削除", 4, 60, int(base['min_area']))
    base['line_low'] = st.slider("線画 Canny low", 20, 150, int(base['line_low']))
    base['line_high'] = st.slider("線画 Canny high", 60, 240, int(base['line_high']))
    base['line_width'] = st.slider("線画の太さ", 1, 3, int(base['line_width']))

    st.divider()
    base['gradient'] = st.toggle("linearGradient を使う", bool(base['gradient']))
    base['underpaint'] = st.toggle("underpaint を作る", False)
    base['lineart'] = st.toggle("main_lineart を作る", True)
    base['face_detail'] = st.toggle("eyes_face_detail を作る", True)
    base['cloth_highlight'] = st.toggle("clothes_highlight を作る", False)
    base['enhance'] = st.toggle("低画質画像を軽く補正", True)

    st.divider()
    st.markdown(
        """
**おすすめの使い方**
- まずは **安定 / まずこれ**
- SVGOMGは **Merge paths OFF**
- まずベースの構成を見て、あとで細部レイヤーを増やす
        """
    )

if uploaded is None:
    st.info("PNG/JPG/WebP画像をアップロードしてください。背景透過PNGだと分離しやすいです。")
    st.stop()

try:
    original = Image.open(uploaded).convert("RGBA")
except Exception as e:
    st.error(f"画像を開けませんでした: {e}")
    st.stop()

if st.button("Paint Stack SVG を生成", type="primary", use_container_width=True):
    with st.spinner("生成中です…"):
        try:
            result = create_package(original, base)
            st.session_state.result = result
            st.session_state.cfg = base.copy()
            st.session_state.base_name = safe_name(uploaded.name)
            st.session_state.zip_data = build_zip(result)
        except Exception as e:
            st.error(f"生成に失敗しました: {e}")
            st.stop()

if 'result' not in st.session_state:
    st.image(original, caption="元画像", use_container_width=True)
    st.stop()

res = st.session_state.result
files = res['files']

c1, c2, c3 = st.columns(3)
c1.metric("SVGファイル数", len(files))
c2.metric("合計サイズ", f"{sum(len(svg.encode('utf-8')) for _, svg in files):,} bytes")
c3.metric("キャンバス", f"{res['work_image'].width}×{res['work_image'].height}")

render_side_by_side(res['work_image'], files, res['work_image'].width, res['work_image'].height)

with st.expander("検出マスク確認", expanded=False):
    cols = st.columns(4)
    masks = [
        ("visible", res['visible']),
        ("skin", res['skin']),
        ("hair", res['hair']),
        ("clothes", res['clothes']),
        ("props", res['props']),
        ("accessories", res['accessories']),
        ("face_details", res['face_details']),
        ("lineart_mask", res['lineart_mask']),
    ]
    for i, (name, mask) in enumerate(masks):
        img = Image.fromarray(mask).convert("L")
        cols[i % 4].image(img, caption=name, use_container_width=True)

with st.expander("生成されたファイル一覧", expanded=True):
    rows = []
    for i, (name, svg) in enumerate(files, start=1):
        rows.append({"order": i, "filename": name, "bytes": len(svg.encode('utf-8'))})
    st.dataframe(rows, use_container_width=True, hide_index=True)

zip_name = f"{st.session_state.base_name}_paint_stack_gt7.zip"
st.download_button("ZIPを保存", st.session_state.zip_data, zip_name, "application/zip", use_container_width=True)

with st.expander("manifest.csv を表示"):
    st.code(res['manifest_csv'], language="csv")

with st.expander("SVGOMGおすすめ設定"):
    st.markdown(
        """
- Prettify markup: OFF
- Remove metadata: ON
- Remove comments: ON
- Convert colors: ON
- Round/rewrite paths: ON
- Merge paths: **OFF**
- Remove viewBox: OFF
        """
    )
