from __future__ import annotations

import base64
import csv
import io
import re
import time
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps, ImageFilter, ImageDraw


APP_TITLE = "GT7 SVG Paint Stack Colorbase v2"
HARD_LIMIT = 2 * 1024 * 1024

PRESETS = {
    "安定 / まずこれ": dict(
        side=1000,
        colors=12,
        max_layers=10,
        target_kb=14,
        simplify=1.8,
        min_area=26,
        smooth=1,
        alpha=16,
        exclude_white=True,
        gradient=True,
        lineart=False,
        line_low=90,
        line_high=180,
    ),
    "高品質 / 仕上げ": dict(
        side=1250,
        colors=16,
        max_layers=12,
        target_kb=14,
        simplify=1.25,
        min_area=16,
        smooth=1,
        alpha=16,
        exclude_white=True,
        gradient=True,
        lineart=False,
        line_low=75,
        line_high=160,
    ),
    "軽量 / 速い": dict(
        side=850,
        colors=10,
        max_layers=8,
        target_kb=12,
        simplify=2.3,
        min_area=40,
        smooth=1,
        alpha=16,
        exclude_white=True,
        gradient=False,
        lineart=False,
        line_low=110,
        line_high=210,
    ),
}


@dataclass
class Shape:
    d: str
    bbox: Tuple[int, int, int, int]
    area: float


@dataclass
class SvgFile:
    name: str
    svg: str
    order: int
    bytes: int


def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def fit_image(img: Image.Image, side: int, enhance: bool) -> Image.Image:
    img = img.convert("RGBA")
    if enhance:
        rgb = ImageOps.autocontrast(img.convert("RGB"))
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=108, threshold=4))
        img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))
    if max(img.size) > side:
        sc = side / max(img.size)
        img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.Resampling.LANCZOS)
    return img


def short_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [int(max(0, min(255, x))) for x in rgb]
    s = f"{r:02x}{g:02x}{b:02x}"
    if s[0] == s[1] and s[2] == s[3] and s[4] == s[5]:
        return f"#{s[0]}{s[2]}{s[4]}"
    return f"#{s}"


def lighten(rgb: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    arr = np.array(rgb, dtype=np.float32)
    out = arr * (1 - amount) + 255 * amount
    return tuple(int(round(x)) for x in np.clip(out, 0, 255))


def darken(rgb: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    arr = np.array(rgb, dtype=np.float32)
    out = arr * (1 - amount)
    return tuple(int(round(x)) for x in np.clip(out, 0, 255))


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def visible_mask_from_image(arr: np.ndarray, alpha_th: int, exclude_white: bool) -> np.ndarray:
    alpha = arr[:, :, 3]
    visible = (alpha >= alpha_th).astype(np.uint8) * 255

    if exclude_white:
        rgb = arr[:, :, :3].astype(np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        # JPEGなどで白背景が入った場合に、ほぼ白だけ背景として除外する
        white_bg = ((v > 245) & (s < 18)).astype(np.uint8) * 255
        visible = cv2.bitwise_and(visible, cv2.bitwise_not(white_bg))

    visible = cv2.morphologyEx(visible, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return visible


def fill_hidden_with_mean(arr: np.ndarray, visible: np.ndarray) -> Image.Image:
    rgb = arr[:, :, :3].copy()
    pts = rgb[visible > 0]
    if len(pts) == 0:
        mean = np.array([128, 128, 128], dtype=np.uint8)
    else:
        mean = np.median(pts, axis=0).astype(np.uint8)
    rgb[visible == 0] = mean
    return Image.fromarray(rgb, "RGB")


def quantize_labels(work: Image.Image, n_colors: int, visible: np.ndarray):
    arr = np.array(work)
    # ノイズを減らして、手作業のベース塗りに近づける
    rgb = arr[:, :, :3]
    rgb_smooth = cv2.bilateralFilter(rgb, d=7, sigmaColor=40, sigmaSpace=40)
    arr2 = np.dstack([rgb_smooth, arr[:, :, 3]])
    q_source = fill_hidden_with_mean(arr2, visible)
    q = q_source.quantize(colors=int(n_colors), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)

    labels = np.array(q, dtype=np.int32)
    labels[visible == 0] = -1

    pal_raw = q.getpalette()[:768]
    used = sorted(int(x) for x in np.unique(labels) if int(x) >= 0)

    remap = np.full_like(labels, -1)
    colors: List[Tuple[int, int, int]] = []
    for new, old in enumerate(used):
        remap[labels == old] = new
        colors.append(tuple(int(x) for x in pal_raw[old * 3: old * 3 + 3]))

    return remap, colors, rgb_smooth


def clean_mask(mask: np.ndarray, min_area: int, smooth: int) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    if smooth > 0:
        k = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=smooth)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=smooth)
    if m.max() == 0:
        return m

    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def contour_to_path(cnt: np.ndarray, eps: float) -> Optional[Shape]:
    if cnt is None or len(cnt) < 3:
        return None
    area = abs(cv2.contourArea(cnt))
    if area < 1:
        return None
    approx = cv2.approxPolyDP(cnt, max(0.2, float(eps)), True)
    if approx is None or len(approx) < 3:
        return None
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        return None
    cmds = [f"M{int(pts[0,0])} {int(pts[0,1])}"]
    for x, y in pts[1:]:
        cmds.append(f"L{int(x)} {int(y)}")
    cmds.append("Z")
    x, y, w, h = cv2.boundingRect(approx)
    return Shape("".join(cmds), (int(x), int(y), int(x + w), int(y + h)), float(area))


def mask_to_shapes(mask: np.ndarray, simplify: float, min_area: int, smooth: int) -> List[Shape]:
    m = clean_mask(mask, min_area, smooth)
    if m.max() == 0:
        return []
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes: List[Shape] = []
    for c in contours:
        if abs(cv2.contourArea(c)) < min_area:
            continue
        s = contour_to_path(c, simplify)
        if s:
            shapes.append(s)
    shapes.sort(key=lambda s: s.area, reverse=True)
    return shapes


def shapes_bbox(shapes: List[Shape]) -> Optional[Tuple[int, int, int, int]]:
    if not shapes:
        return None
    return min(s.bbox[0] for s in shapes), min(s.bbox[1] for s in shapes), max(s.bbox[2] for s in shapes), max(s.bbox[3] for s in shapes)


def estimate_svg_size(shapes: List[Shape], fill: str, width: int, height: int, gradient: bool) -> int:
    total = 120 + len(fill)
    if gradient:
        total += 220
    total += sum(len(s.d) + 18 for s in shapes)
    return total


def gradient_def(gid: str, bbox: Tuple[int, int, int, int], c0: str, c1: str) -> str:
    x0, y0, x1, y1 = bbox
    return (
        f'<defs><linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}">'
        f'<stop offset="0%" stop-color="{c0}"/><stop offset="100%" stop-color="{c1}"/></linearGradient></defs>'
    )


def build_svg(width: int, height: int, shapes: List[Shape], fill: str, gradient_info: Optional[dict] = None) -> str:
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">']
    fill_ref = fill
    if gradient_info and shapes:
        bb = shapes_bbox(shapes)
        if bb:
            gid = gradient_info["id"]
            parts.append(gradient_def(gid, bb, gradient_info["c0"], gradient_info["c1"]))
            fill_ref = f"url(#{gid})"
    parts.append(f'<g fill="{fill_ref}">')
    for s in shapes:
        parts.append(f'<path d="{s.d}"/>')
    parts.append('</g></svg>')
    return "".join(parts)


def split_shapes_to_limit(width: int, height: int, shapes: List[Shape], fill: str, layer_name: str, order: int, target_kb: int, gradient_enabled: bool, color: Tuple[int, int, int]) -> List[SvgFile]:
    if not shapes:
        return []
    target = max(4, target_kb) * 1024
    files: List[SvgFile] = []

    current: List[Shape] = []
    part = 1
    for s in shapes:
        test = current + [s]
        grad = None
        if gradient_enabled and len(test) > 0:
            grad = {"id": f"g{order}_{part}", "c0": short_hex(lighten(color, 0.08)), "c1": short_hex(darken(color, 0.12))}
        est_svg = build_svg(width, height, test, fill, grad)
        if current and len(est_svg.encode("utf-8")) > target:
            grad2 = None
            if gradient_enabled:
                grad2 = {"id": f"g{order}_{part}", "c0": short_hex(lighten(color, 0.08)), "c1": short_hex(darken(color, 0.12))}
            svg = build_svg(width, height, current, fill, grad2)
            name = f"{order:03d}_{layer_name}_{part:02d}.svg"
            files.append(SvgFile(name, svg, order, len(svg.encode("utf-8"))))
            part += 1
            current = [s]
        else:
            current = test

    if current:
        grad3 = None
        if gradient_enabled:
            grad3 = {"id": f"g{order}_{part}", "c0": short_hex(lighten(color, 0.08)), "c1": short_hex(darken(color, 0.12))}
        svg = build_svg(width, height, current, fill, grad3)
        name = f"{order:03d}_{layer_name}_{part:02d}.svg"
        files.append(SvgFile(name, svg, order, len(svg.encode("utf-8"))))
    return files


def auto_simplify_layer(width: int, height: int, mask: np.ndarray, color: Tuple[int, int, int], name: str, order: int, cfg: dict, gradient: bool):
    simplify = float(cfg["simplify"])
    min_area = int(cfg["min_area"])
    smooth = int(cfg["smooth"])
    target = int(cfg["target_kb"]) * 1024

    best_shapes = []
    best_bytes = 10**18

    for attempt in range(7):
        shapes = mask_to_shapes(mask, simplify, min_area, smooth)
        fill = short_hex(color)
        grad_info = None
        if gradient and shapes:
            grad_info = {"id": f"g{order}_1", "c0": short_hex(lighten(color, 0.08)), "c1": short_hex(darken(color, 0.12))}
        svg = build_svg(width, height, shapes, fill, grad_info)
        b = len(svg.encode("utf-8"))
        if b < best_bytes:
            best_bytes = b
            best_shapes = shapes
        if b <= target or not shapes:
            return shapes, simplify, min_area
        simplify *= 1.22
        min_area = int(min_area * 1.25 + 2)

    return best_shapes, simplify, min_area


def raster_preview(width: int, height: int, layers_for_preview: List[Tuple[np.ndarray, Tuple[int, int, int], float]]) -> Image.Image:
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    for mask, color, alpha in layers_for_preview:
        if mask.max() == 0:
            continue
        m = mask > 0
        src_rgb = np.array(color, dtype=np.float32)
        src_a = float(alpha)
        dst = canvas[m].astype(np.float32)
        out_rgb = src_rgb * src_a + dst[:, :3] * (1 - src_a)
        out_a = 255 * src_a + dst[:, 3] * (1 - src_a)
        canvas[m, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        canvas[m, 3] = np.clip(out_a, 0, 255).astype(np.uint8)
    return Image.fromarray(canvas, "RGBA")


def make_checker_bg(size: Tuple[int, int], cell=14) -> Image.Image:
    w, h = size
    img = Image.new("RGBA", size, (245, 245, 245, 255))
    dr = ImageDraw.Draw(img)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            c = (235,235,235,255) if ((x//cell + y//cell) % 2) else (250,250,250,255)
            dr.rectangle([x, y, x+cell-1, y+cell-1], fill=c)
    return img


def alpha_composite_on_checker(img: Image.Image) -> Image.Image:
    bg = make_checker_bg(img.size)
    bg.alpha_composite(img.convert("RGBA"), (0,0))
    return bg


def html_preview(files: List[SvgFile], width: int, height: int) -> str:
    imgs = []
    for f in files:
        enc = base64.b64encode(f.svg.encode("utf-8")).decode("ascii")
        imgs.append(f'<img src="data:image/svg+xml;base64,{enc}" style="position:absolute;left:0;top:0;width:100%;height:100%">')
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>preview</title></head>
<body style="font-family:sans-serif;margin:18px">
<h2>GT7 Paint Stack Colorbase Preview</h2>
<p>Place SVGs in manifest order. Merge paths OFF recommended in SVGOMG.</p>
<div style="position:relative;width:min(94vw,{width}px);aspect-ratio:{width}/{height};border:1px solid #ccc;background:#eee">
{''.join(imgs)}
</div>
</body></html>"""


def make_manifest(files: List[SvgFile]) -> str:
    out = io.StringIO()
    wr = csv.writer(out)
    wr.writerow(["order", "filename", "bytes"])
    for i, f in enumerate(files, start=1):
        wr.writerow([i, f.name, f.bytes])
    return out.getvalue()


def guide_png(files: List[SvgFile], width: int, height: int) -> bytes:
    img = Image.new("RGBA", (900, max(300, 80 + len(files) * 24)), (245,245,245,255))
    dr = ImageDraw.Draw(img)
    dr.text((20, 20), "GT7 Paint Stack Order", fill=(0,0,0,255))
    y = 58
    for i, f in enumerate(files, start=1):
        dr.text((20, y), f"{i:02d}. {f.name}  {f.bytes} bytes", fill=(0,0,0,255))
        y += 24
    return png_bytes(img)


def build_zip(files: List[SvgFile], width: int, height: int) -> bytes:
    manifest = make_manifest(files)
    preview = html_preview(files, width, height)
    readme = """GT7 SVG Paint Stack Colorbase v2

This version builds a light underpaint/base first, then overlays selected color/detail layers.
Recommended SVGOMG:
- Prettify markup OFF
- Remove metadata/comments ON
- Convert colors ON
- Round/rewrite paths ON
- Merge paths OFF
- Remove viewBox OFF
"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.writestr(f.name, f.svg)
        z.writestr("manifest.csv", manifest)
        z.writestr("preview.html", preview)
        z.writestr("README.txt", readme)
        z.writestr("placement_guide.png", guide_png(files, width, height))
    return buf.getvalue()


def safe_name(name: str) -> str:
    name = re.sub(r"\.[^.]+$", "", name)
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return name or "paint_stack"


def create_package(original: Image.Image, cfg: dict):
    work = fit_image(original, int(cfg["side"]), bool(cfg["enhance"]))
    arr = np.array(work)
    visible = visible_mask_from_image(arr, int(cfg["alpha"]), bool(cfg["exclude_white"]))

    if visible.max() == 0:
        raise ValueError("被写体を検出できませんでした。白背景除外をOFFにするか、透明PNGで試してください。")

    labels, colors, rgb_smooth = quantize_labels(work, int(cfg["colors"]), visible)

    area_by_color = []
    for i, color in enumerate(colors):
        area = int((labels == i).sum())
        if area > 0:
            area_by_color.append((area, i, color))
    area_by_color.sort(reverse=True)

    if not area_by_color:
        raise ValueError("色レイヤーを作れませんでした。")

    # ベースは最大色ではなく、可視領域全体の中央値にする。黒ベタ化を防ぐため。
    vis_rgb = arr[:, :, :3][visible > 0]
    base_color = tuple(int(x) for x in np.median(vis_rgb, axis=0))
    files: List[SvgFile] = []
    preview_layers: List[Tuple[np.ndarray, Tuple[int, int, int], float]] = []

    # 00ベース：軽い土台
    base_shapes, _, _ = auto_simplify_layer(work.width, work.height, visible, base_color, "00_base_underpaint", 0, cfg, False)
    base_files = split_shapes_to_limit(work.width, work.height, base_shapes, short_hex(base_color), "00_base_underpaint", 0, int(cfg["target_kb"]), False, base_color)
    files.extend(base_files)
    preview_layers.append((visible, base_color, 1.0))

    # 色レイヤー：大きい順。ただし最大色も入れる。ベースの上に正しい色を置く。
    max_layers = int(cfg["max_layers"])
    selected = area_by_color[:max_layers]
    order = 10
    for rank, (area, idx, color) in enumerate(selected, start=1):
        mask = (labels == idx).astype(np.uint8) * 255
        # 透明外にはみ出さない
        mask = cv2.bitwise_and(mask, visible)
        if int(mask.sum() // 255) < int(cfg["min_area"]):
            continue

        layer_name = f"{order:02d}_color_{rank:02d}_{short_hex(color).replace('#','')}"
        shapes, _, _ = auto_simplify_layer(work.width, work.height, mask, color, layer_name, order, cfg, bool(cfg["gradient"]))
        if not shapes:
            continue
        layer_files = split_shapes_to_limit(work.width, work.height, shapes, short_hex(color), layer_name, order, int(cfg["target_kb"]), bool(cfg["gradient"]), color)
        files.extend(layer_files)
        preview_layers.append((mask, color, 1.0))
        order += 10

    # 補助線画：初期OFF。ONのときも細いエッジだけ。
    line_mask = np.zeros(visible.shape, dtype=np.uint8)
    if cfg.get("lineart", False):
        gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, int(cfg["line_low"]), int(cfg["line_high"]))
        edges = cv2.bitwise_and(edges, visible)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        line_mask = clean_mask(edges, max(8, int(cfg["min_area"]) // 2), 0)
        line_shapes = mask_to_shapes(line_mask, float(cfg["simplify"]) * 0.8, max(8, int(cfg["min_area"]) // 2), 0)
        line_color = (18, 18, 20)
        files.extend(split_shapes_to_limit(work.width, work.height, line_shapes, short_hex(line_color), "90_subtle_lineart", 90, int(cfg["target_kb"]), False, line_color))
        preview_layers.append((line_mask, line_color, 1.0))

    files.sort(key=lambda f: (f.order, f.name))
    preview = raster_preview(work.width, work.height, preview_layers)
    zip_data = build_zip(files, work.width, work.height)

    return {
        "work": work,
        "visible": visible,
        "labels": labels,
        "colors": colors,
        "files": files,
        "preview": preview,
        "line_mask": line_mask,
        "zip": zip_data,
        "total_bytes": sum(f.bytes for f in files),
    }


# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, page_icon="🏁", layout="wide")
st.title("🏁 GT7 SVG Paint Stack Colorbase v2")
st.caption("黒ベタ化を避けるため、髪/服を無理に意味分類せず、軽い全体ベースの上に大きな色レイヤーを順番に積む方式です。")

with st.expander("今回の修正内容", expanded=True):
    st.markdown(
        """
前のPaint Stack版では、暗い領域を「髪」や「線画」として拾いすぎて黒ベタになりました。  
この版では、まず全体の軽いベースを作り、その上に画像から取った大きな色レイヤーを重ねます。

- 意味分類の失敗による黒ベタ化を回避
- 6〜12枚程度の色材レイヤーを目標
- linearGradientは直線タイプのみ
- 実SVGプレビューは重いので、基本は軽量ラスタープレビュー + ZIP内preview.html
        """
    )

uploaded = st.file_uploader("画像を選択", type=["png", "jpg", "jpeg", "webp"])

with st.sidebar:
    st.header("設定")
    preset_name = st.selectbox("品質プリセット", list(PRESETS.keys()) + ["手動設定"], index=0)
    cfg = PRESETS.get(preset_name, PRESETS["安定 / まずこれ"]).copy()

    cfg["side"] = st.slider("処理サイズ（長辺px）", 700, 1600, int(cfg["side"]), step=10)
    cfg["colors"] = st.slider("解析する色数", 6, 32, int(cfg["colors"]))
    cfg["max_layers"] = st.slider("出力する色レイヤー数", 4, 18, int(cfg["max_layers"]))
    cfg["target_kb"] = st.slider("各SVG目標KB", 8, 25, int(cfg["target_kb"]))
    cfg["simplify"] = st.slider("path簡略化", 0.8, 4.0, float(cfg["simplify"]), step=0.1)
    cfg["min_area"] = st.slider("小さい形状の削除", 4, 100, int(cfg["min_area"]))
    cfg["smooth"] = st.slider("マスクなめらか処理", 0, 3, int(cfg["smooth"]))
    cfg["alpha"] = st.slider("透明判定", 0, 255, int(cfg["alpha"]))

    st.divider()
    cfg["exclude_white"] = st.toggle("白背景を除外する", bool(cfg["exclude_white"]))
    cfg["gradient"] = st.toggle("linearGradientを使う", bool(cfg["gradient"]))
    cfg["lineart"] = st.toggle("補助線画を作る", bool(cfg["lineart"]))
    cfg["enhance"] = st.toggle("低画質画像を軽く補正", True)

    if cfg["lineart"]:
        cfg["line_low"] = st.slider("線画Canny low", 20, 160, int(cfg["line_low"]))
        cfg["line_high"] = st.slider("線画Canny high", 60, 260, int(cfg["line_high"]))

    st.divider()
    st.markdown(
        """
**おすすめ**
- まず補助線画OFF
- 黒ベタが出ないか確認
- 良ければ色レイヤー数を10〜14へ
- SVGOMGではMerge paths OFF
        """
    )

if uploaded is None:
    st.info("画像をアップロードしてください。")
    st.stop()

try:
    original = Image.open(uploaded).convert("RGBA")
except Exception as e:
    st.error(f"画像を開けませんでした: {e}")
    st.stop()

if st.button("Paint Stack SVGを生成", type="primary", use_container_width=True):
    with st.spinner("生成中です…"):
        try:
            res = create_package(original, cfg)
            st.session_state.res = res
            st.session_state.cfg = cfg.copy()
            st.session_state.name = safe_name(uploaded.name)
        except Exception as e:
            st.error(f"生成に失敗しました: {e}")
            st.stop()

if "res" not in st.session_state:
    st.image(original, caption="Original", use_container_width=True)
    st.stop()

res = st.session_state.res
files = res["files"]

m1, m2, m3 = st.columns(3)
m1.metric("SVG数", len(files))
m2.metric("合計サイズ", f"{res['total_bytes']:,} bytes")
m3.metric("キャンバス", f"{res['work'].width}×{res['work'].height}")

left, right = st.columns(2)
with left:
    st.subheader("Original")
    st.image(alpha_composite_on_checker(res["work"]), use_container_width=True)
with right:
    st.subheader("Layered Preview")
    st.image(alpha_composite_on_checker(res["preview"]), use_container_width=True)

with st.expander("検出マスク確認", expanded=False):
    c1, c2 = st.columns(2)
    c1.image(Image.fromarray(res["visible"]), caption="visible/base mask", use_container_width=True)
    c2.image(Image.fromarray(res["line_mask"]), caption="lineart mask", use_container_width=True)

with st.expander("生成されたSVG一覧", expanded=True):
    table = [{"order": i + 1, "filename": f.name, "bytes": f.bytes} for i, f in enumerate(files)]
    st.dataframe(table, hide_index=True, use_container_width=True)

st.download_button(
    "ZIPを保存",
    data=res["zip"],
    file_name=f"{st.session_state.name}_paint_stack_colorbase_v2.zip",
    mime="application/zip",
    use_container_width=True,
)

with st.expander("SVGOMGおすすめ設定"):
    st.markdown(
        """
- Prettify markup：OFF
- Remove metadata：ON
- Remove comments：ON
- Convert colors：ON
- Round/rewrite paths：ON
- Merge paths：**OFF**
- Remove viewBox：OFF
        """
    )
