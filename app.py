import base64
import html
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps


# ============================================================
# Anime Illustration SVG Vectorizer / Streamlit App
# ------------------------------------------------------------
# 目的:
# - アップロードしたイラストを、色面を大きい順に重ねるSVGへ変換
# - パーツごとにSVGを分割してZIP保存
# - プレビュー付き
# - スマホでも設定しやすいStreamlit UI
#
# 注意:
# - AIが「髪」「目」「服」を完全認識するものではありません。
# - 色・面積・輪郭を使って、絵を描くように下地→細部→線画の順で重ねます。
# ============================================================


APP_TITLE = "イラスト SVG ベクター化ツール"
DEFAULT_BG = "#ffffff"


@dataclass
class Shape:
    id: int
    d: str
    fill: str
    area: float
    bbox: Tuple[int, int, int, int]
    role: str = "paint"  # paint / line
    gradient: Optional[Dict] = None


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    value = hex_color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        return (255, 255, 255)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Iterable[float]) -> str:
    r, g, b = [clamp_int(round(x), 0, 255) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def color_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return math.sqrt(sum((int(x) - int(y)) ** 2 for x, y in zip(a, b)))


def fmt_num(v: float, decimals: int = 1) -> str:
    # SVGの容量を抑えるため、整数に近いものは整数で出す
    if abs(v - round(v)) < 0.05:
        return str(int(round(v)))
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".")


def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    max_current = max(w, h)
    if max_current <= max_side:
        return img
    scale = max_side / max_current
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def flatten_alpha(img: Image.Image, bg_hex: str) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGBA")
    bg_rgb = hex_to_rgb(bg_hex)
    bg = Image.new("RGBA", img.size, bg_rgb + (255,))
    return Image.alpha_composite(bg, img).convert("RGB")


def preprocess_rgb(rgb: np.ndarray, smooth: str) -> np.ndarray:
    # 変換速度と輪郭の安定を両立するため、軽い平滑化だけ使う
    if smooth == "なし":
        return rgb
    if smooth == "軽く":
        return cv2.bilateralFilter(rgb, d=5, sigmaColor=35, sigmaSpace=35)
    if smooth == "強め":
        out = cv2.bilateralFilter(rgb, d=7, sigmaColor=55, sigmaSpace=55)
        return cv2.medianBlur(out, 3)
    return rgb


def quantize_rgb(rgb: np.ndarray, color_count: int) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    pil = Image.fromarray(rgb, "RGB")
    # PillowのAdaptive Paletteで色数を減らす。anime絵では速度と見た目のバランスが良い。
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
    # 実際に使われているインデックスだけを残して再マップ
    used = sorted(int(x) for x in np.unique(index_map))
    remap = {old: new for new, old in enumerate(used)}
    new_palette = [palette[i] if i < len(palette) else (0, 0, 0) for i in used]
    remapped = np.zeros_like(index_map, dtype=np.uint8)
    for old, new in remap.items():
        remapped[index_map == old] = new
    return remapped, new_palette


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
        # 同じ点が連続すると容量が増えるだけなので除外
        if int(x) == int(last_x) and int(y) == int(last_y):
            continue
        commands.append(f"L{fmt_num(x)} {fmt_num(y)}")
        last_x, last_y = x, y
    commands.append("Z")
    return "".join(commands)


def make_shapes_from_mask(
    mask: np.ndarray,
    fill: str,
    epsilon_ratio: float,
    min_area: int,
    role: str,
    start_id: int,
    skip_huge_area: Optional[float] = None,
) -> List[Shape]:
    shapes: List[Shape] = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return shapes
    hierarchy = hierarchy[0]
    sid = start_id

    for i, h in enumerate(hierarchy):
        parent = h[3]
        if parent != -1:
            continue
        area = float(abs(cv2.contourArea(contours[i])))
        if area < min_area:
            continue
        if skip_huge_area is not None and area > skip_huge_area:
            continue

        d_parts = [contour_to_path(contours[i], epsilon_ratio, reverse=False)]
        child = h[2]
        while child != -1:
            child_area = float(abs(cv2.contourArea(contours[child])))
            if child_area >= max(4, min_area * 0.25):
                child_path = contour_to_path(contours[child], epsilon_ratio, reverse=True)
                if child_path:
                    d_parts.append(child_path)
            child = hierarchy[child][0]

        d = "".join(part for part in d_parts if part)
        if not d:
            continue
        x, y, w, h2 = cv2.boundingRect(contours[i])
        shapes.append(
            Shape(
                id=sid,
                d=d,
                fill=fill,
                area=area,
                bbox=(int(x), int(y), int(w), int(h2)),
                role=role,
            )
        )
        sid += 1
    return shapes


def clean_mask(mask: np.ndarray, clean_px: int) -> np.ndarray:
    if clean_px <= 1:
        return mask
    k = np.ones((clean_px, clean_px), np.uint8)
    # closeで小さい隙間を埋め、openで孤立ノイズを減らす
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)
    return out


def extract_paint_shapes(
    index_map: np.ndarray,
    palette: List[Tuple[int, int, int]],
    epsilon_ratio: float,
    min_area: int,
    clean_px: int,
) -> Tuple[List[Shape], str]:
    h, w = index_map.shape[:2]
    counts = np.bincount(index_map.flatten(), minlength=len(palette))
    dominant_idx = int(np.argmax(counts)) if len(counts) else 0
    bg_hex = rgb_to_hex(palette[dominant_idx]) if palette else "#ffffff"
    shapes: List[Shape] = []
    next_id = 1
    image_area = float(w * h)

    for idx, rgb in enumerate(palette):
        mask = (index_map == idx).astype(np.uint8) * 255
        mask = clean_mask(mask, clean_px)
        skip_huge = image_area * 0.86 if idx == dominant_idx else None
        fill = rgb_to_hex(rgb)
        color_shapes = make_shapes_from_mask(
            mask=mask,
            fill=fill,
            epsilon_ratio=epsilon_ratio,
            min_area=min_area,
            role="paint",
            start_id=next_id,
            skip_huge_area=skip_huge,
        )
        if color_shapes:
            next_id = max(s.id for s in color_shapes) + 1
            shapes.extend(color_shapes)

    # 「下地に大きいパス、上に細部」を実現するため、面積が大きい順に並べる
    shapes.sort(key=lambda s: (s.area, -luminance(hex_to_rgb(s.fill))), reverse=True)
    # idは出力順に振り直し。gradient idとの重複防止にもなる。
    for i, s in enumerate(shapes, start=1):
        s.id = i
    return shapes, bg_hex


def extract_lineart_shapes(
    original_rgb: np.ndarray,
    dark_threshold: int,
    epsilon_ratio: float,
    min_area: int,
    clean_px: int,
    start_id: int,
) -> List[Shape]:
    # 黒や濃い色の線を最後に重ねる簡易線画レイヤー。
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray < dark_threshold).astype(np.uint8) * 255
    if clean_px > 1:
        k = np.ones((clean_px, clean_px), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    shapes = make_shapes_from_mask(
        mask=mask,
        fill="#111111",
        epsilon_ratio=epsilon_ratio,
        min_area=min_area,
        role="line",
        start_id=start_id,
        skip_huge_area=None,
    )
    shapes.sort(key=lambda s: s.area, reverse=True)
    return shapes


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
) -> None:
    if max_gradients <= 0:
        return
    h_img, w_img = original_rgb.shape[:2]
    count = 0
    for s in shapes:
        if s.role != "paint":
            continue
        if s.area < min_area_for_gradient:
            continue
        if count >= max_gradients:
            break
        x, y, w, h = s.bbox
        x1 = clamp_int(x, 0, w_img - 1)
        y1 = clamp_int(y, 0, h_img - 1)
        x2 = clamp_int(x + w, 1, w_img)
        y2 = clamp_int(y + h, 1, h_img)
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
        if use_horizontal:
            coords = (x1, y1, x2, y1)
        else:
            coords = (x1, y1, x1, y2)
        s.gradient = {
            "id": f"g{s.id}",
            "c1": rgb_to_hex(c1),
            "c2": rgb_to_hex(c2),
            "coords": coords,
        }
        count += 1


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
    if not defs:
        return ""
    return "<defs>" + "".join(defs) + "</defs>"


def shape_to_svg_element(s: Shape, stroke_width: float) -> str:
    fill = f'url(#{s.gradient["id"]})' if s.gradient else s.fill
    if s.role == "line":
        # 線画レイヤーは容量を抑えるため基本はfillのみ
        return f'<path d="{s.d}" fill="{s.fill}" fill-rule="evenodd"/>'
    if stroke_width > 0:
        sw = fmt_num(stroke_width)
        return (
            f'<path d="{s.d}" fill="{fill}" stroke="{s.fill}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round" fill-rule="evenodd"/>'
        )
    return f'<path d="{s.d}" fill="{fill}" fill-rule="evenodd"/>'


def make_svg_document(
    width: int,
    height: int,
    bg_hex: str,
    shapes: List[Shape],
    stroke_width: float,
    include_background: bool = True,
    title: str = "vectorized",
) -> str:
    body = []
    if include_background:
        body.append(f'<rect width="{width}" height="{height}" fill="{bg_hex}"/>')
    body.extend(shape_to_svg_element(s, stroke_width=stroke_width) for s in shapes)
    defs = svg_defs_for_shapes(shapes)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" shape-rendering="geometricPrecision">'
        f'<title>{html.escape(title)}</title>'
        f'{defs}'
        f'{"".join(body)}'
        f'</svg>'
    )


def estimate_kb(text: str) -> float:
    return len(text.encode("utf-8")) / 1024.0


def split_shapes_by_size(
    shapes: List[Shape],
    width: int,
    height: int,
    bg_hex: str,
    stroke_width: float,
    target_kb: float,
    include_background_first_part: bool,
    file_prefix: str,
) -> List[Tuple[str, str, float, int]]:
    parts: List[Tuple[str, str, float, int]] = []
    current: List[Shape] = []
    part_no = 1
    target_bytes = int(max(4.0, target_kb) * 1024)

    def finalize(chunk: List[Shape], no: int) -> None:
        include_bg = include_background_first_part and no == 1
        svg = make_svg_document(
            width,
            height,
            bg_hex,
            chunk,
            stroke_width=stroke_width,
            include_background=include_bg,
            title=f"{file_prefix}_{no:03d}",
        )
        name = f"{file_prefix}_{no:03d}_{len(svg.encode('utf-8'))//1024 + 1}KB.svg"
        parts.append((name, svg, estimate_kb(svg), len(chunk)))

    for s in shapes:
        trial = current + [s]
        include_bg = include_background_first_part and part_no == 1
        trial_svg = make_svg_document(
            width,
            height,
            bg_hex,
            trial,
            stroke_width=stroke_width,
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

使い方:
1. preview_combined.svg で全体の見た目を確認します。
2. part_001 から順番に下へ、番号が大きいものほど上に重ねます。
3. lineart が含まれるファイルは最後に重ねると線が締まります。
4. SVGOMGなどでさらに最適化する場合は、見た目を確認しながら行ってください。

注意:
- 自動変換なので、髪・目・服などを完全に意味分解するわけではありません。
- 品質を上げたい場合は「色数」を増やし「輪郭の単純化」を下げます。
- 容量を下げたい場合は「色数」を減らし「最小パーツ面積」を上げます。
"""
        z.writestr("README.txt", readme)
        preview_html = f"""<!doctype html>
<html lang="ja">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SVG Preview</title>
<style>
body{{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:16px;background:#f6f6f6;color:#222}}
.wrap{{max-width:1000px;margin:auto;background:white;padding:16px;border-radius:16px}}
svg{{width:100%;height:auto;border:1px solid #ddd;background:white}}
code{{word-break:break-all}}
</style></head>
<body><div class="wrap">
<h1>SVG Preview</h1>
<p>このHTMLはZIP内の <code>preview_combined.svg</code> と同じ内容を表示します。</p>
{combined_svg}
</div></body></html>"""
        z.writestr("preview.html", preview_html)
    return buf.getvalue()


def html_preview(svg: str, height: int = 720) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"""
<div style="width:100%; background:#f8f8f8; padding:12px; border-radius:14px; box-sizing:border-box;">
  <img src="data:image/svg+xml;base64,{encoded}" style="width:100%; height:auto; max-height:{height}px; object-fit:contain; background:white; border:1px solid #ddd; border-radius:10px;" />
</div>
"""


@st.cache_data(show_spinner=False)
def convert_cached(
    image_bytes: bytes,
    filename: str,
    bg_hex: str,
    max_side: int,
    smooth: str,
    color_count: int,
    epsilon_ratio: float,
    min_area: int,
    clean_px: int,
    add_lineart: bool,
    line_threshold: int,
    line_min_area: int,
    line_epsilon_ratio: float,
    add_gradients: bool,
    gradient_max: int,
    gradient_min_area: int,
    gradient_diff: int,
    gradient_direction: str,
    stroke_width: float,
    target_kb: float,
) -> Dict:
    src = Image.open(io.BytesIO(image_bytes))
    flat = flatten_alpha(src, bg_hex)
    resized = resize_keep_aspect(flat, max_side=max_side)
    original_rgb = np.array(resized, dtype=np.uint8)
    processed = preprocess_rgb(original_rgb, smooth=smooth)
    index_map, palette = quantize_rgb(processed, color_count=color_count)
    paint_shapes, auto_bg = extract_paint_shapes(
        index_map=index_map,
        palette=palette,
        epsilon_ratio=epsilon_ratio,
        min_area=min_area,
        clean_px=clean_px,
    )

    next_id = len(paint_shapes) + 1
    line_shapes: List[Shape] = []
    if add_lineart:
        line_shapes = extract_lineart_shapes(
            original_rgb=original_rgb,
            dark_threshold=line_threshold,
            epsilon_ratio=line_epsilon_ratio,
            min_area=line_min_area,
            clean_px=max(1, clean_px),
            start_id=next_id,
        )

    if add_gradients:
        add_linear_gradients(
            shapes=paint_shapes,
            original_rgb=original_rgb,
            max_gradients=gradient_max,
            min_area_for_gradient=gradient_min_area,
            diff_threshold=gradient_diff,
            direction=gradient_direction,
        )

    width, height = resized.size
    combined_shapes = paint_shapes + line_shapes
    combined_svg = make_svg_document(
        width=width,
        height=height,
        bg_hex=auto_bg,
        shapes=combined_shapes,
        stroke_width=stroke_width,
        include_background=True,
        title="preview_combined",
    )

    paint_parts = split_shapes_by_size(
        shapes=paint_shapes,
        width=width,
        height=height,
        bg_hex=auto_bg,
        stroke_width=stroke_width,
        target_kb=target_kb,
        include_background_first_part=True,
        file_prefix="part_paint",
    )
    line_parts: List[Tuple[str, str, float, int]] = []
    if line_shapes:
        line_parts = split_shapes_by_size(
            shapes=line_shapes,
            width=width,
            height=height,
            bg_hex=auto_bg,
            stroke_width=0.0,
            target_kb=target_kb,
            include_background_first_part=False,
            file_prefix="part_lineart",
        )
    all_parts = paint_parts + line_parts

    settings = {
        "filename": filename,
        "input_size": src.size,
        "output_size": [width, height],
        "background_for_transparency": bg_hex,
        "auto_background_fill": auto_bg,
        "max_side": max_side,
        "smooth": smooth,
        "color_count": color_count,
        "epsilon_ratio": epsilon_ratio,
        "min_area": min_area,
        "clean_px": clean_px,
        "add_lineart": add_lineart,
        "line_threshold": line_threshold,
        "line_min_area": line_min_area,
        "line_epsilon_ratio": line_epsilon_ratio,
        "add_gradients": add_gradients,
        "gradient_max": gradient_max,
        "gradient_min_area": gradient_min_area,
        "gradient_diff": gradient_diff,
        "gradient_direction": gradient_direction,
        "stroke_width": stroke_width,
        "target_kb": target_kb,
        "paint_shape_count": len(paint_shapes),
        "line_shape_count": len(line_shapes),
        "part_count": len(all_parts),
        "combined_kb": round(estimate_kb(combined_svg), 2),
    }
    zip_bytes = make_zip(all_parts, combined_svg, settings, width, height)

    preview_png = io.BytesIO()
    resized.save(preview_png, format="PNG")

    return {
        "width": width,
        "height": height,
        "source_png": preview_png.getvalue(),
        "combined_svg": combined_svg,
        "parts": all_parts,
        "settings": settings,
        "zip_bytes": zip_bytes,
    }


def preset_values(preset: str) -> Dict:
    if preset == "高速・軽量":
        return dict(max_side=700, color_count=14, epsilon=0.010, min_area=22, clean_px=2, target_kb=14.0)
    if preset == "高品質":
        return dict(max_side=1100, color_count=34, epsilon=0.0045, min_area=8, clean_px=1, target_kb=24.0)
    if preset == "15KB分割重視":
        return dict(max_side=850, color_count=20, epsilon=0.008, min_area=18, clean_px=2, target_kb=14.0)
    return dict(max_side=900, color_count=24, epsilon=0.0065, min_area=12, clean_px=1, target_kb=18.0)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎨", layout="wide")
    st.title("🎨 イラスト SVG ベクター化ツール")
    st.caption("アップロード画像を、下地→色面→細部→線画の順で重ねるSVGに変換し、パーツ分割ZIPで保存します。")

    with st.expander("このツールの考え方", expanded=False):
        st.markdown(
            """
- 完全なAI手描き分解ではなく、**色面・輪郭・面積**を使って自動的にパーツ化します。
- 大きな色面を先に描き、その上に小さな形を重ねるため、普通の単純トレースよりイラスト向けです。
- 容量を抑えたい場合は、色数を減らし、最小パーツ面積を上げ、輪郭の単純化を強めてください。
- 品質を上げたい場合は、色数を増やし、輪郭の単純化を弱め、最大辺を大きくしてください。
            """
        )

    uploaded = st.file_uploader(
        "画像をアップロードしてください",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    preset = st.selectbox(
        "プリセット",
        ["標準", "高速・軽量", "高品質", "15KB分割重視"],
        index=0,
        help="迷ったら標準。容量制限が厳しい場合は15KB分割重視。",
    )
    pv = preset_values(preset)

    col1, col2, col3 = st.columns(3)
    with col1:
        max_side = st.slider("最大辺px", 400, 1600, pv["max_side"], 50)
        color_count = st.slider("色数", 6, 64, pv["color_count"], 1)
        smooth = st.selectbox("平滑化", ["なし", "軽く", "強め"], index=1)
    with col2:
        epsilon_ratio = st.slider("輪郭の単純化", 0.002, 0.020, pv["epsilon"], 0.0005, format="%.4f")
        min_area = st.slider("最小パーツ面積", 2, 120, pv["min_area"], 1)
        clean_px = st.slider("ノイズ整理px", 1, 5, pv["clean_px"], 1)
    with col3:
        target_kb = st.slider("1ファイル目標KB", 6.0, 80.0, pv["target_kb"], 1.0)
        stroke_width = st.slider("隙間隠しストローク", 0.0, 2.0, 0.4, 0.1)
        bg_hex = st.text_input("透過画像の背景色", DEFAULT_BG)

    with st.expander("線画・グラデーション設定", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            add_lineart = st.checkbox("濃い線を最後に線画レイヤーとして追加", value=True)
            line_threshold = st.slider("線画として拾う暗さ", 20, 150, 78, 1)
            line_min_area = st.slider("線画の最小面積", 1, 80, max(2, min_area // 2), 1)
            line_epsilon_ratio = st.slider("線画の単純化", 0.002, 0.020, max(0.004, epsilon_ratio * 0.8), 0.0005, format="%.4f")
        with c2:
            add_gradients = st.checkbox("大きい面に直線グラデーションを試す", value=False)
            gradient_direction = st.selectbox("グラデーション方向", ["自動", "縦", "横"], index=0)
            gradient_max = st.slider("最大グラデーション数", 0, 30, 8, 1)
            gradient_min_area = st.slider("グラデーション対象の最小面積", 200, 20000, 1800, 100)
            gradient_diff = st.slider("色差がこの値以上なら使用", 10, 120, 36, 1)

    if uploaded is None:
        st.info("画像をアップロードすると、ここにプレビューとZIP保存ボタンが表示されます。")
        return

    image_bytes = uploaded.getvalue()

    if st.button("SVGに変換する", type="primary", use_container_width=True):
        with st.spinner("変換中です。大きい面からパス化し、パーツ分割しています。"):
            result = convert_cached(
                image_bytes=image_bytes,
                filename=uploaded.name,
                bg_hex=bg_hex,
                max_side=max_side,
                smooth=smooth,
                color_count=color_count,
                epsilon_ratio=epsilon_ratio,
                min_area=min_area,
                clean_px=clean_px,
                add_lineart=add_lineart,
                line_threshold=line_threshold,
                line_min_area=line_min_area,
                line_epsilon_ratio=line_epsilon_ratio,
                add_gradients=add_gradients,
                gradient_max=gradient_max,
                gradient_min_area=gradient_min_area,
                gradient_diff=gradient_diff,
                gradient_direction=gradient_direction,
                stroke_width=stroke_width,
                target_kb=target_kb,
            )
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if not result:
        st.warning("設定後、上の「SVGに変換する」を押してください。")
        return

    settings = result["settings"]
    st.success(
        f"変換完了: {settings['output_size'][0]}×{settings['output_size'][1]}px / "
        f"ペイント{settings['paint_shape_count']}個 / 線画{settings['line_shape_count']}個 / "
        f"分割SVG {settings['part_count']}個"
    )

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "ZIPを一括ダウンロード",
            data=result["zip_bytes"],
            file_name="vectorized_svg_parts.zip",
            mime="application/zip",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            "結合プレビューSVGをダウンロード",
            data=result["combined_svg"].encode("utf-8"),
            file_name="preview_combined.svg",
            mime="image/svg+xml",
            use_container_width=True,
        )

    left, right = st.columns(2)
    with left:
        st.subheader("元画像")
        st.image(result["source_png"], use_container_width=True)
    with right:
        st.subheader("SVGプレビュー")
        components.html(html_preview(result["combined_svg"]), height=760, scrolling=True)

    st.subheader("分割ファイル一覧")
    rows = []
    for i, (name, svg, kb, count) in enumerate(result["parts"], start=1):
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
        st.json(settings)


if __name__ == "__main__":
    main()
