from __future__ import annotations

import io
import math
import os
import re
import traceback
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


# ============================================================
# GT7 Anime Layer Splitter
# ------------------------------------------------------------
# 目的:
# - アニメ画像を GT7 デカール向けに「下地色面」「線画」「顔周辺高精度」へ分解
# - SVGOMG/SVGO に通しやすい、シンプルな path + fill だけの SVG を作る
# - 15KB 目標で自動分割した SVG を ZIP で出力
#
# 注意:
# - 完全な人間クオリティのベジェ清書ではなく、GT7で重ねて完成させるための半自動素材生成です。
# - SVGの最終15KB保証は、GT7側やSVGOMG後の変化もあるため「目標」です。
# ============================================================

APP_TITLE = "GT7 Anime Layer Splitter v1"
SVG_NS = "http://www.w3.org/2000/svg"


@dataclass
class SvgPath:
    element: str
    area: float
    layer: str
    color: str


# -----------------------------
# Basic image helpers
# -----------------------------

def pil_to_rgb_image(uploaded_file) -> Image.Image:
    """Load image, apply EXIF orientation, composite transparency on white, return RGB PIL image."""
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")

    return img.convert("RGB")


def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_side:
        return img
    scale = max_side / float(long_side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def rgb_to_hex(rgb: Sequence[int]) -> str:
    r, g, b = [int(x) for x in rgb[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def safe_name(name: str) -> str:
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_")
    return name or "image"


def pretty_kb(data: str | bytes) -> str:
    if isinstance(data, str):
        n = len(data.encode("utf-8"))
    else:
        n = len(data)
    return f"{n / 1024:.1f} KB"


# -----------------------------
# Quantization / masks
# -----------------------------

def quantize_rgb(img: Image.Image, color_count: int) -> Tuple[np.ndarray, List[Tuple[Tuple[int, int, int], int]]]:
    """Return quantized RGB array and palette [(rgb, pixel_count), ...] sorted by count desc."""
    rgb = img.convert("RGB")

    # Pillow versions differ. This fallback keeps the app copy-paste friendly.
    try:
        method = Image.Quantize.MEDIANCUT
    except AttributeError:  # old Pillow
        method = Image.MEDIANCUT

    q = rgb.quantize(colors=int(color_count), method=method)
    q_rgb = q.convert("RGB")
    arr = np.array(q_rgb)

    flat = arr.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    palette = [(tuple(map(int, c)), int(n)) for c, n in zip(colors, counts)]
    palette.sort(key=lambda x: x[1], reverse=True)
    return arr, palette


def should_skip_color(rgb: Tuple[int, int, int], skip_white: bool, white_threshold: int) -> bool:
    if not skip_white:
        return False
    return min(rgb) >= white_threshold


def make_color_mask(q_arr: np.ndarray, rgb: Tuple[int, int, int]) -> np.ndarray:
    mask = np.all(q_arr == np.array(rgb, dtype=np.uint8), axis=2).astype(np.uint8) * 255
    return mask


# -----------------------------
# SVG path creation
# -----------------------------

def contour_to_path(contour: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> str:
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return ""

    # Integer coordinates keep SVG small and GT7/SVGOMG friendly.
    first_x, first_y = pts[0]
    commands = [f"M{int(first_x) + offset_x} {int(first_y) + offset_y}"]
    for x, y in pts[1:]:
        commands.append(f"L{int(x) + offset_x} {int(y) + offset_y}")
    commands.append("Z")
    return "".join(commands)


def mask_to_svg_paths(
    mask: np.ndarray,
    fill: str,
    layer: str,
    min_area: float,
    simplify_ratio: float,
    dilation_px: int = 0,
    offset_x: int = 0,
    offset_y: int = 0,
    max_paths: int = 99999,
) -> List[SvgPath]:
    """Convert a binary mask to simple filled SVG path elements."""
    if mask.ndim != 2:
        raise ValueError("mask must be grayscale")

    work = (mask > 0).astype(np.uint8) * 255

    if dilation_px > 0:
        k = max(1, int(dilation_px))
        kernel = np.ones((k, k), np.uint8)
        work = cv2.dilate(work, kernel, iterations=1)

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Larger areas first. This works like manual underpainting.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    out: List[SvgPath] = []

    for contour in contours[:max_paths]:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        peri = cv2.arcLength(contour, True)
        epsilon = max(0.0, float(simplify_ratio)) * peri
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue

        d = contour_to_path(approx, offset_x=offset_x, offset_y=offset_y)
        if not d:
            continue

        # Avoid style attributes; simple fill path is easiest for SVGOMG and GT7.
        element = f'<path fill="{fill}" d="{d}"/>'
        out.append(SvgPath(element=element, area=area, layer=layer, color=fill))

    return out


def svg_wrap(path_elements: Sequence[str], width: int, height: int, title: str | None = None) -> str:
    title_el = ""
    if title:
        title_safe = (
            title.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        title_el = f"<title>{title_safe}</title>"

    # No XML declaration to save bytes.
    body = "".join(path_elements)
    return f'<svg xmlns="{SVG_NS}" viewBox="0 0 {width} {height}" width="{width}" height="{height}">{title_el}{body}</svg>'


def svg_from_records(records: Sequence[SvgPath], width: int, height: int, title: str) -> str:
    return svg_wrap([r.element for r in records], width, height, title=title)


# -----------------------------
# Line art extraction
# -----------------------------

def extract_line_mask(
    img: Image.Image,
    adaptive_block_size: int,
    adaptive_c: int,
    canny_low: int,
    canny_high: int,
    line_dilate: int,
    use_canny: bool,
    use_adaptive: bool,
) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Bilateral filter keeps edges while smoothing color noise.
    smooth = cv2.bilateralFilter(gray, 7, 50, 50)

    masks: List[np.ndarray] = []

    if use_adaptive:
        block = int(adaptive_block_size)
        if block % 2 == 0:
            block += 1
        block = max(3, block)
        adaptive = cv2.adaptiveThreshold(
            smooth,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            int(adaptive_c),
        )
        masks.append(adaptive)

    if use_canny:
        edges = cv2.Canny(smooth, int(canny_low), int(canny_high))
        masks.append(edges)

    if not masks:
        return np.zeros(gray.shape, dtype=np.uint8)

    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)

    if line_dilate > 0:
        k = max(1, int(line_dilate))
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# -----------------------------
# Face/detail region
# -----------------------------

def get_focus_rect(width: int, height: int, cx_pct: float, cy_pct: float, size_pct: float) -> Tuple[int, int, int, int]:
    # size_pct is based on shorter image side.
    side = int(round(min(width, height) * float(size_pct) / 100.0))
    side = max(8, min(side, max(width, height)))
    cx = int(round(width * float(cx_pct) / 100.0))
    cy = int(round(height * float(cy_pct) / 100.0))

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)

    # If clipped at edge, keep rectangle as close to requested side as possible.
    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)
    return x1, y1, x2, y2


def draw_focus_preview(img: Image.Image, rect: Tuple[int, int, int, int]) -> Image.Image:
    arr = np.array(img.convert("RGB")).copy()
    x1, y1, x2, y2 = rect
    color = (255, 0, 0)
    thickness = max(2, int(round(max(img.size) / 300)))
    cv2.rectangle(arr, (x1, y1), (x2 - 1, y2 - 1), color, thickness)
    return Image.fromarray(arr)


def build_color_records(
    img: Image.Image,
    color_count: int,
    min_area: float,
    simplify_ratio: float,
    dilation_px: int,
    skip_white: bool,
    white_threshold: int,
    layer_name: str,
    offset_x: int = 0,
    offset_y: int = 0,
) -> Tuple[List[SvgPath], List[Tuple[Tuple[int, int, int], int]]]:
    q_arr, palette = quantize_rgb(img, color_count)
    records: List[SvgPath] = []

    for rgb, _count in palette:
        if should_skip_color(rgb, skip_white=skip_white, white_threshold=white_threshold):
            continue
        mask = make_color_mask(q_arr, rgb)
        fill = rgb_to_hex(rgb)
        records.extend(
            mask_to_svg_paths(
                mask=mask,
                fill=fill,
                layer=layer_name,
                min_area=min_area,
                simplify_ratio=simplify_ratio,
                dilation_px=dilation_px,
                offset_x=offset_x,
                offset_y=offset_y,
            )
        )

    records.sort(key=lambda r: r.area, reverse=True)
    return records, palette


def build_face_detail_records(
    img: Image.Image,
    rect: Tuple[int, int, int, int],
    face_colors: int,
    face_min_area: float,
    face_simplify: float,
    skip_white: bool,
    white_threshold: int,
    line_settings: Dict[str, int | bool],
) -> List[SvgPath]:
    x1, y1, x2, y2 = rect
    crop = img.crop((x1, y1, x2, y2))

    color_records, _ = build_color_records(
        crop,
        color_count=face_colors,
        min_area=face_min_area,
        simplify_ratio=face_simplify,
        dilation_px=0,
        skip_white=skip_white,
        white_threshold=white_threshold,
        layer_name="face_color",
        offset_x=x1,
        offset_y=y1,
    )

    line_mask = extract_line_mask(
        crop,
        adaptive_block_size=int(line_settings["adaptive_block_size"]),
        adaptive_c=int(line_settings["adaptive_c"]),
        canny_low=int(line_settings["canny_low"]),
        canny_high=int(line_settings["canny_high"]),
        line_dilate=int(line_settings["line_dilate"]),
        use_canny=bool(line_settings["use_canny"]),
        use_adaptive=bool(line_settings["use_adaptive"]),
    )
    line_records = mask_to_svg_paths(
        mask=line_mask,
        fill="#111111",
        layer="face_line",
        min_area=max(1.0, face_min_area * 0.4),
        simplify_ratio=max(0.0005, face_simplify * 0.8),
        dilation_px=0,
        offset_x=x1,
        offset_y=y1,
    )

    # Put color details first, then face line details on top.
    return color_records + line_records


# -----------------------------
# Splitting / ZIP
# -----------------------------

def split_records_by_size(
    records: Sequence[SvgPath],
    width: int,
    height: int,
    max_kb: float,
    prefix: str,
) -> List[Tuple[str, str]]:
    """Greedy split SVG path records into multiple SVG strings around target size."""
    max_bytes = int(float(max_kb) * 1024)
    if max_bytes < 2048:
        max_bytes = 2048

    files: List[Tuple[str, str]] = []
    current: List[str] = []
    part = 1

    def current_svg(elements: Sequence[str], part_no: int) -> str:
        return svg_wrap(elements, width, height, title=f"{prefix} part {part_no:02d}")

    for rec in records:
        candidate = current + [rec.element]
        candidate_svg = current_svg(candidate, part)
        candidate_size = len(candidate_svg.encode("utf-8"))

        if current and candidate_size > max_bytes:
            svg = current_svg(current, part)
            files.append((f"{prefix}_part_{part:02d}.svg", svg))
            part += 1
            current = [rec.element]
        else:
            current = candidate

    if current:
        svg = current_svg(current, part)
        files.append((f"{prefix}_part_{part:02d}.svg", svg))

    return files


def make_zip(files: Sequence[Tuple[str, str]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, text in files:
            zf.writestr(filename, text.encode("utf-8"))
    return buf.getvalue()


def make_stats_table(files: Sequence[Tuple[str, str]]) -> List[Dict[str, str]]:
    rows = []
    for name, data in files:
        rows.append({"file": name, "size": pretty_kb(data)})
    return rows


# -----------------------------
# Streamlit UI
# -----------------------------

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("画像 → 下地色面SVG / 線画SVG / 顔周辺高精度SVG / GT7向け分割SVG ZIP")

    with st.expander("使い方", expanded=False):
        st.markdown(
            """
1. 画像をアップロードします。  
2. まずは初期設定のまま **生成する** を押します。  
3. 線が太すぎる場合は **線画の太さ** を下げます。  
4. SVGが重すぎる場合は **色数** と **最大辺** を下げ、**パス単純化** と **最小面積カット** を上げます。  
5. ZIP内の `gt7_part_XX.svg` をSVGOMG/SVGOで圧縮してからGT7にアップします。  
6. GT7内では、同じサイズ・同じ位置で重ねる想定です。
            """
        )

    uploaded = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("PNG/JPG/WebP をアップロードしてください。")
        return

    original_name = safe_name(uploaded.name)

    try:
        img_original = pil_to_rgb_image(uploaded)
    except Exception as e:
        st.error(f"画像を読み込めませんでした: {e}")
        return

    st.sidebar.header("基本設定")
    max_side = st.sidebar.slider("最大辺 px", 300, 1600, 900, 50)
    color_count = st.sidebar.slider("下地の色数", 4, 48, 18, 1)
    min_area = st.sidebar.slider("最小面積カット", 1, 800, 35, 1)
    simplify_ratio = st.sidebar.slider("パス単純化", 0.000, 0.040, 0.006, 0.001, format="%.3f")
    color_dilation = st.sidebar.slider("下地の少しはみ出し", 0, 5, 1, 1)
    skip_white = st.sidebar.checkbox("白背景をSVG化しない", True)
    white_threshold = st.sidebar.slider("白判定", 220, 255, 246, 1)

    st.sidebar.header("線画設定")
    use_adaptive = st.sidebar.checkbox("線画: 明暗しきい値", True)
    adaptive_block_size = st.sidebar.slider("線画検出サイズ", 5, 81, 31, 2)
    adaptive_c = st.sidebar.slider("線画検出強度", 1, 30, 8, 1)
    use_canny = st.sidebar.checkbox("線画: エッジ検出も混ぜる", True)
    canny_low = st.sidebar.slider("Canny low", 10, 200, 60, 5)
    canny_high = st.sidebar.slider("Canny high", 40, 300, 160, 5)
    line_dilate = st.sidebar.slider("線画の太さ", 0, 5, 1, 1)
    line_min_area = st.sidebar.slider("線画のゴミ除去", 1, 500, 12, 1)
    line_simplify = st.sidebar.slider("線画の単純化", 0.000, 0.030, 0.004, 0.001, format="%.3f")

    st.sidebar.header("顔周辺高精度")
    enable_face = st.sidebar.checkbox("顔/重要部分を別レイヤー化", True)
    focus_cx = st.sidebar.slider("重要部分 中心X %", 0, 100, 50, 1)
    focus_cy = st.sidebar.slider("重要部分 中心Y %", 0, 100, 38, 1)
    focus_size = st.sidebar.slider("重要部分 サイズ %", 10, 100, 45, 1)
    face_colors = st.sidebar.slider("重要部分の色数", 8, 64, 34, 1)
    face_min_area = st.sidebar.slider("重要部分 最小面積", 1, 250, 8, 1)
    face_simplify = st.sidebar.slider("重要部分 単純化", 0.000, 0.030, 0.003, 0.001, format="%.3f")

    st.sidebar.header("GT7分割")
    target_kb = st.sidebar.slider("1 SVGの目標KB", 8.0, 30.0, 14.2, 0.1)
    include_face_in_parts = st.sidebar.checkbox("分割SVGに顔高精度も含める", True)

    img = resize_keep_aspect(img_original, int(max_side))
    width, height = img.size
    focus_rect = get_focus_rect(width, height, focus_cx, focus_cy, focus_size)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("元画像 / リサイズ後")
        st.image(img, caption=f"{width} x {height}px", use_container_width=True)
    with col2:
        st.subheader("顔/重要部分の範囲")
        st.image(draw_focus_preview(img, focus_rect), use_container_width=True)

    if not st.button("生成する", type="primary"):
        return

    try:
        with st.spinner("SVGを生成中..."):
            base_records, palette = build_color_records(
                img,
                color_count=color_count,
                min_area=float(min_area),
                simplify_ratio=float(simplify_ratio),
                dilation_px=int(color_dilation),
                skip_white=skip_white,
                white_threshold=int(white_threshold),
                layer_name="base_color",
            )

            line_mask = extract_line_mask(
                img,
                adaptive_block_size=int(adaptive_block_size),
                adaptive_c=int(adaptive_c),
                canny_low=int(canny_low),
                canny_high=int(canny_high),
                line_dilate=int(line_dilate),
                use_canny=use_canny,
                use_adaptive=use_adaptive,
            )
            line_records = mask_to_svg_paths(
                mask=line_mask,
                fill="#111111",
                layer="line_art",
                min_area=float(line_min_area),
                simplify_ratio=float(line_simplify),
                dilation_px=0,
            )

            face_records: List[SvgPath] = []
            if enable_face:
                line_settings = {
                    "adaptive_block_size": int(adaptive_block_size),
                    "adaptive_c": int(adaptive_c),
                    "canny_low": int(canny_low),
                    "canny_high": int(canny_high),
                    "line_dilate": int(line_dilate),
                    "use_canny": bool(use_canny),
                    "use_adaptive": bool(use_adaptive),
                }
                face_records = build_face_detail_records(
                    img,
                    rect=focus_rect,
                    face_colors=int(face_colors),
                    face_min_area=float(face_min_area),
                    face_simplify=float(face_simplify),
                    skip_white=skip_white,
                    white_threshold=int(white_threshold),
                    line_settings=line_settings,
                )

            # Recommended layer order:
            # 1 base colors large to small
            # 2 line art
            # 3 face details on top
            all_records = base_records + line_records + face_records
            split_source_records = base_records + line_records + (face_records if include_face_in_parts else [])

            base_svg = svg_from_records(base_records, width, height, "base colors")
            line_svg = svg_from_records(line_records, width, height, "line art")
            face_svg = svg_from_records(face_records, width, height, "face detail") if face_records else svg_wrap([], width, height, "face detail empty")
            all_svg = svg_from_records(all_records, width, height, "all layers preview")

            split_files = split_records_by_size(
                split_source_records,
                width=width,
                height=height,
                max_kb=float(target_kb),
                prefix="gt7",
            )

            files: List[Tuple[str, str]] = [
                (f"{original_name}_00_all_layers.svg", all_svg),
                (f"{original_name}_01_base_colors.svg", base_svg),
                (f"{original_name}_02_line_art.svg", line_svg),
                (f"{original_name}_03_face_detail.svg", face_svg),
            ]
            files.extend(split_files)
            zip_bytes = make_zip(files)

        st.success("生成完了")

        st.subheader("出力ファイル")
        st.dataframe(make_stats_table(files), use_container_width=True, hide_index=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button(
                "全部入りSVG",
                data=all_svg.encode("utf-8"),
                file_name=f"{original_name}_00_all_layers.svg",
                mime="image/svg+xml",
            )
        with c2:
            st.download_button(
                "下地色面SVG",
                data=base_svg.encode("utf-8"),
                file_name=f"{original_name}_01_base_colors.svg",
                mime="image/svg+xml",
            )
        with c3:
            st.download_button(
                "線画SVG",
                data=line_svg.encode("utf-8"),
                file_name=f"{original_name}_02_line_art.svg",
                mime="image/svg+xml",
            )
        with c4:
            st.download_button(
                "全部ZIP",
                data=zip_bytes,
                file_name=f"{original_name}_gt7_svg_parts.zip",
                mime="application/zip",
            )

        st.subheader("プレビュー")
        st.caption("ブラウザ表示用の簡易プレビューです。GT7/SVGOMG後の見た目とは少し変わる場合があります。")
        st.image(all_svg.encode("utf-8"), caption="all_layers.svg", use_container_width=True)

        st.subheader("調整の目安")
        st.markdown(
            f"""
- パス数: 下地 `{len(base_records)}` / 線画 `{len(line_records)}` / 重要部分 `{len(face_records)}`
- 全部入り: `{pretty_kb(all_svg)}`
- GT7分割: `{len(split_files)}` 個
- 15KBを超えるパートがある場合: **最大辺pxを下げる**、**色数を下げる**、**パス単純化を上げる**、**最小面積カットを上げる** の順で調整してください。
- 線が汚い場合: **線画の太さ**を0〜1にし、**線画のゴミ除去**を上げてください。
- 顔が崩れる場合: 重要部分の赤枠を顔に合わせ、**重要部分の色数**を増やし、**重要部分 単純化**を下げてください。
            """
        )

        with st.expander("検出された主な色", expanded=False):
            color_cols = st.columns(6)
            for i, (rgb, count) in enumerate(palette[:36]):
                with color_cols[i % 6]:
                    hex_color = rgb_to_hex(rgb)
                    st.markdown(
                        f"<div style='width:100%;height:28px;background:{hex_color};border:1px solid #999'></div>"
                        f"<small>{hex_color}<br>{count} px</small>",
                        unsafe_allow_html=True,
                    )

        with st.expander("SVGOMGに通す前の注意", expanded=False):
            st.markdown(
                """
SVGOMG/SVGOでは、まず以下を試してください。

- `Prettify markup`: OFF
- `Remove metadata`: ON
- `Convert colors`: ON
- `Round/rewrite paths`: ON
- `Merge paths`: ONにして確認。見た目が壊れる場合はOFF
- `Remove viewBox`: OFF

GT7で位置合わせしやすいように、各SVGは同じ `viewBox` で出しています。
                """
            )

    except Exception:
        st.error("生成中にエラーが出ました。設定を軽くして再試行してください。")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
