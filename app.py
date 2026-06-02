from __future__ import annotations

import base64
import html
import io
import re
import time
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFilter, ImageOps

HARD_LIMIT = 2 * 1024 * 1024

PRESETS = {
    "高速 / 30秒目標": {
        "colors": 48,
        "side": 900,
        "simp": 1.0,
        "area": 3,
        "dec": 0,
        "attempts": 1,
    },
    "標準 / 1分目標": {
        "colors": 64,
        "side": 1100,
        "simp": 0.75,
        "area": 2,
        "dec": 1,
        "attempts": 2,
    },
    "高品質 / 少し時間がかかる": {
        "colors": 96,
        "side": 1350,
        "simp": 0.55,
        "area": 1,
        "dec": 1,
        "attempts": 3,
    },
    "元画像優先 / 重い": {
        "colors": 128,
        "side": 1600,
        "simp": 0.40,
        "area": 1,
        "dec": 1,
        "attempts": 4,
    },
}


def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def safe_svg(svg: str) -> str:
    for tag in [
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
    ]:
        svg = re.sub(rf"<\s*{tag}\b.*?<\s*/\s*{tag}\s*>", "", svg, flags=re.I | re.S)
        svg = re.sub(rf"<\s*{tag}\b[^>]*/\s*>", "", svg, flags=re.I | re.S)

    svg = re.sub(r"\s+", " ", svg)
    svg = svg.replace("> <", "><").strip()
    return svg


def checkerboard(size: tuple[int, int], cell: int = 12) -> Image.Image:
    w, h = size
    img = Image.new("RGBA", size, (245, 245, 245, 255))
    dr = ImageDraw.Draw(img)

    c1 = (250, 250, 250, 255)
    c2 = (225, 225, 225, 255)

    for y in range(0, h, cell):
        for x in range(0, w, cell):
            dr.rectangle(
                [x, y, x + cell - 1, y + cell - 1],
                fill=c1 if ((x // cell) + (y // cell)) % 2 == 0 else c2,
            )

    return img


def paste_on_checkerboard(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    bg = checkerboard(img.size)
    bg.alpha_composite(img, (0, 0))
    return bg


def fit_image(img: Image.Image, side: int, enhance: bool) -> Image.Image:
    img = img.convert("RGBA")

    if enhance:
        rgb = img.convert("RGB")
        rgb = ImageOps.autocontrast(rgb)
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=115, threshold=3))
        img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))

    if max(img.size) > side:
        scale = side / max(img.size)
        new_size = (
            max(1, int(img.width * scale)),
            max(1, int(img.height * scale)),
        )
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    return img


def quantize_image(
    img: Image.Image,
    n_colors: int,
    alpha_threshold: int,
    white_bg: bool,
):
    img = img.convert("RGBA")
    alpha = np.array(img.getchannel("A"))
    visible = alpha >= alpha_threshold

    if white_bg:
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
        visible[:] = True

    rgb = img.convert("RGB")

    q = rgb.quantize(
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
        palette.append(
            tuple(palette_raw[old_idx * 3 : old_idx * 3 + 3])
        )

    preview_arr = np.zeros((remapped.shape[0], remapped.shape[1], 4), dtype=np.uint8)

    for i, c in enumerate(palette):
        mask = remapped == i
        preview_arr[mask, 0] = c[0]
        preview_arr[mask, 1] = c[1]
        preview_arr[mask, 2] = c[2]
        preview_arr[mask, 3] = 255

    if white_bg:
        preview_arr[:, :, 3] = 255

    preview = Image.fromarray(preview_arr, "RGBA")
    return remapped, palette, preview


def fmt_num(x: float, decimals: int) -> str:
    if decimals <= 0:
        return str(int(round(x)))
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".").replace("-0", "0")


def contour_area_cv(contour: np.ndarray) -> float:
    try:
        return float(abs(cv2.contourArea(contour)))
    except Exception:
        return 0.0


def contour_to_path_cv(
    contour: np.ndarray,
    simplify: float,
    min_area: int,
    decimals: int,
) -> Optional[str]:
    if contour is None or len(contour) < 3:
        return None

    if contour_area_cv(contour) < min_area:
        return None

    epsilon = max(0.05, float(simplify))
    approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)

    if approx is None or len(approx) < 3:
        return None

    pts = approx.reshape(-1, 2).astype(np.float32)

    if len(pts) < 3:
        return None

    commands = [f"M{fmt_num(pts[0, 0], decimals)} {fmt_num(pts[0, 1], decimals)}"]

    for x, y in pts[1:]:
        commands.append(f"L{fmt_num(x, decimals)} {fmt_num(y, decimals)}")

    commands.append("Z")
    return "".join(commands)


def clean_mask_fast(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(np.uint8)

    binary = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        return binary

    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def build_svg_fast(
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    width: int,
    height: int,
    simplify: float,
    min_area: int,
    decimals: int,
    white_bg: bool,
):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    path_count = 0
    contour_count = 0

    if white_bg:
        parts.append(f'<path fill="#ffffff" d="M0 0H{width}V{height}H0Z"/>')
        path_count += 1

    order = sorted(
        [(int((labels == i).sum()), i) for i in range(len(palette))],
        reverse=True,
    )

    for _, color_index in order:
        mask = labels == color_index

        if int(mask.sum()) < min_area:
            continue

        binary = clean_mask_fast(mask, min_area=min_area)
        if binary.max() == 0:
            continue

        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        sub_paths = []

        for contour in contours:
            d = contour_to_path_cv(
                contour=contour,
                simplify=simplify,
                min_area=min_area,
                decimals=decimals,
            )
            if d:
                sub_paths.append(d)
                contour_count += 1

        if not sub_paths:
            continue

        r, g, b = palette[color_index]
        parts.append(
            f'<path fill="#{r:02x}{g:02x}{b:02x}" fill-rule="evenodd" d="{html.escape("".join(sub_paths), quote=True)}"/>'
        )
        path_count += 1

    parts.append("</svg>")
    svg = safe_svg("".join(parts))

    return svg, path_count, contour_count


def make_compare_image(original: Image.Image, vector_preview: Image.Image) -> bytes:
    def fit_for_preview(im: Image.Image) -> Image.Image:
        im = im.convert("RGBA")
        if im.height > 900:
            scale = 900 / im.height
            im = im.resize((int(im.width * scale), 900), Image.Resampling.LANCZOS)
        return im

    a = fit_for_preview(original)
    b = fit_for_preview(vector_preview)

    a = paste_on_checkerboard(a)
    b = paste_on_checkerboard(b)

    margin = 16
    gap = 24
    title_h = 36

    w = a.width + b.width + gap + margin * 2
    h = max(a.height, b.height) + title_h + margin * 2

    canvas = Image.new("RGBA", (w, h), (245, 245, 245, 255))
    dr = ImageDraw.Draw(canvas)

    dr.text((margin, 10), "Original", fill=(0, 0, 0))
    dr.text((margin + a.width + gap, 10), "Vector preview", fill=(0, 0, 0))

    canvas.alpha_composite(a, (margin, title_h + margin))
    canvas.alpha_composite(b, (margin + a.width + gap, title_h + margin))

    return png_bytes(canvas)


def convert_once(
    img: Image.Image,
    colors: int,
    side: int,
    simplify: float,
    min_area: int,
    decimals: int,
    alpha_threshold: int,
    enhance: bool,
    white_bg: bool,
):
    work = fit_image(img, side=side, enhance=enhance)

    labels, palette, preview = quantize_image(
        work,
        n_colors=colors,
        alpha_threshold=alpha_threshold,
        white_bg=white_bg,
    )

    svg, path_count, contour_count = build_svg_fast(
        labels=labels,
        palette=palette,
        width=work.width,
        height=work.height,
        simplify=simplify,
        min_area=min_area,
        decimals=decimals,
        white_bg=white_bg,
    )

    return {
        "svg": svg,
        "size": len(svg.encode("utf-8")),
        "preview": preview,
        "compare": make_compare_image(work, preview),
        "colors": len(palette),
        "paths": path_count,
        "contours": contour_count,
        "width": work.width,
        "height": work.height,
    }


def convert_image(img: Image.Image, cfg: dict):
    target = min(int(cfg["target"]), HARD_LIMIT)

    colors = int(cfg["colors"])
    side = int(cfg["side"])
    simplify = float(cfg["simp"])
    min_area = int(cfg["area"])
    decimals = int(cfg["dec"])
    max_attempts = int(cfg["attempts"])

    best = None

    for i in range(max_attempts):
        result = convert_once(
            img=img,
            colors=max(2, colors),
            side=max(128, side),
            simplify=max(0.05, simplify),
            min_area=max(1, min_area),
            decimals=max(0, decimals),
            alpha_threshold=cfg["alpha"],
            enhance=cfg["enhance"],
            white_bg=cfg["white_bg"],
        )

        result["attempts"] = i + 1

        if best is None:
            best = result
        else:
            if result["size"] <= target and best["size"] <= target:
                if result["size"] > best["size"]:
                    best = result
            elif result["size"] <= target < best["size"]:
                best = result
            elif result["size"] > target and best["size"] > target:
                if result["size"] < best["size"]:
                    best = result

        if result["size"] <= target:
            return result

        colors = max(8, int(colors * 0.82))
        side = max(550, int(side * 0.90))
        simplify = min(4.0, simplify * 1.25)
        min_area = min(24, int(min_area * 1.4 + 1))
        decimals = 0

    return best


def data_url_svg(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return "data:image/svg+xml;base64," + encoded


def target_from_choice(choice: str, custom_kb: int) -> int:
    mapping = {
        "2MB 絶対上限": 2 * 1024 * 1024,
        "1MB": 1024 * 1024,
        "500KB": 500 * 1024,
        "250KB": 250 * 1024,
    }
    if choice == "カスタムKB":
        return max(1, min(2048, int(custom_kb))) * 1024
    return mapping.get(choice, 2 * 1024 * 1024)


st.set_page_config(
    page_title="GT7 SVG Vectorizer Fast",
    page_icon="🏁",
    layout="wide",
)

st.title("🏁 GT7 SVG Vectorizer Fast")
st.caption("30秒〜1分程度での変換を狙った高速版です。まず高速に出して、必要なら品質を上げる方式です。")

with st.expander("使い方", expanded=False):
    st.markdown(
        """
1. 画像をアップロード  
2. まず **標準 / 1分目標** で変換  
3. 崩れが大きい場合は **高品質** に変更  
4. 遅い場合は **高速 / 30秒目標** に変更  
5. SVGを保存
"""
    )

uploaded = st.file_uploader(
    "画像を選択",
    type=["png", "jpg", "jpeg", "webp"],
)

with st.sidebar:
    st.header("設定")

    preset_name = st.selectbox(
        "品質プリセット",
        list(PRESETS.keys()) + ["手動設定"],
        index=1,
    )

    base = PRESETS.get(preset_name, PRESETS["標準 / 1分目標"]).copy()

    size_choice = st.selectbox(
        "目標サイズ",
        ["2MB 絶対上限", "1MB", "500KB", "250KB", "カスタムKB"],
        index=0,
    )

    custom_kb = st.number_input(
        "カスタムKB",
        min_value=1,
        max_value=2048,
        value=2048,
    )

    enhance = st.toggle("低画質画像を軽く補正", True)
    white_bg = st.toggle("透明部分を白背景にする", False)

    st.divider()
    st.subheader("詳細調整")

    base["colors"] = st.slider(
        "色数",
        2,
        160,
        int(base["colors"]),
    )

    base["side"] = st.slider(
        "処理サイズ（長辺px）",
        256,
        1800,
        int(base["side"]),
        step=16,
    )

    base["simp"] = st.slider(
        "パス簡略化（小さいほど元絵優先・遅い）",
        0.10,
        5.0,
        float(base["simp"]),
        step=0.05,
    )

    base["area"] = st.slider(
        "小さい形状の削除",
        1,
        40,
        int(base["area"]),
    )

    base["alpha"] = st.slider(
        "透明判定",
        0,
        255,
        16,
    )

    base["dec"] = st.slider(
        "座標の小数桁",
        0,
        2,
        int(base["dec"]),
    )

    base["attempts"] = st.slider(
        "自動軽量化の試行回数",
        1,
        5,
        int(base["attempts"]),
    )

    base["target"] = target_from_choice(size_choice, custom_kb)
    base["enhance"] = enhance
    base["white_bg"] = white_bg

    st.divider()
    st.markdown(
        """
**速度重視のおすすめ**
- 標準 / 1分目標
- 色数: 48〜64
- 処理サイズ: 900〜1100
- パス簡略化: 0.75〜1.2
- 試行回数: 1〜2

**画質重視**
- 高品質
- 色数: 80〜96
- 処理サイズ: 1200〜1350
- パス簡略化: 0.45〜0.65
"""
    )

if uploaded is None:
    st.info("画像をアップロードしてください。")
    st.stop()

try:
    original_img = Image.open(uploaded).convert("RGBA")
except Exception as e:
    st.error(f"画像を開けませんでした: {e}")
    st.stop()

if st.button("SVGへ変換", type="primary", use_container_width=True):
    start = time.time()

    with st.spinner("変換中です…"):
        st.session_state.result = convert_image(original_img, base)
        st.session_state.elapsed = time.time() - start
        st.session_state.filename = uploaded.name
        st.session_state.used_settings = base.copy()

if "result" not in st.session_state:
    st.subheader("元画像")
    st.image(paste_on_checkerboard(original_img), use_container_width=True)
    st.stop()

result = st.session_state.result

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("SVGサイズ", f"{result['size']:,} bytes")
m2.metric("目標サイズ", f"{st.session_state.used_settings['target']:,} bytes")
m3.metric("色数", str(result["colors"]))
m4.metric("path数", str(result["paths"]))
m5.metric("輪郭数", str(result["contours"]))
m6.metric("変換時間", f"{st.session_state.elapsed:.1f} 秒")

if result["size"] <= st.session_state.used_settings["target"]:
    st.success("目標サイズ内です。")
else:
    st.warning("目標サイズを超えています。色数・処理サイズを下げるか、パス簡略化を少し上げてください。")

if result["size"] > HARD_LIMIT:
    st.error("2MBの絶対上限を超えています。設定を軽くしてください。")

if st.session_state.elapsed > 60:
    st.warning(
        "1分を超えました。次回は「高速 / 30秒目標」にするか、処理サイズを900〜1100、色数を48〜64にしてください。"
    )
elif st.session_state.elapsed <= 60:
    st.info("速度目標内です。画質を上げたい場合は色数や処理サイズを少し上げてください。")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("元画像")
    st.image(paste_on_checkerboard(original_img), use_container_width=True)

with col_right:
    st.subheader("SVGプレビュー")
    svg_url = data_url_svg(result["svg"])

    components.html(
        f"""
        <div style="
            width:100%;
            min-height:520px;
            border:1px solid #ddd;
            border-radius:12px;
            padding:14px;
            background:
              linear-gradient(45deg, #f8f8f8 25%, transparent 25%),
              linear-gradient(-45deg, #f8f8f8 25%, transparent 25%),
              linear-gradient(45deg, transparent 75%, #f8f8f8 75%),
              linear-gradient(-45deg, transparent 75%, #f8f8f8 75%);
            background-size:24px 24px;
            background-position:0 0, 0 12px, 12px -12px, -12px 0px;
            text-align:center;
        ">
            <img src="{svg_url}" style="max-width:100%; height:auto;" />
        </div>
        """,
        height=560,
        scrolling=True,
    )

st.subheader("比較プレビュー")
st.image(result["compare"], use_container_width=True)

base_name = re.sub(r"\.[^.]+$", "", st.session_state.filename)
base_name = re.sub(r"[^A-Za-z0-9_-]+", "_", base_name).strip("_") or "converted"

d1, d2 = st.columns(2)

with d1:
    st.download_button(
        "SVGを保存",
        data=result["svg"].encode("utf-8"),
        file_name=f"{base_name}_vectorized.svg",
        mime="image/svg+xml",
        use_container_width=True,
    )

with d2:
    st.download_button(
        "比較PNGを保存",
        data=result["compare"],
        file_name=f"{base_name}_compare.png",
        mime="image/png",
        use_container_width=True,
    )

with st.expander("SVGコードを表示 / コピー"):
    st.code(result["svg"], language="xml")