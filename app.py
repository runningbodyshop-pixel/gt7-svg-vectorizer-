from __future__ import annotations

import base64
import html
import io
import re
import time
from typing import Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from skimage import measure

# ---------------------------------
# 基本設定
# ---------------------------------
HARD_LIMIT = 2 * 1024 * 1024  # 2MB 絶対上限

PRESETS = {
    "元画像優先 / 超高品質": {
        "colors": 128,
        "side": 1800,
        "simp": 0.35,
        "area": 1,
        "smooth": 0,
        "dec": 1,
    },
    "元画像優先 / 高品質": {
        "colors": 96,
        "side": 1500,
        "simp": 0.45,
        "area": 1,
        "smooth": 0,
        "dec": 1,
    },
    "標準 / バランス": {
        "colors": 64,
        "side": 1200,
        "simp": 0.70,
        "area": 2,
        "smooth": 0,
        "dec": 1,
    },
    "軽量 / 確認用": {
        "colors": 32,
        "side": 900,
        "simp": 1.10,
        "area": 4,
        "smooth": 0,
        "dec": 0,
    },
}

# ---------------------------------
# 共通ユーティリティ
# ---------------------------------
def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def safe_svg(s: str) -> str:
    # 念のため GT 系で嫌われやすいタグは排除
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
        s = re.sub(rf"<\s*{tag}\b.*?<\s*/\s*{tag}\s*>", "", s, flags=re.I | re.S)
        s = re.sub(rf"<\s*{tag}\b[^>]*/\s*>", "", s, flags=re.I | re.S)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("> <", "><").strip()
    return s


def checkerboard(size=(300, 300), cell=12) -> Image.Image:
    w, h = size
    img = Image.new("RGBA", size, (240, 240, 240, 255))
    dr = ImageDraw.Draw(img)
    c1 = (248, 248, 248, 255)
    c2 = (225, 225, 225, 255)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            dr.rectangle(
                [x, y, x + cell - 1, y + cell - 1],
                fill=c1 if ((x // cell) + (y // cell)) % 2 == 0 else c2,
            )
    return img


def paste_on_checkerboard(img: Image.Image) -> Image.Image:
    bg = checkerboard(img.size)
    bg.alpha_composite(img.convert("RGBA"), (0, 0))
    return bg


def fit_image(img: Image.Image, side: int, enhance: bool) -> Image.Image:
    img = img.convert("RGBA")

    if enhance:
        rgb = img.convert("RGB")
        rgb = ImageOps.autocontrast(rgb)
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
        img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))

        # 小さい画像なら軽く拡大してから処理
        if max(img.size) < 900:
            scale = min(2.0, 900 / max(img.size))
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.Resampling.LANCZOS,
            )

    # 最終処理サイズ
    if max(img.size) > side:
        scale = side / max(img.size)
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.LANCZOS,
        )

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

    palette = q.getpalette()[:768]
    used = sorted(int(x) for x in np.unique(labels) if int(x) >= 0)

    remapped = np.full_like(labels, -1)
    colors = []
    for new_idx, old_idx in enumerate(used):
        remapped[labels == old_idx] = new_idx
        colors.append(tuple(palette[old_idx * 3 : old_idx * 3 + 3]))

    preview = np.zeros((remapped.shape[0], remapped.shape[1], 4), dtype=np.uint8)
    for i, c in enumerate(colors):
        m = remapped == i
        preview[m, 0] = c[0]
        preview[m, 1] = c[1]
        preview[m, 2] = c[2]
        preview[m, 3] = 255
    if white_bg:
        preview[:, :, 3] = 255

    preview_img = Image.fromarray(preview, "RGBA")
    return remapped, colors, preview_img


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) < 3 or epsilon <= 0:
        return points

    start = points[0]
    end = points[-1]
    line = end - start
    line_len = float(np.linalg.norm(line))

    if line_len == 0:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        dists = np.abs(
            line[0] * (start[1] - points[:, 1]) - line[1] * (start[0] - points[:, 0])
        ) / line_len

    idx = int(np.argmax(dists))
    if float(dists[idx]) > epsilon:
        left = rdp(points[: idx + 1], epsilon)
        right = rdp(points[idx:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.vstack((start, end))


def chaikin(points: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0 or len(points) < 4:
        return points

    p = points
    for _ in range(steps):
        new_points = []
        for i in range(len(p)):
            a = p[i]
            b = p[(i + 1) % len(p)]
            new_points.append(0.75 * a + 0.25 * b)
            new_points.append(0.25 * a + 0.75 * b)
        p = np.array(new_points, dtype=np.float32)
        if len(p) > 8000:
            break
    return p


def fmt_num(x: float, dec: int) -> str:
    if dec <= 0:
        return str(int(round(x)))
    return f"{x:.{dec}f}".rstrip("0").rstrip(".").replace("-0", "0")


def remove_small_regions(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1 or not mask.any():
        return mask

    labels = measure.label(mask, connectivity=1)
    if labels.max() == 0:
        return mask

    counts = np.bincount(labels.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[labels]


def contour_to_path(
    contour: np.ndarray,
    simplify: float,
    min_area: int,
    smooth: int,
    decimals: int,
) -> Optional[str]:
    pts = np.stack([contour[:, 1], contour[:, 0]], axis=1).astype(np.float32)
    if len(pts) < 3:
        return None

    # 閉じる
    if np.linalg.norm(pts[0] - pts[-1]) > 0.01:
        pts = np.vstack([pts, pts[0]])

    if polygon_area(pts) < min_area:
        return None

    pts = rdp(pts, simplify)
    if len(pts) < 3:
        return None

    if np.linalg.norm(pts[0] - pts[-1]) < 0.01:
        pts = pts[:-1]

    pts = chaikin(pts, smooth)
    pts = rdp(pts, max(0.03, simplify * 0.65))

    if len(pts) < 3:
        return None

    commands = [f"M{fmt_num(pts[0,0], decimals)} {fmt_num(pts[0,1], decimals)}"]
    for x, y in pts[1:]:
        commands.append(f"L{fmt_num(x, decimals)} {fmt_num(y, decimals)}")
    commands.append("Z")
    return "".join(commands)


def build_svg(
    labels: np.ndarray,
    colors: list[tuple[int, int, int]],
    width: int,
    height: int,
    simplify: float,
    min_area: int,
    smooth: int,
    decimals: int,
    white_bg: bool,
):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    path_count = 0

    if white_bg:
        parts.append(f'<path fill="#ffffff" d="M0 0H{width}V{height}H0Z"/>')
        path_count += 1

    # 面積の大きい色から積む
    ordering = sorted(
        [(int((labels == i).sum()), i) for i in range(len(colors))],
        reverse=True,
    )

    for _, i in ordering:
        mask = labels == i
        if int(mask.sum()) < min_area:
            continue

        # ゴミだけ少し除去
        clean_mask = remove_small_regions(mask, max(1, min_area))

        padded = np.pad(clean_mask.astype(np.uint8), 1)
        sub_paths = []
        contours = measure.find_contours(padded, 0.5, fully_connected="high")

        for c in contours:
            d = contour_to_path(
                c - 1.0,
                simplify=simplify,
                min_area=min_area,
                smooth=smooth,
                decimals=decimals,
            )
            if d:
                sub_paths.append(d)

        if not sub_paths:
            continue

        r, g, b = colors[i]
        parts.append(
            f'<path fill="#{r:02x}{g:02x}{b:02x}" fill-rule="evenodd" d="{html.escape("".join(sub_paths), quote=True)}"/>'
        )
        path_count += 1

    parts.append("</svg>")
    svg = safe_svg("".join(parts))
    return svg, path_count


def make_compare_image(original: Image.Image, vector_preview: Image.Image) -> bytes:
    def fit_for_preview(im: Image.Image) -> Image.Image:
        im = im.convert("RGBA")
        if im.height > 900:
            scale = 900 / im.height
            im = im.resize(
                (int(im.width * scale), 900),
                Image.Resampling.LANCZOS,
            )
        return im

    a = fit_for_preview(original)
    b = fit_for_preview(vector_preview)

    a_bg = paste_on_checkerboard(a)
    b_bg = paste_on_checkerboard(b)

    margin = 16
    gap = 24
    title_h = 36
    w = a_bg.width + b_bg.width + gap + margin * 2
    h = max(a_bg.height, b_bg.height) + title_h + margin * 2

    canvas = Image.new("RGBA", (w, h), (245, 245, 245, 255))
    dr = ImageDraw.Draw(canvas)

    dr.text((margin, 10), "Original", fill=(0, 0, 0))
    dr.text((margin + a_bg.width + gap, 10), "Vector preview", fill=(0, 0, 0))

    canvas.alpha_composite(a_bg, (margin, title_h + margin))
    canvas.alpha_composite(b_bg, (margin + a_bg.width + gap, title_h + margin))
    return png_bytes(canvas)


def convert_once(
    img: Image.Image,
    colors: int,
    side: int,
    simplify: float,
    min_area: int,
    smooth: int,
    decimals: int,
    alpha_th: int,
    enhance: bool,
    white_bg: bool,
):
    work = fit_image(img, side=side, enhance=enhance)
    labels, palette, preview = quantize_image(
        work,
        n_colors=colors,
        alpha_threshold=alpha_th,
        white_bg=white_bg,
    )
    svg, path_count = build_svg(
        labels=labels,
        colors=palette,
        width=work.width,
        height=work.height,
        simplify=simplify,
        min_area=min_area,
        smooth=smooth,
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
        "width": work.width,
        "height": work.height,
    }


def convert_image(img: Image.Image, cfg: dict):
    best = None
    target = min(int(cfg["target"]), HARD_LIMIT)

    colors = int(cfg["colors"])
    side = int(cfg["side"])
    simplify = float(cfg["simp"])
    min_area = int(cfg["area"])
    smooth = int(cfg["smooth"])
    decimals = int(cfg["dec"])

    loops = 20 if cfg["auto"] else 1

    for i in range(loops):
        result = convert_once(
            img=img,
            colors=max(2, colors),
            side=max(128, side),
            simplify=max(0.05, simplify),
            min_area=max(1, min_area),
            smooth=max(0, smooth),
            decimals=max(0, decimals),
            alpha_th=cfg["alpha"],
            enhance=cfg["enhance"],
            white_bg=cfg["white_bg"],
        )
        result["attempts"] = i + 1

        if best is None:
            best = result
        else:
            # 目標以下なら元画像に近そうなもの（大きい方）を優先
            if result["size"] <= target and best["size"] <= target:
                if result["size"] > best["size"]:
                    best = result
            # どちらも超過なら小さい方
            elif result["size"] > target and best["size"] > target:
                if result["size"] < best["size"]:
                    best = result
            # 片方だけ目標内なら目標内を優先
            elif result["size"] <= target < best["size"]:
                best = result

        if result["size"] <= target:
            return result

        # 超過時だけ、ゆるやかに軽量化
        colors = max(8, int(colors * 0.92))
        side = max(700, int(side * 0.97))
        simplify = min(3.0, simplify * 1.08)

        if i >= 4:
            min_area = min(16, int(min_area * 1.15 + 1))
        if i >= 6:
            decimals = 0

    return best


def data_url_svg(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


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


# ---------------------------------
# UI
# ---------------------------------
st.set_page_config(page_title="GT7 SVG Vectorizer", page_icon="🏁", layout="wide")
st.title("🏁 GT7 SVG Vectorizer")
st.caption("15KB前提を外し、元画像をできるだけそのままトレースする方向にした版です。")

with st.expander("使い方", expanded=False):
    st.markdown(
        """
1. 画像をアップロード  
2. 品質を選ぶ（迷ったら **元画像優先 / 高品質**）  
3. **SVGへ変換** を押す  
4. プレビューを見て、必要なら色数や簡略化を調整  
5. SVGを保存
"""
    )

uploaded = st.file_uploader("画像を選択", type=["png", "jpg", "jpeg", "webp"])

with st.sidebar:
    st.header("設定")

    preset_name = st.selectbox(
        "品質プリセット",
        list(PRESETS.keys()) + ["手動設定"],
        index=1,
    )
    base = PRESETS.get(preset_name, PRESETS["元画像優先 / 高品質"]).copy()

    size_choice = st.selectbox(
        "目標サイズ",
        ["2MB 絶対上限", "1MB", "500KB", "250KB", "カスタムKB"],
        index=0,
    )
    custom_kb = st.number_input("カスタムKB", min_value=1, max_value=2048, value=1024)
    auto_opt = st.toggle("サイズ超過時だけ自動軽量化", True)
    enhance = st.toggle("低画質画像を軽く補正", True)
    white_bg = st.toggle("透明部分を白背景にする", False)

    st.divider()
    st.subheader("詳細調整")

    base["colors"] = st.slider("色数", 2, 192, int(base["colors"]))
    base["side"] = st.slider("処理サイズ（長辺px）", 256, 2200, int(base["side"]), step=16)
    base["simp"] = st.slider("パス簡略化（小さいほど元絵優先）", 0.05, 6.0, float(base["simp"]), step=0.05)
    base["area"] = st.slider("小さい形状の削除（小さいほど元絵優先）", 1, 40, int(base["area"]))
    base["smooth"] = st.slider("角の平滑化", 0, 2, int(base["smooth"]))
    base["alpha"] = st.slider("透明判定", 0, 255, 16)
    base["dec"] = st.slider("座標の小数桁", 0, 2, int(base["dec"]))

    base["target"] = target_from_choice(size_choice, custom_kb)
    base["auto"] = auto_opt
    base["enhance"] = enhance
    base["white_bg"] = white_bg

    st.divider()
    st.markdown(
        """
**おすすめ設定（元画像重視）**
- 品質: **元画像優先 / 高品質**
- 色数: **64〜128**
- 処理サイズ: **1200〜1800**
- パス簡略化: **0.35〜0.70**
- 小さい形状の削除: **1〜2**
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

if "result" not in st.session_state:
    st.subheader("元画像")
    st.image(paste_on_checkerboard(original_img), use_container_width=True)
    st.stop()

result = st.session_state.result

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("SVGサイズ", f"{result['size']:,} bytes")
m2.metric("目標サイズ", f"{base['target']:,} bytes")
m3.metric("色数", str(result["colors"]))
m4.metric("path数", str(result["paths"]))
m5.metric("変換時間", f"{st.session_state.elapsed:.1f} 秒")

if result["size"] <= base["target"]:
    st.success("目標サイズ内です。")
else:
    st.warning("目標サイズを超えています。色数や処理サイズを少し下げて再実行してください。")

if result["size"] > HARD_LIMIT:
    st.error("2MBの絶対上限を超えています。設定を軽くしてください。")

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