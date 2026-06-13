import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageOps


APP_TITLE = "スマホ用 Inkscape風 オートトレース"


# ---------- small utilities ----------

def has_binary(name: str) -> bool:
    return shutil.which(name) is not None


def read_upload_as_image(uploaded_file) -> Image.Image:
    img = Image.open(uploaded_file)
    img.load()
    return img


def flatten_alpha(img: Image.Image, bg_hex: str = "#ffffff") -> Image.Image:
    """Return RGB image. Transparent pixels are composited over bg_hex."""
    img = ImageOps.exif_transpose(img)
    if img.mode == "RGBA":
        bg_hex = bg_hex.strip().lstrip("#")
        if len(bg_hex) != 6:
            bg_hex = "ffffff"
        bg = tuple(int(bg_hex[i:i + 2], 16) for i in (0, 2, 4))
        canvas = Image.new("RGB", img.size, bg)
        canvas.paste(img, mask=img.getchannel("A"))
        return canvas
    return img.convert("RGB")


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    ratio = max_side / max(img.size)
    new_size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def clean_svg(svg: str) -> str:
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S)
    svg = re.sub(r">\s+<", "><", svg)
    svg = re.sub(r"\s{2,}", " ", svg)
    return svg.strip()


def run_cmd(cmd, timeout=60):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "変換に失敗しました。")
    return result


# ---------- potrace path creation ----------

def make_pbm_from_luma(
    img_rgb: Image.Image,
    out_pbm: Path,
    threshold: int,
    invert: bool,
    blur_radius: float,
    median_size: int,
):
    gray = img_rgb.convert("L")
    if blur_radius > 0:
        gray = gray.filter(ImageFilter.GaussianBlur(blur_radius))
    if median_size >= 3:
        # PIL median size must be odd
        if median_size % 2 == 0:
            median_size += 1
        gray = gray.filter(ImageFilter.MedianFilter(median_size))

    arr = np.asarray(gray)
    if invert:
        obj = arr > threshold
    else:
        obj = arr < threshold
    pbm_arr = np.where(obj, 0, 255).astype(np.uint8)  # potrace traces black pixels
    Image.fromarray(pbm_arr, mode="L").convert("1").save(out_pbm)


def make_pbm_from_mask(mask_bool: np.ndarray, out_pbm: Path):
    pbm_arr = np.where(mask_bool, 0, 255).astype(np.uint8)
    Image.fromarray(pbm_arr, mode="L").convert("1").save(out_pbm)


def potrace_to_svg(
    pbm_path: Path,
    svg_path: Path,
    turdsize: int,
    alphamax: float,
    opttolerance: float,
    turnpolicy: str,
    longcurve: bool,
):
    cmd = [
        "potrace",
        str(pbm_path),
        "-s",
        "-o",
        str(svg_path),
        "--turdsize",
        str(turdsize),
        "--alphamax",
        str(alphamax),
        "--opttolerance",
        str(opttolerance),
        "--turnpolicy",
        turnpolicy,
    ]
    if longcurve:
        cmd.append("--longcurve")
    run_cmd(cmd, timeout=120)


def extract_potrace_group(svg_text: str, fill_hex: str) -> str:
    """Keep potrace's transform, replace fill color, and keep path elements."""
    group_match = re.search(r"<g\b([^>]*)>(.*?)</g>", svg_text, flags=re.S)
    if not group_match:
        # Fallback: keep only paths if a different potrace build emits no group.
        paths = "".join(re.findall(r"<path\b[^>]*/>", svg_text, flags=re.S))
        return f'<g fill="{fill_hex}" stroke="none">{paths}</g>' if paths else ""

    attrs = group_match.group(1)
    inner = group_match.group(2)
    transform = ""
    transform_match = re.search(r'transform="([^"]+)"', attrs)
    if transform_match:
        transform = f' transform="{transform_match.group(1)}"'

    paths = "".join(re.findall(r"<path\b[^>]*/>", inner, flags=re.S))
    if not paths:
        return ""
    paths = re.sub(r'\sfill="[^"]*"', "", paths)
    paths = re.sub(r'\sstroke="[^"]*"', "", paths)
    return f'<g{transform} fill="{fill_hex}" stroke="none">{paths}</g>'


def trace_bw_svg(
    img_rgb: Image.Image,
    threshold: int,
    invert: bool,
    blur_radius: float,
    median_size: int,
    turdsize: int,
    alphamax: float,
    opttolerance: float,
    turnpolicy: str,
    longcurve: bool,
) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pbm = tmp / "input.pbm"
        svg = tmp / "output.svg"
        make_pbm_from_luma(img_rgb, pbm, threshold, invert, blur_radius, median_size)
        potrace_to_svg(pbm, svg, turdsize, alphamax, opttolerance, turnpolicy, longcurve)
        return clean_svg(svg.read_text(encoding="utf-8", errors="ignore"))


# ---------- color multi-scan: Inkscape-like stacked scans ----------

def quantize_image(img_rgb: Image.Image, color_count: int, dither: bool) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    q = img_rgb.quantize(
        colors=color_count,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE,
    )
    idx = np.asarray(q)
    raw_palette = q.getpalette()[: color_count * 3]
    palette = []
    for i in range(color_count):
        j = i * 3
        palette.append((raw_palette[j], raw_palette[j + 1], raw_palette[j + 2]))
    return idx, palette


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def trace_color_svg(
    img_rgb: Image.Image,
    color_count: int,
    dither: bool,
    min_region_percent: float,
    underpaint: bool,
    turdsize: int,
    alphamax: float,
    opttolerance: float,
    turnpolicy: str,
    longcurve: bool,
) -> str:
    idx, palette = quantize_image(img_rgb, color_count=color_count, dither=dither)
    h, w = idx.shape
    total = h * w
    present = []
    for color_index in np.unique(idx):
        area = int(np.sum(idx == color_index))
        if area / total * 100 >= min_region_percent:
            present.append((int(color_index), area, palette[int(color_index)]))

    # Large areas first. This hides small tracing gaps and feels closer to practical layered scans.
    present.sort(key=lambda x: x[1], reverse=True)

    groups = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for n, (color_index, area, rgb) in enumerate(present, start=1):
            mask = idx == color_index
            pbm = tmp / f"mask_{n}.pbm"
            svg = tmp / f"mask_{n}.svg"
            make_pbm_from_mask(mask, pbm)
            try:
                potrace_to_svg(pbm, svg, turdsize, alphamax, opttolerance, turnpolicy, longcurve)
            except RuntimeError:
                continue
            raw = svg.read_text(encoding="utf-8", errors="ignore")
            group = extract_potrace_group(raw, rgb_to_hex(rgb))
            if group:
                groups.append(group)

    if not groups:
        raise RuntimeError("有効なパスを作れませんでした。色数・最小面積・画像サイズを調整してください。")

    bg_rect = ""
    if underpaint and present:
        bg_rect = f'<rect width="100%" height="100%" fill="{rgb_to_hex(present[0][2])}"/>'

    svg_text = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'{bg_rect}'
        f'{"".join(groups)}'
        f'</svg>'
    )
    return clean_svg(svg_text)


# ---------- autotrace centerline ----------

def trace_centerline_autotrace(img_rgb: Image.Image, color_count: int, despeckle: int) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_png = tmp / "input.png"
        out_svg = tmp / "centerline.svg"
        img_rgb.save(input_png)
        cmd = [
            "autotrace",
            str(input_png),
            "-centerline",
            "-color-count",
            str(color_count),
            "-despeckle-level",
            str(despeckle),
            "-output-file",
            str(out_svg),
        ]
        run_cmd(cmd, timeout=180)
        return clean_svg(out_svg.read_text(encoding="utf-8", errors="ignore"))


# ---------- Streamlit UI ----------

st.set_page_config(page_title=APP_TITLE, page_icon="🖋️", layout="wide")
st.title("🖋️ スマホ用 Inkscape風 オートトレース")
st.caption("PNG/JPG/WebP → SVG。通常トレースは Potrace、線の中心線トレースは Autotrace を使います。")

with st.sidebar:
    st.header("基本設定")
    mode = st.selectbox(
        "変換モード",
        [
            "白黒トレース / Potrace",
            "カラー多重スキャン / Potrace",
            "中心線トレース / Autotrace",
        ],
    )
    max_side = st.slider("最大辺サイズ（速さ重視なら小さく）", 512, 4096, 1600, 128)
    bg_hex = st.text_input("透過PNGの背景色", "#ffffff")

    st.header("Potrace品質")
    turdsize = st.slider("ノイズ除去 turdsize", 0, 50, 2)
    alphamax = st.slider("角の丸め alphamax", 0.0, 1.5, 1.0, 0.05)
    opttolerance = st.slider("曲線最適化 opttolerance", 0.0, 2.0, 0.2, 0.05)
    turnpolicy = st.selectbox("曖昧部分 turnpolicy", ["minority", "majority", "black", "white", "left", "right", "random"])
    longcurve = st.checkbox("高精度寄り：曲線最適化を弱める --longcurve", value=False)

uploaded = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg", "webp", "bmp"])

if uploaded is None:
    st.info("スマホなら、ここにPNGをアップロードしてから変換ボタンを押してください。")
    st.stop()

try:
    original = read_upload_as_image(uploaded)
    img_rgb = flatten_alpha(original, bg_hex=bg_hex)
    img_rgb = resize_max_side(img_rgb, max_side=max_side)
except Exception as e:
    st.error(f"画像を読み込めませんでした: {e}")
    st.stop()

left, right = st.columns([1, 1])
with left:
    st.subheader("元画像 / 処理サイズ")
    st.image(img_rgb, use_container_width=True)
    st.write(f"処理サイズ: {img_rgb.width} × {img_rgb.height}px")

with right:
    st.subheader("モード別設定")

    if mode == "白黒トレース / Potrace":
        threshold = st.slider("明るさしきい値", 0, 255, 128)
        invert = st.checkbox("反転：明るい部分をトレース", value=False)
        blur_radius = st.slider("軽いぼかし", 0.0, 3.0, 0.0, 0.1)
        median_size = st.select_slider("点ノイズ除去", options=[1, 3, 5, 7], value=1)

    elif mode == "カラー多重スキャン / Potrace":
        color_count = st.slider("色数 / scans", 2, 64, 12)
        dither = st.checkbox("ディザを使う（写真向け。ロゴ/アニメはOFF推奨）", value=False)
        min_region_percent = st.slider("小さすぎる色面を捨てる %", 0.0, 5.0, 0.05, 0.05)
        underpaint = st.checkbox("最大色を下地に敷く（白い隙間対策）", value=True)

    else:
        color_count = st.slider("Autotrace色数", 2, 32, 2)
        despeckle = st.slider("Autotraceノイズ除去", 0, 20, 2)
        st.warning("このモードはサーバー側に autotrace コマンドが必要です。Streamlit Cloudだけでは入らない場合があります。")

convert = st.button("SVGに変換", type="primary", use_container_width=True)

if not convert:
    st.stop()

try:
    if mode in ["白黒トレース / Potrace", "カラー多重スキャン / Potrace"] and not has_binary("potrace"):
        st.error("potrace が見つかりません。packages.txt に `potrace` を入れてデプロイしてください。")
        st.stop()

    with st.spinner("変換中..."):
        if mode == "白黒トレース / Potrace":
            svg_text = trace_bw_svg(
                img_rgb,
                threshold=threshold,
                invert=invert,
                blur_radius=blur_radius,
                median_size=median_size,
                turdsize=turdsize,
                alphamax=alphamax,
                opttolerance=opttolerance,
                turnpolicy=turnpolicy,
                longcurve=longcurve,
            )
            filename = "inkscape_like_bw_trace.svg"

        elif mode == "カラー多重スキャン / Potrace":
            svg_text = trace_color_svg(
                img_rgb,
                color_count=color_count,
                dither=dither,
                min_region_percent=min_region_percent,
                underpaint=underpaint,
                turdsize=turdsize,
                alphamax=alphamax,
                opttolerance=opttolerance,
                turnpolicy=turnpolicy,
                longcurve=longcurve,
            )
            filename = "inkscape_like_color_multiscan.svg"

        else:
            if not has_binary("autotrace"):
                st.error("autotrace が見つかりません。中心線トレースを使うには、Docker/Render等で autotrace を入れてください。")
                st.stop()
            svg_text = trace_centerline_autotrace(img_rgb, color_count=color_count, despeckle=despeckle)
            filename = "inkscape_like_centerline_trace.svg"

    st.success("SVG変換完了")
    st.download_button(
        "SVGをダウンロード",
        data=svg_text.encode("utf-8"),
        file_name=filename,
        mime="image/svg+xml",
        use_container_width=True,
    )

    with st.expander("SVGコードを表示"):
        st.code(svg_text[:200000], language="xml")
        if len(svg_text) > 200000:
            st.info("表示は先頭のみです。ダウンロードされるSVGは完全版です。")

except subprocess.TimeoutExpired:
    st.error("処理時間が長すぎました。最大辺サイズ・色数・ノイズを下げてください。")
except Exception as e:
    st.error(f"変換に失敗しました: {e}")
