# app.py
# GT7 / Streamlit SVG Vectorizer 150KB版
# GitHub/Streamlit Cloud: このファイルを app.py として置き、requirements.txt も同じ階層に置いてください。

import base64
import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
import vtracer
from scour import scour


DEFAULT_MAX_BYTES = 150_000  # 150KBを1000byte換算で厳守。153600ではなく安全側。


# 今回の成功設定に近い順。上から高品質寄り、下へ行くほど容量優先。
BASE_CONFIGS = [
    ("HQ fs5 ld14 len4 cp5 polygon", dict(colormode="color", hierarchical="stacked", mode="polygon", filter_speckle=5, color_precision=5, layer_difference=14, corner_threshold=60, length_threshold=4, max_iterations=10, splice_threshold=45, path_precision=1)),
    ("HQ fs5 ld15 len4 cp5 polygon", dict(colormode="color", hierarchical="stacked", mode="polygon", filter_speckle=5, color_precision=5, layer_difference=15, corner_threshold=60, length_threshold=4, max_iterations=10, splice_threshold=45, path_precision=1)),
    ("HQ fs5 ld16 len4 cp5 polygon", dict(colormode="color", hierarchical="stacked", mode="polygon", filter_speckle=5, color_precision=5, layer_difference=16, corner_threshold=60, length_threshold=4, max_iterations=10, splice_threshold=45, path_precision=1)),
    ("Safe fs5 ld18 len4 cp5 polygon", dict(colormode="color", hierarchical="stacked", mode="polygon", filter_speckle=5, color_precision=5, layer_difference=18, corner_threshold=60, length_threshold=4, max_iterations=10, splice_threshold=45, path_precision=1)),
    ("Safe fs5 ld20 len4 cp5 polygon", dict(colormode="color", hierarchical="stacked", mode="polygon", filter_speckle=5, color_precision=5, layer_difference=20, corner_threshold=60, length_threshold=4, max_iterations=10, splice_threshold=45, path_precision=1)),
    ("Compact fs6 ld18 len5 cp5 polygon", dict(colormode="color", hierarchical="stacked", mode="polygon", filter_speckle=6, color_precision=5, layer_difference=18, corner_threshold=60, length_threshold=5, max_iterations=10, splice_threshold=45, path_precision=1)),
    ("Compact fs6 ld22 len5 cp5 polygon", dict(colormode="color", hierarchical="stacked", mode="polygon", filter_speckle=6, color_precision=5, layer_difference=22, corner_threshold=60, length_threshold=5, max_iterations=10, splice_threshold=45, path_precision=1)),
    ("Small fs8 ld26 len6 cp4 polygon", dict(colormode="color", hierarchical="stacked", mode="polygon", filter_speckle=8, color_precision=4, layer_difference=26, corner_threshold=60, length_threshold=6, max_iterations=8, splice_threshold=45, path_precision=1)),
    ("Tiny fs10 ld32 len7 cp4 polygon", dict(colormode="color", hierarchical="stacked", mode="polygon", filter_speckle=10, color_precision=4, layer_difference=32, corner_threshold=60, length_threshold=7, max_iterations=8, splice_threshold=45, path_precision=1)),
    ("Emergency fs14 ld42 len9 cp3 polygon", dict(colormode="color", hierarchical="stacked", mode="polygon", filter_speckle=14, color_precision=3, layer_difference=42, corner_threshold=60, length_threshold=9, max_iterations=6, splice_threshold=45, path_precision=1)),
]


def load_and_resize(uploaded_file, max_side: int, keep_transparency: bool = True) -> Image.Image:
    """画像を読み込み、長辺をmax_sideに制限。透明PNGはRGBA維持。"""
    img = Image.open(uploaded_file)
    if keep_transparency:
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return img


def scour_minify(svg_text: str) -> str:
    """ScourでSVGを最適化。CLI不要のPython API版。"""
    opts = scour.parse_args([
        "--enable-viewboxing",
        "--enable-id-stripping",
        "--enable-comment-stripping",
        "--shorten-ids",
        "--indent=none",
        "--strip-xml-prolog",
        "--remove-metadata",
        "--no-line-breaks",
    ])
    return scour.scourString(svg_text, opts)


def parse_attrs(attr_text: str) -> dict:
    return {m.group(1): m.group(2) for m in re.finditer(r'([A-Za-z_:][-A-Za-z0-9_:.]*)="([^"]*)"', attr_text)}


def merge_adjacent_paths(svg_text: str) -> str:
    """
    連続していて、d以外の属性が完全一致するpathだけ結合。
    見た目が変わる可能性のある無理な結合はしない。
    """
    if "<path" not in svg_text:
        return svg_text

    path_re = re.compile(r"<path\b([^>]*)/>")
    matches = list(path_re.finditer(svg_text))
    if not matches:
        return svg_text

    header = svg_text[:matches[0].start()]
    footer = svg_text[matches[-1].end():]
    chunks = []
    last_key = None
    last_d = ""

    preferred_order = ["transform", "fill", "fill-rule", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "opacity"]

    def flush():
        nonlocal last_key, last_d
        if last_key is None:
            return
        attrs = "".join(f' {k}="{v}"' for k, v in last_key)
        chunks.append(f'<path{attrs} d="{last_d}"/>')
        last_key = None
        last_d = ""

    for m in matches:
        attrs = parse_attrs(m.group(1))
        d = attrs.pop("d", "")
        ordered = []
        for k in preferred_order:
            if k in attrs:
                ordered.append((k, attrs.pop(k)))
        ordered.extend(sorted(attrs.items()))
        key = tuple(ordered)
        if key == last_key:
            last_d += d
        else:
            flush()
            last_key = key
            last_d = d
    flush()
    return header + "".join(chunks) + footer


def hard_minify(svg_text: str) -> str:
    """最後の安全なテキスト圧縮。"""
    s = svg_text.replace("\ufeff", "")
    s = re.sub(r"<\?xml[^>]*>", "", s)
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r">\s+<", "><", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = s.replace(" />", "/>")
    return s.strip()


def vectorize_once(img: Image.Image, cfg: dict) -> str:
    """1設定でvtracer変換し、SVG文字列を返す。"""
    with tempfile.TemporaryDirectory() as td:
        png_path = os.path.join(td, "input.png")
        raw_svg_path = os.path.join(td, "raw.svg")
        img.save(png_path)
        vtracer.convert_image_to_svg_py(png_path, raw_svg_path, **cfg)
        svg = Path(raw_svg_path).read_text(encoding="utf-8")
    svg = scour_minify(svg)
    svg = merge_adjacent_paths(svg)
    svg = hard_minify(svg)
    return svg


def svg_size(svg_text: str) -> int:
    return len(svg_text.encode("utf-8"))


def auto_vectorize(uploaded_file, max_bytes: int, start_max_side: int, keep_transparency: bool):
    """
    150KBを超えない候補だけを採用。
    まず元サイズに近いまま複数設定を試し、無理なら段階的に縮小して再試行。
    """
    # 画像縮小を最後の手段にする。元サイズに近いほど高品質。
    side_candidates = []
    for s in [start_max_side, 1188, 1080, 960, 860, 760, 700, 640, 560, 480]:
        s = int(s)
        if s > 0 and s not in side_candidates:
            side_candidates.append(s)

    logs = []
    best_under = None
    smallest_over = None

    for side in side_candidates:
        img = load_and_resize(uploaded_file, side, keep_transparency)
        for name, cfg in BASE_CONFIGS:
            try:
                svg = vectorize_once(img, cfg)
                size = svg_size(svg)
                path_count = svg.count("<path")
                logs.append({"side": side, "setting": name, "bytes": size, "paths": path_count, "ok": size <= max_bytes})
                if size <= max_bytes:
                    # 最初のOKは品質優先順なので採用。ただし一応保持。
                    best_under = (svg, size, side, name, path_count, logs)
                    return best_under
                if smallest_over is None or size < smallest_over[1]:
                    smallest_over = (svg, size, side, name, path_count, logs)
            except Exception as e:
                logs.append({"side": side, "setting": name, "bytes": None, "paths": None, "ok": False, "error": str(e)})

    # ここまで来ることは少ないが、絶対に超えないため非常用にさらに小さくして返す。
    for side in [420, 360, 300]:
        img = load_and_resize(uploaded_file, side, keep_transparency)
        cfg = dict(colormode="color", hierarchical="stacked", mode="polygon", filter_speckle=18, color_precision=3, layer_difference=55, corner_threshold=70, length_threshold=11, max_iterations=5, splice_threshold=45, path_precision=1)
        svg = vectorize_once(img, cfg)
        size = svg_size(svg)
        path_count = svg.count("<path")
        logs.append({"side": side, "setting": "Emergency ultra compact", "bytes": size, "paths": path_count, "ok": size <= max_bytes})
        if size <= max_bytes:
            return (svg, size, side, "Emergency ultra compact", path_count, logs)

    raise RuntimeError("指定容量以下のSVGを生成できませんでした。上限を上げるか、入力画像を小さくしてください。")


def svg_preview(svg_text: str, height: int = 520):
    b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    html = f"""
    <div style="background:#111;padding:12px;border-radius:10px;overflow:auto;text-align:center;">
      <img src="data:image/svg+xml;base64,{b64}" style="max-width:100%;height:{height}px;object-fit:contain;" />
    </div>
    """
    st.components.v1.html(html, height=height + 40, scrolling=True)


st.set_page_config(page_title="GT7 SVG 150KB Vectorizer", layout="wide")
st.title("GT7 SVG 150KB Vectorizer")
st.caption("画像をpathベースSVGへ変換し、指定容量を超えない候補だけを出力します。埋め込み画像は使いません。")

with st.sidebar:
    st.header("設定")
    max_kb = st.number_input("最大容量 KB（1000byte換算）", min_value=5, max_value=5000, value=150, step=5)
    max_bytes = int(max_kb * 1000)
    start_max_side = st.slider("変換開始サイズ：長辺px", min_value=300, max_value=1800, value=1188, step=20)
    keep_transparency = st.checkbox("透明背景を維持", value=True)
    st.info("品質優先で複数設定を試し、容量オーバーなら自動で作り方を軽くします。")

uploaded = st.file_uploader("PNG / JPG / WEBP をアップロード", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("元画像")
        st.image(uploaded, use_container_width=True)

    if st.button("SVG変換を開始", type="primary"):
        with st.spinner("変換中です。複数設定を試すため少し時間がかかる場合があります。"):
            # file_uploaderは読み取り位置が進むので、各試行の前にseekできるBytesIOへ寄せる。
            from io import BytesIO
            data = BytesIO(uploaded.getvalue())
            svg, size, side, setting_name, path_count, logs = auto_vectorize(data, max_bytes, start_max_side, keep_transparency)

        st.success(f"完成: {size:,} bytes / 上限 {max_bytes:,} bytes / path {path_count:,} / 長辺 {side}px / {setting_name}")
        with col2:
            st.subheader("SVGプレビュー")
            svg_preview(svg)

        st.download_button(
            "SVGをダウンロード",
            data=svg.encode("utf-8"),
            file_name="gt7_vector_150kb.svg",
            mime="image/svg+xml",
        )

        st.subheader("試行ログ")
        st.dataframe(logs, use_container_width=True)

        with st.expander("SVGコードを表示"):
            st.code(svg[:20000] + ("\n...省略..." if len(svg) > 20000 else ""), language="xml")
else:
    st.warning("画像をアップロードしてください。")
