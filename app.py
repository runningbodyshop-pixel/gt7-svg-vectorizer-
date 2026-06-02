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
        "overlap": 1,
    },
    "標準 / 1分目標": {
        "colors": 64,
        "side": 1100,
        "simp": 0.75,
        "area": 2,
        "dec": 1,
        "attempts": 2,
        "overlap": 1,
    },
    "高品質 / SVGOMG前提": {
        "colors": 96,
        "side": 1350,
        "simp": 0.50,
        "area": 1,
        "dec": 1,
        "attempts": 2,
        "overlap": 1,
    },
    "元画像優先 / 重い": {
        "colors": 128,
        "side": 1600,
        "simp": 0.38,
        "area": 1,
        "dec": 1,
        "attempts": 3,
        "overlap": 1,
    },
}


def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def short_hex(r: int, g: int, b: int) -> str:
    s = f"{r:02x}{g:02x}{b:02x}"
    if s[0] == s[1] and s[2] == s[3] and s[4] == s[5]:
        return f"#{s[0]}{s[2]}{s[4]}"
    return f"#{s}"


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

    palette_raw = q.getpalette()[:768]
    used = sorted(int(x) for x in np.unique(labels) if int(x) >= 0)

    remapped = np.full_like(labels, -1)
    palette: list[tuple[int, int, int]] = []

    for new_idx, old_idx in enumerate(used):
        remapped[labels == old_idx] = new_idx
        palette.append(tuple(palette_raw[old_idx * 3 : old_idx * 3 + 3]))

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
    return remapped, palette, preview, visible.astype(np.uint8)


def fmt_num(x: float, decimals: int) -> str:
    if decimals <= 0:
        return str(int(round(x)))
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".").replace("-0", "0")


def contour_to_path_cv(
    contour: np.ndarray,
    simplify: float,
    min_area: int,
    decimals: int,
) -> Optional[str]:
    if contour is None or len(contour) < 3:
        return None

    area = abs(cv2.contourArea(contour))
    if area < min_area:
        return None

    approx = cv2.approxPolyDP(
        contour,
        epsilon=max(0.05, float(simplify)),
        closed=True,
    )

    if approx is None or len(approx) < 3:
        return None

    pts = approx.reshape(-1, 2).astype(np.float32)

    if len(pts) < 3:
        return None

    cmds = [f"M{fmt_num(pts[0, 0], decimals)} {fmt_num(pts[0, 1], decimals)}"]
    for x, y in pts[1:]:
        cmds.append(f"L{fmt_num(x, decimals)} {fmt_num(y, decimals)}")
    cmds.append("Z")
    return "".join(cmds)


def connected_clean(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = mask.astype(np.uint8)

    if min_area <= 1:
        return binary * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cleaned = np.zeros_like(binary, dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def mask_to_paths_external(
    binary: np.ndarray,
    simplify: float,
    min_area: int,
    decimals: int,
):
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    paths = []
    for contour in contours:
        d = contour_to_path_cv(
            contour=contour,
            simplify=simplify,
            min_area=min_area,
            decimals=decimals,
        )
        if d:
            paths.append(d)

    return paths


def dominant_color(palette: list[tuple[int, int, int]], labels: np.ndarray) -> tuple[int, int, int]:
    best_i = 0
    best_count = -1
    for i in range(len(palette)):
        count = int((labels == i).sum())
        if count > best_count:
            best_count = count
            best_i = i
    if not palette:
        return (0, 0, 0)
    return palette[best_i]


def build_svg_layered(
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    visible_mask: np.ndarray,
    width: int,
    height: int,
    simplify: float,
    min_area: int,
    decimals: int,
    white_bg: bool,
    use_underpaint: bool,
    overlap_px: int,
):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]

    path_count = 0
    contour_count = 0

    if white_bg:
        parts.append(f'<path fill="#fff" d="M0 0H{width}V{height}H0Z"/>')
        path_count += 1

    # 透明の隙間が出ないように、最初に見えている範囲だけを下塗りする
    if use_underpaint and visible_mask is not None and visible_mask.max() > 0:
        base_binary = connected_clean(visible_mask > 0, min_area=max(1, min_area))
        paths = mask_to_paths_external(
            base_binary,
            simplify=max(0.2, simplify * 1.2),
            min_area=max(1, min_area),
            decimals=decimals,
        )

        if paths:
            r, g, b = dominant_color(palette, labels)
            parts.append(
                f'<path fill="{short_hex(r, g, b)}" d="{html.escape("".join(paths), quote=True)}"/>'
            )
            path_count += 1
            contour_count += len(paths)

    order = sorted(
        [(int((labels == i).sum()), i) for i in range(len(palette))],
        reverse=True,
    )

    kernel = None
    if overlap_px > 0:
        k = overlap_px * 2 + 1
        kernel = np.ones((k, k), np.uint8)

    for _, color_index in order:
        mask = labels == color_index

        if int(mask.sum()) < min_area:
            continue

        binary = connected_clean(mask, min_area=max(1, min_area))

        # 色同士の境界に透明な隙間が出ないよう、ほんの少しだけ重ねる
        if kernel is not None:
            binary = cv2.dilate(binary, kernel, iterations=1)

            # 背景の透明部分へ大きくはみ出さないよう、元の見えている範囲も少しだけ拡張して制限
            visible_binary = (visible_mask > 0).astype(np.uint8) * 255
            visible_binary = cv2.dilate(visible_binary, kernel, iterations=1)
            binary = cv2.bitwise_and(binary, visible_binary)

        if binary.max() == 0:
            continue

        paths = mask_to_paths_external(
            binary=binary,
            simplify=simplify,
            min_area=max(1, min_area),
            decimals=decimals,
        )

        if not paths:
            continue

        r, g, b = palette[color_index]
        parts.append(
            f'<path fill="{short_hex(r, g, b)}" d="{html.escape("".join(paths), quote=True)}"/>'
        )
        path_count += 1
        contour_count += len(paths)

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

    a = paste_on_checkerboard(fit_for_preview(original))
    b = paste_on_checkerboard(fit_for_preview(vector_preview))

    margin = 16
    gap = 24
    title_h = 36

    w = a.width + b.width + gap + margin * 2
    h = max(a.height, b.height) + title_h + margin * 2

    canvas = Image.new("RGBA", (w, h), (245, 245, 245, 255))
    dr = ImageDraw.Draw(canvas)

    dr.text((margin, 10), "Original", fill=(0, 0, 0))
    dr.text((margin + a.width + gap, 10), "Quantized reference", fill=(0, 0, 0))

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
    use_underpaint: bool,
    overlap_px: int,
):
    work = fit_image(img, side=side, enhance=enhance)

    labels, palette, quant_preview, visible_mask = quantize_image(
        work,
        n_colors=colors,
        alpha_threshold=alpha_threshold,
        white_bg=white_bg,
    )

    svg, path_count, contour_count = build_svg_layered(
        labels=labels,
        palette=palette,
        visible_mask=visible_mask,
        width=work.width,
        height=work.height,
        simplify=simplify,
        min_area=min_area,
        decimals=decimals,
        white_bg=white_bg,
        use_underpaint=use_underpaint,
        overlap_px=overlap_px,
    )

    return {
        "svg": svg,
        "size": len(svg.encode("utf-8")),
        "quant_preview": quant_preview,
        "compare": make_compare_image(work, quant_preview),
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
            use_underpaint=cfg["underpaint"],
            overlap_px=cfg["overlap"],
        )

        result["attempts"] = i + 1

        if best is None:
            best = result
        else:
            if result["size"] <= target and best["size"] <= target:
                # SVGOMG前提なので、目標内なら情報量が多い方を優先
                if result["size"] > best["size"]:
                    best = result
            elif result["size"] <= target < best["size"]:
                best = result
            elif result["size"] > target and best["size"] > target:
                if result["size"] < best["size"]:
                    best = result

        if result["size"] <= target:
            return result

        colors = max(8, int(colors * 0.86))
        side = max(600, int(side * 0.92))
        simplify = min(4.0, simplify * 1.18)
        min_area = min(20, int(min_area * 1.25 + 1))
        decimals = 0

    return best


def data_url_svg(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return "data:image/svg+xml;base64," + encoded


def target_from_choice(choice: str, custom_kb: int) -> int:
    mapping = {
        "2MB 絶対上限": 2 * 1024 * 1024,
        "1.8MB SVGOMG前提": int(1.8 * 1024 * 1024),
        "1.5MB SVGOMG前提": int(1.5 * 1024 * 1024),
        "1MB": 1024 * 1024,
        "500KB": 500 * 1024,
    }
    if choice == "カスタムKB":
        return max(1, min(2048, int(custom_kb))) * 1024
    return mapping.get(choice, 2 * 1024 * 1024)


st.set_page_config(
    page_title="GT7 SVG Vectorizer SVGOMG",
    page_icon="🏁",
    layout="wide",
)

st.title("🏁 GT7 SVG Vectorizer")
st.caption("実際のSVG表示を基準にし、SVGOMGで軽量化しやすい形にした版です。")

with st.expander("重要：プレビューについて", expanded=True):
    st.markdown(
        """
この版では **SVGプレビューが実際のSVGデータそのもの** です。  
下の「比較PNG」は参考用で、実際のSVG描画とは完全一致しません。

前の版でチェック柄がキャラの中に強く見えていたのは、色ごとのパスの境界に透明な隙間ができていたためです。  
この版では **下塗りシルエット** と **境界の少し重ね描き** で、その隙間を減らしています。
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
        index=2,
    )

    base = PRESETS.get(preset_name, PRESETS["高品質 / SVGOMG前提"]).copy()

    size_choice = st.selectbox(
        "目標サイズ",
        ["2MB 絶対上限", "1.8MB SVGOMG前提", "1.5MB SVGOMG前提", "1MB", "500KB", "カスタムKB"],
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
    st.subheader("隙間対策")

    underpaint = st.toggle("下塗りシルエットを入れる", True)
    base["overlap"] = st.slider(
        "色境界の重ね描きpx",
        0,
        2,
        int(base["overlap"]),
    )

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
        "パス簡略化（小さいほど元絵優先・大きいほど軽い）",
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
    base["underpaint"] = underpaint

    st.divider()
    st.markdown(
        """
**おすすめ**
- 高品質 / SVGOMG前提
- 色数: 80〜120
- 処理サイズ: 1200〜1500
- パス簡略化: 0.45〜0.65
- 下塗りシルエット: ON
- 重ね描きpx: 1
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
used = st.session_state.used_settings

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("SVGサイズ", f"{result['size']:,} bytes")
m2.metric("目標サイズ", f"{used['target']:,} bytes")
m3.metric("色数", str(result["colors"]))
m4.metric("path数", str(result["paths"]))
m5.metric("輪郭数", str(result["contours"]))
m6.metric("変換時間", f"{st.session_state.elapsed:.1f} 秒")

if result["size"] <= used["target"]:
    st.success("目標サイズ内です。SVGOMGに通すと、さらに軽くなる可能性があります。")
else:
    st.warning("目標サイズを超えています。SVGOMG後に下がる可能性はありますが、色数・処理サイズ・簡略化も調整してください。")

if result["size"] > HARD_LIMIT:
    st.error("2MBの絶対上限を超えています。SVGOMG前に一度軽くしてください。")

if st.session_state.elapsed > 60:
    st.warning("1分を超えました。処理サイズを下げる、色数を下げる、試行回数を1〜2にすると速くなります。")

svg_url = data_url_svg(result["svg"])

st.subheader("実データSVGプレビュー")
components.html(
    f"""
    <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap;">
      <div style="flex:1;min-width:260px;">
        <div style="font-weight:bold;margin-bottom:8px;">Original</div>
        <div style="
          border:1px solid #ddd;
          border-radius:12px;
          padding:12px;
          background:#eee;
          text-align:center;
        ">
          <img src="data:image/png;base64,{base64.b64encode(png_bytes(paste_on_checkerboard(original_img))).decode("ascii")}" style="max-width:100%;height:auto;">
        </div>
      </div>
      <div style="flex:1;min-width:260px;">
        <div style="font-weight:bold;margin-bottom:8px;">Actual SVG</div>
        <div style="
          border:1px solid #ddd;
          border-radius:12px;
          padding:12px;
          background:#eee;
          text-align:center;
        ">
          <img src="{svg_url}" style="max-width:100%;height:auto;">
        </div>
      </div>
    </div>
    """,
    height=620,
    scrolling=True,
)

with st.expander("参考用：量子化プレビューPNG"):
    st.markdown("これは実際のSVG描画ではなく、SVG化する前の色分け確認用です。")
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
        "参考PNGを保存",
        data=result["compare"],
        file_name=f"{base_name}_reference.png",
        mime="image/png",
        use_container_width=True,
    )

with st.expander("SVGコードを表示 / コピー"):
    st.code(result["svg"], language="xml")

with st.expander("SVGOMGに通す時のおすすめ"):
    st.markdown(
        """
SVGOMGでは、まず以下の方向がおすすめです。

- **Prettify markup**：OFF
- **Remove metadata**：ON
- **Remove comments**：ON
- **Collapse useless groups**：ON
- **Convert colors**：ON
- **Round/rewrite paths**：ON
- **Merge paths**：ONでもOK
- **Remove viewBox**：OFF推奨

このアプリは、SVGOMGに通しやすいように、基本的に `path` と `fill` だけで出力します。
"""
    )