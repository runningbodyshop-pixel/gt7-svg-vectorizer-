import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import vtracer
from PIL import Image, ImageFilter, ImageOps

APP_TITLE = "スマホ用 VTracer SVG化"


# ---------- utilities ----------

def pil_to_cv_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def cv_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def rgba_from_rgb_alpha(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    ratio = max_side / max(img.size)
    new_size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def svg_preview_html(svg_text: str, height: int = 700) -> str:
    return f"""
    <div style=\"background:#111;padding:10px;border-radius:12px;\">
      <div style=\"background: repeating-conic-gradient(#1a1a1a 0% 25%, #222 0% 50%) 50% / 24px 24px; min-height:{height}px; display:flex; align-items:center; justify-content:center; overflow:auto; border-radius:10px;\">
        {svg_text}
      </div>
    </div>
    """


# ---------- background removal ----------

def make_barrier_mask(rgb: np.ndarray, barrier_radius: int) -> np.ndarray:
    mx = rgb.max(axis=2).astype(np.int16)
    mn = rgb.min(axis=2).astype(np.int16)
    chroma = mx - mn
    strong = (((chroma > 20) & (mx < 252)) | (mx < 120) | (mn < 90)).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(strong, 8)
    clean = np.zeros_like(strong)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] > 8:
            clean[labels == i] = 1
    if barrier_radius <= 0:
        return clean.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (barrier_radius * 2 + 1, barrier_radius * 2 + 1))
    barrier = cv2.dilate(clean * 255, k, iterations=1) > 0
    return barrier


def remove_light_border_background(
    img: Image.Image,
    neutral_max_chroma: int = 28,
    light_min: int = 145,
    force_white_min: int = 245,
    barrier_radius: int = 4,
    alpha_blur: float = 0.35,
    crop_padding: int = 12,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Remove border-connected light/neutral background while preserving white clothing."""
    rgb = pil_to_cv_rgb(img)
    h, w = rgb.shape[:2]

    mx = rgb.max(axis=2).astype(np.int16)
    mn = rgb.min(axis=2).astype(np.int16)
    chroma = mx - mn

    eligible = (((chroma <= neutral_max_chroma) & (mx >= light_min)) | (mx >= force_white_min))
    barrier = make_barrier_mask(rgb, barrier_radius)
    passable = eligible & (~barrier)

    bg = np.zeros((h, w), dtype=np.uint8)
    q = []
    for x in range(w):
        if passable[0, x]:
            bg[0, x] = 1
            q.append((x, 0))
        if passable[h - 1, x]:
            bg[h - 1, x] = 1
            q.append((x, h - 1))
    for y in range(h):
        if passable[y, 0]:
            bg[y, 0] = 1
            q.append((0, y))
        if passable[y, w - 1]:
            bg[y, w - 1] = 1
            q.append((w - 1, y))

    head = 0
    while head < len(q):
        x, y = q[head]
        head += 1
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and passable[ny, nx] and bg[ny, nx] == 0:
                bg[ny, nx] = 1
                q.append((nx, ny))

    fg_mask = (bg == 0).astype(np.uint8) * 255
    alpha = fg_mask.copy()
    if alpha_blur > 0:
        alpha = np.array(Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(alpha_blur)))

    rgba = rgba_from_rgb_alpha(rgb, alpha)
    bbox = Image.fromarray(alpha, mode="L").getbbox()
    if bbox:
        left = max(bbox[0] - crop_padding, 0)
        top = max(bbox[1] - crop_padding, 0)
        right = min(bbox[2] + crop_padding, w)
        bottom = min(bbox[3] + crop_padding, h)
        rgba = rgba.crop((left, top, right, bottom))
        fg_mask = fg_mask[top:bottom, left:right]
        alpha = alpha[top:bottom, left:right]
    else:
        fg_mask = fg_mask.copy()
        alpha = alpha.copy()

    return rgba, fg_mask, alpha


# ---------- halo cleanup ----------

def choke_alpha(alpha: np.ndarray, shrink_px: int, hard_threshold: int) -> np.ndarray:
    a = alpha.copy()
    if hard_threshold > 0:
        a = np.where(a >= hard_threshold, 255, 0).astype(np.uint8)
    if shrink_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink_px * 2 + 1, shrink_px * 2 + 1))
        a = cv2.erode(a, k, iterations=1)
    return a


def bleed_subject_colors(rgb: np.ndarray, alpha: np.ndarray, bleed_px: int) -> np.ndarray:
    if bleed_px <= 0:
        return rgb.copy()

    result = rgb.copy()
    filled = alpha > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(bleed_px):
        dil_r = cv2.dilate(result[:, :, 0], kernel, iterations=1)
        dil_g = cv2.dilate(result[:, :, 1], kernel, iterations=1)
        dil_b = cv2.dilate(result[:, :, 2], kernel, iterations=1)
        neighbor = cv2.dilate(filled.astype(np.uint8), kernel, iterations=1).astype(bool)
        ring = neighbor & (~filled)
        result[:, :, 0][ring] = dil_r[ring]
        result[:, :, 1][ring] = dil_g[ring]
        result[:, :, 2][ring] = dil_b[ring]
        filled[ring] = True
    return result


def cleanup_white_halo(
    rgba: Image.Image,
    edge_shrink_px: int,
    alpha_threshold: int,
    color_bleed_px: int,
    soft_edge_blur: float,
) -> Image.Image:
    arr = np.array(rgba.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    alpha = choke_alpha(alpha, edge_shrink_px, alpha_threshold)
    rgb = bleed_subject_colors(rgb, alpha, color_bleed_px)

    if soft_edge_blur > 0:
        alpha = np.array(Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(soft_edge_blur)))

    return rgba_from_rgb_alpha(rgb, alpha)


# ---------- vtracer ----------

def preset_params(name: str) -> dict:
    if name == "爆速":
        return dict(
            filter_speckle=5,
            color_precision=6,
            layer_difference=18,
            corner_threshold=60,
            length_threshold=5.5,
            max_iterations=10,
            splice_threshold=45,
            path_precision=2,
        )
    if name == "高精細":
        return dict(
            filter_speckle=2,
            color_precision=8,
            layer_difference=8,
            corner_threshold=60,
            length_threshold=3.0,
            max_iterations=12,
            splice_threshold=45,
            path_precision=3,
        )
    return dict(
        filter_speckle=3,
        color_precision=7,
        layer_difference=12,
        corner_threshold=58,
        length_threshold=4.0,
        max_iterations=11,
        splice_threshold=45,
        path_precision=2,
    )


def apply_quality_sliders(params: dict, detail: int, smoothness: int, lightness: int) -> dict:
    # detail: + => more detail
    params["color_precision"] = int(np.clip(params["color_precision"] + round(detail * 0.8), 4, 10))
    params["layer_difference"] = int(np.clip(params["layer_difference"] - detail * 2, 4, 24))
    params["length_threshold"] = float(np.clip(params["length_threshold"] - detail * 0.35, 1.5, 8.0))

    # smoothness: + => smoother / less jagged
    params["filter_speckle"] = int(np.clip(params["filter_speckle"] + max(0, smoothness), 1, 12))
    params["max_iterations"] = int(np.clip(params["max_iterations"] + max(0, smoothness), 8, 20))
    params["corner_threshold"] = int(np.clip(params["corner_threshold"] - smoothness * 3, 30, 90))

    # lightness: + => lighter/smaller SVG
    params["path_precision"] = int(np.clip(params["path_precision"] - max(0, lightness // 2), 1, 8))
    params["layer_difference"] = int(np.clip(params["layer_difference"] + max(0, lightness), 4, 28))
    params["length_threshold"] = float(np.clip(params["length_threshold"] + max(0, lightness) * 0.4, 1.5, 10.0))
    params["filter_speckle"] = int(np.clip(params["filter_speckle"] + max(0, lightness // 2), 1, 12))

    return params


def run_vtracer(rgba_img: Image.Image, params: dict) -> tuple[str, bytes]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        in_png = tmp_dir / "input.png"
        out_svg = tmp_dir / "output.svg"
        rgba_img.save(in_png)

        vtracer.convert_image_to_svg_py(
            str(in_png),
            str(out_svg),
            colormode="color",
            hierarchical="stacked",
            mode="spline",
            filter_speckle=int(params["filter_speckle"]),
            color_precision=int(params["color_precision"]),
            layer_difference=int(params["layer_difference"]),
            corner_threshold=int(params["corner_threshold"]),
            length_threshold=float(params["length_threshold"]),
            max_iterations=int(params["max_iterations"]),
            splice_threshold=int(params["splice_threshold"]),
            path_precision=int(params["path_precision"]),
        )
        svg_bytes = out_svg.read_bytes()
        return out_svg.read_text(encoding="utf-8", errors="ignore"), svg_bytes


# ---------- app ----------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("PNG/JPG/WebP → 背景処理 → 白フチ補正 → VTracerでSVG化")

    upload = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg", "webp"])
    if not upload:
        st.info("画像を入れると、スマホ向けの調整UIでSVG化できます。")
        return

    src = Image.open(upload)
    src = ImageOps.exif_transpose(src)
    if src.mode not in ("RGB", "RGBA"):
        src = src.convert("RGBA")

    with st.sidebar:
        st.header("基本設定")
        max_side = st.slider("最大辺", 800, 3000, 1800, 100)
        preset = st.selectbox("プリセット", ["爆速", "標準", "高精細"], index=1)
        detail = st.slider("ディテール", -3, 3, 0)
        smoothness = st.slider("なめらかさ", -3, 3, 0)
        lightness = st.slider("軽量化", -3, 3, 0)

        st.header("背景処理")
        enable_bg = st.checkbox("外側の白/薄灰背景を透明化", value=True)
        neutral_max_chroma = st.slider("背景の無彩色判定", 8, 50, 28)
        light_min = st.slider("背景の明るさ下限", 100, 240, 145)
        force_white_min = st.slider("強制白判定", 220, 255, 245)
        barrier_radius = st.slider("白服保護", 0, 8, 4)

        st.header("白フチ / 輪郭補正")
        edge_shrink_px = st.slider("白フチ削り（輪郭を少し内側へ）", 0, 4, 1)
        alpha_threshold = st.slider("半透明エッジのしきい値", 0, 255, 180)
        color_bleed_px = st.slider("輪郭の色にじみ補正", 0, 4, 2)
        soft_edge_blur = st.slider("輪郭のやわらかさ", 0.0, 1.2, 0.2, 0.05)

    src = resize_max_side(src, max_side)

    if enable_bg:
        processed_rgba, _, _ = remove_light_border_background(
            src,
            neutral_max_chroma=neutral_max_chroma,
            light_min=light_min,
            force_white_min=force_white_min,
            barrier_radius=barrier_radius,
        )
    else:
        if src.mode != "RGBA":
            src = src.convert("RGBA")
        processed_rgba = src

    processed_rgba = cleanup_white_halo(
        processed_rgba,
        edge_shrink_px=edge_shrink_px,
        alpha_threshold=alpha_threshold,
        color_bleed_px=color_bleed_px,
        soft_edge_blur=soft_edge_blur,
    )

    params = preset_params(preset)
    params = apply_quality_sliders(params, detail, smoothness, lightness)

    with st.expander("現在のVTracerパラメータ", expanded=False):
        st.json(params)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("入力画像")
        st.image(src, use_container_width=True)
    with col2:
        st.subheader("前処理後PNG")
        st.image(processed_rgba, use_container_width=True)
        st.download_button(
            "前処理PNGを保存",
            data=png_bytes(processed_rgba),
            file_name="vtracer_preprocessed.png",
            mime="image/png",
            use_container_width=True,
        )

    with st.spinner("VTracerでSVG化しています..."):
        svg_text, svg_bytes = run_vtracer(processed_rgba, params)

    st.subheader("SVGプレビュー")
    st.components.v1.html(svg_preview_html(svg_text, height=780), height=820, scrolling=True)

    st.download_button(
        "SVGをダウンロード",
        data=svg_bytes,
        file_name="vtracer_output.svg",
        mime="image/svg+xml",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown(
        """
        ### おすすめ調整
        - **白い輪郭が出る** → `白フチ削り` を 1〜2、`輪郭の色にじみ補正` を 1〜3
        - **細部がガタつく** → `なめらかさ` を +1〜+2
        - **細部が消えすぎる** → `ディテール` を +1〜+2、`軽量化` を下げる
        - **白服まで消える** → `白服保護` を上げる / `背景の明るさ下限` を少し下げる
        """
    )


if __name__ == "__main__":
    main()
