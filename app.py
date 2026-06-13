import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageFilter, ImageOps

try:
    import vtracer
    VTRACER_IMPORT_ERROR = None
except Exception as exc:  # App should open even when dependency import fails.
    vtracer = None
    VTRACER_IMPORT_ERROR = str(exc)

APP_TITLE = "スマホ用 VTracer SVG化"


# =========================================================
# Basic helpers
# =========================================================

def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    scale = max_side / max(img.size)
    size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(size, Image.Resampling.LANCZOS)


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ensure_odd_size(px: int) -> int:
    px = int(max(1, px))
    return px if px % 2 == 1 else px + 1


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    im = im.filter(ImageFilter.MaxFilter(ensure_odd_size(radius * 2 + 1)))
    return np.array(im) > 0


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    im = im.filter(ImageFilter.MinFilter(ensure_odd_size(radius * 2 + 1)))
    return np.array(im) > 0


def alpha_to_bbox(alpha: np.ndarray):
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_rgba_by_alpha(rgba: Image.Image, padding: int) -> Image.Image:
    arr = np.array(rgba.convert("RGBA"))
    bbox = alpha_to_bbox(arr[:, :, 3])
    if bbox is None:
        return rgba
    l, t, r, b = bbox
    l = max(0, l - padding)
    t = max(0, t - padding)
    r = min(rgba.width, r + padding)
    b = min(rgba.height, b + padding)
    return rgba.crop((l, t, r, b))


# =========================================================
# Background removal
# =========================================================

def flood_fill_border(passable: np.ndarray) -> np.ndarray:
    """Return mask of pixels connected to image border through passable pixels."""
    h, w = passable.shape
    out = np.zeros((h, w), dtype=bool)
    queue = []

    for x in range(w):
        if passable[0, x] and not out[0, x]:
            out[0, x] = True
            queue.append((x, 0))
        if passable[h - 1, x] and not out[h - 1, x]:
            out[h - 1, x] = True
            queue.append((x, h - 1))

    for y in range(h):
        if passable[y, 0] and not out[y, 0]:
            out[y, 0] = True
            queue.append((0, y))
        if passable[y, w - 1] and not out[y, w - 1]:
            out[y, w - 1] = True
            queue.append((w - 1, y))

    head = 0
    while head < len(queue):
        x, y = queue[head]
        head += 1
        nx = x + 1
        if nx < w and passable[y, nx] and not out[y, nx]:
            out[y, nx] = True
            queue.append((nx, y))
        nx = x - 1
        if nx >= 0 and passable[y, nx] and not out[y, nx]:
            out[y, nx] = True
            queue.append((nx, y))
        ny = y + 1
        if ny < h and passable[ny, x] and not out[ny, x]:
            out[ny, x] = True
            queue.append((x, ny))
        ny = y - 1
        if ny >= 0 and passable[ny, x] and not out[ny, x]:
            out[ny, x] = True
            queue.append((x, ny))

    return out


def remove_border_light_background(
    img: Image.Image,
    neutral_max_chroma: int,
    light_min: int,
    force_white_min: int,
    barrier_radius: int,
    alpha_blur: float,
    crop_padding: int,
) -> Image.Image:
    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    mx = rgb.max(axis=2).astype(np.int16)
    mn = rgb.min(axis=2).astype(np.int16)
    chroma = mx - mn

    # Candidate background: bright low-chroma pixels, but only if border-connected.
    eligible_bg = (((chroma <= neutral_max_chroma) & (mx >= light_min)) | (mx >= force_white_min))

    # Barrier: colored pixels, dark line art, and strong edges.
    strong_subject = (((chroma > 20) & (mx < 252)) | (mx < 120) | (mn < 90))
    barrier = dilate_mask(strong_subject, barrier_radius)

    passable = eligible_bg & (~barrier)
    bg = flood_fill_border(passable)

    alpha = np.where(bg, 0, 255).astype(np.uint8)
    if alpha_blur > 0:
        alpha = np.array(Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(alpha_blur)))

    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    out = Image.fromarray(rgba, mode="RGBA")
    return crop_rgba_by_alpha(out, crop_padding)


# =========================================================
# White halo cleanup
# =========================================================

def color_bleed_under_transparent(rgb: np.ndarray, alpha: np.ndarray, px: int) -> np.ndarray:
    """Extend nearby subject colors into transparent/edge pixels to reduce white halos."""
    if px <= 0:
        return rgb.copy()

    out = rgb.copy()
    filled = alpha > 0

    for _ in range(px):
        # 8-neighbor maximum/minimum is not ideal for color, so copy from nearest existing neighbor by simple sweeps.
        new_out = out.copy()
        new_filled = filled.copy()

        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            src_y0 = max(0, -dy)
            src_y1 = out.shape[0] - max(0, dy)
            src_x0 = max(0, -dx)
            src_x1 = out.shape[1] - max(0, dx)

            dst_y0 = max(0, dy)
            dst_y1 = out.shape[0] - max(0, -dy)
            dst_x0 = max(0, dx)
            dst_x1 = out.shape[1] - max(0, -dx)

            src_filled = filled[src_y0:src_y1, src_x0:src_x1]
            dst_filled = new_filled[dst_y0:dst_y1, dst_x0:dst_x1]
            can_copy = src_filled & (~dst_filled)
            if np.any(can_copy):
                block = new_out[dst_y0:dst_y1, dst_x0:dst_x1]
                block[can_copy] = out[src_y0:src_y1, src_x0:src_x1][can_copy]
                new_out[dst_y0:dst_y1, dst_x0:dst_x1] = block
                dst_filled[can_copy] = True
                new_filled[dst_y0:dst_y1, dst_x0:dst_x1] = dst_filled

        out = new_out
        filled = new_filled

    return out


def cleanup_white_halo(
    rgba: Image.Image,
    edge_shrink_px: int,
    alpha_threshold: int,
    color_bleed_px: int,
    soft_edge_blur: float,
) -> Image.Image:
    arr = np.array(rgba.convert("RGBA"), dtype=np.uint8)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    if alpha_threshold > 0:
        alpha = np.where(alpha >= alpha_threshold, 255, 0).astype(np.uint8)

    if edge_shrink_px > 0:
        fg = erode_mask(alpha > 0, edge_shrink_px)
        alpha = np.where(fg, 255, 0).astype(np.uint8)

    rgb = color_bleed_under_transparent(rgb, alpha, color_bleed_px)

    if soft_edge_blur > 0:
        alpha = np.array(Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(soft_edge_blur)))

    return Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), mode="RGBA")


# =========================================================
# VTracer settings
# =========================================================

def preset_params(name: str) -> dict:
    if name == "爆速":
        return {
            "filter_speckle": 5,
            "color_precision": 6,
            "layer_difference": 18,
            "corner_threshold": 60,
            "length_threshold": 5.5,
            "max_iterations": 10,
            "splice_threshold": 45,
            "path_precision": 2,
        }
    if name == "高精細":
        return {
            "filter_speckle": 2,
            "color_precision": 8,
            "layer_difference": 8,
            "corner_threshold": 60,
            "length_threshold": 3.0,
            "max_iterations": 12,
            "splice_threshold": 45,
            "path_precision": 3,
        }
    return {
        "filter_speckle": 4,
        "color_precision": 6,
        "layer_difference": 16,
        "corner_threshold": 60,
        "length_threshold": 5.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 2,
    }


def apply_sliders(params: dict, detail: int, smoothness: int, lightness: int) -> dict:
    # detail +: keep more small shapes/colors.
    params["color_precision"] = int(np.clip(params["color_precision"] + detail, 4, 10))
    params["layer_difference"] = int(np.clip(params["layer_difference"] - detail * 2, 4, 28))
    params["length_threshold"] = float(np.clip(params["length_threshold"] - detail * 0.45, 1.5, 10.0))

    # smoothness +: remove more tiny speckles and allow more fitting.
    params["filter_speckle"] = int(np.clip(params["filter_speckle"] + max(0, smoothness), 1, 14))
    params["max_iterations"] = int(np.clip(params["max_iterations"] + max(0, smoothness), 8, 20))
    params["corner_threshold"] = int(np.clip(params["corner_threshold"] - smoothness * 3, 30, 90))

    # lightness +: smaller SVG.
    params["path_precision"] = int(np.clip(params["path_precision"] - max(0, lightness // 2), 1, 8))
    params["layer_difference"] = int(np.clip(params["layer_difference"] + max(0, lightness) * 2, 4, 32))
    params["length_threshold"] = float(np.clip(params["length_threshold"] + max(0, lightness) * 0.5, 1.5, 12.0))
    params["filter_speckle"] = int(np.clip(params["filter_speckle"] + max(0, lightness // 2), 1, 14))

    return params


def run_vtracer(rgba_img: Image.Image, params: dict) -> bytes:
    if vtracer is None:
        raise RuntimeError(f"vtracer を読み込めませんでした: {VTRACER_IMPORT_ERROR}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_png = td / "input.png"
        out_svg = td / "output.svg"
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
        return out_svg.read_bytes()


def svg_preview(svg_bytes: bytes, height: int = 620) -> None:
    svg_text = svg_bytes.decode("utf-8", errors="ignore")
    html = f"""
    <div style="background:#111;padding:10px;border-radius:12px;">
      <div style="background:repeating-conic-gradient(#1c1c1c 0% 25%, #272727 0% 50%) 50% / 22px 22px;min-height:{height}px;display:flex;align-items:center;justify-content:center;overflow:auto;border-radius:10px;">
        {svg_text}
      </div>
    </div>
    """
    components.html(html, height=height + 40, scrolling=True)


# =========================================================
# Streamlit UI
# =========================================================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("PNG/JPG/WebP → 背景透明化 → 白フチ補正 → VTracer SVG出力")

    if VTRACER_IMPORT_ERROR:
        st.warning("vtracer の読み込みに失敗しています。requirements.txt と runtime.txt を確認してください。")
        st.code(VTRACER_IMPORT_ERROR)

    upload = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg", "webp"])
    if upload is None:
        st.info("まず画像をアップロードしてください。")
        return

    img = Image.open(upload)
    img = ImageOps.exif_transpose(img)

    with st.form("settings"):
        st.subheader("変換設定")
        c1, c2 = st.columns(2)
        with c1:
            max_side = st.slider("最大辺", 700, 2600, 1600, 100)
            preset = st.selectbox("プリセット", ["爆速", "標準", "高精細"], index=1)
            detail = st.slider("ディテール", -3, 3, 0)
            smoothness = st.slider("なめらかさ", -3, 3, 1)
            lightness = st.slider("軽量化", -3, 3, 0)
        with c2:
            remove_bg = st.checkbox("外側の白/薄灰背景を透明化", value=True)
            barrier_radius = st.slider("白服保護", 0, 8, 4)
            edge_shrink_px = st.slider("白フチ削り", 0, 4, 1)
            alpha_threshold = st.slider("半透明エッジしきい値", 0, 255, 180)
            color_bleed_px = st.slider("輪郭色にじみ補正", 0, 4, 2)
            soft_edge_blur = st.slider("輪郭のやわらかさ", 0.0, 1.2, 0.15, 0.05)

        with st.expander("背景判定の詳細設定"):
            neutral_max_chroma = st.slider("背景の無彩色判定", 8, 50, 28)
            light_min = st.slider("背景の明るさ下限", 100, 240, 145)
            force_white_min = st.slider("強制白判定", 220, 255, 245)
            crop_padding = st.slider("切り抜き余白", 0, 40, 12)
            bg_alpha_blur = st.slider("背景除去エッジぼかし", 0.0, 1.2, 0.35, 0.05)

        submitted = st.form_submit_button("SVG化を実行", use_container_width=True)

    img = resize_max_side(img, max_side)

    if remove_bg:
        pre = remove_border_light_background(
            img,
            neutral_max_chroma=neutral_max_chroma,
            light_min=light_min,
            force_white_min=force_white_min,
            barrier_radius=barrier_radius,
            alpha_blur=bg_alpha_blur,
            crop_padding=crop_padding,
        )
    else:
        pre = img.convert("RGBA")

    pre = cleanup_white_halo(
        pre,
        edge_shrink_px=edge_shrink_px,
        alpha_threshold=alpha_threshold,
        color_bleed_px=color_bleed_px,
        soft_edge_blur=soft_edge_blur,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("元画像")
        st.image(img, use_container_width=True)
    with col2:
        st.subheader("前処理後PNG")
        st.image(pre, use_container_width=True)
        st.download_button(
            "前処理PNGを保存",
            data=image_to_png_bytes(pre),
            file_name="vtracer_preprocessed.png",
            mime="image/png",
            use_container_width=True,
        )

    params = apply_sliders(preset_params(preset), detail, smoothness, lightness)
    with st.expander("現在のVTracerパラメータ"):
        st.json(params)

    if not submitted:
        st.info("設定を確認してから「SVG化を実行」を押してください。スライダー変更だけでは重い処理を走らせないようにしています。")
        return

    try:
        with st.spinner("VTracerでSVG化しています..."):
            svg_bytes = run_vtracer(pre, params)
    except Exception as exc:
        st.error("SVG化中にエラーが出ました。まず最大辺を1200〜1600に下げて再実行してください。")
        st.code(str(exc))
        return

    st.subheader("SVGプレビュー")
    svg_preview(svg_bytes)
    st.download_button(
        "SVGをダウンロード",
        data=svg_bytes,
        file_name="vtracer_output.svg",
        mime="image/svg+xml",
        use_container_width=True,
    )

    st.success(f"完了: {len(svg_bytes) / 1024:.1f} KB")

    st.markdown(
        """
### 調整メモ
- 白い輪郭が出る: **白フチ削り 1〜2**、**輪郭色にじみ補正 2〜3**
- 細部がガタつく: **なめらかさ +1〜+2**
- 細部が消える: **ディテール +1〜+2**、**白フチ削りを下げる**
- 変換が落ちる/重い: **最大辺を1200〜1600**、プリセットを**標準/爆速**
        """
    )


if __name__ == "__main__":
    main()
