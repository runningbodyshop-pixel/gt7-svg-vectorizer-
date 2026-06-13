import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter

st.set_page_config(page_title="Anime SVG Vectorizer 1.8 Clean", layout="wide")

SVG_NS = "http://www.w3.org/2000/svg"


# =========================================================
# command / basic helpers
# =========================================================

def run_cmd(cmd, timeout=120):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout, p.stderr


def color_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(np.clip(rgb[0], 0, 255)),
        int(np.clip(rgb[1], 0, 255)),
        int(np.clip(rgb[2], 0, 255)),
    )


def hex_to_rgb(hex_color):
    s = str(hex_color).strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return None


def make_download_name(original_name, suffix):
    stem = Path(original_name).stem if original_name else "vectorized"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "vectorized"
    return f"{safe}_{suffix}.svg"


@st.cache_data(show_spinner=False)
def tool_versions():
    info = {}
    for exe in ["potrace", "scour"]:
        path = shutil.which(exe)
        if not path:
            info[exe] = "not found"
            continue
        try:
            code, out, err = run_cmd([exe, "--version"], timeout=20)
            info[exe] = (out or err or "").strip() or path
        except Exception as e:
            info[exe] = f"error: {e}"
    return info


# =========================================================
# masks / image preprocess
# =========================================================

def normalize_rgba_image(uploaded_file, out_png, max_side, alpha_threshold, harden_alpha):
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img).convert("RGBA")

    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    arr = np.array(img)
    alpha = arr[:, :, 3]

    # 白背景合成しない。透明/半透明外周だけ整理。
    arr[alpha < alpha_threshold, 3] = 0

    if harden_alpha:
        arr[arr[:, :, 3] >= alpha_threshold, 3] = 255

    cleaned = Image.fromarray(arr, mode="RGBA")
    cleaned.save(out_png)
    return cleaned


def mask_from_bool(mask_bool):
    return Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")


def bool_from_mask(mask_img):
    return np.array(mask_img.convert("L")) > 0


def morph_mask(mask_img, px):
    px = int(px)
    if px == 0:
        return mask_img

    size = abs(px) * 2 + 1

    if px > 0:
        return mask_img.filter(ImageFilter.MaxFilter(size))

    return mask_img.filter(ImageFilter.MinFilter(size))


def clean_mask(mask_img, open_px=0, close_px=0):
    out = mask_img

    if open_px > 0:
        size = open_px * 2 + 1
        out = out.filter(ImageFilter.MinFilter(size))
        out = out.filter(ImageFilter.MaxFilter(size))

    if close_px > 0:
        size = close_px * 2 + 1
        out = out.filter(ImageFilter.MaxFilter(size))
        out = out.filter(ImageFilter.MinFilter(size))

    return out


def foreground_mask_from_alpha(alpha, alpha_threshold, edge_trim_px):
    fg = alpha >= alpha_threshold
    mask = mask_from_bool(fg)

    if edge_trim_px != 0:
        mask = morph_mask(mask, -edge_trim_px)

    return bool_from_mask(mask)


def make_smoothed_rgb(rgba_img, median_px=0, blur_radius=0.0):
    rgb_img = rgba_img.convert("RGB")

    if median_px > 0:
        size = int(median_px) * 2 + 1
        rgb_img = rgb_img.filter(ImageFilter.MedianFilter(size=size))

    if blur_radius > 0:
        rgb_img = rgb_img.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    return np.array(rgb_img)


# =========================================================
# color metrics
# =========================================================

def rgb_metrics(rgb):
    rgb_f = rgb.astype(np.float32)

    mx = rgb_f.max(axis=-1)
    mn = rgb_f.min(axis=-1)

    chroma = mx - mn
    sat = chroma / np.maximum(mx, 1.0)

    lum = (
        rgb_f[..., 0] * 0.2126
        + rgb_f[..., 1] * 0.7152
        + rgb_f[..., 2] * 0.0722
    )

    return lum, sat, chroma


def rgb_to_hsv_np(rgb):
    arr = rgb.astype(np.float32) / 255.0

    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]

    mx = np.max(arr, axis=-1)
    mn = np.min(arr, axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff != 0

    idx = (mx == r) & mask
    h[idx] = (60.0 * ((g[idx] - b[idx]) / diff[idx]) + 360.0) % 360.0

    idx = (mx == g) & mask
    h[idx] = 60.0 * (((b[idx] - r[idx]) / diff[idx]) + 2.0)

    idx = (mx == b) & mask
    h[idx] = 60.0 * (((r[idx] - g[idx]) / diff[idx]) + 4.0)

    s = np.zeros_like(mx)
    nonzero = mx != 0
    s[nonzero] = diff[nonzero] / mx[nonzero]

    v = mx
    return h, s, v


def estimate_dark_color(rgba_img, mask, fallback="#11151b"):
    arr = np.array(rgba_img)
    rgb = arr[:, :, :3]

    pixels = rgb[mask]
    if len(pixels) == 0:
        return fallback

    lum, sat, chroma = rgb_metrics(pixels)
    cutoff = np.percentile(lum, 18)
    dark_pixels = pixels[lum <= cutoff]

    if len(dark_pixels) == 0:
        dark_pixels = pixels

    c = np.median(dark_pixels, axis=0).astype(int)
    c = np.clip(c, 8, 75)

    return color_hex(c)


# =========================================================
# SVG / potrace helpers
# =========================================================

def round_svg_numbers(svg_text, decimals):
    if decimals < 0:
        return svg_text

    decimals = int(decimals)

    def repl(m):
        x = float(m.group(0))
        s = f"{x:.{decimals}f}"
        s = s.rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        return s

    return re.sub(r"-?\d+\.\d+", repl, svg_text)


def optimize_svg_text(svg_text, precision_decimals=2, use_scour=True):
    svg_text = re.sub(r"<\?xml[^>]*>\s*", "", svg_text)
    svg_text = re.sub(r"<!--.*?-->", "", svg_text, flags=re.S)
    svg_text = re.sub(r">\s+<", "><", svg_text).strip()

    svg_text = round_svg_numbers(svg_text, precision_decimals)

    if not use_scour:
        return svg_text

    scour = shutil.which("scour")
    if not scour:
        return svg_text

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.svg"
        dst = Path(td) / "out.svg"

        src.write_text(svg_text, encoding="utf-8")

        cmd = [
            scour,
            "-i",
            str(src),
            "-o",
            str(dst),
            "--enable-viewboxing",
            "--enable-id-stripping",
            "--shorten-ids",
            "--strip-xml-prolog",
            "--remove-metadata",
            "--indent=none",
        ]

        try:
            code, out, err = run_cmd(cmd, timeout=60)
            if code == 0 and dst.exists() and dst.stat().st_size > 0:
                out_text = dst.read_text(encoding="utf-8", errors="ignore")
                return round_svg_numbers(out_text, precision_decimals)
        except Exception:
            pass

    return svg_text


def extract_potrace_group(svg_text, fill, group_id=None):
    group_match = re.search(r"<g\b([^>]*)>(.*?)</g>", svg_text, flags=re.S)

    transform = ""
    inner = svg_text

    if group_match:
        attrs = group_match.group(1)
        inner = group_match.group(2)
        t = re.search(r'transform="([^"]+)"', attrs)
        if t:
            transform = f' transform="{t.group(1)}"'

    paths = re.findall(r'<path\b[^>]*\bd="([^"]+)"[^>]*/?>', inner)

    if not paths:
        paths = re.findall(r'<path\b[^>]*\bd="([^"]+)"[^>]*>', inner)

    paths = [d for d in paths if len(d) > 5]

    if not paths:
        return ""

    gid = f' id="{group_id}"' if group_id else ""
    body = "\n".join(f'<path d="{d}"/>' for d in paths)

    return f'<g{gid}{transform} fill="{fill}" stroke="none">\n{body}\n</g>'


def trace_binary_mask_to_group(mask_img, fill, group_id, speckles, smooth_corners, optimize):
    if not shutil.which("potrace"):
        return ""

    arr = np.array(mask_img.convert("L"))
    if not np.any(arr > 0):
        return ""

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pbm = td / "mask.pbm"
        part_svg = td / "part.svg"

        # Potrace は黒をトレース対象にする
        pbm_arr = np.where(arr > 0, 0, 255).astype(np.uint8)
        Image.fromarray(pbm_arr, mode="L").convert("1").save(pbm)

        cmd = [
            "potrace",
            str(pbm),
            "-b",
            "svg",
            "-o",
            str(part_svg),
            "--turdsize",
            str(int(speckles)),
            "--alphamax",
            str(float(smooth_corners)),
            "--opttolerance",
            str(float(optimize)),
            "--unit",
            "1",
        ]

        code, out, err = run_cmd(cmd, timeout=80)

        if code != 0 or not part_svg.exists():
            return ""

        svg_text = part_svg.read_text(encoding="utf-8", errors="ignore")
        return extract_potrace_group(svg_text, fill=fill, group_id=group_id)


def svg_preview_png(svg_text, width=900):
    try:
        import cairosvg
        return cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=width,
        )
    except Exception:
        return None


# =========================================================
# palette helpers
# =========================================================

def sample_pixels(pixels, max_sample=240_000, seed=1234):
    if len(pixels) <= max_sample:
        return pixels

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pixels), size=max_sample, replace=False)

    return pixels[idx]


def quantize_palette_from_pixels(pixels, colors):
    if len(pixels) == 0 or colors <= 0:
        return []

    pixels = sample_pixels(pixels).astype(np.uint8)

    if len(pixels) == 1:
        return [(tuple(int(v) for v in pixels[0]), 1)]

    img = Image.fromarray(pixels.reshape((len(pixels), 1, 3)), mode="RGB")
    q = img.quantize(colors=int(colors), method=Image.Quantize.MEDIANCUT)

    raw = q.getpalette()[: int(colors) * 3]
    labels = np.array(q).reshape(-1)
    counts = np.bincount(labels, minlength=int(colors))

    result = []

    for i in np.argsort(-counts):
        if counts[i] <= 0:
            continue

        base = i * 3
        c = tuple(int(v) for v in raw[base:base + 3])

        if len(c) == 3:
            result.append((c, int(counts[i])))

    return result


def rgb_distance(c1, c2):
    a = np.array(c1, dtype=np.float32)
    b = np.array(c2, dtype=np.float32)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def merge_palette_candidates(anchor_colors, base_colors, max_colors, merge_distance):
    final = []

    def add_color(c, protected=False):
        c = tuple(int(x) for x in c)

        if not final:
            final.append((c, protected))
            return

        distances = [rgb_distance(c, existing[0]) for existing in final]

        if min(distances) >= merge_distance:
            final.append((c, protected))

    for c in anchor_colors:
        add_color(c, protected=True)

    for c in base_colors:
        add_color(c, protected=False)

    protected = [c for c, p in final if p]
    normal = [c for c, p in final if not p]

    clipped = protected[:max_colors]

    for c in normal:
        if len(clipped) >= max_colors:
            break
        clipped.append(c)

    if not clipped:
        clipped = base_colors[:max_colors]

    return np.array(clipped, dtype=np.uint8)


def detect_skin_pixels(pixels):
    if len(pixels) == 0:
        return np.zeros((0,), dtype=bool)

    lum, sat, chroma = rgb_metrics(pixels)
    h, s_hsv, v_hsv = rgb_to_hsv_np(pixels)

    r = pixels[:, 0].astype(np.float32)
    g = pixels[:, 1].astype(np.float32)
    b = pixels[:, 2].astype(np.float32)

    warm_hue = (h <= 58) | (h >= 330)

    skin_like = (
        warm_hue
        & (sat >= 0.03)
        & (sat <= 0.58)
        & (lum >= 78)
        & (lum <= 248)
        & (r >= g * 0.90)
        & (g >= b * 0.68)
        & ((r - b) >= 5)
    )

    return skin_like


def build_palette(
    rgb_for_palette,
    fg_mask,
    colors,
    use_priority,
    keep_neutrals,
    accent_count,
    light_tone_count,
    skin_tone_count,
    shadow_tone_count,
    merge_distance,
    manual_colors,
):
    pixels = rgb_for_palette[fg_mask]

    if len(pixels) == 0:
        raise RuntimeError("不透明ピクセルが見つかりません。alphaしきい値か外周トリムを下げてください。")

    base_pairs = quantize_palette_from_pixels(pixels, colors)
    base_colors = [c for c, count in base_pairs]

    if not use_priority:
        return merge_palette_candidates(manual_colors, base_colors, colors, merge_distance)

    lum, sat, chroma = rgb_metrics(pixels)
    anchors = list(manual_colors)

    if keep_neutrals:
        dark_pixels = pixels[lum <= np.percentile(lum, 12)]
        if len(dark_pixels) > 0:
            anchors.append(tuple(np.median(dark_pixels, axis=0).astype(int)))

        light_mask = (lum >= np.percentile(lum, 84)) & (sat <= 0.36)
        light_pixels = pixels[light_mask]
        if len(light_pixels) > 0:
            anchors.append(tuple(np.median(light_pixels, axis=0).astype(int)))

        mid_neutral_mask = (sat <= 0.24) & (lum > 45) & (lum < 220)
        mid_neutral_pixels = pixels[mid_neutral_mask]
        if len(mid_neutral_pixels) > 0:
            mid_pairs = quantize_palette_from_pixels(
                mid_neutral_pixels,
                min(3, max(1, colors // 10)),
            )
            anchors.extend([c for c, count in mid_pairs])

    if accent_count > 0:
        accent_mask = (
            (sat >= 0.32)
            & (chroma >= 34)
            & (lum >= 28)
            & (lum <= 248)
        )
        accent_pairs = quantize_palette_from_pixels(pixels[accent_mask], accent_count)
        anchors.extend([c for c, count in accent_pairs])

    if light_tone_count > 0:
        light_mask = (lum >= np.percentile(lum, 68)) & (sat <= 0.48)
        light_pairs = quantize_palette_from_pixels(pixels[light_mask], light_tone_count)
        anchors.extend([c for c, count in light_pairs])

    if skin_tone_count > 0:
        skin_pairs = quantize_palette_from_pixels(
            pixels[detect_skin_pixels(pixels)],
            skin_tone_count,
        )
        anchors.extend([c for c, count in skin_pairs])

    if shadow_tone_count > 0:
        shadow_mask = (
            (lum >= 32)
            & (lum <= np.percentile(lum, 58))
            & (sat <= 0.58)
        )
        shadow_pairs = quantize_palette_from_pixels(pixels[shadow_mask], shadow_tone_count)
        anchors.extend([c for c, count in shadow_pairs])

    return merge_palette_candidates(
        anchor_colors=anchors,
        base_colors=base_colors,
        max_colors=colors,
        merge_distance=merge_distance,
    )


def assign_nearest_palette(rgb_for_labels, fg_mask, palette):
    h, w, _ = rgb_for_labels.shape
    labels = np.full((h, w), -1, dtype=np.int16)

    flat_rgb = rgb_for_labels.reshape(-1, 3)
    flat_fg = fg_mask.reshape(-1)
    fg_indices = np.flatnonzero(flat_fg)

    flat_labels = labels.reshape(-1)
    palette_i = palette.astype(np.int32)

    chunk = 80_000

    for start in range(0, len(fg_indices), chunk):
        part_idx = fg_indices[start:start + chunk]
        part = flat_rgb[part_idx].astype(np.int32)

        diff = part[:, None, :] - palette_i[None, :, :]
        dist = np.sum(diff * diff, axis=2)

        flat_labels[part_idx] = np.argmin(dist, axis=1).astype(np.int16)

    return labels


def sampled_fill_color(rgb_original, rgb_smoothed, mask, method, vibrance=0, source="元画像"):
    rgb_src = rgb_original if source == "元画像" else rgb_smoothed
    pixels = rgb_src[mask]

    if len(pixels) == 0:
        return "#000000"

    if method == "中央値":
        c = np.median(pixels, axis=0).astype(np.float32)
    else:
        c = np.mean(pixels, axis=0).astype(np.float32)

    if vibrance != 0:
        gray = np.mean(c)
        factor = 1.0 + float(vibrance) / 100.0
        c = gray + (c - gray) * factor

    return color_hex(c.astype(int))


# =========================================================
# line masks
# =========================================================

def make_lineart_mask(
    rgba_img,
    alpha_threshold,
    edge_trim_px,
    mode,
    dark_threshold,
    contrast_threshold,
    width_adjust,
    open_px,
    close_px,
    max_lum_allowed,
    line_blur_radius,
):
    arr = np.array(rgba_img)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    fg = foreground_mask_from_alpha(alpha, alpha_threshold, edge_trim_px)

    lum, sat, chroma = rgb_metrics(rgb)

    gray_img = Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8), mode="L")

    if line_blur_radius > 0:
        gray_img = gray_img.filter(ImageFilter.GaussianBlur(radius=float(line_blur_radius)))

    gray = np.array(gray_img).astype(np.float32)

    local_max = np.array(gray_img.filter(ImageFilter.MaxFilter(3))).astype(np.float32)
    local_min = np.array(gray_img.filter(ImageFilter.MinFilter(3))).astype(np.float32)
    local_contrast = local_max - local_min

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)

    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :])

    edge = gx + gy

    dark = lum <= dark_threshold
    edge_like = (
        (local_contrast >= contrast_threshold)
        | (edge >= contrast_threshold * 1.25)
    )

    if mode == "エッジ優先・汎用":
        mask = fg & dark & edge_like

    elif mode == "黒インク強め":
        very_dark = lum <= max(18, dark_threshold * 0.72)
        mask = fg & (
            (dark & edge_like)
            | (very_dark & (local_contrast >= contrast_threshold * 0.55))
        )

    elif mode == "淡いグレー線も拾う":
        soft_dark = lum <= min(max_lum_allowed, dark_threshold + 48)
        low_sat = sat <= 0.45
        mask = fg & soft_dark & low_sat & edge_like

    else:
        # 細部専用
        soft = lum <= max_lum_allowed
        mask = fg & soft & edge_like & (local_contrast >= contrast_threshold)

    mask_img = mask_from_bool(mask)
    mask_img = clean_mask(mask_img, open_px=open_px, close_px=close_px)
    mask_img = morph_mask(mask_img, width_adjust)

    return mask_img


def make_line_group(
    rgba_img,
    alpha_threshold,
    edge_trim_px,
    mode,
    dark_threshold,
    contrast_threshold,
    width_adjust,
    color_mode,
    manual_color,
    speckles,
    smooth_corners,
    optimize,
    open_px,
    close_px,
    max_lum_allowed,
    line_blur_radius,
    group_id,
):
    mask = make_lineart_mask(
        rgba_img=rgba_img,
        alpha_threshold=alpha_threshold,
        edge_trim_px=edge_trim_px,
        mode=mode,
        dark_threshold=dark_threshold,
        contrast_threshold=contrast_threshold,
        width_adjust=width_adjust,
        open_px=open_px,
        close_px=close_px,
        max_lum_allowed=max_lum_allowed,
        line_blur_radius=line_blur_radius,
    )

    mask_bool = bool_from_mask(mask)

    if not mask_bool.any():
        return "", mask

    if color_mode == "自動":
        fill = estimate_dark_color(rgba_img, mask_bool, fallback="#14171c")
    else:
        fill = manual_color

    group = trace_binary_mask_to_group(
        mask_img=mask,
        fill=fill,
        group_id=group_id,
        speckles=speckles,
        smooth_corners=smooth_corners,
        optimize=optimize,
    )

    return group, mask


# =========================================================
# main trace
# =========================================================

def make_underpaint_group(
    rgba_img,
    alpha_threshold,
    edge_trim_px,
    expand_px,
    color_mode,
    manual_color,
    speckles,
    smooth_corners,
    optimize,
):
    arr = np.array(rgba_img)
    alpha = arr[:, :, 3]

    fg = foreground_mask_from_alpha(alpha, alpha_threshold, edge_trim_px=0)
    mask = mask_from_bool(fg)
    mask = morph_mask(mask, expand_px)

    # 外側に謎の輪郭が出にくいよう、必要なら外周トリムも反映
    if edge_trim_px > 0 and expand_px <= 0:
        mask = morph_mask(mask, -edge_trim_px)

    mask_bool = bool_from_mask(mask)

    if not mask_bool.any():
        return ""

    if color_mode == "自動":
        fill = estimate_dark_color(rgba_img, mask_bool)
    else:
        fill = manual_color

    return trace_binary_mask_to_group(
        mask_img=mask,
        fill=fill,
        group_id="underpaint",
        speckles=speckles,
        smooth_corners=smooth_corners,
        optimize=optimize,
    )


def run_trace(
    in_png,
    out_svg,
    colors,
    alpha_threshold,
    edge_trim_px,
    remove_largest_color,
    underpaint,
    underpaint_expand,
    underpaint_color_mode,
    underpaint_manual_color,
    clip_fills_to_foreground,
    color_layer_overlap,
    fill_open_px,
    fill_close_px,
    fill_smooth_median_px,
    fill_smooth_blur,
    resample_color,
    color_sample_method,
    color_sample_source,
    vibrance,
    use_priority_palette,
    keep_neutrals,
    accent_count,
    light_tone_count,
    skin_tone_count,
    shadow_tone_count,
    palette_merge_distance,
    min_color_area_px,
    manual_palette_colors,
    add_main_line,
    main_line_mode,
    main_line_dark_threshold,
    main_line_contrast_threshold,
    main_line_width_adjust,
    main_line_color_mode,
    main_line_manual_color,
    main_line_open_px,
    main_line_close_px,
    main_line_max_lum,
    main_line_blur,
    add_detail_line,
    detail_line_dark_threshold,
    detail_line_contrast_threshold,
    detail_line_width_adjust,
    detail_line_color_mode,
    detail_line_manual_color,
    detail_line_open_px,
    detail_line_close_px,
    detail_line_max_lum,
    detail_line_blur,
    cut_line_from_fills_px,
    fill_speckles,
    fill_smooth_corners,
    fill_optimize,
    line_speckles,
    line_smooth_corners,
    line_optimize,
    under_speckles,
    under_smooth_corners,
    under_optimize,
    precision_decimals,
    use_scour,
):
    if not shutil.which("potrace"):
        raise RuntimeError("potrace が見つかりません。packages.txt に potrace を入れてください。")

    img = Image.open(in_png).convert("RGBA")

    w, h = img.size
    arr = np.array(img)

    rgb_original = arr[:, :, :3]
    alpha = arr[:, :, 3]

    rgb_smoothed = make_smoothed_rgb(
        rgba_img=img,
        median_px=fill_smooth_median_px,
        blur_radius=fill_smooth_blur,
    )

    fg_mask = foreground_mask_from_alpha(alpha, alpha_threshold, edge_trim_px)

    if not fg_mask.any():
        raise RuntimeError("トレース対象がありません。alphaしきい値か外周トリムを下げてください。")

    palette = build_palette(
        rgb_for_palette=rgb_smoothed,
        fg_mask=fg_mask,
        colors=colors,
        use_priority=use_priority_palette,
        keep_neutrals=keep_neutrals,
        accent_count=accent_count,
        light_tone_count=light_tone_count,
        skin_tone_count=skin_tone_count,
        shadow_tone_count=shadow_tone_count,
        merge_distance=palette_merge_distance,
        manual_colors=manual_palette_colors,
    )

    labels = assign_nearest_palette(rgb_smoothed, fg_mask, palette)

    counts = [int(np.sum(labels == i)) for i in range(len(palette))]
    order = list(np.argsort(-np.array(counts)))

    if remove_largest_color and order:
        order = order[1:]

    groups = []
    debug_masks = {}

    if underpaint:
        under_group = make_underpaint_group(
            rgba_img=img,
            alpha_threshold=alpha_threshold,
            edge_trim_px=edge_trim_px,
            expand_px=underpaint_expand,
            color_mode=underpaint_color_mode,
            manual_color=underpaint_manual_color,
            speckles=under_speckles,
            smooth_corners=under_smooth_corners,
            optimize=under_optimize,
        )
        if under_group:
            groups.append(under_group)

    line_groups = []
    line_cut_mask = None

    if add_main_line:
        group, mask = make_line_group(
            rgba_img=img,
            alpha_threshold=alpha_threshold,
            edge_trim_px=edge_trim_px,
            mode=main_line_mode,
            dark_threshold=main_line_dark_threshold,
            contrast_threshold=main_line_contrast_threshold,
            width_adjust=main_line_width_adjust,
            color_mode=main_line_color_mode,
            manual_color=main_line_manual_color,
            speckles=line_speckles,
            smooth_corners=line_smooth_corners,
            optimize=line_optimize,
            open_px=main_line_open_px,
            close_px=main_line_close_px,
            max_lum_allowed=main_line_max_lum,
            line_blur_radius=main_line_blur,
            group_id="main_line_top",
        )

        debug_masks["main_line_mask"] = mask

        if group:
            line_groups.append(group)
            line_cut_mask = bool_from_mask(mask)

    if add_detail_line:
        group, mask = make_line_group(
            rgba_img=img,
            alpha_threshold=alpha_threshold,
            edge_trim_px=edge_trim_px,
            mode="細部専用",
            dark_threshold=detail_line_dark_threshold,
            contrast_threshold=detail_line_contrast_threshold,
            width_adjust=detail_line_width_adjust,
            color_mode=detail_line_color_mode,
            manual_color=detail_line_manual_color,
            speckles=max(0, min(line_speckles, 2)),
            smooth_corners=max(0.2, min(line_smooth_corners, 0.75)),
            optimize=min(line_optimize, 0.18),
            open_px=detail_line_open_px,
            close_px=detail_line_close_px,
            max_lum_allowed=detail_line_max_lum,
            line_blur_radius=detail_line_blur,
            group_id="detail_line_top",
        )

        debug_masks["detail_line_mask"] = mask

        if group:
            line_groups.append(group)
            detail_bool = bool_from_mask(mask)
            line_cut_mask = detail_bool if line_cut_mask is None else (line_cut_mask | detail_bool)

    if line_cut_mask is not None and cut_line_from_fills_px > 0:
        line_cut_mask = bool_from_mask(
            morph_mask(mask_from_bool(line_cut_mask), cut_line_from_fills_px)
        )

    min_area = max(1, int(min_color_area_px))

    for idx in order:
        if idx < 0 or idx >= len(palette):
            continue

        if counts[idx] < min_area:
            continue

        mask_bool = labels == idx

        if line_cut_mask is not None and cut_line_from_fills_px > 0:
            mask_bool = mask_bool & (~line_cut_mask)

        if clip_fills_to_foreground:
            mask_bool = mask_bool & fg_mask

        if not mask_bool.any():
            continue

        mask_img = mask_from_bool(mask_bool)
        mask_img = clean_mask(mask_img, open_px=fill_open_px, close_px=fill_close_px)
        mask_img = morph_mask(mask_img, color_layer_overlap)

        if clip_fills_to_foreground:
            mask_bool_after = bool_from_mask(mask_img) & fg_mask
            mask_img = mask_from_bool(mask_bool_after)

        final_mask_bool = bool_from_mask(mask_img)
        if not final_mask_bool.any():
            continue

        if resample_color:
            fill = sampled_fill_color(
                rgb_original=rgb_original,
                rgb_smoothed=rgb_smoothed,
                mask=mask_bool,
                method=color_sample_method,
                vibrance=vibrance,
                source=color_sample_source,
            )
        else:
            fill = color_hex(palette[idx])

        group = trace_binary_mask_to_group(
            mask_img=mask_img,
            fill=fill,
            group_id=f"color_{idx}",
            speckles=fill_speckles,
            smooth_corners=fill_smooth_corners,
            optimize=fill_optimize,
        )

        if group:
            groups.append(group)

    groups.extend(line_groups)

    svg = (
        f'<svg xmlns="{SVG_NS}" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" shape-rendering="geometricPrecision">\n'
        + "\n".join(groups)
        + "\n</svg>"
    )

    svg = optimize_svg_text(
        svg,
        precision_decimals=precision_decimals,
        use_scour=use_scour,
    )

    Path(out_svg).write_text(svg, encoding="utf-8")

    return svg, palette, debug_masks


# =========================================================
# UI presets
# =========================================================

PRESETS = {
    "クリーン軽量": {
        "max_side": 1200,
        "colors": 20,
        "fill_median": 1,
        "fill_blur": 0.35,
        "detail_line": False,
        "precision": 1,
        "fill_optimize": 0.28,
        "line_optimize": 0.14,
        "main_line_width": -2,
        "main_line_contrast": 38,
        "min_area": 8,
    },
    "バランス": {
        "max_side": 1400,
        "colors": 28,
        "fill_median": 1,
        "fill_blur": 0.25,
        "detail_line": False,
        "precision": 2,
        "fill_optimize": 0.18,
        "line_optimize": 0.10,
        "main_line_width": -2,
        "main_line_contrast": 34,
        "min_area": 4,
    },
    "品質寄り": {
        "max_side": 1600,
        "colors": 36,
        "fill_median": 0,
        "fill_blur": 0.15,
        "detail_line": True,
        "precision": 2,
        "fill_optimize": 0.12,
        "line_optimize": 0.06,
        "main_line_width": -1,
        "main_line_contrast": 30,
        "min_area": 2,
    },
}


st.title("Anime SVG Vectorizer 1.8 Clean")
st.caption(
    "ザラつき低減・謎輪郭対策・軽量化調整版。"
    "塗り用スムージング、線画ブラー、外周クリップ、座標丸めを追加しています。"
)

with st.expander("環境チェック", expanded=False):
    st.write(tool_versions())

uploaded = st.file_uploader(
    "画像をアップロード",
    type=["png", "jpg", "jpeg", "webp"],
)

with st.sidebar:
    st.header("プリセット")

    preset_name = st.selectbox(
        "変換モード",
        ["クリーン軽量", "バランス", "品質寄り"],
        index=1,
    )

    P = PRESETS[preset_name]

    st.header("基本")

    max_side = st.slider("処理前の最大辺px", 512, 2600, P["max_side"], 64)

    st.subheader("外周・謎の輪郭対策")

    alpha_threshold = st.slider("alphaしきい値", 0, 254, 58)

    harden_alpha = st.checkbox("半透明の縁を締める", value=True)

    edge_trim_px = st.slider(
        "外周トリムpx",
        -2,
        5,
        1,
        help="謎の外周線や灰色フチが出る時は1〜3。欠ける時は0。",
    )

    underpaint = st.checkbox("下地シルエットを敷く", value=True)

    underpaint_expand = st.slider(
        "下地の拡張/縮小px",
        -4,
        6,
        -1,
        help="謎の輪郭が見える時は-1〜0。隙間が出る時は1。",
    )

    underpaint_color_mode = st.radio(
        "下地色",
        ["自動", "手動"],
        index=1,
        horizontal=True,
    )

    underpaint_manual_color = st.color_picker("手動下地色", "#07090c")

    clip_fills_to_foreground = st.checkbox(
        "色レイヤーを外周内にクリップ",
        value=True,
        help="ON推奨。色レイヤーのはみ出しによる謎輪郭を減らします。",
    )

    color_layer_overlap = st.slider(
        "色レイヤーの重ね/縮小px",
        -2,
        3,
        0,
        help="謎輪郭が出る時は0〜-1。色の隙間が出る時だけ1。",
    )

    fill_open_px = st.slider("塗りノイズ除去px", 0, 2, 0)

    fill_close_px = st.slider("塗り小穴埋めpx", 0, 2, 1)

    st.subheader("ザラつき低減")

    fill_smooth_median_px = st.slider(
        "塗り用メディアン平滑化px",
        0,
        3,
        P["fill_median"],
        help="1がおすすめ。細かい色ノイズを減らします。",
    )

    fill_smooth_blur = st.slider(
        "塗り用ぼかし",
        0.0,
        2.0,
        float(P["fill_blur"]),
        0.05,
        help="0.2〜0.5で色面が滑らかになります。強すぎると細部が溶けます。",
    )

    st.subheader("色パレット")

    colors = st.slider("総色数", 2, 64, P["colors"])

    use_priority_palette = st.checkbox("優先色保護を使う", value=True)

    keep_neutrals = st.checkbox("黒/白/グレー系を保護", value=True)

    accent_count = st.slider("高彩度アクセント色の保護数", 0, 10, 3)

    light_tone_count = st.slider("明るい薄色の保護数", 0, 12, 5)

    skin_tone_count = st.slider("肌色保護数", 0, 12, 4)

    shadow_tone_count = st.slider("影色/中間色の保護数", 0, 12, 4)

    palette_merge_distance = st.slider(
        "近い色を統合する強さ",
        0,
        80,
        16 if preset_name == "クリーン軽量" else 12,
        help="小さいほど色を分ける。大きいほど軽い。",
    )

    min_color_area_px = st.slider(
        "小さい色領域を捨てる基準px",
        1,
        250,
        P["min_area"],
        help="ザラつきが出る時は上げる。細部色を残したい時は下げる。",
    )

    resample_color = st.checkbox("元画像/平滑化画像から塗り色を取り直す", value=True)

    color_sample_source = st.radio(
        "塗り色の参照元",
        ["元画像", "平滑化後"],
        index=1,
        horizontal=True,
        help="ザラつき軽減なら平滑化後。鮮やかさ優先なら元画像。",
    )

    color_sample_method = st.radio(
        "塗り色の取り方",
        ["中央値", "平均"],
        index=0,
        horizontal=True,
    )

    vibrance = st.slider("色の鮮やかさ補正", -30, 50, 6)

    remove_largest_color = st.checkbox("最大面積色を背景として削除", value=False)

    with st.expander("手動保護色", expanded=False):
        use_manual_1 = st.checkbox("手動色1を保護", value=False)
        manual_color_1 = st.color_picker("手動色1", "#5fd5e8")

        use_manual_2 = st.checkbox("手動色2を保護", value=False)
        manual_color_2 = st.color_picker("手動色2", "#f1ded5")

        use_manual_3 = st.checkbox("手動色3を保護", value=False)
        manual_color_3 = st.color_picker("手動色3", "#f6f6f2")

        use_manual_4 = st.checkbox("手動色4を保護", value=False)
        manual_color_4 = st.color_picker("手動色4", "#4b4c55")

    st.subheader("主線レイヤー")

    add_main_line = st.checkbox("主線を最上段に追加", value=True)

    main_line_mode = st.radio(
        "主線抽出モード",
        ["エッジ優先・汎用", "黒インク強め", "淡いグレー線も拾う"],
        index=0,
    )

    main_line_dark_threshold = st.slider("主線: 暗さしきい値", 20, 180, 90)

    main_line_contrast_threshold = st.slider(
        "主線: コントラストしきい値",
        4,
        100,
        P["main_line_contrast"],
    )

    main_line_width_adjust = st.slider("主線の太さ調整", -4, 3, P["main_line_width"])

    main_line_blur = st.slider(
        "主線用ぼかし",
        0.0,
        2.0,
        0.35,
        0.05,
        help="細部のザラつきが出る時は0.3〜0.7。",
    )

    main_line_open_px = st.slider("主線ノイズ除去px", 0, 2, 0)

    main_line_close_px = st.slider("主線の切れ補修px", 0, 2, 0)

    main_line_max_lum = st.slider("主線: 最大明るさ", 60, 220, 145)

    main_line_color_mode = st.radio(
        "主線色",
        ["自動", "手動"],
        index=1,
        horizontal=True,
    )

    main_line_manual_color = st.color_picker("手動主線色", "#111318")

    st.subheader("細部線レイヤー")

    add_detail_line = st.checkbox(
        "細部線を別レイヤーで追加",
        value=P["detail_line"],
        help="ザラつきや軽量化が気になる場合はOFF推奨。",
    )

    detail_line_dark_threshold = st.slider("細部線: 暗さしきい値", 20, 220, 142)

    detail_line_contrast_threshold = st.slider("細部線: コントラストしきい値", 4, 100, 52)

    detail_line_width_adjust = st.slider("細部線の太さ調整", -3, 2, -1)

    detail_line_blur = st.slider(
        "細部線用ぼかし",
        0.0,
        2.0,
        0.65,
        0.05,
    )

    detail_line_open_px = st.slider("細部線ノイズ除去px", 0, 2, 1)

    detail_line_close_px = st.slider("細部線の切れ補修px", 0, 2, 0)

    detail_line_max_lum = st.slider("細部線: 最大明るさ", 80, 240, 178)

    detail_line_color_mode = st.radio(
        "細部線色",
        ["自動", "手動"],
        index=1,
        horizontal=True,
    )

    detail_line_manual_color = st.color_picker("手動細部線色", "#3b424a")

    cut_line_from_fills_px = st.slider(
        "塗りから線画部分を抜くpx",
        0,
        4,
        1,
        help="線が二重に太る時は1〜2。欠ける時は0。",
    )

    st.subheader("Potrace / 軽量化")

    fill_speckles = st.slider("塗り: 小ゴミ除去", 0, 100, 2)

    fill_smooth_corners = st.slider("塗り: 角の滑らかさ", 0.0, 2.0, 0.85, 0.05)

    fill_optimize = st.slider("塗り: パス最適化", 0.0, 5.0, float(P["fill_optimize"]), 0.02)

    line_speckles = st.slider("線: 小ゴミ除去", 0, 50, 1)

    line_smooth_corners = st.slider("線: 角の滑らかさ", 0.0, 2.0, 0.60, 0.05)

    line_optimize = st.slider("線: パス最適化", 0.0, 2.0, float(P["line_optimize"]), 0.02)

    under_speckles = st.slider("下地: 小ゴミ除去", 0, 100, 4)

    under_smooth_corners = st.slider("下地: 角の滑らかさ", 0.0, 2.0, 1.0, 0.05)

    under_optimize = st.slider("下地: パス最適化", 0.0, 5.0, 0.35, 0.05)

    precision_decimals = st.slider(
        "SVG座標精度",
        0,
        3,
        P["precision"],
        help="1でかなり軽く、2で品質寄り。0は軽いが崩れやすい。",
    )

    use_scour = st.checkbox("ScourでSVG軽量化", value=True)

    show_debug = st.checkbox("デバッグ表示", value=False)


if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded, caption="元画像", use_container_width=True)

    if st.button("ベクター化する", type="primary", use_container_width=True):
        manual_palette_colors = []

        for enabled, c in [
            (use_manual_1, manual_color_1),
            (use_manual_2, manual_color_2),
            (use_manual_3, manual_color_3),
            (use_manual_4, manual_color_4),
        ]:
            if enabled:
                rgb_c = hex_to_rgb(c)
                if rgb_c is not None:
                    manual_palette_colors.append(rgb_c)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            in_png = td / "input_clean.png"
            out_svg = td / "output.svg"

            rgba_img = normalize_rgba_image(
                uploaded_file=uploaded,
                out_png=in_png,
                max_side=max_side,
                alpha_threshold=alpha_threshold,
                harden_alpha=harden_alpha,
            )

            try:
                with st.spinner("SVG変換中..."):
                    svg_text, debug_palette, debug_masks = run_trace(
                        in_png=in_png,
                        out_svg=out_svg,
                        colors=colors,
                        alpha_threshold=alpha_threshold,
                        edge_trim_px=edge_trim_px,
                        remove_largest_color=remove_largest_color,
                        underpaint=underpaint,
                        underpaint_expand=underpaint_expand,
                        underpaint_color_mode=underpaint_color_mode,
                        underpaint_manual_color=underpaint_manual_color,
                        clip_fills_to_foreground=clip_fills_to_foreground,
                        color_layer_overlap=color_layer_overlap,
                        fill_open_px=fill_open_px,
                        fill_close_px=fill_close_px,
                        fill_smooth_median_px=fill_smooth_median_px,
                        fill_smooth_blur=fill_smooth_blur,
                        resample_color=resample_color,
                        color_sample_method=color_sample_method,
                        color_sample_source=color_sample_source,
                        vibrance=vibrance,
                        use_priority_palette=use_priority_palette,
                        keep_neutrals=keep_neutrals,
                        accent_count=accent_count,
                        light_tone_count=light_tone_count,
                        skin_tone_count=skin_tone_count,
                        shadow_tone_count=shadow_tone_count,
                        palette_merge_distance=palette_merge_distance,
                        min_color_area_px=min_color_area_px,
                        manual_palette_colors=manual_palette_colors,
                        add_main_line=add_main_line,
                        main_line_mode=main_line_mode,
                        main_line_dark_threshold=main_line_dark_threshold,
                        main_line_contrast_threshold=main_line_contrast_threshold,
                        main_line_width_adjust=main_line_width_adjust,
                        main_line_color_mode=main_line_color_mode,
                        main_line_manual_color=main_line_manual_color,
                        main_line_open_px=main_line_open_px,
                        main_line_close_px=main_line_close_px,
                        main_line_max_lum=main_line_max_lum,
                        main_line_blur=main_line_blur,
                        add_detail_line=add_detail_line,
                        detail_line_dark_threshold=detail_line_dark_threshold,
                        detail_line_contrast_threshold=detail_line_contrast_threshold,
                        detail_line_width_adjust=detail_line_width_adjust,
                        detail_line_color_mode=detail_line_color_mode,
                        detail_line_manual_color=detail_line_manual_color,
                        detail_line_open_px=detail_line_open_px,
                        detail_line_close_px=detail_line_close_px,
                        detail_line_max_lum=detail_line_max_lum,
                        detail_line_blur=detail_line_blur,
                        cut_line_from_fills_px=cut_line_from_fills_px,
                        fill_speckles=fill_speckles,
                        fill_smooth_corners=fill_smooth_corners,
                        fill_optimize=fill_optimize,
                        line_speckles=line_speckles,
                        line_smooth_corners=line_smooth_corners,
                        line_optimize=line_optimize,
                        under_speckles=under_speckles,
                        under_smooth_corners=under_smooth_corners,
                        under_optimize=under_optimize,
                        precision_decimals=precision_decimals,
                        use_scour=use_scour,
                    )

                out_svg.write_text(svg_text, encoding="utf-8")

                svg_bytes = svg_text.encode("utf-8")
                kb = len(svg_bytes) / 1024

                st.success(
                    f"完了: {rgba_img.size[0]}×{rgba_img.size[1]}px / {kb:.1f}KB"
                )

                with col2:
                    preview = svg_preview_png(svg_text)

                    if preview:
                        st.image(
                            preview,
                            caption="SVGプレビュー",
                            use_container_width=True,
                        )
                    else:
                        st.code(svg_text[:2500], language="xml")

                st.download_button(
                    "SVGをダウンロード",
                    data=svg_bytes,
                    file_name=make_download_name(uploaded.name, "anime_v18_clean"),
                    mime="image/svg+xml",
                    use_container_width=True,
                )

                if kb > 15:
                    st.warning(
                        "GT7の15KB制限には大きいです。"
                        "クリーン軽量プリセット、総色数↓、細部線OFF、SVG座標精度1、"
                        "塗り/線のパス最適化↑を試してください。"
                    )

                if show_debug:
                    with st.expander("デバッグ", expanded=True):
                        st.image(
                            in_png.as_posix(),
                            caption="alpha整理後の入力PNG",
                            use_container_width=True,
                        )

                        for name, mask in debug_masks.items():
                            st.image(
                                mask,
                                caption=name,
                                use_container_width=True,
                            )

                        if debug_palette is not None and len(debug_palette) > 0:
                            swatch_h = 54
                            swatch_w = max(54, 54 * len(debug_palette))
                            swatch = Image.new(
                                "RGB",
                                (swatch_w, swatch_h),
                                (255, 255, 255),
                            )

                            for i, c in enumerate(debug_palette):
                                block = Image.new(
                                    "RGB",
                                    (54, swatch_h),
                                    tuple(int(x) for x in c),
                                )
                                swatch.paste(block, (i * 54, 0))

                            st.image(
                                swatch,
                                caption="使用パレット",
                                use_container_width=True,
                            )

                with st.expander("SVGコードを見る"):
                    st.code(svg_text, language="xml")

            except Exception as e:
                st.error(str(e))

else:
    st.info("まず画像をアップロードしてください。")