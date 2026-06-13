import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter

st.set_page_config(page_title="Anime SVG Vectorizer 1.7", layout="wide")

SVG_NS = "http://www.w3.org/2000/svg"


# =========================================================
# basic helpers
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
# image / mask helpers
# =========================================================

def normalize_rgba_image(uploaded_file, out_png, max_side, alpha_threshold, harden_alpha):
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img).convert("RGBA")

    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    arr = np.array(img)
    alpha = arr[:, :, 3]

    # 白背景合成はしない。透明/半透明外周だけ整理する。
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
    """
    positive = expand
    negative = shrink
    """
    px = int(px)

    if px == 0:
        return mask_img

    size = abs(px) * 2 + 1

    if px > 0:
        return mask_img.filter(ImageFilter.MaxFilter(size))

    return mask_img.filter(ImageFilter.MinFilter(size))


def clean_mask(mask_img, open_px=0, close_px=0):
    # open: remove tiny noise
    # close: fill tiny holes/gaps
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

    if edge_trim_px > 0:
        mask = morph_mask(mask, -edge_trim_px)
    elif edge_trim_px < 0:
        mask = morph_mask(mask, abs(edge_trim_px))

    return bool_from_mask(mask)


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
# svg / potrace helpers
# =========================================================

def optimize_svg_text(svg_text):
    svg_text = re.sub(r"<\?xml[^>]*>\s*", "", svg_text)
    svg_text = re.sub(r"<!--.*?-->", "", svg_text, flags=re.S)
    svg_text = re.sub(r">\s+<", "><", svg_text).strip()

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
                return dst.read_text(encoding="utf-8", errors="ignore")
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

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        pbm = td / "mask.pbm"
        part_svg = td / "part.svg"

        arr = np.array(mask_img.convert("L"))

        if not np.any(arr > 0):
            return ""

        # Potraceは黒をトレース対象にする。
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
        & (sat >= 0.035)
        & (sat <= 0.56)
        & (lum >= 82)
        & (lum <= 245)
        & (r >= g * 0.93)
        & (g >= b * 0.70)
        & ((r - b) >= 6)
    )

    return skin_like


def build_palette(
    rgb,
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
    pixels = rgb[fg_mask]

    if len(pixels) == 0:
        raise RuntimeError("不透明ピクセルが見つかりません。alphaしきい値を下げてください。")

    base_pairs = quantize_palette_from_pixels(pixels, colors)
    base_colors = [c for c, count in base_pairs]

    if not use_priority:
        anchors = manual_colors
        return merge_palette_candidates(anchors, base_colors, colors, merge_distance)

    lum, sat, chroma = rgb_metrics(pixels)
    anchors = list(manual_colors)

    if keep_neutrals:
        dark_pixels = pixels[lum <= np.percentile(lum, 12)]

        if len(dark_pixels) > 0:
            anchors.append(tuple(np.median(dark_pixels, axis=0).astype(int)))

        light_mask = (lum >= np.percentile(lum, 84)) & (sat <= 0.34)
        light_pixels = pixels[light_mask]

        if len(light_pixels) > 0:
            anchors.append(tuple(np.median(light_pixels, axis=0).astype(int)))

        mid_neutral_mask = (sat <= 0.22) & (lum > 45) & (lum < 220)
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
        light_mask = (lum >= np.percentile(lum, 68)) & (sat <= 0.46)
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
            (lum >= 35)
            & (lum <= np.percentile(lum, 55))
            & (sat <= 0.55)
        )
        shadow_pairs = quantize_palette_from_pixels(pixels[shadow_mask], shadow_tone_count)
        anchors.extend([c for c, count in shadow_pairs])

    return merge_palette_candidates(anchors, base_colors, colors, merge_distance)


def assign_nearest_palette(rgb, fg_mask, palette):
    h, w, _ = rgb.shape
    labels = np.full((h, w), -1, dtype=np.int16)

    flat_rgb = rgb.reshape(-1, 3)
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


def sampled_fill_color(rgb, mask, method, vibrance=0):
    pixels = rgb[mask]

    if len(pixels) == 0:
        return "#000000"

    if method == "中央値":
        c = np.median(pixels, axis=0).astype(np.float32)
    else:
        c = np.mean(pixels, axis=0).astype(np.float32)

    # 少しだけ色を立たせる。0なら無加工。
    if vibrance != 0:
        gray = np.mean(c)
        factor = 1.0 + float(vibrance) / 100.0
        c = gray + (c - gray) * factor

    return color_hex(c.astype(int))


# =========================================================
# line / detail masks
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
):
    arr = np.array(rgba_img)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    fg = foreground_mask_from_alpha(alpha, alpha_threshold, edge_trim_px)

    lum, sat, chroma = rgb_metrics(rgb)

    gray = lum.astype(np.float32)
    gray_img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")

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
            | (very_dark & (local_contrast >= contrast_threshold * 0.50))
        )

    elif mode == "淡いグレー線も拾う":
        soft_dark = lum <= min(max_lum_allowed, dark_threshold + 48)
        low_sat = sat <= 0.45
        mask = fg & soft_dark & low_sat & edge_like

    else:
        # 細部専用。
        # 薄い服のシワや髪の細線を拾いやすいが、面ではなくエッジだけに寄せる。
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
# core trace
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
    color_layer_overlap,
    fill_open_px,
    fill_close_px,
    resample_color,
    color_sample_method,
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
    add_detail_line,
    detail_line_dark_threshold,
    detail_line_contrast_threshold,
    detail_line_width_adjust,
    detail_line_color_mode,
    detail_line_manual_color,
    detail_line_open_px,
    detail_line_close_px,
    detail_line_max_lum,
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
):
    if not shutil.which("potrace"):
        raise RuntimeError("potrace が見つかりません。packages.txt に potrace を入れてください。")

    img = Image.open(in_png).convert("RGBA")

    w, h = img.size

    arr = np.array(img)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    fg_mask = foreground_mask_from_alpha(alpha, alpha_threshold, edge_trim_px)

    if not fg_mask.any():
        raise RuntimeError("トレース対象がありません。alphaしきい値か外周トリムを下げてください。")

    palette = build_palette(
        rgb=rgb,
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

    labels = assign_nearest_palette(rgb, fg_mask, palette)

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
            group_id="main_line_top",
        )

        debug_masks["main_line_mask"] = mask

        if group:
            line_groups.append(group)
            line_cut_mask = (
                bool_from_mask(mask)
                if line_cut_mask is None
                else (line_cut_mask | bool_from_mask(mask))
            )

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
            speckles=max(0, min(line_speckles, 1)),
            smooth_corners=max(0.2, min(line_smooth_corners, 0.75)),
            optimize=min(line_optimize, 0.18),
            open_px=detail_line_open_px,
            close_px=detail_line_close_px,
            max_lum_allowed=detail_line_max_lum,
            group_id="detail_line_top",
        )

        debug_masks["detail_line_mask"] = mask

        if group:
            line_groups.append(group)
            line_cut_mask = (
                bool_from_mask(mask)
                if line_cut_mask is None
                else (line_cut_mask | bool_from_mask(mask))
            )

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

        if not mask_bool.any():
            continue

        mask_img = mask_from_bool(mask_bool)
        mask_img = clean_mask(mask_img, open_px=fill_open_px, close_px=fill_close_px)
        mask_img = morph_mask(mask_img, color_layer_overlap)

        final_mask_bool = bool_from_mask(mask_img)

        if not final_mask_bool.any():
            continue

        if resample_color:
            fill = sampled_fill_color(
                rgb,
                mask_bool,
                color_sample_method,
                vibrance=vibrance,
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

    svg = optimize_svg_text(svg)

    Path(out_svg).write_text(svg, encoding="utf-8")

    return svg, palette, debug_masks


# =========================================================
# UI
# =========================================================

st.title("Anime SVG Vectorizer 1.7")
st.caption(
    "謎の輪郭・線画太り・細部の粗さを詰める版。"
    "外周トリム、下地の拡張/縮小、線画/細部線の分離、肌色/薄色保護を調整できます。"
)

with st.expander("環境チェック", expanded=False):
    st.write(tool_versions())

uploaded = st.file_uploader(
    "画像をアップロード",
    type=["png", "jpg", "jpeg", "webp"],
)

with st.sidebar:
    st.header("基本")

    max_side = st.slider("