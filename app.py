import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter

st.set_page_config(page_title="Anime SVG Vectorizer 1.5", layout="wide")

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def run_cmd(cmd, timeout=120):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout, p.stderr


def maybe_xvfb(cmd):
    if shutil.which("xvfb-run"):
        return ["xvfb-run", "-a", "-s", "-screen 0 1280x960x24"] + cmd
    return cmd


@st.cache_data(show_spinner=False)
def tool_versions():
    info = {}
    for exe in ["inkscape", "potrace"]:
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


@st.cache_data(show_spinner=False)
def find_inkscape_trace_action():
    if not shutil.which("inkscape"):
        return None, "inkscape not found"
    try:
        cmd = maybe_xvfb(["inkscape", "--action-list"])
        code, out, err = run_cmd(cmd, timeout=60)
        text = (out or "") + "\n" + (err or "")

        for name in ("object_trace", "object-trace"):
            if name in text:
                return name, text

        return None, text[-4000:]
    except Exception as e:
        return None, str(e)


def normalize_rgba_image(uploaded_file, out_png, max_side, alpha_threshold, harden_alpha):
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img).convert("RGBA")

    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    arr = np.array(img)
    alpha = arr[:, :, 3]

    # 白合成しない。透明/半透明外周をalphaで整理。
    arr[alpha < alpha_threshold, 3] = 0

    if harden_alpha:
        arr[arr[:, :, 3] >= alpha_threshold, 3] = 255

    cleaned = Image.fromarray(arr, mode="RGBA")
    cleaned.save(out_png)
    return cleaned


def foreground_mask(rgba_img, alpha_threshold):
    arr = np.array(rgba_img)
    return arr[:, :, 3] >= alpha_threshold


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


def color_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(np.clip(rgb[0], 0, 255)),
        int(np.clip(rgb[1], 0, 255)),
        int(np.clip(rgb[2], 0, 255)),
    )


def estimate_dark_color(rgba_img, mask, fallback="#11151b"):
    arr = np.array(rgba_img)
    rgb = arr[:, :, :3]

    pixels = rgb[mask]

    if len(pixels) == 0:
        return fallback

    lum = (
        pixels[:, 0].astype(np.float32) * 0.2126
        + pixels[:, 1].astype(np.float32) * 0.7152
        + pixels[:, 2].astype(np.float32) * 0.0722
    )

    cutoff = np.percentile(lum, 18)
    dark_pixels = pixels[lum <= cutoff]

    if len(dark_pixels) == 0:
        dark_pixels = pixels

    c = np.median(dark_pixels, axis=0).astype(int)
    c = np.clip(c, 8, 70)
    return color_hex(c)


def remove_raster_images(svg_text):
    try:
        root = ET.fromstring(svg_text.encode("utf-8"))
        parents = {child: parent for parent in root.iter() for child in parent}

        for el in list(root.iter()):
            if el.tag.endswith("image"):
                parent = parents.get(el)
                if parent is not None:
                    parent.remove(el)

        return ET.tostring(root, encoding="unicode")
    except Exception:
        return re.sub(
            r"<image\b[^>]*(?:/>|>.*?</image>)",
            "",
            svg_text,
            flags=re.S,
        )


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


def insert_group_after_svg_open(svg_text, group_text):
    if not group_text:
        return svg_text

    m = re.search(r"<svg\b[^>]*>", svg_text)

    if not m:
        return svg_text

    pos = m.end()
    return svg_text[:pos] + "\n" + group_text + "\n" + svg_text[pos:]


def svg_preview_png(svg_text, width=900):
    try:
        import cairosvg

        return cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=width,
        )
    except Exception:
        return None


def make_download_name(original_name, suffix):
    stem = Path(original_name).stem if original_name else "vectorized"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "vectorized"
    return f"{safe}_{suffix}.svg"


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

    return (
        f'<g{gid}{transform} fill="{fill}" stroke="none">\n'
        f"{body}\n"
        f"</g>"
    )


def trace_binary_mask_to_group(
    mask_img,
    fill,
    group_id,
    speckles,
    smooth_corners,
    optimize,
):
    if not shutil.which("potrace"):
        return ""

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        pbm = td / "mask.pbm"
        part_svg = td / "part.svg"

        # Potraceは黒をトレース対象にする。
        # mask_img は 255=対象, 0=背景。
        arr = np.array(mask_img.convert("L"))
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

        code, out, err = run_cmd(cmd, timeout=60)

        if code != 0 or not part_svg.exists():
            return ""

        svg_text = part_svg.read_text(encoding="utf-8", errors="ignore")
        return extract_potrace_group(svg_text, fill=fill, group_id=group_id)


def make_underpaint_group(
    rgba_img,
    alpha_threshold,
    expand_px,
    speckles,
    smooth_corners,
    optimize,
    color_mode,
    manual_color,
):
    fg = foreground_mask(rgba_img, alpha_threshold)

    if not fg.any():
        return ""

    mask = Image.fromarray((fg.astype(np.uint8) * 255), mode="L")

    if expand_px > 0:
        mask = mask.filter(ImageFilter.MaxFilter(expand_px * 2 + 1))

    if color_mode == "自動":
        fill = estimate_dark_color(rgba_img, np.array(mask) > 0)
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


def sample_pixels(pixels, max_sample=220_000, seed=1234):
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
        return [(tuple(pixels[0]), 1)]

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
        c = tuple(int(v) for v in raw[base : base + 3])

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
        nonlocal final

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


def build_generic_anime_palette(
    rgb,
    fg_mask,
    colors,
    use_priority,
    keep_neutrals,
    accent_count,
    merge_distance,
):
    pixels = rgb[fg_mask]

    if len(pixels) == 0:
        raise RuntimeError("不透明ピクセルが見つかりません。alphaしきい値を下げてください。")

    base_pairs = quantize_palette_from_pixels(pixels, colors)
    base_colors = [c for c, count in base_pairs]

    if not use_priority:
        return np.array(base_colors[:colors], dtype=np.uint8)

    lum, sat, chroma = rgb_metrics(pixels)
    anchors = []

    if keep_neutrals:
        # 黒線、白服、淡色肌、グレー影などを汎用的に保護。
        dark_cut = np.percentile(lum, 12)
        dark_pixels = pixels[lum <= dark_cut]

        if len(dark_pixels) > 0:
            anchors.append(tuple(np.median(dark_pixels, axis=0).astype(int)))

        light_mask = (lum >= np.percentile(lum, 82)) & (sat <= 0.32)
        light_pixels = pixels[light_mask]

        if len(light_pixels) > 0:
            anchors.append(tuple(np.median(light_pixels, axis=0).astype(int)))

        mid_neutral_mask = (sat <= 0.18) & (lum > 55) & (lum < 210)
        mid_neutral_pixels = pixels[mid_neutral_mask]

        if len(mid_neutral_pixels) > 0:
            anchors.append(tuple(np.median(mid_neutral_pixels, axis=0).astype(int)))

    if accent_count > 0:
        # シアン固定ではなく、高彩度アクセント色を汎用的に拾う。
        accent_mask = (
            (sat >= 0.34)
            & (chroma >= 38)
            & (lum >= 35)
            & (lum <= 240)
        )

        accent_pixels = pixels[accent_mask]
        accent_pairs = quantize_palette_from_pixels(accent_pixels, accent_count)

        anchors.extend([c for c, count in accent_pairs])

    palette = merge_palette_candidates(
        anchor_colors=anchors,
        base_colors=base_colors,
        max_colors=colors,
        merge_distance=merge_distance,
    )

    return palette


def assign_nearest_palette(rgb, fg_mask, palette):
    h, w, _ = rgb.shape

    labels = np.full((h, w), -1, dtype=np.int16)

    flat_rgb = rgb.reshape(-1, 3)
    flat_fg = fg_mask.reshape(-1)
    fg_indices = np.flatnonzero(flat_fg)

    palette_i = palette.astype(np.int32)
    flat_labels = labels.reshape(-1)

    chunk = 80_000

    for start in range(0, len(fg_indices), chunk):
        part_idx = fg_indices[start : start + chunk]
        part = flat_rgb[part_idx].astype(np.int32)

        diff = part[:, None, :] - palette_i[None, :, :]
        dist = np.sum(diff * diff, axis=2)

        flat_labels[part_idx] = np.argmin(dist, axis=1).astype(np.int16)

    return labels


def sampled_fill_color(rgb, mask, method):
    pixels = rgb[mask]

    if len(pixels) == 0:
        return "#000000"

    if method == "中央値":
        c = np.median(pixels, axis=0)
    else:
        c = np.mean(pixels, axis=0)

    return color_hex(c.astype(int))


def make_lineart_mask(
    rgba_img,
    alpha_threshold,
    mode,
    dark_threshold,
    contrast_threshold,
    line_width,
    remove_tiny_px,
):
    arr = np.array(rgba_img)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    fg = alpha >= alpha_threshold

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
    edge_like = (local_contrast >= contrast_threshold) | (
        edge >= contrast_threshold * 1.25
    )

    if mode == "エッジ優先・汎用":
        # 黒髪や黒服を全面線画化しにくい安全寄り。
        mask = fg & dark & edge_like

    elif mode == "黒インク強め":
        # 黒線を強く拾う。黒髪や黒服の内部も少し拾いやすい。
        very_dark = lum <= max(20, dark_threshold * 0.72)
        mask = fg & (
            (dark & edge_like)
            | (very_dark & (local_contrast >= contrast_threshold * 0.55))
        )

    else:
        # 白服の薄いシワやグレー線を拾いやすい。
        soft_dark = lum <= min(170, dark_threshold + 35)
        low_sat_shadow = sat <= 0.36
        mask = fg & soft_dark & low_sat_shadow & edge_like

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")

    if remove_tiny_px > 0:
        mask_img = mask_img.filter(ImageFilter.MinFilter(remove_tiny_px * 2 + 1))
        mask_img = mask_img.filter(ImageFilter.MaxFilter(remove_tiny_px * 2 + 1))

    if line_width > 0:
        mask_img = mask_img.filter(ImageFilter.MaxFilter(line_width * 2 + 1))

    return mask_img


def make_lineart_group(
    rgba_img,
    alpha_threshold,
    mode,
    dark_threshold,
    contrast_threshold,
    line_width,
    line_color_mode,
    manual_color,
    speckles,
    smooth_corners,
    optimize,
    remove_tiny_px,
):
    mask = make_lineart_mask(
        rgba_img=rgba_img,
        alpha_threshold=alpha_threshold,
        mode=mode,
        dark_threshold=dark_threshold,
        contrast_threshold=contrast_threshold,
        line_width=line_width,
        remove_tiny_px=remove_tiny_px,
    )

    mask_bool = np.array(mask) > 0

    if not mask_bool.any():
        return "", mask

    if line_color_mode == "自動":
        fill = estimate_dark_color(rgba_img, mask_bool, fallback="#101216")
    else:
        fill = manual_color

    group = trace_binary_mask_to_group(
        mask_img=mask,
        fill=fill,
        group_id="lineart_top",
        speckles=speckles,
        smooth_corners=smooth_corners,
        optimize=optimize,
    )

    return group, mask


def run_potrace_color_trace(
    in_png,
    out_svg,
    colors,
    alpha_threshold,
    remove_largest_color,
    underpaint,
    underpaint_expand,
    underpaint_color_mode,
    underpaint_manual_color,
    color_layer_overlap,
    resample_color,
    color_sample_method,
    use_priority_palette,
    keep_neutrals,
    accent_count,
    palette_merge_distance,
    add_lineart,
    line_mode,
    line_dark_threshold,
    line_contrast_threshold,
    line_width,
    line_color_mode,
    line_manual_color,
    line_remove_tiny_px,
    speckles,
    smooth_corners,
    optimize,
):
    if not shutil.which("potrace"):
        raise RuntimeError("potrace が見つかりません。packages.txt に potrace を入れてください。")

    img = Image.open(in_png).convert("RGBA")

    w, h = img.size

    arr = np.array(img)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    fg_mask = alpha >= alpha_threshold

    if not fg_mask.any():
        raise RuntimeError("トレース対象がありません。alphaしきい値を下げてください。")

    palette = build_generic_anime_palette(
        rgb=rgb,
        fg_mask=fg_mask,
        colors=colors,
        use_priority=use_priority_palette,
        keep_neutrals=keep_neutrals,
        accent_count=accent_count,
        merge_distance=palette_merge_distance,
    )

    labels = assign_nearest_palette(rgb, fg_mask, palette)

    counts = [int(np.sum(labels == i)) for i in range(len(palette))]
    order = list(np.argsort(-np.array(counts)))

    if remove_largest_color and order:
        order = order[1:]

    groups = []

    if underpaint:
        under_group = make_underpaint_group(
            rgba_img=img,
            alpha_threshold=alpha_threshold,
            expand_px=underpaint_expand,
            speckles=speckles,
            smooth_corners=smooth_corners,
            optimize=optimize,
            color_mode=underpaint_color_mode,
            manual_color=underpaint_manual_color,
        )

        if under_group:
            groups.append(under_group)

    min_area = max(8, int(w * h * 0.000035))

    for idx in order:
        if idx < 0 or idx >= len(palette):
            continue

        if counts[idx] < min_area:
            continue

        mask_bool = labels == idx

        if not mask_bool.any():
            continue

        mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")

        if color_layer_overlap > 0:
            mask_img = mask_img.filter(ImageFilter.MaxFilter(color_layer_overlap * 2 + 1))

        if resample_color:
            fill = sampled_fill_color(rgb, mask_bool, color_sample_method)
        else:
            fill = color_hex(palette[idx])

        group = trace_binary_mask_to_group(
            mask_img=mask_img,
            fill=fill,
            group_id=f"color_{idx}",
            speckles=speckles,
            smooth_corners=smooth_corners,
            optimize=optimize,
        )

        if group:
            groups.append(group)

    line_mask = None

    if add_lineart:
        line_group, line_mask = make_lineart_group(
            rgba_img=img,
            alpha_threshold=alpha_threshold,
            mode=line_mode,
            dark_threshold=line_dark_threshold,
            contrast_threshold=line_contrast_threshold,
            line_width=line_width,
            line_color_mode=line_color_mode,
            manual_color=line_manual_color,
            speckles=max(0, min(speckles, 2)),
            smooth_corners=max(0.4, min(smooth_corners, 1.2)),
            optimize=min(optimize, 0.35),
            remove_tiny_px=line_remove_tiny_px,
        )

        if line_group:
            groups.append(line_group)

    svg = (
        f'<svg xmlns="{SVG_NS}" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" shape-rendering="geometricPrecision">\n'
        + "\n".join(groups)
        + "\n</svg>"
    )

    svg = optimize_svg_text(svg)
    Path(out_svg).write_text(svg, encoding="utf-8")

    return svg, line_mask, palette


def run_inkscape_trace(
    in_png,
    out_svg,
    rgba_img,
    alpha_threshold,
    add_underpaint,
    underpaint_expand,
    underpaint_color_mode,
    underpaint_manual_color,
    add_lineart,
    line_mode,
    line_dark_threshold,
    line_contrast_threshold,
    line_width,
    line_color_mode,
    line_manual_color,
    line_remove_tiny_px,
    scans,
    smooth,
    stack,
    remove_background,
    speckles,
    smooth_corners,
    optimize,
):
    action, action_log = find_inkscape_trace_action()

    if not action:
        raise RuntimeError(
            "この環境のInkscapeにCLIトレース action が見つかりません。"
            "Inkscape 1.4以上が必要です。Potraceフォールバックを使ってください。\n\n"
            + str(action_log)[-1200:]
        )

    params = ",".join(
        [
            str(int(scans)),
            "true" if smooth else "false",
            "true" if stack else "false",
            "true" if remove_background else "false",
            str(int(speckles)),
            str(float(smooth_corners)),
            str(float(optimize)),
        ]
    )

    actions = f"select-all;{action}:{params};export-filename:{out_svg};export-do"

    cmd = maybe_xvfb(
        [
            "inkscape",
            str(in_png),
            f"--actions={actions}",
            "--export-overwrite",
        ]
    )

    code, out, err = run_cmd(cmd, timeout=180)

    if code != 0 or not Path(out_svg).exists() or Path(out_svg).stat().st_size == 0:
        raise RuntimeError(
            "Inkscape trace failed.\n\nSTDOUT:\n"
            + out
            + "\n\nSTDERR:\n"
            + err
        )

    svg = Path(out_svg).read_text(encoding="utf-8", errors="ignore")
    svg = remove_raster_images(svg)

    if add_underpaint:
        under_group = make_underpaint_group(
            rgba_img=rgba_img,
            alpha_threshold=alpha_threshold,
            expand_px=underpaint_expand,
            speckles=speckles,
            smooth_corners=smooth_corners,
            optimize=optimize,
            color_mode=underpaint_color_mode,
            manual_color=underpaint_manual_color,
        )

        svg = insert_group_after_svg_open(svg, under_group)

    if add_lineart:
        line_group, line_mask = make_lineart_group(
            rgba_img=rgba_img,
            alpha_threshold=alpha_threshold,
            mode=line_mode,
            dark_threshold=line_dark_threshold,
            contrast_threshold=line_contrast_threshold,
            line_width=line_width,
            line_color_mode=line_color_mode,
            manual_color=line_manual_color,
            speckles=max(0, min(speckles, 2)),
            smooth_corners=max(0.4, min(smooth_corners, 1.2)),
            optimize=min(optimize, 0.35),
            remove_tiny_px=line_remove_tiny_px,
        )

        if line_group:
            svg = svg.replace("</svg>", line_group + "\n</svg>")

    svg = optimize_svg_text(svg)
    Path(out_svg).write_text(svg, encoding="utf-8")

    return svg


st.title("Anime SVG Vectorizer 1.5")
st.caption(
    "汎用アニメ絵向け。白フチ対策、線画レイヤー最上段追加、"
    "黒/白/グレー/高彩度アクセント色の保護を入れた版です。"
)

with st.expander("環境チェック", expanded=False):
    st.write(tool_versions())
    action, _ = find_inkscape_trace_action()
    st.write({"inkscape_trace_action": action or "not available"})

uploaded = st.file_uploader(
    "画像をアップロード",
    type=["png", "jpg", "jpeg", "webp"],
)

with st.sidebar:
    st.header("基本")

    engine = st.radio(
        "エンジン",
        ["Auto: Inkscape優先", "Inkscapeのみ", "Potraceのみ"],
        index=2,
    )

    max_side = st.slider("処理前の最大辺px", 512, 2400, 1200, 64)

    st.subheader("白フチ対策")

    alpha_threshold = st.slider(
        "alphaしきい値",
        0,
        254,
        60,
        help="外周の白/灰色フチが出るなら上げる。輪郭が欠けるなら下げる。",
    )

    harden_alpha = st.checkbox("半透明の縁を締める", value=True)

    underpaint = st.checkbox("下地シルエットを敷く", value=True)

    underpaint_expand = st.slider("下地の拡張px", 0, 6, 1)

    underpaint_color_mode = st.radio(
        "下地色",
        ["自動", "手動"],
        index=1,
        horizontal=True,
    )

    underpaint_manual_color = st.color_picker("手動下地色", "#10151b")

    color_layer_overlap = st.slider("色レイヤーの重ねpx", 0, 3, 1)

    st.subheader("汎用パレット補正")

    fallback_colors = st.slider("色数", 2, 36, 14)

    use_priority_palette = st.checkbox(
        "黒・白・アクセント色を優先保持",
        value=True,
    )

    keep_neutrals = st.checkbox("黒/白/グレー系を保護", value=True)

    accent_count = st.slider("アクセント色の保護数", 0, 8, 3)

    palette_merge_distance = st.slider("近い色を統合する強さ", 0, 80, 26)

    resample_color = st.checkbox(
        "元画像から塗り色を取り直す",
        value=True,
    )

    color_sample_method = st.radio(
        "色の取り方",
        ["中央値", "平均"],
        index=0,
        horizontal=True,
    )

    remove_largest_color = st.checkbox(
        "最大面積色を背景として削除",
        value=False,
    )

    st.subheader("線画レイヤー")

    add_lineart = st.checkbox("線画を最上段に追加", value=True)

    line_mode = st.radio(
        "線画抽出モード",
        ["エッジ優先・汎用", "黒インク強め", "淡いグレー線も拾う"],
        index=0,
    )

    line_dark_threshold = st.slider("線画: 暗さしきい値", 20, 180, 105)

    line_contrast_threshold = st.slider("線画: コントラストしきい値", 4, 90, 24)

    line_width = st.slider("線画の太さ補正px", 0, 3, 1)

    line_remove_tiny_px = st.slider("線画ノイズ除去", 0, 2, 0)

    line_color_mode = st.radio(
        "線画色",
        ["自動", "手動"],
        index=1,
        horizontal=True,
    )

    line_manual_color = st.color_picker("手動線画色", "#111318")

    st.subheader("Inkscape Trace Bitmap")

    scans = st.slider("Inkscape スキャン数", 2, 32, 8)
    smooth = st.checkbox("Smooth", value=True)
    stack = st.checkbox("Stack", value=True)
    remove_background = st.checkbox("Inkscape Remove background", value=True)

    st.subheader("Potrace")

    speckles = st.slider("小ゴミ除去", 0, 100, 1)

    smooth_corners = st.slider("角の滑らかさ", 0.0, 2.0, 0.85, 0.05)

    optimize = st.slider("パス最適化", 0.0, 5.0, 0.20, 0.05)

    do_opt = st.checkbox("SVG軽量化", value=True)

    show_debug = st.checkbox("デバッグ表示", value=False)

if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded, caption="元画像", use_container_width=True)

    if st.button("ベクター化する", type="primary", use_container_width=True):
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

            svg_text = None
            used = None
            errors = []
            debug_line_mask = None
            debug_palette = None

            if engine in ["Auto: Inkscape優先", "Inkscapeのみ"]:
                try:
                    with st.spinner("Inkscapeでトレース中..."):
                        svg_text = run_inkscape_trace(
                            in_png=in_png,
                            out_svg=out_svg,
                            rgba_img=rgba_img,
                            alpha_threshold=alpha_threshold,
                            add_underpaint=underpaint,
                            underpaint_expand=underpaint_expand,
                            underpaint_color_mode=underpaint_color_mode,
                            underpaint_manual_color=underpaint_manual_color,
                            add_lineart=add_lineart,
                            line_mode=line_mode,
                            line_dark_threshold=line_dark_threshold,
                            line_contrast_threshold=line_contrast_threshold,
                            line_width=line_width,
                            line_color_mode=line_color_mode,
                            line_manual_color=line_manual_color,
                            line_remove_tiny_px=line_remove_tiny_px,
                            scans=scans,
                            smooth=smooth,
                            stack=stack,
                            remove_background=remove_background,
                            speckles=speckles,
                            smooth_corners=smooth_corners,
                            optimize=optimize,
                        )

                        used = "Inkscape + underpaint + lineart"

                except Exception as e:
                    errors.append(str(e))

                    if engine == "Inkscapeのみ":
                        st.error(str(e))

            if svg_text is None and engine in ["Auto: Inkscape優先", "Potraceのみ"]:
                try:
                    with st.spinner("Potraceで汎用アニメ補正トレース中..."):
                        svg_text, debug_line_mask, debug_palette = run_potrace_color_trace(
                            in_png=in_png,
                            out_svg=out_svg,
                            colors=fallback_colors,
                            alpha_threshold=alpha_threshold,
                            remove_largest_color=remove_largest_color,
                            underpaint=underpaint,
                            underpaint_expand=underpaint_expand,
                            underpaint_color_mode=underpaint_color_mode,
                            underpaint_manual_color=underpaint_manual_color,
                            color_layer_overlap=color_layer_overlap,
                            resample_color=resample_color,
                            color_sample_method=color_sample_method,
                            use_priority_palette=use_priority_palette,
                            keep_neutrals=keep_neutrals,
                            accent_count=accent_count,
                            palette_merge_distance=palette_merge_distance,
                            add_lineart=add_lineart,
                            line_mode=line_mode,
                            line_dark_threshold=line_dark_threshold,
                            line_contrast_threshold=line_contrast_threshold,
                            line_width=line_width,
                            line_color_mode=line_color_mode,
                            line_manual_color=line_manual_color,
                            line_remove_tiny_px=line_remove_tiny_px,
                            speckles=speckles,
                            smooth_corners=smooth_corners,
                            optimize=optimize,
                        )

                        used = "Potrace + generic palette + lineart"

                except Exception as e:
                    errors.append(str(e))
                    st.error(str(e))

            if svg_text:
                if do_opt:
                    svg_text = optimize_svg_text(svg_text)
                    out_svg.write_text(svg_text, encoding="utf-8")

                svg_bytes = svg_text.encode("utf-8")
                kb = len(svg_bytes) / 1024

                st.success(
                    f"完了: {used} / {rgba_img.size[0]}×{rgba_img.size[1]}px / {kb:.1f}KB"
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
                    file_name=make_download_name(uploaded.name, "anime_v15"),
                    mime="image/svg+xml",
                    use_container_width=True,
                )

                if kb > 15:
                    st.warning(
                        "GT7の15KB制限には大きいです。"
                        "色数・最大辺px・線画の太さ・重ねpxを下げるか、"
                        "次段階でパーツ分割してください。"
                    )

                if show_debug:
                    with st.expander("デバッグ"):
                        st.image(
                            in_png.as_posix(),
                            caption="alpha整理後の入力PNG",
                            use_container_width=True,
                        )

                        if debug_line_mask is not None:
                            st.image(
                                debug_line_mask,
                                caption="線画マスク",
                                use_container_width=True,
                            )

                        if debug_palette is not None:
                            swatch_h = 48
                            swatch_w = 48 * len(debug_palette)
                            swatch = Image.new(
                                "RGB",
                                (max(swatch_w, 48), swatch_h),
                                (255, 255, 255),
                            )

                            for i, c in enumerate(debug_palette):
                                block = Image.new(
                                    "RGB",
                                    (48, swatch_h),
                                    tuple(int(x) for x in c),
                                )
                                swatch.paste(block, (i * 48, 0))

                            st.image(
                                swatch,
                                caption="使用パレット",
                                use_container_width=True,
                            )

                with st.expander("SVGコードを見る"):
                    st.code(svg_text, language="xml")

            if errors and engine.startswith("Auto"):
                with st.expander("失敗ログ / fallback理由"):
                    for e in errors:
                        st.text(e[-3000:])
else:
    st.info("まず画像をアップロードしてください。")