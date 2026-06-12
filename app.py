import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter

st.set_page_config(page_title="White Fringe Fixed Vectorizer", layout="wide")

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


# =========================
# command helpers
# =========================

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


# =========================
# image preprocessing
# =========================

def normalize_rgba_image(
    uploaded_file,
    out_png,
    max_side,
    alpha_threshold,
    harden_alpha,
):
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img).convert("RGBA")

    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    arr = np.array(img)
    alpha = arr[:, :, 3]

    # ここが白フチ対策の核心。
    # 透明部分を白で合成しない。
    # alpha_threshold 未満は完全透明として扱う。
    arr[alpha < alpha_threshold, 3] = 0

    if harden_alpha:
        arr[arr[:, :, 3] >= alpha_threshold, 3] = 255

    cleaned = Image.fromarray(arr, mode="RGBA")
    cleaned.save(out_png)

    return cleaned


def get_foreground_mask(rgba_img, alpha_threshold):
    arr = np.array(rgba_img)
    return arr[:, :, 3] >= alpha_threshold


def estimate_dark_underpaint_color(rgba_img, fg_mask):
    arr = np.array(rgba_img)
    rgb = arr[:, :, :3]

    pixels = rgb[fg_mask]
    if len(pixels) == 0:
        return "#20242a"

    lum = (
        pixels[:, 0].astype(np.float32) * 0.2126
        + pixels[:, 1].astype(np.float32) * 0.7152
        + pixels[:, 2].astype(np.float32) * 0.0722
    )

    cutoff = np.percentile(lum, 20)
    dark_pixels = pixels[lum <= cutoff]

    if len(dark_pixels) == 0:
        dark_pixels = pixels

    color = np.median(dark_pixels, axis=0).astype(int)

    # 真っ黒すぎるとGT7上で潰れやすいので少しだけ持ち上げる
    color = np.clip(color, 12, 80)

    return "#{:02x}{:02x}{:02x}".format(
        int(color[0]),
        int(color[1]),
        int(color[2]),
    )


def color_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(np.clip(rgb[0], 0, 255)),
        int(np.clip(rgb[1], 0, 255)),
        int(np.clip(rgb[2], 0, 255)),
    )


# =========================
# SVG helpers
# =========================

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


# =========================
# Potrace helpers
# =========================

def extract_potrace_group(svg_text, fill, group_id=None):
    """
    PotraceのSVGはtransform付きgにpathが入ることが多い。
    pathだけ抜くと上下反転/倍率ズレが起きる環境があるので、
    transformを維持してpathを取り出す。
    """
    group_match = re.search(r"<g\b([^>]*)>(.*?)</g>", svg_text, flags=re.S)

    transform = ""
    inner = svg_text

    if group_match:
        attrs = group_match.group(1)
        inner = group_match.group(2)
        t = re.search(r'transform="([^"]+)"', attrs)
        if t:
            transform = f' transform="{t.group(1)}"'

    paths = re.findall(r"<path\b[^>]*\bd=\"([^\"]+)\"[^>]*/?>", inner)

    if not paths:
        paths = re.findall(r"<path\b[^>]*\bd=\"([^\"]+)\"[^>]*>", inner)

    if not paths:
        return ""

    gid = f' id="{group_id}"' if group_id else ""

    body = "\n".join(
        f'<path d="{d}"/>'
        for d in paths
        if len(d) > 5
    )

    if not body:
        return ""

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
        # mask_img: 255=対象, 0=背景
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
    fg = get_foreground_mask(rgba_img, alpha_threshold)

    if not fg.any():
        return ""

    mask = Image.fromarray((fg.astype(np.uint8) * 255), mode="L")

    if expand_px > 0:
        size = expand_px * 2 + 1
        mask = mask.filter(ImageFilter.MaxFilter(size))

    if color_mode == "自動":
        fill = estimate_dark_underpaint_color(rgba_img, np.array(mask) > 0)
    else:
        fill = manual_color

    group = trace_binary_mask_to_group(
        mask_img=mask,
        fill=fill,
        group_id="underpaint",
        speckles=max(0, speckles),
        smooth_corners=smooth_corners,
        optimize=optimize,
    )

    return group


# =========================
# palette / color layers
# =========================

def make_palette_from_foreground(rgb, fg_mask, colors):
    pixels = rgb[fg_mask]

    if len(pixels) == 0:
        raise RuntimeError("不透明ピクセルが見つかりません。alphaしきい値を下げてください。")

    max_sample = 200_000

    if len(pixels) > max_sample:
        rng = np.random.default_rng(1234)
        idx = rng.choice(len(pixels), size=max_sample, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    sample = sample.astype(np.uint8)

    sample_img = Image.fromarray(sample.reshape((len(sample), 1, 3)), mode="RGB")
    q = sample_img.quantize(colors=int(colors), method=Image.Quantize.MEDIANCUT)

    raw = q.getpalette()[: int(colors) * 3]
    palette = []

    for i in range(0, len(raw), 3):
        c = tuple(raw[i : i + 3])
        if len(c) == 3 and c not in palette:
            palette.append(c)

    if not palette:
        raise RuntimeError("パレット生成に失敗しました。")

    return np.array(palette, dtype=np.uint8)


def assign_nearest_palette(rgb, fg_mask, palette):
    h, w, _ = rgb.shape
    labels = np.full((h, w), -1, dtype=np.int16)

    flat_rgb = rgb.reshape(-1, 3)
    flat_fg = fg_mask.reshape(-1)
    fg_indices = np.flatnonzero(flat_fg)

    palette_i = palette.astype(np.int16)

    chunk = 150_000

    for start in range(0, len(fg_indices), chunk):
        part_idx = fg_indices[start : start + chunk]
        part = flat_rgb[part_idx].astype(np.int16)

        diff = part[:, None, :] - palette_i[None, :, :]
        dist = np.sum(diff * diff, axis=2)
        nearest = np.argmin(dist, axis=1).astype(np.int16)

        labels.reshape(-1)[part_idx] = nearest

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


def run_potrace_color_fallback(
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

    palette = make_palette_from_foreground(rgb, fg_mask, colors)
    labels = assign_nearest_palette(rgb, fg_mask, palette)

    counts = []
    for i in range(len(palette)):
        counts.append(int(np.sum(labels == i)))

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

    min_area = max(8, int(w * h * 0.00004))

    for idx in order:
        if idx < 0 or idx >= len(palette):
            continue

        if counts[idx] < min_area:
            continue

        mask_bool = labels == idx

        if not mask_bool.any():
            continue

        mask = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")

        # 色レイヤーを少し重ねる。
        # これで色面同士の細い白いスキマがかなり減る。
        if color_layer_overlap > 0:
            size = color_layer_overlap * 2 + 1
            mask = mask.filter(ImageFilter.MaxFilter(size))

        expanded_mask_bool = np.array(mask) > 0

        if resample_color:
            fill = sampled_fill_color(
                rgb=rgb,
                mask=mask_bool,
                method=color_sample_method,
            )
        else:
            fill = color_hex(palette[idx])

        group = trace_binary_mask_to_group(
            mask_img=mask,
            fill=fill,
            group_id=f"color_{idx}",
            speckles=speckles,
            smooth_corners=smooth_corners,
            optimize=optimize,
        )

        if group:
            groups.append(group)

    svg = (
        f'<svg xmlns="{SVG_NS}" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" shape-rendering="geometricPrecision">\n'
        + "\n".join(groups)
        + "\n</svg>"
    )

    svg = optimize_svg_text(svg)
    Path(out_svg).write_text(svg, encoding="utf-8")
    return svg


# =========================
# Inkscape trace
# =========================

def run_inkscape_trace(
    in_png,
    out_svg,
    rgba_img,
    alpha_threshold,
    add_underpaint,
    underpaint_expand,
    underpaint_color_mode,
    underpaint_manual_color,
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

    svg = optimize_svg_text(svg)
    Path(out_svg).write_text(svg, encoding="utf-8")
    return svg


# =========================
# UI
# =========================

st.title("White Fringe Fixed Vectorizer")
st.caption(
    "白フチ対策版。透明を白合成せず、alphaを使って色分解し、"
    "下地シルエットと色レイヤー重ねでスキマを減らします。"
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
        index=0,
    )

    max_side = st.slider("処理前の最大辺px", 512, 2400, 1200, 64)

    st.subheader("白フチ対策")

    alpha_threshold = st.slider(
        "alphaしきい値",
        0,
        254,
        72,
        help="透明〜半透明の縁をどこまで捨てるか。白フチが出るなら上げる。欠けるなら下げる。",
    )

    harden_alpha = st.checkbox(
        "半透明の縁を締める",
        value=True,
        help="ONにすると、残す部分を完全不透明化します。白っぽい半透明フチ対策に有効。",
    )

    underpaint = st.checkbox(
        "下地シルエットを敷く",
        value=True,
        help="キャラ全体の下に少し大きい暗色シルエットを置いて、色面のスキマを隠します。",
    )

    underpaint_expand = st.slider(
        "下地の拡張px",
        0,
        6,
        2,
        help="白い隙間が目立つなら1〜3がおすすめ。",
    )

    underpaint_color_mode = st.radio(
        "下地色",
        ["自動", "手動"],
        index=0,
        horizontal=True,
    )

    underpaint_manual_color = st.color_picker(
        "手動下地色",
        "#20242a",
    )

    color_layer_overlap = st.slider(
        "色レイヤーの重ねpx",
        0,
        3,
        1,
        help="色ごとの境界に出る白い線を減らします。強すぎると細部が太ります。",
    )

    st.subheader("色")

    fallback_colors = st.slider("色数", 2, 32, 10)

    resample_color = st.checkbox(
        "元画像から塗り色を取り直す",
        value=True,
        help="量子化後の色ではなく、元画像の実ピクセルから塗り色を再計算します。",
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
        help="透明PNGなら基本OFF。JPG背景が混ざる時だけON。",
    )

    st.subheader("Inkscape Trace Bitmap")

    scans = st.slider("Inkscape スキャン数", 2, 32, 8)
    smooth = st.checkbox("Smooth", value=True)
    stack = st.checkbox("Stack", value=True)
    remove_background = st.checkbox("Inkscape Remove background", value=True)

    st.subheader("Potrace")

    speckles = st.slider("小ゴミ除去", 0, 100, 2)
    smooth_corners = st.slider("角の滑らかさ", 0.0, 2.0, 1.0, 0.05)
    optimize = st.slider("パス最適化", 0.0, 5.0, 0.25, 0.05)

    do_opt = st.checkbox("SVG軽量化", value=True)


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
                            scans=scans,
                            smooth=smooth,
                            stack=stack,
                            remove_background=remove_background,
                            speckles=speckles,
                            smooth_corners=smooth_corners,
                            optimize=optimize,
                        )
                        used = "Inkscape object_trace + white-fringe fix"
                except Exception as e:
                    errors.append(str(e))

                    if engine == "Inkscapeのみ":
                        st.error(str(e))

            if svg_text is None and engine in ["Auto: Inkscape優先", "Potraceのみ"]:
                try:
                    with st.spinner("Potraceで白フチ対策トレース中..."):
                        svg_text = run_potrace_color_fallback(
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
                            speckles=speckles,
                            smooth_corners=smooth_corners,
                            optimize=optimize,
                        )
                        used = "Potrace alpha-safe color trace"
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
                    file_name=make_download_name(uploaded.name, "fringe_fixed"),
                    mime="image/svg+xml",
                    use_container_width=True,
                )

                if kb > 15:
                    st.warning(
                        "GT7の15KB制限にはまだ大きいです。"
                        "色数・最大辺px・下地拡張・色レイヤー重ねを調整するか、"
                        "次段階でパーツ分割してください。"
                    )

                with st.expander("SVGコードを見る"):
                    st.code(svg_text, language="xml")

            if errors and engine.startswith("Auto"):
                with st.expander("失敗ログ / fallback理由"):
                    for e in errors:
                        st.text(e[-3000:])
else:
    st.info("まず画像をアップロードしてください。")