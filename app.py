import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Inkscape Auto Trace Vectorizer", layout="wide")

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


def normalize_image(uploaded_file, out_png, max_side, background):
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img).convert("RGBA")

    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    if background == "透明を白で合成":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg.convert("RGB")
    else:
        img = img.convert("RGBA")

    img.save(out_png)
    return img.size


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
                return dst.read_text(encoding="utf-8")
        except Exception:
            pass

    return svg_text


def run_inkscape_trace(
    in_png,
    out_svg,
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
    svg = optimize_svg_text(svg)
    Path(out_svg).write_text(svg, encoding="utf-8")
    return svg


def color_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0]),
        int(rgb[1]),
        int(rgb[2]),
    )


def extract_paths(svg_text, fill):
    paths = re.findall(r"<path\b[^>]*\bd=\"([^\"]+)\"[^>]*/?>", svg_text)
    out = []

    for d in paths:
        if len(d) < 5:
            continue
        out.append(f'<path d="{d}" fill="{fill}" stroke="none"/>')

    return "\n".join(out)


def run_potrace_color_fallback(
    in_png,
    out_svg,
    colors,
    remove_background,
    speckles,
    smooth_corners,
    optimize,
):
    if not shutil.which("potrace"):
        raise RuntimeError("potrace が見つかりません。packages.txt に potrace を入れてください。")

    img = Image.open(in_png).convert("RGBA")
    w, h = img.size

    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    rgb = bg.convert("RGB")

    q = rgb.quantize(colors=int(colors), method=Image.Quantize.MEDIANCUT)
    palette = q.getpalette()[: int(colors) * 3]
    pal = [tuple(palette[i : i + 3]) for i in range(0, len(palette), 3)]

    arr = np.array(q)
    counts = np.bincount(arr.reshape(-1), minlength=len(pal))

    order = list(np.argsort(-counts))

    if remove_background and order:
        order = order[1:]

    layer_paths = []

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        for idx in order:
            if idx >= len(pal) or counts[idx] < max(8, w * h * 0.00008):
                continue

            mask_arr = np.where(arr == idx, 0, 255).astype(np.uint8)
            mask = Image.fromarray(mask_arr, mode="L").convert("1")

            pbm = td / f"mask_{idx}.pbm"
            part_svg = td / f"part_{idx}.svg"

            mask.save(pbm)

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
                continue

            fill = color_hex(pal[idx])
            paths = extract_paths(
                part_svg.read_text(encoding="utf-8", errors="ignore"),
                fill,
            )

            if paths:
                layer_paths.append(
                    f'<g id="color_{idx}" data-count="{int(counts[idx])}">\n'
                    f"{paths}\n"
                    f"</g>"
                )

    svg = (
        f'<svg xmlns="{SVG_NS}" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" shape-rendering="geometricPrecision">\n'
        + "\n".join(layer_paths)
        + "\n</svg>"
    )

    svg = optimize_svg_text(svg)
    Path(out_svg).write_text(svg, encoding="utf-8")
    return svg


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


st.title("Inkscape Auto Trace Vectorizer")
st.caption(
    "画像をアップロード → InkscapeのCLIトレースを試行 → "
    "失敗時はPotraceで色レイヤーSVGを生成します。"
)

with st.expander("環境チェック", expanded=False):
    versions = tool_versions()
    st.write(versions)

    action, _ = find_inkscape_trace_action()
    st.write({"inkscape_trace_action": action or "not available"})

uploaded = st.file_uploader(
    "画像をアップロード",
    type=["png", "jpg", "jpeg", "webp"],
)

with st.sidebar:
    st.header("設定")

    engine = st.radio(
        "エンジン",
        ["Auto: Inkscape優先", "Inkscapeのみ", "Potraceのみ"],
        index=0,
    )

    max_side = st.slider("処理前の最大辺px", 512, 2400, 1200, 64)

    background = st.radio(
        "透明背景",
        ["透明を白で合成", "透明を保持"],
        index=0,
    )

    st.subheader("Inkscape Trace Bitmap")

    scans = st.slider("スキャン数 / 色数", 2, 32, 8)
    smooth = st.checkbox("Smooth", value=True)
    stack = st.checkbox("Stack", value=True)
    remove_background = st.checkbox("Remove background", value=True)
    speckles = st.slider("Speckles / 小ゴミ除去", 0, 100, 2)
    smooth_corners = st.slider("Smooth corners", 0.0, 2.0, 1.0, 0.05)
    optimize = st.slider("Optimize", 0.0, 5.0, 0.2, 0.05)

    st.subheader("Potrace fallback")

    fallback_colors = st.slider("fallback色数", 2, 24, 8)
    do_opt = st.checkbox("SVG軽量化", value=True)

if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded, caption="元画像", use_container_width=True)

    if st.button("ベクター化する", type="primary", use_container_width=True):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            in_png = td / "input.png"
            out_svg = td / "output.svg"

            size = normalize_image(uploaded, in_png, max_side, background)

            svg_text = None
            used = None
            errors = []

            if engine in ["Auto: Inkscape優先", "Inkscapeのみ"]:
                try:
                    with st.spinner("Inkscapeでトレース中..."):
                        svg_text = run_inkscape_trace(
                            in_png,
                            out_svg,
                            scans,
                            smooth,
                            stack,
                            remove_background,
                            speckles,
                            smooth_corners,
                            optimize,
                        )
                        used = "Inkscape object_trace"
                except Exception as e:
                    errors.append(str(e))

                    if engine == "Inkscapeのみ":
                        st.error(str(e))

            if svg_text is None and engine in ["Auto: Inkscape優先", "Potraceのみ"]:
                try:
                    with st.spinner("Potrace fallbackでトレース中..."):
                        svg_text = run_potrace_color_fallback(
                            in_png,
                            out_svg,
                            fallback_colors,
                            remove_background,
                            speckles,
                            smooth_corners,
                            optimize,
                        )
                        used = "Potrace color fallback"
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
                    f"完了: {used} / {size[0]}×{size[1]}px / {kb:.1f}KB"
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
                        st.code(svg_text[:2000], language="xml")

                st.download_button(
                    "SVGをダウンロード",
                    data=svg_bytes,
                    file_name=make_download_name(uploaded.name, "trace"),
                    mime="image/svg+xml",
                    use_container_width=True,
                )

                if kb > 15:
                    st.warning(
                        "GT7の15KB制限には大きいです。"
                        "最大辺px・色数・Optimizeを下げるか、"
                        "顔/髪/服などに分割して別SVGにしてください。"
                    )

                with st.expander("SVGコードを見る"):
                    st.code(svg_text, language="xml")

            if errors and engine.startswith("Auto"):
                with st.expander("Inkscape失敗ログ / fallback理由"):
                    for e in errors:
                        st.text(e[-3000:])
else:
    st.info("まず画像をアップロードしてください。")