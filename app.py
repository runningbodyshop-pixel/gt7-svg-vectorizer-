import io
import zipfile
import base64
import html
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


# ============================================================
# Smartphone-first path-only SVG vectorizer
# - Upload raster image
# - Quantize colors
# - Extract colored regions as SVG <path>
# - Sort layers by largest area -> smallest area
# - Fit result toward total 100 KB / part 15 KB
# ============================================================


@dataclass
class Shape:
    area: float
    fill: str
    d: str


def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [int(max(0, min(255, x))) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def prepare_image(uploaded_file, max_side: int, background_rgb=(255, 255, 255)) -> Image.Image:
    img = Image.open(uploaded_file)

    # Flatten transparency because this exporter creates opaque path fills.
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, background_rgb + (255,))
        bg.alpha_composite(img.convert("RGBA"))
        img = bg.convert("RGB")
    else:
        img = img.convert("RGB")

    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def quantize_image_kmeans(arr: np.ndarray, color_count: int, sample_limit: int = 45000):
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3).astype(np.float32)

    unique_count = min(color_count, len(np.unique(pixels.astype(np.uint8), axis=0)))
    color_count = max(2, int(unique_count))

    rng = np.random.default_rng(1234)
    if len(pixels) > sample_limit:
        idx = rng.choice(len(pixels), sample_limit, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    cv2.setRNGSeed(1234)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        35,
        0.6,
    )

    _, _, centers = cv2.kmeans(
        sample,
        color_count,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = np.clip(centers, 0, 255).astype(np.float32)

    # Assign every pixel to nearest center in chunks to avoid high memory use.
    labels = np.empty((len(pixels),), dtype=np.int32)
    chunk_size = 60000
    for start in range(0, len(pixels), chunk_size):
        chunk = pixels[start:start + chunk_size]
        distances = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels[start:start + chunk_size] = np.argmin(distances, axis=1)

    label_img = labels.reshape(h, w)
    centers_u8 = centers.astype(np.uint8)

    return label_img, centers_u8


def guess_corner_background_label(label_img: np.ndarray) -> int:
    h, w = label_img.shape
    corner = max(4, min(h, w) // 20)

    samples = np.concatenate([
        label_img[:corner, :corner].ravel(),
        label_img[:corner, -corner:].ravel(),
        label_img[-corner:, :corner].ravel(),
        label_img[-corner:, -corner:].ravel(),
    ])

    labels, counts = np.unique(samples, return_counts=True)
    return int(labels[np.argmax(counts)])


def fmt_num(x: float) -> str:
    return str(int(round(float(x))))


def contour_to_path(contour) -> str:
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return ""

    parts = [f"M{fmt_num(pts[0][0])},{fmt_num(pts[0][1])}"]
    for x, y in pts[1:]:
        parts.append(f"L{fmt_num(x)},{fmt_num(y)}")
    parts.append("Z")
    return "".join(parts)


def extract_shapes(
    label_img: np.ndarray,
    centers: np.ndarray,
    detail: int,
    min_area_px: int,
    remove_background: bool,
    smooth_masks: bool,
    force_epsilon_multiplier: float = 1.0,
    force_min_area_multiplier: float = 1.0,
) -> List[Shape]:
    h, w = label_img.shape

    bg_label = guess_corner_background_label(label_img) if remove_background else None

    # Higher detail = lower simplification.
    base_epsilon = max(0.35, (101 - detail) * 0.035)
    epsilon = base_epsilon * force_epsilon_multiplier
    effective_min_area = max(1, int(min_area_px * force_min_area_multiplier))

    shapes: List[Shape] = []

    kernel = np.ones((3, 3), np.uint8)

    for label_id, rgb in enumerate(centers):
        if remove_background and label_id == bg_label:
            continue

        mask = (label_img == label_id).astype(np.uint8) * 255

        if smooth_masks:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if hierarchy is None:
            continue

        hierarchy = hierarchy[0]
        fill = rgb_to_hex(tuple(int(x) for x in rgb))

        for i, contour in enumerate(contours):
            parent = hierarchy[i][3]
            if parent != -1:
                continue

            area = abs(cv2.contourArea(contour))
            if area < effective_min_area:
                continue

            approx = cv2.approxPolyDP(contour, epsilon, True)
            d = contour_to_path(approx)

            # Add direct holes as subpaths. fill-rule="evenodd" will cut them out.
            child = hierarchy[i][2]
            while child != -1:
                hole = contours[child]
                hole_area = abs(cv2.contourArea(hole))
                if hole_area >= effective_min_area:
                    hole_approx = cv2.approxPolyDP(hole, epsilon, True)
                    d += contour_to_path(hole_approx)
                child = hierarchy[child][0]

            if d:
                shapes.append(Shape(area=area, fill=fill, d=d))

    # Largest area first. Smaller details are drawn later on top.
    shapes.sort(key=lambda s: s.area, reverse=True)
    return shapes


def make_svg(
    shapes: List[Shape],
    width: int,
    height: int,
    title: str = "path_layer_vector",
    include_comments: bool = False,
) -> str:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" shape-rendering="geometricPrecision">',
    ]

    if include_comments:
        lines.append(f"<title>{html.escape(title)}</title>")

    for i, shape in enumerate(shapes, start=1):
        # Every vector object is path data only.
        # Groups are just layer containers.
        lines.append(
            f'<g id="layer_{i:04d}_area_{int(shape.area)}">'
            f'<path d="{shape.d}" fill="{shape.fill}" fill-rule="evenodd"/>'
            f"</g>"
        )

    lines.append("</svg>")
    return "".join(lines)


def fit_shapes_to_budget(
    label_img: np.ndarray,
    centers: np.ndarray,
    width: int,
    height: int,
    detail: int,
    min_area_px: int,
    remove_background: bool,
    smooth_masks: bool,
    total_limit_bytes: int,
):
    best_shapes: List[Shape] = []
    best_svg = ""

    # First try progressively stronger simplification.
    for attempt in range(18):
        eps_mul = 1.0 + (attempt * 0.22)
        area_mul = 1.0 + (attempt * 0.50)

        shapes = extract_shapes(
            label_img=label_img,
            centers=centers,
            detail=detail,
            min_area_px=min_area_px,
            remove_background=remove_background,
            smooth_masks=smooth_masks,
            force_epsilon_multiplier=eps_mul,
            force_min_area_multiplier=area_mul,
        )

        svg = make_svg(shapes, width, height)

        if not best_svg or byte_len(svg) < byte_len(best_svg):
            best_shapes = shapes
            best_svg = svg

        if byte_len(svg) <= total_limit_bytes:
            return shapes, svg, {
                "mode": "simplified",
                "attempt": attempt + 1,
                "dropped_shapes": 0,
                "epsilon_multiplier": eps_mul,
                "min_area_multiplier": area_mul,
            }

    # If simplification alone is not enough, drop smallest layers last.
    shapes = list(best_shapes)
    dropped = 0

    while shapes:
        svg = make_svg(shapes, width, height)
        if byte_len(svg) <= total_limit_bytes:
            return shapes, svg, {
                "mode": "simplified_and_trimmed",
                "attempt": 18,
                "dropped_shapes": dropped,
                "epsilon_multiplier": None,
                "min_area_multiplier": None,
            }

        shapes.pop()
        dropped += 1

    empty_svg = make_svg([], width, height)
    return [], empty_svg, {
        "mode": "empty_after_trim",
        "attempt": 18,
        "dropped_shapes": dropped,
        "epsilon_multiplier": None,
        "min_area_multiplier": None,
    }


def pack_layer_parts(
    shapes: List[Shape],
    width: int,
    height: int,
    per_part_limit_bytes: int,
    total_limit_bytes: int,
):
    parts: List[List[Shape]] = []
    current: List[Shape] = []

    for shape in shapes:
        test = current + [shape]
        test_svg = make_svg(test, width, height)

        if current and byte_len(test_svg) > per_part_limit_bytes:
            parts.append(current)
            current = [shape]
        else:
            current = test

    if current:
        parts.append(current)

    # Enforce total ZIP-independent SVG layer size.
    # Because each part has its own SVG header, total parts can exceed full SVG size.
    while parts:
        part_svgs = [make_svg(p, width, height, title=f"part_{i+1:02d}") for i, p in enumerate(parts)]
        total = sum(byte_len(s) for s in part_svgs)

        if total <= total_limit_bytes:
            return part_svgs, parts

        # Remove the smallest remaining shape from the final non-empty part.
        for idx in range(len(parts) - 1, -1, -1):
            if parts[idx]:
                parts[idx].pop()
                if not parts[idx]:
                    parts.pop(idx)
                break

    return [], []


def make_zip(full_svg: str, part_svgs: List[str], report_txt: str) -> bytes:
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("full_vector.svg", full_svg)
        for i, svg in enumerate(part_svgs, start=1):
            z.writestr(f"layer_part_{i:02d}.svg", svg)
        z.writestr("report.txt", report_txt)

    return buf.getvalue()


def render_mobile_preview(base_img: Image.Image, svg_text: str, show_base: bool):
    data_uri = pil_to_data_uri(base_img)
    safe_svg = svg_text

    base_html = ""
    if show_base:
        base_html = f'<img class="base" src="{data_uri}" />'

    preview_html = f"""
    <style>
      .preview-wrap {{
        width: 100%;
        max-width: 100%;
        margin: 0 auto;
        border: 1px solid #ddd;
        border-radius: 14px;
        overflow: hidden;
        background: repeating-conic-gradient(#f7f7f7 0% 25%, #ffffff 0% 50%) 50% / 24px 24px;
      }}
      .stage {{
        position: relative;
        width: 100%;
        line-height: 0;
      }}
      .stage .base {{
        width: 100%;
        display: block;
        opacity: 0.35;
      }}
      .stage .vector {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
      }}
      .stage svg {{
        width: 100%;
        height: 100%;
        display: block;
      }}
      .only-vector svg {{
        position: relative;
        width: 100%;
        height: auto;
        display: block;
      }}
    </style>

    <div class="preview-wrap">
      <div class="stage {'only-vector' if not show_base else ''}">
        {base_html}
        <div class="vector">
          {safe_svg}
        </div>
      </div>
    </div>
    """

    components.html(preview_html, height=min(740, max(360, int(base_img.height * 1.2))), scrolling=True)


def main():
    st.set_page_config(
        page_title="Path Layer SVG Vectorizer",
        page_icon="🧩",
        layout="wide",
    )

    st.title("🧩 Path Layer SVG Vectorizer")
    st.caption("Smartphone-first image → layered path-only SVG. Largest area layers are drawn first.")

    with st.expander("What this app does", expanded=False):
        st.write(
            """
            This app converts an uploaded raster image into colored SVG path layers.
            It does not use bitmap tracing tags, embedded images, rectangles, circles, or polygons for the artwork.
            The artwork is exported as SVG `<path>` data only.

            The layer order is automatic:
            largest colored area first → smallest detail last.
            This is useful for decal-style workflows where big underpaint shapes are placed below small details.
            """
        )

    uploaded = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    st.sidebar.header("Vector settings")

    max_side = st.sidebar.slider(
        "Processing size",
        min_value=160,
        max_value=900,
        value=420,
        step=20,
        help="Smaller size creates smaller SVG. Larger size keeps more detail.",
    )

    color_count = st.sidebar.slider(
        "Color layers",
        min_value=3,
        max_value=36,
        value=14,
        step=1,
        help="More colors improve similarity but increase SVG size.",
    )

    detail = st.sidebar.slider(
        "Path detail",
        min_value=10,
        max_value=100,
        value=72,
        step=1,
        help="Higher keeps more contour points. Lower creates smaller files.",
    )

    min_area_px = st.sidebar.slider(
        "Remove tiny islands",
        min_value=1,
        max_value=500,
        value=18,
        step=1,
        help="Higher removes more tiny regions and reduces file size.",
    )

    total_kb = st.sidebar.slider(
        "Total SVG target KB",
        min_value=20,
        max_value=200,
        value=100,
        step=5,
    )

    part_kb = st.sidebar.slider(
        "Layer part target KB",
        min_value=5,
        max_value=40,
        value=15,
        step=1,
    )

    smooth_masks = st.sidebar.checkbox(
        "Smooth color regions",
        value=True,
        help="Reduces noisy pixels before path extraction.",
    )

    remove_background = st.sidebar.checkbox(
        "Remove corner background",
        value=False,
        help="Skips the most common corner color. Turn on for white/flat backgrounds.",
    )

    show_base_preview = st.sidebar.checkbox(
        "Preview over base image",
        value=True,
        help="Preview only. Exported SVG files remain vector-only.",
    )

    if not uploaded:
        st.info("Upload an image to generate layered SVG paths.")
        return

    img = prepare_image(uploaded, max_side=max_side)
    arr = np.array(img)
    h, w, _ = arr.shape

    with st.spinner("Generating path layers..."):
        label_img, centers = quantize_image_kmeans(arr, color_count=color_count)

        total_limit_bytes = int(total_kb * 1024)
        per_part_limit_bytes = int(part_kb * 1024)

        shapes, full_svg, fit_info = fit_shapes_to_budget(
            label_img=label_img,
            centers=centers,
            width=w,
            height=h,
            detail=detail,
            min_area_px=min_area_px,
            remove_background=remove_background,
            smooth_masks=smooth_masks,
            total_limit_bytes=total_limit_bytes,
        )

        part_svgs, part_shape_groups = pack_layer_parts(
            shapes=shapes,
            width=w,
            height=h,
            per_part_limit_bytes=per_part_limit_bytes,
            total_limit_bytes=total_limit_bytes,
        )

        part_sizes = [byte_len(s) for s in part_svgs]
        total_parts_size = sum(part_sizes)

        report = []
        report.append("Path Layer SVG Vectorizer Report")
        report.append("--------------------------------")
        report.append(f"Image size: {w} x {h}")
        report.append(f"Requested colors: {color_count}")
        report.append(f"Exported shape layers: {len(shapes)}")
        report.append(f"Full SVG size: {byte_len(full_svg)} bytes")
        report.append(f"Layer part count: {len(part_svgs)}")
        report.append(f"Total layer part size: {total_parts_size} bytes")
        report.append(f"Target total size: {total_limit_bytes} bytes")
        report.append(f"Target part size: {per_part_limit_bytes} bytes")
        report.append(f"Fit mode: {fit_info['mode']}")
        report.append(f"Dropped small shapes: {fit_info['dropped_shapes']}")
        report.append("")
        report.append("Layer order:")
        report.append("Largest area first, smallest details last.")
        report.append("")
        for i, shape in enumerate(shapes[:200], start=1):
            report.append(f"{i:04d}: area={int(shape.area)} fill={shape.fill} path_bytes={byte_len(shape.d)}")
        report_txt = "\n".join(report)

        zip_bytes = make_zip(full_svg, part_svgs, report_txt)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Full SVG", f"{byte_len(full_svg) / 1024:.1f} KB")
    col2.metric("Layer parts total", f"{total_parts_size / 1024:.1f} KB")
    col3.metric("Parts", str(len(part_svgs)))
    col4.metric("Paths", str(len(shapes)))

    if byte_len(full_svg) > total_limit_bytes or total_parts_size > total_limit_bytes:
        st.warning(
            "The result is still above the target. Lower Processing size, Color layers, or Path detail."
        )
    else:
        st.success("Result is inside the requested total SVG size target.")

    st.subheader("Preview")
    render_mobile_preview(img, full_svg, show_base=show_base_preview)

    st.subheader("Download")
    st.download_button(
        "Download ZIP: full SVG + layer parts",
        data=zip_bytes,
        file_name="path_layer_svg_export.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.download_button(
        "Download full_vector.svg",
        data=full_svg,
        file_name="full_vector.svg",
        mime="image/svg+xml",
        use_container_width=True,
    )

    st.download_button(
        "Download report.txt",
        data=report_txt,
        file_name="report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.subheader("Copy full SVG")
    st.text_area(
        "Copy and paste this SVG",
        value=full_svg,
        height=260,
    )

    if part_svgs:
        st.subheader("Copy layer parts")
        selected = st.selectbox(
            "Select layer part",
            options=list(range(1, len(part_svgs) + 1)),
            format_func=lambda i: f"layer_part_{i:02d}.svg — {byte_len(part_svgs[i - 1]) / 1024:.1f} KB",
        )

        st.text_area(
            f"layer_part_{selected:02d}.svg",
            value=part_svgs[selected - 1],
            height=220,
        )

    with st.expander("Layer size details"):
        for i, svg in enumerate(part_svgs, start=1):
            st.write(f"layer_part_{i:02d}.svg — {byte_len(svg) / 1024:.2f} KB")


if __name__ == "__main__":
    main()