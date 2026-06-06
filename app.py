import io
import zipfile
import base64
import html
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


# ============================================================
# High-fidelity smartphone SVG vectorizer
# ============================================================
# Goals:
# - Path-only vector data
# - Colored shape overlays
# - Largest area first, smallest detail last
# - Automatic candidate search for better fidelity
# - Keep total exported SVG layer size under target, default 100 KB
# - Split layer parts around target size, default 15 KB each
# ============================================================


@dataclass
class Shape:
    area: float
    fill_rgb: Tuple[int, int, int]
    fill_hex: str
    outer: np.ndarray
    holes: List[np.ndarray]
    d: str
    has_holes: bool
    is_detail: bool = False


@dataclass
class Candidate:
    score: float
    shapes: List[Shape]
    full_svg: str
    part_svgs: List[str]
    part_shape_groups: List[List[Shape]]
    preview_rgb: np.ndarray
    info: Dict[str, Any]


def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [int(max(0, min(255, x))) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def prepare_image(uploaded_file, max_side: int, background_rgb=(255, 255, 255)) -> Image.Image:
    img = Image.open(uploaded_file)

    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, background_rgb + (255,))
        bg.alpha_composite(img.convert("RGBA"))
        img = bg.convert("RGB")
    else:
        img = img.convert("RGB")

    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def ndarray_to_data_uri(arr_rgb: np.ndarray) -> str:
    img = Image.fromarray(np.clip(arr_rgb, 0, 255).astype(np.uint8), "RGB")
    return pil_to_data_uri(img)


def dominant_corner_rgb(arr_rgb: np.ndarray) -> Tuple[int, int, int]:
    h, w, _ = arr_rgb.shape
    c = max(3, min(h, w) // 18)

    samples = np.concatenate(
        [
            arr_rgb[:c, :c].reshape(-1, 3),
            arr_rgb[:c, -c:].reshape(-1, 3),
            arr_rgb[-c:, :c].reshape(-1, 3),
            arr_rgb[-c:, -c:].reshape(-1, 3),
        ],
        axis=0,
    )

    return tuple(np.median(samples, axis=0).astype(np.uint8).tolist())


def preprocess_image(arr_rgb: np.ndarray, method: str) -> np.ndarray:
    src = arr_rgb.astype(np.uint8)

    if method == "none":
        return src

    if method == "bilateral":
        return cv2.bilateralFilter(src, 5, 38, 38)

    if method == "meanshift":
        return cv2.pyrMeanShiftFiltering(src, sp=7, sr=18)

    if method == "hybrid":
        x = cv2.bilateralFilter(src, 5, 34, 34)
        x = cv2.pyrMeanShiftFiltering(x, sp=6, sr=16)
        return x

    return src


def quantize_lab(
    processed_rgb: np.ndarray,
    original_rgb: np.ndarray,
    color_count: int,
    sample_limit: int = 65000,
):
    h, w, _ = processed_rgb.shape

    lab = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2LAB)
    lab_pixels = lab.reshape(-1, 3).astype(np.float32)
    original_pixels = original_rgb.reshape(-1, 3).astype(np.float32)

    unique_count = len(np.unique(lab.reshape(-1, 3), axis=0))
    k = max(2, min(int(color_count), unique_count))

    rng = np.random.default_rng(2026)
    if len(lab_pixels) > sample_limit:
        idx = rng.choice(len(lab_pixels), sample_limit, replace=False)
        sample = lab_pixels[idx]
    else:
        sample = lab_pixels

    cv2.setRNGSeed(2026)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        38,
        0.45,
    )

    _, _, centers_lab = cv2.kmeans(
        sample,
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = np.empty((len(lab_pixels),), dtype=np.int32)
    chunk_size = 70000

    for start in range(0, len(lab_pixels), chunk_size):
        chunk = lab_pixels[start:start + chunk_size]
        dist = ((chunk[:, None, :] - centers_lab[None, :, :]) ** 2).sum(axis=2)
        labels[start:start + chunk_size] = np.argmin(dist, axis=1)

    label_img = labels.reshape(h, w)

    # Use original image colors for final fill values.
    # This gives better visual matching than using the smoothed image colors.
    centers_rgb = np.zeros((k, 3), dtype=np.uint8)
    for i in range(k):
        mask = labels == i
        if np.any(mask):
            centers_rgb[i] = np.median(original_pixels[mask], axis=0).astype(np.uint8)
        else:
            lab_color = np.uint8([[centers_lab[i]]])
            rgb_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2RGB)[0, 0]
            centers_rgb[i] = rgb_color

    return label_img, centers_rgb


def absorb_small_components(
    label_img: np.ndarray,
    centers_rgb: np.ndarray,
    min_area: int,
    passes: int = 1,
) -> np.ndarray:
    labels = label_img.copy()
    h, w = labels.shape
    k = len(centers_rgb)
    kernel = np.ones((3, 3), np.uint8)

    min_area = max(1, int(min_area))

    for _ in range(max(1, passes)):
        changed = False

        for label_id in range(k):
            mask = (labels == label_id).astype(np.uint8)
            if mask.sum() == 0:
                continue

            count, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

            for comp_id in range(1, count):
                area = int(stats[comp_id, cv2.CC_STAT_AREA])
                if area >= min_area:
                    continue

                comp_mask = comp == comp_id
                dilated = cv2.dilate(comp_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                border = dilated & (~comp_mask)

                neighbor_labels = labels[border]
                neighbor_labels = neighbor_labels[neighbor_labels != label_id]

                if len(neighbor_labels) == 0:
                    continue

                vals, counts = np.unique(neighbor_labels, return_counts=True)

                # Prefer frequent neighbor; break ties by color distance.
                comp_color = centers_rgb[label_id].astype(np.float32)
                best_label = int(vals[0])
                best_key = None

                for v, c in zip(vals, counts):
                    dist = np.linalg.norm(comp_color - centers_rgb[int(v)].astype(np.float32))
                    key = (-int(c), float(dist))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_label = int(v)

                labels[comp_mask] = best_label
                changed = True

        if not changed:
            break

    return labels


def contour_to_compact_path(contour: np.ndarray) -> str:
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return ""

    pts = np.rint(pts).astype(int)

    # Absolute compact form:
    # Mx y x y x yZ
    abs_path = "M" + " ".join(f"{x} {y}" for x, y in pts) + "Z"

    # Relative compact form:
    # Mx yl dx dy dx dyZ
    rel_pairs = []
    prev = pts[0]
    for p in pts[1:]:
        dx = int(p[0] - prev[0])
        dy = int(p[1] - prev[1])
        rel_pairs.append(f"{dx} {dy}")
        prev = p

    if rel_pairs:
        rel_path = f"M{pts[0][0]} {pts[0][1]}l" + " ".join(rel_pairs) + "Z"
        if len(rel_path) < len(abs_path):
            return rel_path

    return abs_path


def extract_shapes_from_labels(
    label_img: np.ndarray,
    centers_rgb: np.ndarray,
    detail: int,
    min_area: int,
    epsilon_multiplier: float,
    smooth_masks: bool,
) -> List[Shape]:
    shapes: List[Shape] = []
    h, w = label_img.shape

    detail = int(np.clip(detail, 1, 100))
    base_epsilon = max(0.25, (101 - detail) * 0.022)
    epsilon = base_epsilon * float(epsilon_multiplier)

    min_area = max(1, int(min_area))
    kernel = np.ones((3, 3), np.uint8)

    for label_id, rgb in enumerate(centers_rgb):
        mask = (label_img == label_id).astype(np.uint8) * 255

        if smooth_masks:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if hierarchy is None:
            continue

        hierarchy = hierarchy[0]
        fill_rgb = tuple(int(x) for x in rgb)
        fill_hex = rgb_to_hex(fill_rgb)

        for i, contour in enumerate(contours):
            parent = hierarchy[i][3]
            if parent != -1:
                continue

            area = abs(cv2.contourArea(contour))
            if area < min_area:
                continue

            outer = cv2.approxPolyDP(contour, epsilon, True)
            if len(outer) < 3:
                continue

            d = contour_to_compact_path(outer)
            if not d:
                continue

            holes: List[np.ndarray] = []
            child = hierarchy[i][2]

            while child != -1:
                hole = contours[child]
                hole_area = abs(cv2.contourArea(hole))

                if hole_area >= min_area:
                    hole_approx = cv2.approxPolyDP(hole, epsilon, True)
                    if len(hole_approx) >= 3:
                        hole_d = contour_to_compact_path(hole_approx)
                        if hole_d:
                            holes.append(hole_approx)
                            d += hole_d

                child = hierarchy[child][0]

            shapes.append(
                Shape(
                    area=float(area),
                    fill_rgb=fill_rgb,
                    fill_hex=fill_hex,
                    outer=outer,
                    holes=holes,
                    d=d,
                    has_holes=len(holes) > 0,
                    is_detail=False,
                )
            )

    shapes.sort(key=lambda s: s.area, reverse=True)
    return shapes


def extract_detail_shapes(
    original_rgb: np.ndarray,
    detail: int,
    min_area: int,
    epsilon_multiplier: float,
    strength: str,
) -> List[Shape]:
    if strength == "off":
        return []

    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.bilateralFilter(gray, 5, 30, 30)

    med = float(np.median(gray_blur))
    lower = int(max(0, 0.62 * med))
    upper = int(min(255, 1.38 * med))

    edges = cv2.Canny(gray_blur, lower, upper)

    if strength == "medium":
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=1)
    else:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=1)
        dark_thr = np.percentile(gray_blur, 24)
        dark_mask = (gray_blur <= dark_thr).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(dark_mask, cv2.dilate(edges, kernel, iterations=1)))

    min_area = max(1, int(min_area))
    detail = int(np.clip(detail, 1, 100))
    base_epsilon = max(0.25, (101 - detail) * 0.018)
    epsilon = base_epsilon * float(epsilon_multiplier)

    pixels = original_rgb[mask > 0]
    if len(pixels) > 0:
        dark_rgb = tuple(np.percentile(pixels, 28, axis=0).astype(np.uint8).tolist())
    else:
        dark_rgb = (30, 30, 30)

    # Force line color slightly dark for readability.
    dark_rgb = tuple(int(max(0, min(90, x))) for x in dark_rgb)
    fill_hex = rgb_to_hex(dark_rgb)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes: List[Shape] = []

    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if area < min_area:
            continue

        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue

        d = contour_to_compact_path(approx)
        if not d:
            continue

        shapes.append(
            Shape(
                area=float(area),
                fill_rgb=dark_rgb,
                fill_hex=fill_hex,
                outer=approx,
                holes=[],
                d=d,
                has_holes=False,
                is_detail=True,
            )
        )

    return shapes


def shape_importance_per_byte(shape: Shape) -> float:
    cost = max(1, byte_len(shape.d))
    boost = 2.2 if shape.is_detail else 1.0
    return (shape.area * boost) / cost


def make_svg(
    shapes: List[Shape],
    width: int,
    height: int,
    merge_consecutive_same_color: bool = True,
) -> str:
    # Compact SVG.
    # The artwork itself is path-only.
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">']

    if not merge_consecutive_same_color:
        for s in shapes:
            rule = ' fill-rule="evenodd"' if s.has_holes else ""
            out.append(f'<path fill="{s.fill_hex}" d="{s.d}"{rule}/>')
        out.append("</svg>")
        return "".join(out)

    i = 0
    n = len(shapes)

    while i < n:
        fill = shapes[i].fill_hex
        d = [shapes[i].d]
        has_holes = shapes[i].has_holes
        j = i + 1

        # Merge only consecutive same-color paths.
        # This preserves the largest-to-smallest draw order while reducing tag overhead.
        while j < n and shapes[j].fill_hex == fill:
            d.append(shapes[j].d)
            has_holes = has_holes or shapes[j].has_holes
            j += 1

        rule = ' fill-rule="evenodd"' if has_holes else ""
        out.append(f'<path fill="{fill}" d="{"".join(d)}"{rule}/>')

        i = j

    out.append("</svg>")
    return "".join(out)


def pack_layer_parts(
    shapes: List[Shape],
    width: int,
    height: int,
    per_part_limit_bytes: int,
) -> Tuple[List[str], List[List[Shape]]]:
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

    part_svgs = [make_svg(p, width, height) for p in parts]
    return part_svgs, parts


def fit_shapes_to_budget(
    shapes: List[Shape],
    width: int,
    height: int,
    total_limit_bytes: int,
    per_part_limit_bytes: int,
) -> Tuple[List[Shape], str, List[str], List[List[Shape]], int]:
    shapes = sorted(shapes, key=lambda s: s.area, reverse=True)
    removed = 0

    while True:
        full_svg = make_svg(shapes, width, height)
        part_svgs, part_groups = pack_layer_parts(shapes, width, height, per_part_limit_bytes)
        total_part_size = sum(byte_len(x) for x in part_svgs)

        if byte_len(full_svg) <= total_limit_bytes and total_part_size <= total_limit_bytes:
            return shapes, full_svg, part_svgs, part_groups, removed

        if len(shapes) <= 1:
            return shapes, full_svg, part_svgs, part_groups, removed

        # Remove the lowest visual value per byte.
        # This usually removes noisy tiny contours before important large surfaces.
        scores = [(idx, shape_importance_per_byte(s)) for idx, s in enumerate(shapes)]

        # Protect the very largest layer unless absolutely necessary.
        if len(shapes) > 8:
            scores = [(idx, val) for idx, val in scores if idx != 0]

        drop_idx = min(scores, key=lambda x: x[1])[0]
        shapes.pop(drop_idx)
        removed += 1


def render_shapes(
    shapes: List[Shape],
    width: int,
    height: int,
    background_rgb: Tuple[int, int, int],
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = np.array(background_rgb, dtype=np.uint8)

    for s in sorted(shapes, key=lambda x: x.area, reverse=True):
        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.drawContours(mask, [s.outer], -1, 255, cv2.FILLED)

        for hole in s.holes:
            cv2.drawContours(mask, [hole], -1, 0, cv2.FILLED)

        canvas[mask > 0] = np.array(s.fill_rgb, dtype=np.uint8)

    return canvas


def score_vector_result(original_rgb: np.ndarray, vector_rgb: np.ndarray, size_bytes: int, path_count: int) -> float:
    orig_lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    vec_lab = cv2.cvtColor(vector_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    lab_delta = np.linalg.norm(orig_lab - vec_lab, axis=2).mean()

    orig_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    vec_gray = cv2.cvtColor(vector_rgb, cv2.COLOR_RGB2GRAY)

    orig_edges = cv2.Canny(orig_gray, 70, 150)
    vec_edges = cv2.Canny(vec_gray, 70, 150)

    edge_delta = np.mean(np.abs(orig_edges.astype(np.float32) - vec_edges.astype(np.float32))) / 255.0

    # Small penalties prevent bloated solutions from winning when visual error is similar.
    size_penalty = size_bytes / 1024.0 * 0.015
    path_penalty = path_count * 0.004

    return float(lab_delta + edge_delta * 24.0 + size_penalty + path_penalty)


def build_trials(
    base_colors: int,
    detail: int,
    min_area: int,
    strength: str,
    line_strength: str,
    max_trials: int,
):
    if strength == "Fast":
        preps = ["meanshift", "bilateral"]
        color_opts = [base_colors, base_colors + 4]
        eps_opts = [1.0, 0.78]
        area_opts = [1.0, 1.3]

    elif strength == "Balanced":
        preps = ["meanshift", "hybrid", "bilateral"]
        color_opts = [base_colors, base_colors + 4, max(3, base_colors - 4), base_colors + 8]
        eps_opts = [0.72, 1.0, 1.35]
        area_opts = [0.75, 1.0, 1.55]

    else:
        preps = ["meanshift", "hybrid", "bilateral", "none"]
        color_opts = [base_colors, base_colors + 4, base_colors + 8, max(3, base_colors - 4), base_colors + 12]
        eps_opts = [0.62, 0.78, 1.0, 1.28, 1.65]
        area_opts = [0.65, 0.9, 1.25, 1.75]

    color_opts = [int(np.clip(c, 2, 48)) for c in color_opts]
    color_opts = list(dict.fromkeys(color_opts))

    line_opts = ["off"]
    if line_strength != "off":
        line_opts.append(line_strength)

    all_trials = []

    for prep in preps:
        for colors in color_opts:
            for eps in eps_opts:
                for area_mul in area_opts:
                    for line in line_opts:
                        all_trials.append(
                            {
                                "prep": prep,
                                "colors": colors,
                                "epsilon_multiplier": eps,
                                "area_multiplier": area_mul,
                                "line_strength": line,
                            }
                        )

    # Interleave high-quality and compact attempts.
    all_trials.sort(
        key=lambda t: (
            abs(t["colors"] - base_colors),
            t["epsilon_multiplier"],
            t["area_multiplier"],
            0 if t["prep"] in ("meanshift", "hybrid") else 1,
            0 if t["line_strength"] != "off" else 1,
        )
    )

    return all_trials[:max(1, int(max_trials))]


def generate_candidate(
    original_rgb: np.ndarray,
    trial: Dict[str, Any],
    detail: int,
    min_area: int,
    smooth_masks: bool,
    total_limit_bytes: int,
    per_part_limit_bytes: int,
) -> Candidate:
    h, w, _ = original_rgb.shape

    processed = preprocess_image(original_rgb, trial["prep"])

    label_img, centers_rgb = quantize_lab(
        processed_rgb=processed,
        original_rgb=original_rgb,
        color_count=trial["colors"],
    )

    effective_min_area = max(1, int(min_area * trial["area_multiplier"]))

    label_img = absorb_small_components(
        label_img=label_img,
        centers_rgb=centers_rgb,
        min_area=effective_min_area,
        passes=2,
    )

    shapes = extract_shapes_from_labels(
        label_img=label_img,
        centers_rgb=centers_rgb,
        detail=detail,
        min_area=effective_min_area,
        epsilon_multiplier=trial["epsilon_multiplier"],
        smooth_masks=smooth_masks,
    )

    detail_shapes = extract_detail_shapes(
        original_rgb=original_rgb,
        detail=detail,
        min_area=max(1, int(effective_min_area * 0.45)),
        epsilon_multiplier=trial["epsilon_multiplier"],
        strength=trial["line_strength"],
    )

    shapes.extend(detail_shapes)
    shapes.sort(key=lambda s: s.area, reverse=True)

    shapes, full_svg, part_svgs, part_groups, removed = fit_shapes_to_budget(
        shapes=shapes,
        width=w,
        height=h,
        total_limit_bytes=total_limit_bytes,
        per_part_limit_bytes=per_part_limit_bytes,
    )

    background = dominant_corner_rgb(original_rgb)
    preview_rgb = render_shapes(shapes, w, h, background)

    total_part_size = sum(byte_len(x) for x in part_svgs)
    size_for_score = max(byte_len(full_svg), total_part_size)

    score = score_vector_result(
        original_rgb=original_rgb,
        vector_rgb=preview_rgb,
        size_bytes=size_for_score,
        path_count=len(shapes),
    )

    info = {
        **trial,
        "removed_shapes": removed,
        "shape_count": len(shapes),
        "full_svg_bytes": byte_len(full_svg),
        "part_count": len(part_svgs),
        "total_part_bytes": total_part_size,
        "score": score,
    }

    return Candidate(
        score=score,
        shapes=shapes,
        full_svg=full_svg,
        part_svgs=part_svgs,
        part_shape_groups=part_groups,
        preview_rgb=preview_rgb,
        info=info,
    )


def make_zip(full_svg: str, part_svgs: List[str], report_txt: str) -> bytes:
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("full_vector.svg", full_svg)

        for i, svg in enumerate(part_svgs, start=1):
            z.writestr(f"layer_part_{i:02d}.svg", svg)

        z.writestr("report.txt", report_txt)

    return buf.getvalue()


def render_preview_html(base_rgb: np.ndarray, vector_rgb: np.ndarray, svg_text: str, mode: str):
    base_uri = ndarray_to_data_uri(base_rgb)
    vector_uri = ndarray_to_data_uri(vector_rgb)

    if mode == "Original":
        body = f'<img src="{base_uri}" class="fit"/>'

    elif mode == "Vector raster preview":
        body = f'<img src="{vector_uri}" class="fit"/>'

    elif mode == "Overlay":
        body = f"""
        <div class="stage">
          <img src="{base_uri}" class="fit base"/>
          <div class="svg-layer">{svg_text}</div>
        </div>
        """

    else:
        body = f"""
        <div class="compare">
          <div>
            <div class="label">Original</div>
            <img src="{base_uri}" class="fit"/>
          </div>
          <div>
            <div class="label">Vector preview</div>
            <img src="{vector_uri}" class="fit"/>
          </div>
        </div>
        """

    html_doc = f"""
    <style>
      body {{
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      }}
      .fit {{
        width: 100%;
        height: auto;
        display: block;
      }}
      .stage {{
        position: relative;
        width: 100%;
        line-height: 0;
        background:
          repeating-conic-gradient(#f3f3f3 0% 25%, #ffffff 0% 50%) 50% / 24px 24px;
      }}
      .stage .base {{
        opacity: 0.38;
      }}
      .svg-layer {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
      }}
      .svg-layer svg {{
        width: 100%;
        height: 100%;
        display: block;
      }}
      .compare {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }}
      .label {{
        font-size: 13px;
        padding: 6px;
        background: #f2f2f2;
        color: #333;
      }}
    </style>
    {body}
    """

    h = base_rgb.shape[0]
    components.html(html_doc, height=max(360, min(760, int(h * 1.45))), scrolling=True)


def build_report(candidate: Candidate, width: int, height: int, total_limit_bytes: int, per_part_limit_bytes: int) -> str:
    lines = []
    lines.append("Optimized Path Layer SVG Report")
    lines.append("--------------------------------")
    lines.append(f"Image size: {width} x {height}")
    lines.append(f"Score: {candidate.score:.4f}")
    lines.append(f"Full SVG size: {byte_len(candidate.full_svg)} bytes")
    lines.append(f"Layer part count: {len(candidate.part_svgs)}")
    lines.append(f"Total layer part size: {sum(byte_len(x) for x in candidate.part_svgs)} bytes")
    lines.append(f"Total target: {total_limit_bytes} bytes")
    lines.append(f"Part target: {per_part_limit_bytes} bytes")
    lines.append("")
    lines.append("Winning settings:")
    for k, v in candidate.info.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Layer order:")
    lines.append("Largest area first, smallest detail last.")
    lines.append("")
    for i, s in enumerate(candidate.shapes, start=1):
        kind = "detail" if s.is_detail else "color"
        lines.append(
            f"{i:04d}: area={int(s.area)} fill={s.fill_hex} kind={kind} path_bytes={byte_len(s.d)}"
        )
    return "\n".join(lines)


def main():
    st.set_page_config(
        page_title="Optimized Path Layer SVG Vectorizer",
        page_icon="🧩",
        layout="wide",
    )

    st.title("🧩 Optimized Path Layer SVG Vectorizer")
    st.caption("High-fidelity image → path-only layered SVG. Largest surfaces first, smallest details last.")

    uploaded = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    st.sidebar.header("Quality / size")

    max_side = st.sidebar.slider(
        "Processing size",
        min_value=220,
        max_value=900,
        value=520,
        step=20,
        help="Higher = better accuracy but larger SVG.",
    )

    base_colors = st.sidebar.slider(
        "Base color count",
        min_value=4,
        max_value=42,
        value=20,
        step=1,
        help="Higher = better color accuracy but larger SVG.",
    )

    detail = st.sidebar.slider(
        "Path detail",
        min_value=20,
        max_value=100,
        value=82,
        step=1,
        help="Higher = more accurate contours.",
    )

    min_area = st.sidebar.slider(
        "Noise removal",
        min_value=1,
        max_value=420,
        value=18,
        step=1,
        help="Higher removes tiny regions and reduces path count.",
    )

    total_kb = st.sidebar.slider(
        "Total SVG layer limit KB",
        min_value=30,
        max_value=200,
        value=100,
        step=5,
    )

    part_kb = st.sidebar.slider(
        "Target KB per layer part",
        min_value=5,
        max_value=40,
        value=15,
        step=1,
    )

    strength = st.sidebar.selectbox(
        "Optimization strength",
        ["Fast", "Balanced", "Maximum"],
        index=1,
        help="Maximum tries more candidates and usually gives better accuracy.",
    )

    max_trials = st.sidebar.slider(
        "Candidate trials",
        min_value=4,
        max_value=48,
        value=18,
        step=1,
        help="More trials = better chance of high quality, but slower.",
    )

    line_strength = st.sidebar.selectbox(
        "Detail / line-art reinforcement",
        ["off", "medium", "strong"],
        index=1,
        help="Adds small dark path layers for edges and linework.",
    )

    smooth_masks = st.sidebar.checkbox(
        "Smooth color regions",
        value=True,
        help="Reduces jagged noise before path extraction.",
    )

    preview_mode = st.sidebar.selectbox(
        "Preview mode",
        ["Compare", "Overlay", "Vector raster preview", "Original"],
        index=0,
    )

    if uploaded is None:
        st.info("Upload an image to start.")
        return

    img = prepare_image(uploaded, max_side=max_side)
    original_rgb = np.array(img).astype(np.uint8)
    h, w, _ = original_rgb.shape

    total_limit_bytes = int(total_kb * 1024)
    per_part_limit_bytes = int(part_kb * 1024)

    trials = build_trials(
        base_colors=base_colors,
        detail=detail,
        min_area=min_area,
        strength=strength,
        line_strength=line_strength,
        max_trials=max_trials,
    )

    best: Candidate | None = None

    progress = st.progress(0, text="Preparing optimization...")

    for i, trial in enumerate(trials):
        progress.progress(
            (i + 1) / len(trials),
            text=f"Testing candidate {i + 1}/{len(trials)}...",
        )

        try:
            candidate = generate_candidate(
                original_rgb=original_rgb,
                trial=trial,
                detail=detail,
                min_area=min_area,
                smooth_masks=smooth_masks,
                total_limit_bytes=total_limit_bytes,
                per_part_limit_bytes=per_part_limit_bytes,
            )

            if best is None or candidate.score < best.score:
                best = candidate

        except Exception as e:
            st.warning(f"Skipped one failed candidate: {e}")

    progress.empty()

    if best is None:
        st.error("Vectorization failed. Try a smaller processing size or fewer colors.")
        return

    report_txt = build_report(
        candidate=best,
        width=w,
        height=h,
        total_limit_bytes=total_limit_bytes,
        per_part_limit_bytes=per_part_limit_bytes,
    )

    zip_bytes = make_zip(best.full_svg, best.part_svgs, report_txt)

    full_size = byte_len(best.full_svg)
    total_part_size = sum(byte_len(x) for x in best.part_svgs)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best score", f"{best.score:.2f}")
    col2.metric("Full SVG", f"{full_size / 1024:.1f} KB")
    col3.metric("Layer parts total", f"{total_part_size / 1024:.1f} KB")
    col4.metric("Paths", str(len(best.shapes)))

    if full_size <= total_limit_bytes and total_part_size <= total_limit_bytes:
        st.success("Optimized result fits inside the requested total layer size.")
    else:
        st.warning("Result is still above the target. Lower Processing size, Base color count, or Path detail.")

    st.subheader("Preview")
    render_preview_html(original_rgb, best.preview_rgb, best.full_svg, preview_mode)

    st.subheader("Download")
    st.download_button(
        "Download ZIP: full SVG + layer parts + report",
        data=zip_bytes,
        file_name="optimized_path_layer_svg_export.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.download_button(
        "Download full_vector.svg",
        data=best.full_svg,
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
        "full_vector.svg",
        value=best.full_svg,
        height=260,
    )

    if best.part_svgs:
        st.subheader("Copy layer parts")

        selected = st.selectbox(
            "Select part",
            options=list(range(1, len(best.part_svgs) + 1)),
            format_func=lambda i: f"layer_part_{i:02d}.svg — {byte_len(best.part_svgs[i - 1]) / 1024:.1f} KB",
        )

        st.text_area(
            f"layer_part_{selected:02d}.svg",
            value=best.part_svgs[selected - 1],
            height=220,
        )

    with st.expander("Winning optimization settings"):
        st.json(best.info)

    with st.expander("Layer size details"):
        for i, svg in enumerate(best.part_svgs, start=1):
            st.write(f"layer_part_{i:02d}.svg — {byte_len(svg) / 1024:.2f} KB")

    with st.expander("Important note"):
        st.write(
            """
            This version tries to reproduce the original image as faithfully as possible inside the size limit.
            However, a strict 100 KB total SVG limit cannot perfectly reproduce every high-detail image,
            especially photos, gradients, noisy backgrounds, and very small facial details.

            For best results, use cropped anime-style images, simple backgrounds, and Maximum optimization.
            """
        )


if __name__ == "__main__":
    main()