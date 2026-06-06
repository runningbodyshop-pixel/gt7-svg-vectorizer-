import io
import zipfile
import base64
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


# ============================================================
# Fine-detail optimized path-only SVG vectorizer
# ============================================================
# Features:
# - Path-only SVG artwork
# - Largest-area layer order
# - Lab color quantization
# - Automatic candidate testing
# - Edge snapping
# - Smooth curve path generation
# - Residual-error micro correction layers
# - Optional line-art/detail reinforcement
# - Size-constrained output
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
    kind: str = "color"


@dataclass
class Candidate:
    score: float
    shapes: List[Shape]
    full_svg: str
    part_svgs: List[str]
    preview_rgb: np.ndarray
    info: Dict[str, Any]


def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [int(max(0, min(255, x))) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def fmt_num(x: float) -> str:
    x = float(x)
    if abs(x - round(x)) < 0.01:
        return str(int(round(x)))
    return f"{x:.1f}".rstrip("0").rstrip(".")


def prepare_image(uploaded_file, max_side: int) -> Image.Image:
    img = Image.open(uploaded_file)

    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img.convert("RGBA"))
        img = bg.convert("RGB")
    else:
        img = img.convert("RGB")

    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def image_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def array_to_data_uri(arr_rgb: np.ndarray) -> str:
    img = Image.fromarray(np.clip(arr_rgb, 0, 255).astype(np.uint8), "RGB")
    return image_to_data_uri(img)


def dominant_corner_rgb(arr_rgb: np.ndarray) -> Tuple[int, int, int]:
    h, w, _ = arr_rgb.shape
    c = max(4, min(h, w) // 18)

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
        return cv2.bilateralFilter(src, 5, 34, 34)

    if method == "meanshift":
        return cv2.pyrMeanShiftFiltering(src, sp=7, sr=17)

    if method == "hybrid":
        x = cv2.bilateralFilter(src, 5, 30, 30)
        x = cv2.pyrMeanShiftFiltering(x, sp=6, sr=15)
        return x

    if method == "detail_preserve":
        x = cv2.detailEnhance(src, sigma_s=8, sigma_r=0.12)
        x = cv2.bilateralFilter(x, 3, 22, 22)
        return x

    return src


def source_edge_map(arr_rgb: np.ndarray, strength: str) -> np.ndarray:
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 28, 28)

    if strength == "soft":
        lower, upper = 50, 135
    elif strength == "strong":
        lower, upper = 28, 105
    else:
        lower, upper = 38, 125

    edges = cv2.Canny(blur, lower, upper)

    # Add thin dark anime-like line hints.
    dark_thr = np.percentile(blur, 24)
    dark = (blur <= dark_thr).astype(np.uint8) * 255
    dark_edges = cv2.bitwise_and(cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1), dark)

    edges = cv2.bitwise_or(edges, dark_edges)
    return edges


def quantize_lab(
    processed_rgb: np.ndarray,
    original_rgb: np.ndarray,
    color_count: int,
    sample_limit: int = 70000,
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
        45,
        0.42,
    )

    _, _, centers_lab = cv2.kmeans(
        sample,
        k,
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = np.empty((len(lab_pixels),), dtype=np.int32)
    chunk_size = 70000

    for start in range(0, len(lab_pixels), chunk_size):
        chunk = lab_pixels[start:start + chunk_size]
        dist = ((chunk[:, None, :] - centers_lab[None, :, :]) ** 2).sum(axis=2)
        labels[start:start + chunk_size] = np.argmin(dist, axis=1)

    label_img = labels.reshape(h, w)

    centers_rgb = np.zeros((k, 3), dtype=np.uint8)

    for i in range(k):
        mask = labels == i
        if np.any(mask):
            centers_rgb[i] = np.median(original_pixels[mask], axis=0).astype(np.uint8)
        else:
            lab_color = np.uint8([[centers_lab[i]]])
            centers_rgb[i] = cv2.cvtColor(lab_color, cv2.COLOR_LAB2RGB)[0, 0]

    return label_img, centers_rgb


def absorb_small_components(
    label_img: np.ndarray,
    centers_rgb: np.ndarray,
    min_area: int,
    passes: int,
) -> np.ndarray:
    labels = label_img.copy()
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
                current_color = centers_rgb[label_id].astype(np.float32)

                best_label = int(vals[0])
                best_key = None

                for v, c in zip(vals, counts):
                    dist = np.linalg.norm(current_color - centers_rgb[int(v)].astype(np.float32))
                    key = (-int(c), float(dist))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_label = int(v)

                labels[comp_mask] = best_label
                changed = True

        if not changed:
            break

    return labels


def snap_points_to_edges(
    pts: np.ndarray,
    edge_map: np.ndarray,
    radius: int,
    blend: float,
) -> np.ndarray:
    if radius <= 0:
        return pts

    h, w = edge_map.shape
    out = pts.astype(np.float32).copy()

    for i, p in enumerate(out):
        x = int(round(p[0]))
        y = int(round(p[1]))

        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)

        window = edge_map[y1:y2, x1:x2]
        ys, xs = np.where(window > 0)

        if len(xs) == 0:
            continue

        xs = xs + x1
        ys = ys + y1

        dist = (xs - x) ** 2 + (ys - y) ** 2
        j = int(np.argmin(dist))

        target = np.array([xs[j], ys[j]], dtype=np.float32)
        out[i] = out[i] * (1.0 - blend) + target * blend

    return out


def contour_to_path(
    contour: np.ndarray,
    edge_map: np.ndarray | None,
    edge_snap_radius: int,
    curve_mode: str,
    area: float,
) -> str:
    pts = contour.reshape(-1, 2).astype(np.float32)

    if len(pts) < 3:
        return ""

    if edge_map is not None and edge_snap_radius > 0:
        pts = snap_points_to_edges(
            pts,
            edge_map=edge_map,
            radius=edge_snap_radius,
            blend=0.72,
        )

    # Tiny detail shapes are better kept sharper.
    use_curve = curve_mode != "off" and len(pts) >= 5 and area >= 8

    if not use_curve:
        p = pts
        d = f"M{fmt_num(p[0][0])} {fmt_num(p[0][1])}"
        if len(p) > 1:
            d += "L" + " ".join(f"{fmt_num(x)} {fmt_num(y)}" for x, y in p[1:])
        d += "Z"
        return d

    # Smooth closed quadratic path.
    # This removes jagged stair-step outlines while still following contour vertices.
    mids = []
    n = len(pts)

    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        mids.append((a + b) / 2.0)

    d = f"M{fmt_num(mids[-1][0])} {fmt_num(mids[-1][1])}"

    for i in range(n):
        c = pts[i]
        end = mids[i]
        d += f"Q{fmt_num(c[0])} {fmt_num(c[1])} {fmt_num(end[0])} {fmt_num(end[1])}"

    d += "Z"
    return d


def extract_shapes_from_labels(
    label_img: np.ndarray,
    centers_rgb: np.ndarray,
    detail: int,
    min_area: int,
    epsilon_multiplier: float,
    smooth_masks: bool,
    edge_map: np.ndarray | None,
    edge_snap_radius: int,
    curve_mode: str,
) -> List[Shape]:
    shapes: List[Shape] = []

    detail = int(np.clip(detail, 1, 100))
    base_epsilon = max(0.16, (101 - detail) * 0.018)
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

            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            d = contour_to_path(
                approx,
                edge_map=edge_map,
                edge_snap_radius=edge_snap_radius,
                curve_mode=curve_mode,
                area=area,
            )

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
                        hole_d = contour_to_path(
                            hole_approx,
                            edge_map=edge_map,
                            edge_snap_radius=edge_snap_radius,
                            curve_mode=curve_mode,
                            area=hole_area,
                        )

                        if hole_d:
                            holes.append(hole_approx)
                            d += hole_d

                child = hierarchy[child][0]

            shapes.append(
                Shape(
                    area=float(area),
                    fill_rgb=fill_rgb,
                    fill_hex=fill_hex,
                    outer=approx,
                    holes=holes,
                    d=d,
                    has_holes=len(holes) > 0,
                    kind="color",
                )
            )

    shapes.sort(key=lambda s: s.area, reverse=True)
    return shapes


def extract_line_detail_shapes(
    original_rgb: np.ndarray,
    detail: int,
    min_area: int,
    strength: str,
    edge_map: np.ndarray,
    curve_mode: str,
) -> List[Shape]:
    if strength == "off":
        return []

    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 24, 24)

    if strength == "medium":
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(edge_map, kernel, iterations=1)
    else:
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(edge_map, kernel, iterations=1)

        dark_thr = np.percentile(blur, 26)
        dark = (blur <= dark_thr).astype(np.uint8) * 255

        mask = cv2.bitwise_or(mask, cv2.bitwise_and(mask, dark))

    pixels = original_rgb[mask > 0]

    if len(pixels) > 0:
        line_rgb = tuple(np.percentile(pixels, 20, axis=0).astype(np.uint8).tolist())
    else:
        line_rgb = (28, 28, 28)

    # Keep line reinforcement dark but not always pure black.
    line_rgb = tuple(int(max(0, min(95, x))) for x in line_rgb)
    line_hex = rgb_to_hex(line_rgb)

    detail = int(np.clip(detail, 1, 100))
    epsilon = max(0.12, (101 - detail) * 0.012)

    min_area = max(1, int(min_area))

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    shapes: List[Shape] = []

    for contour in contours:
        area = abs(cv2.contourArea(contour))

        if area < min_area:
            continue

        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        d = contour_to_path(
            approx,
            edge_map=edge_map,
            edge_snap_radius=1,
            curve_mode=curve_mode,
            area=area,
        )

        if not d:
            continue

        shapes.append(
            Shape(
                area=float(area),
                fill_rgb=line_rgb,
                fill_hex=line_hex,
                outer=approx,
                holes=[],
                d=d,
                has_holes=False,
                kind="line",
            )
        )

    return shapes


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


def extract_residual_micro_shapes(
    original_rgb: np.ndarray,
    vector_rgb: np.ndarray,
    detail: int,
    min_area: int,
    max_micro_shapes: int,
    edge_map: np.ndarray,
    curve_mode: str,
) -> List[Shape]:
    h, w, _ = original_rgb.shape

    orig_lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    vec_lab = cv2.cvtColor(vector_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    delta = np.linalg.norm(orig_lab - vec_lab, axis=2)

    orig_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    vec_gray = cv2.cvtColor(vector_rgb, cv2.COLOR_RGB2GRAY)

    orig_edges = cv2.Canny(orig_gray, 42, 130)
    vec_edges = cv2.Canny(vec_gray, 42, 130)
    vec_edges_dilated = cv2.dilate(vec_edges, np.ones((3, 3), np.uint8), iterations=1)

    missed_edges = cv2.bitwise_and(orig_edges, cv2.bitwise_not(vec_edges_dilated))

    color_threshold = max(16.0, float(np.percentile(delta, 84)))
    color_error = (delta >= color_threshold).astype(np.uint8) * 255

    residual = cv2.bitwise_or(color_error, missed_edges)
    residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    count, comp, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)

    detail = int(np.clip(detail, 1, 100))
    epsilon = max(0.08, (101 - detail) * 0.010)

    min_area = max(1, int(min_area))
    max_area = int(h * w * 0.035)

    candidates = []

    for comp_id in range(1, count):
        area_px = int(stats[comp_id, cv2.CC_STAT_AREA])

        if area_px < min_area:
            continue

        if area_px > max_area:
            continue

        ys, xs = np.where(comp == comp_id)

        if len(xs) == 0:
            continue

        mean_error = float(delta[ys, xs].mean())
        priority = mean_error * np.sqrt(area_px)

        candidates.append((priority, comp_id, area_px))

    candidates.sort(reverse=True)
    candidates = candidates[:max(1, int(max_micro_shapes))]

    shapes: List[Shape] = []

    for _, comp_id, area_px in candidates:
        mask = (comp == comp_id).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        pixels = original_rgb[mask > 0]
        if len(pixels) == 0:
            continue

        rgb = tuple(np.median(pixels, axis=0).astype(np.uint8).tolist())
        fill_hex = rgb_to_hex(rgb)

        for contour in contours:
            area = abs(cv2.contourArea(contour))

            if area < min_area:
                continue

            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            d = contour_to_path(
                approx,
                edge_map=edge_map,
                edge_snap_radius=1,
                curve_mode=curve_mode,
                area=area,
            )

            if not d:
                continue

            shapes.append(
                Shape(
                    area=float(area),
                    fill_rgb=rgb,
                    fill_hex=fill_hex,
                    outer=approx,
                    holes=[],
                    d=d,
                    has_holes=False,
                    kind="micro",
                )
            )

    return shapes


def shape_importance(shape: Shape) -> float:
    cost = max(1, byte_len(shape.d))

    if shape.kind == "micro":
        boost = 9.0
        base = 140.0
    elif shape.kind == "line":
        boost = 7.0
        base = 120.0
    else:
        boost = 1.0
        base = 0.0

    return (shape.area * boost + base) / cost


def make_svg(shapes: List[Shape], width: int, height: int) -> str:
    ordered = sorted(shapes, key=lambda s: s.area, reverse=True)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
    ]

    i = 0
    n = len(ordered)

    while i < n:
        fill = ordered[i].fill_hex
        d_parts = [ordered[i].d]
        has_holes = ordered[i].has_holes
        j = i + 1

        # Only merge adjacent same-color layers.
        # This reduces tag size without breaking area order.
        while j < n and ordered[j].fill_hex == fill:
            d_parts.append(ordered[j].d)
            has_holes = has_holes or ordered[j].has_holes
            j += 1

        rule = ' fill-rule="evenodd"' if has_holes else ""
        out.append(f'<path fill="{fill}" d="{"".join(d_parts)}"{rule}/>')

        i = j

    out.append("</svg>")
    return "".join(out)


def pack_layer_parts(
    shapes: List[Shape],
    width: int,
    height: int,
    per_part_limit_bytes: int,
) -> List[str]:
    ordered = sorted(shapes, key=lambda s: s.area, reverse=True)

    parts: List[List[Shape]] = []
    current: List[Shape] = []

    for shape in ordered:
        test = current + [shape]
        test_svg = make_svg(test, width, height)

        if current and byte_len(test_svg) > per_part_limit_bytes:
            parts.append(current)
            current = [shape]
        else:
            current = test

    if current:
        parts.append(current)

    return [make_svg(p, width, height) for p in parts]


def fit_shapes_to_budget(
    shapes: List[Shape],
    width: int,
    height: int,
    total_limit_bytes: int,
    per_part_limit_bytes: int,
    protect_details: bool,
):
    shapes = sorted(shapes, key=lambda s: s.area, reverse=True)
    removed = 0

    while True:
        full_svg = make_svg(shapes, width, height)
        part_svgs = pack_layer_parts(shapes, width, height, per_part_limit_bytes)
        total_part_size = sum(byte_len(x) for x in part_svgs)

        if byte_len(full_svg) <= total_limit_bytes and total_part_size <= total_limit_bytes:
            return shapes, full_svg, part_svgs, removed

        if len(shapes) <= 1:
            return shapes, full_svg, part_svgs, removed

        scores = []

        for idx, s in enumerate(shapes):
            # Avoid deleting the largest base surface until necessary.
            if idx == 0 and len(shapes) > 8:
                continue

            # In detail-protection mode, fine details are removed later.
            if protect_details and s.kind in ("micro", "line") and len(shapes) > 12:
                penalty = shape_importance(s) * 1.8
            else:
                penalty = shape_importance(s)

            scores.append((idx, penalty))

        drop_idx = min(scores, key=lambda x: x[1])[0]
        shapes.pop(drop_idx)
        removed += 1


def score_vector_result(original_rgb: np.ndarray, vector_rgb: np.ndarray, size_bytes: int, path_count: int) -> float:
    orig_lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    vec_lab = cv2.cvtColor(vector_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    lab_delta = np.linalg.norm(orig_lab - vec_lab, axis=2).mean()

    orig_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    vec_gray = cv2.cvtColor(vector_rgb, cv2.COLOR_RGB2GRAY)

    orig_edges = cv2.Canny(orig_gray, 42, 130)
    vec_edges = cv2.Canny(vec_gray, 42, 130)

    missed = cv2.bitwise_and(orig_edges, cv2.bitwise_not(cv2.dilate(vec_edges, np.ones((3, 3), np.uint8), iterations=1)))
    extra = cv2.bitwise_and(vec_edges, cv2.bitwise_not(cv2.dilate(orig_edges, np.ones((3, 3), np.uint8), iterations=1)))

    missed_edge_penalty = missed.mean() / 255.0
    extra_edge_penalty = extra.mean() / 255.0

    # Small penalty so oversized / too-many-path results do not win when visually similar.
    size_penalty = size_bytes / 1024.0 * 0.010
    path_penalty = path_count * 0.002

    return float(
        lab_delta
        + missed_edge_penalty * 36.0
        + extra_edge_penalty * 14.0
        + size_penalty
        + path_penalty
    )


def build_trials(
    base_colors: int,
    strength: str,
    max_trials: int,
):
    if strength == "Fast":
        preps = ["meanshift", "hybrid"]
        color_opts = [base_colors, base_colors + 4]
        eps_opts = [0.75, 1.0]
        area_opts = [0.8, 1.2]

    elif strength == "Balanced":
        preps = ["meanshift", "hybrid", "bilateral", "detail_preserve"]
        color_opts = [base_colors, base_colors + 4, base_colors + 8, max(3, base_colors - 4)]
        eps_opts = [0.55, 0.75, 1.0, 1.25]
        area_opts = [0.55, 0.8, 1.1, 1.55]

    else:
        preps = ["detail_preserve", "hybrid", "meanshift", "bilateral", "none"]
        color_opts = [base_colors, base_colors + 4, base_colors + 8, base_colors + 12, max(3, base_colors - 4)]
        eps_opts = [0.42, 0.58, 0.75, 1.0, 1.28]
        area_opts = [0.42, 0.62, 0.85, 1.15, 1.6]

    color_opts = [int(np.clip(c, 2, 54)) for c in color_opts]
    color_opts = list(dict.fromkeys(color_opts))

    trials = []

    for prep in preps:
        for colors in color_opts:
            for eps in eps_opts:
                for area_mul in area_opts:
                    trials.append(
                        {
                            "prep": prep,
                            "colors": colors,
                            "epsilon_multiplier": eps,
                            "area_multiplier": area_mul,
                        }
                    )

    trials.sort(
        key=lambda t: (
            abs(t["colors"] - base_colors),
            t["epsilon_multiplier"],
            t["area_multiplier"],
            0 if t["prep"] in ("detail_preserve", "hybrid") else 1,
        )
    )

    return trials[:max(1, int(max_trials))]


def generate_candidate(
    original_rgb: np.ndarray,
    trial: Dict[str, Any],
    detail: int,
    min_area: int,
    smooth_masks: bool,
    total_limit_bytes: int,
    per_part_limit_bytes: int,
    line_strength: str,
    micro_corrections: int,
    max_micro_shapes: int,
    edge_snap_radius: int,
    curve_mode: str,
    protect_details: bool,
) -> Candidate:
    h, w, _ = original_rgb.shape

    edge_map = source_edge_map(original_rgb, "strong" if line_strength == "strong" else "normal")

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

    base_shapes = extract_shapes_from_labels(
        label_img=label_img,
        centers_rgb=centers_rgb,
        detail=detail,
        min_area=effective_min_area,
        epsilon_multiplier=trial["epsilon_multiplier"],
        smooth_masks=smooth_masks,
        edge_map=edge_map,
        edge_snap_radius=edge_snap_radius,
        curve_mode=curve_mode,
    )

    # Reserve some space for micro corrections.
    reserve_ratio = 0.80 if micro_corrections > 0 or line_strength != "off" else 1.0

    base_shapes, _, _, _ = fit_shapes_to_budget(
        shapes=base_shapes,
        width=w,
        height=h,
        total_limit_bytes=int(total_limit_bytes * reserve_ratio),
        per_part_limit_bytes=per_part_limit_bytes,
        protect_details=False,
    )

    background = dominant_corner_rgb(original_rgb)

    current_shapes = list(base_shapes)

    if line_strength != "off":
        line_shapes = extract_line_detail_shapes(
            original_rgb=original_rgb,
            detail=detail,
            min_area=max(1, int(effective_min_area * 0.38)),
            strength=line_strength,
            edge_map=edge_map,
            curve_mode=curve_mode,
        )
        current_shapes.extend(line_shapes)

    for _ in range(max(0, int(micro_corrections))):
        preview = render_shapes(current_shapes, w, h, background)

        micro_shapes = extract_residual_micro_shapes(
            original_rgb=original_rgb,
            vector_rgb=preview,
            detail=detail,
            min_area=max(1, int(effective_min_area * 0.28)),
            max_micro_shapes=max_micro_shapes,
            edge_map=edge_map,
            curve_mode=curve_mode,
        )

        if not micro_shapes:
            break

        current_shapes.extend(micro_shapes)

        current_shapes, _, _, _ = fit_shapes_to_budget(
            shapes=current_shapes,
            width=w,
            height=h,
            total_limit_bytes=total_limit_bytes,
            per_part_limit_bytes=per_part_limit_bytes,
            protect_details=protect_details,
        )

    final_shapes, full_svg, part_svgs, removed = fit_shapes_to_budget(
        shapes=current_shapes,
        width=w,
        height=h,
        total_limit_bytes=total_limit_bytes,
        per_part_limit_bytes=per_part_limit_bytes,
        protect_details=protect_details,
    )

    preview_rgb = render_shapes(final_shapes, w, h, background)

    total_part_size = sum(byte_len(x) for x in part_svgs)
    score_size = max(byte_len(full_svg), total_part_size)

    score = score_vector_result(
        original_rgb=original_rgb,
        vector_rgb=preview_rgb,
        size_bytes=score_size,
        path_count=len(final_shapes),
    )

    info = {
        **trial,
        "shape_count": len(final_shapes),
        "removed_shapes": removed,
        "full_svg_bytes": byte_len(full_svg),
        "part_count": len(part_svgs),
        "total_part_bytes": total_part_size,
        "line_strength": line_strength,
        "micro_corrections": micro_corrections,
        "edge_snap_radius": edge_snap_radius,
        "curve_mode": curve_mode,
        "score": score,
    }

    return Candidate(
        score=score,
        shapes=final_shapes,
        full_svg=full_svg,
        part_svgs=part_svgs,
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


def build_report(candidate: Candidate, width: int, height: int, total_limit_bytes: int, per_part_limit_bytes: int) -> str:
    lines = []
    lines.append("Fine Detail Optimized Path Layer SVG Report")
    lines.append("------------------------------------------")
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
    for i, s in enumerate(sorted(candidate.shapes, key=lambda x: x.area, reverse=True), start=1):
        lines.append(
            f"{i:04d}: area={int(s.area)} fill={s.fill_hex} kind={s.kind} path_bytes={byte_len(s.d)}"
        )
    return "\n".join(lines)


def render_preview_html(original_rgb: np.ndarray, vector_rgb: np.ndarray, svg_text: str, mode: str):
    original_uri = array_to_data_uri(original_rgb)
    vector_uri = array_to_data_uri(vector_rgb)

    if mode == "Original":
        body = f'<img class="fit" src="{original_uri}"/>'

    elif mode == "Vector":
        body = f'<img class="fit" src="{vector_uri}"/>'

    elif mode == "Overlay":
        body = f"""
        <div class="stage">
          <img class="fit base" src="{original_uri}"/>
          <div class="svg-layer">{svg_text}</div>
        </div>
        """

    else:
        body = f"""
        <div class="compare">
          <div>
            <div class="label">Original</div>
            <img class="fit" src="{original_uri}"/>
          </div>
          <div>
            <div class="label">Vector</div>
            <img class="fit" src="{vector_uri}"/>
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
      .base {{
        opacity: 0.42;
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
        padding: 6px;
        font-size: 13px;
        background: #f0f0f0;
      }}
    </style>
    {body}
    """

    h = original_rgb.shape[0]
    components.html(html_doc, height=max(360, min(820, int(h * 1.55))), scrolling=True)


def main():
    st.set_page_config(
        page_title="Fine Detail Path Layer SVG Vectorizer",
        page_icon="🧩",
        layout="wide",
    )

    st.title("🧩 Fine Detail Path Layer SVG Vectorizer")
    st.caption("Path-only layered SVG with curve smoothing, edge correction, and micro-detail repair.")

    uploaded = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    st.sidebar.header("Quality")

    max_side = st.sidebar.slider(
        "Processing size",
        min_value=260,
        max_value=1000,
        value=640,
        step=20,
    )

    base_colors = st.sidebar.slider(
        "Base color count",
        min_value=6,
        max_value=54,
        value=28,
        step=1,
    )

    detail = st.sidebar.slider(
        "Contour detail",
        min_value=30,
        max_value=100,
        value=92,
        step=1,
    )

    min_area = st.sidebar.slider(
        "Noise removal",
        min_value=1,
        max_value=280,
        value=10,
        step=1,
    )

    strength = st.sidebar.selectbox(
        "Optimization strength",
        ["Fast", "Balanced", "Maximum"],
        index=2,
    )

    max_trials = st.sidebar.slider(
        "Candidate trials",
        min_value=4,
        max_value=64,
        value=32,
        step=1,
    )

    st.sidebar.header("Fine detail correction")

    curve_mode = st.sidebar.selectbox(
        "Outline smoothing",
        ["smooth curves", "off"],
        index=0,
    )

    edge_snap_radius = st.sidebar.slider(
        "Edge snap radius",
        min_value=0,
        max_value=4,
        value=2,
        step=1,
        help="Moves contour points toward source-image edges to reduce jagged misalignment.",
    )

    line_strength = st.sidebar.selectbox(
        "Line/detail reinforcement",
        ["off", "medium", "strong"],
        index=2,
    )

    micro_corrections = st.sidebar.slider(
        "Residual correction passes",
        min_value=0,
        max_value=3,
        value=2,
        step=1,
        help="Compares vector result with original and adds small correction path layers.",
    )

    max_micro_shapes = st.sidebar.slider(
        "Max micro-correction shapes per pass",
        min_value=5,
        max_value=160,
        value=56,
        step=1,
    )

    protect_details = st.sidebar.checkbox(
        "Protect small details during size trimming",
        value=True,
    )

    smooth_masks = st.sidebar.checkbox(
        "Smooth color regions",
        value=True,
    )

    st.sidebar.header("Size")

    total_kb = st.sidebar.slider(
        "Total SVG layer limit KB",
        min_value=40,
        max_value=220,
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

    preview_mode = st.sidebar.selectbox(
        "Preview",
        ["Compare", "Overlay", "Vector", "Original"],
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
        strength=strength,
        max_trials=max_trials,
    )

    best: Candidate | None = None

    progress = st.progress(0, text="Starting fine-detail optimization...")

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
                line_strength=line_strength,
                micro_corrections=micro_corrections,
                max_micro_shapes=max_micro_shapes,
                edge_snap_radius=edge_snap_radius,
                curve_mode="smooth" if curve_mode == "smooth curves" else "off",
                protect_details=protect_details,
            )

            if best is None or candidate.score < best.score:
                best = candidate

        except Exception as e:
            st.warning(f"Skipped failed candidate: {e}")

    progress.empty()

    if best is None:
        st.error("Vectorization failed. Try lowering Processing size, Base color count, or Candidate trials.")
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
        st.success("Result fits inside the requested size limit.")
    else:
        st.warning("Result is still above the size target. Increase Noise removal or reduce Base color count.")

    st.subheader("Preview")
    render_preview_html(original_rgb, best.preview_rgb, best.full_svg, preview_mode)

    st.subheader("Download")
    st.download_button(
        "Download ZIP",
        data=zip_bytes,
        file_name="fine_detail_path_layer_svg_export.zip",
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
            "Select layer part",
            options=list(range(1, len(best.part_svgs) + 1)),
            format_func=lambda i: f"layer_part_{i:02d}.svg — {byte_len(best.part_svgs[i - 1]) / 1024:.1f} KB",
        )

        st.text_area(
            f"layer_part_{selected:02d}.svg",
            value=best.part_svgs[selected - 1],
            height=220,
        )

    with st.expander("Winning settings"):
        st.json(best.info)

    with st.expander("Layer details"):
        ordered = sorted(best.shapes, key=lambda s: s.area, reverse=True)
        for i, s in enumerate(ordered, start=1):
            st.write(
                f"{i:04d} | area={int(s.area)} | color={s.fill_hex} | kind={s.kind} | path={byte_len(s.d)} bytes"
            )


if __name__ == "__main__":
    main()