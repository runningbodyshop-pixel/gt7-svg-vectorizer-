from __future__ import annotations

import base64
import io
import math
import re
import time
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFilter, ImageOps

HARD_LIMIT = 2 * 1024 * 1024
DEFAULT_SAFE_PATH_BYTES = 12 * 1024

PRESETS = {
    "高速 / 30秒目標": {
        "colors": 64,
        "side": 1050,
        "simp": 0.70,
        "area": 2,
        "dec": 1,
        "attempts": 1,
        "overlap": 1,
        "smooth": 1,
        "tile": 256,
        "quant": "高速",
        "line": True,
    },
    "標準 / 30秒〜1分": {
        "colors": 96,
        "side": 1350,
        "simp": 0.45,
        "area": 1,
        "dec": 1,
        "attempts": 2,
        "overlap": 1,
        "smooth": 1,
        "tile": 256,
        "quant": "高精度",
        "line": True,
    },
    "高精度 / SVGOMG前提": {
        "colors": 128,
        "side": 1600,
        "simp": 0.32,
        "area": 1,
        "dec": 1,
        "attempts": 2,
        "overlap": 1,
        "smooth": 2,
        "tile": 224,
        "quant": "高精度",
        "line": True,
    },
    "元画像優先 / 重い": {
        "colors": 160,
        "side": 1800,
        "simp": 0.25,
        "area": 1,
        "dec": 1,
        "attempts": 2,
        "overlap": 1,
        "smooth": 2,
        "tile": 192,
        "quant": "高精度",
        "line": True,
    },
}


def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def short_hex(r: int, g: int, b: int) -> str:
    s = f"{r:02x}{g:02x}{b:02x}"
    if s[0] == s[1] and s[2] == s[3] and s[4] == s[5]:
        return f"#{s[0]}{s[2]}{s[4]}"
    return f"#{s}"


def clean_svg(svg: str) -> str:
    # SVGOMGに渡しやすいように余分な空白だけ削る。危険・非対応タグは出力しない設計。
    svg = re.sub(r"\s+", " ", svg)
    svg = svg.replace("> <", "><")
    return svg.strip()


def checkerboard(size: tuple[int, int], cell: int = 12) -> Image.Image:
    w, h = size
    img = Image.new("RGBA", size, (245, 245, 245, 255))
    dr = ImageDraw.Draw(img)
    c1 = (250, 250, 250, 255)
    c2 = (225, 225, 225, 255)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            dr.rectangle(
                [x, y, x + cell - 1, y + cell - 1],
                fill=c1 if ((x // cell) + (y // cell)) % 2 == 0 else c2,
            )
    return img


def paste_on_checkerboard(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    bg = checkerboard(img.size)
    bg.alpha_composite(img, (0, 0))
    return bg


def fit_image(img: Image.Image, side: int, enhance: bool, smooth_color: bool) -> Image.Image:
    img = img.convert("RGBA")

    if enhance:
        rgb = img.convert("RGB")
        rgb = ImageOps.autocontrast(rgb)
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=115, threshold=3))
        img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))

    if max(img.size) > side:
        scale = side / max(img.size)
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.LANCZOS,
        )

    if smooth_color:
        arr = np.array(img)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]
        # 色面だけ軽くなめらかにする。線を潰しすぎない値。
        smooth_rgb = cv2.bilateralFilter(rgb, d=5, sigmaColor=24, sigmaSpace=6)
        arr[:, :, :3] = smooth_rgb
        arr[:, :, 3] = alpha
        img = Image.fromarray(arr, "RGBA")

    return img


def quantize_pillow(img: Image.Image, n_colors: int, alpha_threshold: int, white_bg: bool):
    img = img.convert("RGBA")
    alpha = np.array(img.getchannel("A"))
    visible = alpha >= alpha_threshold

    if white_bg:
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
        visible[:] = True

    q = img.convert("RGB").quantize(
        colors=max(2, int(n_colors)),
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )

    labels = np.array(q, dtype=np.int32)
    labels[~visible] = -1

    pal_raw = q.getpalette()[:768]
    used = sorted(int(x) for x in np.unique(labels) if int(x) >= 0)

    remapped = np.full_like(labels, -1)
    palette: list[tuple[int, int, int]] = []
    for new_i, old_i in enumerate(used):
        remapped[labels == old_i] = new_i
        palette.append(tuple(int(v) for v in pal_raw[old_i * 3 : old_i * 3 + 3]))

    return remapped, palette, visible.astype(np.uint8), img


def quantize_lab_kmeans(
    img: Image.Image,
    n_colors: int,
    alpha_threshold: int,
    white_bg: bool,
    sample_max: int = 120_000,
):
    img = img.convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]
    visible = alpha >= alpha_threshold

    if white_bg:
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
        arr = np.array(img)
        alpha = arr[:, :, 3]
        visible[:] = True

    rgb = arr[:, :, :3]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    pix = lab[visible].reshape(-1, 3).astype(np.float32)

    if pix.size == 0:
        labels = np.full((img.height, img.width), -1, dtype=np.int32)
        return labels, [], visible.astype(np.uint8), img

    k = max(2, min(int(n_colors), len(pix)))
    cv2.setRNGSeed(12345)

    if len(pix) > sample_max:
        idx = np.linspace(0, len(pix) - 1, sample_max).astype(np.int64)
        sample = pix[idx]
    else:
        sample = pix

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 22, 0.6)
    _, _, centers = cv2.kmeans(
        sample,
        k,
        None,
        criteria,
        1,
        cv2.KMEANS_PP_CENTERS,
    )

    # 全ピクセルを中心色へ割り当てる。メモリ節約のため分割計算。
    out_visible = np.empty((len(pix),), dtype=np.int32)
    chunk = 80_000
    for start in range(0, len(pix), chunk):
        p = pix[start : start + chunk]
        d = ((p[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        out_visible[start : start + chunk] = np.argmin(d, axis=1).astype(np.int32)

    labels = np.full((img.height, img.width), -1, dtype=np.int32)
    labels[visible] = out_visible

    # 使用数順に並べ替え。大きい色を先に積むため。
    counts = np.bincount(out_visible, minlength=k)
    order = list(np.argsort(-counts))
    remap = np.full(k, -1, dtype=np.int32)
    for new_i, old_i in enumerate(order):
        if counts[old_i] > 0:
            remap[old_i] = new_i
    valid = labels >= 0
    labels[valid] = remap[labels[valid]]

    centers_ordered = centers[order]
    centers_lab = np.clip(centers_ordered.reshape(1, -1, 3), 0, 255).astype(np.uint8)
    centers_rgb = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB).reshape(-1, 3)

    palette: list[tuple[int, int, int]] = []
    for i, c in enumerate(centers_rgb):
        if i < len(order) and counts[order[i]] > 0:
            palette.append((int(c[0]), int(c[1]), int(c[2])))

    return labels, palette, visible.astype(np.uint8), img


def labels_to_preview(labels: np.ndarray, palette: list[tuple[int, int, int]], white_bg: bool) -> Image.Image:
    h, w = labels.shape
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for i, (r, g, b) in enumerate(palette):
        m = labels == i
        arr[m, 0] = r
        arr[m, 1] = g
        arr[m, 2] = b
        arr[m, 3] = 255
    if white_bg:
        arr[:, :, 3] = 255
    return Image.fromarray(arr, "RGBA")


def fmt_num(x: float, decimals: int) -> str:
    if decimals <= 0:
        return str(int(round(x)))
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".").replace("-0", "0")


def smooth_closed_points(pts: np.ndarray, passes: int) -> np.ndarray:
    if passes <= 0 or len(pts) < 5:
        return pts
    out = pts.astype(np.float32)
    for _ in range(passes):
        prev_pts = np.roll(out, 1, axis=0)
        next_pts = np.roll(out, -1, axis=0)
        out = 0.25 * prev_pts + 0.50 * out + 0.25 * next_pts
    return out


def path_tag_size(d: str) -> int:
    return len(f'<path d="{d}"/>'.encode("utf-8"))


def contour_to_path(
    contour: np.ndarray,
    simplify: float,
    min_area: int,
    decimals: int,
    smooth_passes: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> Optional[str]:
    if contour is None or len(contour) < 3:
        return None

    area = abs(cv2.contourArea(contour))
    if area < min_area:
        return None

    # CHAIN_APPROX_NONEの輪郭を受けて、まず軽く近似。その後なめらか処理。
    approx = cv2.approxPolyDP(contour, epsilon=max(0.03, float(simplify)), closed=True)
    if approx is None or len(approx) < 3:
        return None

    pts = approx.reshape(-1, 2).astype(np.float32)
    pts = smooth_closed_points(pts, smooth_passes)

    # なめらか処理後にもう一度だけ軽く近似して、点数爆発を防ぐ。
    if len(pts) >= 5 and simplify > 0.08:
        tmp = pts.reshape(-1, 1, 2).astype(np.float32)
        approx2 = cv2.approxPolyDP(tmp, epsilon=max(0.02, simplify * 0.45), closed=True)
        if approx2 is not None and len(approx2) >= 3:
            pts = approx2.reshape(-1, 2).astype(np.float32)

    if len(pts) < 3:
        return None

    pts[:, 0] += offset_x
    pts[:, 1] += offset_y

    cmds = [f"M{fmt_num(pts[0, 0], decimals)} {fmt_num(pts[0, 1], decimals)}"]
    for x, y in pts[1:]:
        cmds.append(f"L{fmt_num(x, decimals)} {fmt_num(y, decimals)}")
    cmds.append("Z")
    return "".join(cmds)


def connected_clean(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if min_area <= 1:
        return binary * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def split_contour_mask(
    binary: np.ndarray,
    contour: np.ndarray,
    overlap: int = 1,
) -> list[tuple[np.ndarray, int, int]]:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 2 or h <= 2:
        return []

    local = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, :, 0] -= x
    shifted[:, :, 1] -= y
    cv2.drawContours(local, [shifted], -1, 255, thickness=-1)

    pieces: list[tuple[np.ndarray, int, int]] = []
    if w >= h:
        mid = w // 2
        ranges = [(0, min(w, mid + overlap)), (max(0, mid - overlap), w)]
        for sx0, sx1 in ranges:
            crop = local[:, sx0:sx1]
            if crop.size and crop.max() > 0:
                pieces.append((crop, x + sx0, y))
    else:
        mid = h // 2
        ranges = [(0, min(h, mid + overlap)), (max(0, mid - overlap), h)]
        for sy0, sy1 in ranges:
            crop = local[sy0:sy1, :]
            if crop.size and crop.max() > 0:
                pieces.append((crop, x, y + sy0))
    return pieces


def paths_from_binary_no_tile(
    binary: np.ndarray,
    simplify: float,
    min_area: int,
    decimals: int,
    smooth_passes: int,
    max_path_bytes: int,
    offset_x: int = 0,
    offset_y: int = 0,
    depth: int = 0,
    max_depth: int = 8,
) -> list[str]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    paths: list[str] = []

    for contour in contours:
        d = contour_to_path(contour, simplify, min_area, decimals, smooth_passes, offset_x, offset_y)
        if not d:
            continue

        if path_tag_size(d) <= max_path_bytes:
            paths.append(d)
            continue

        # なるべく精度を落とさず、まず分割で15KB超えを避ける。
        if depth < max_depth:
            pieces = split_contour_mask(binary, contour, overlap=1)
            if pieces:
                for crop, ox, oy in pieces:
                    sub = paths_from_binary_no_tile(
                        crop,
                        simplify=simplify,
                        min_area=min_area,
                        decimals=decimals,
                        smooth_passes=smooth_passes,
                        max_path_bytes=max_path_bytes,
                        offset_x=offset_x + ox,
                        offset_y=offset_y + oy,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                    paths.extend(sub)
                continue

        # どうしても長すぎる場合だけ、段階的に簡略化する。
        forced = simplify
        for _ in range(6):
            forced *= 1.45
            d2 = contour_to_path(contour, forced, min_area, decimals, smooth_passes=0, offset_x=offset_x, offset_y=offset_y)
            if d2 and path_tag_size(d2) <= max_path_bytes:
                paths.append(d2)
                break
        else:
            # 最後の保険。完全に捨てるよりは簡略版を残す。
            if d2:
                paths.append(d2)

    return paths


def paths_from_binary_tiled(
    binary: np.ndarray,
    simplify: float,
    min_area: int,
    decimals: int,
    smooth_passes: int,
    max_path_bytes: int,
    tile_size: int,
) -> list[str]:
    h, w = binary.shape
    if tile_size <= 0 or (w <= tile_size and h <= tile_size):
        return paths_from_binary_no_tile(
            binary,
            simplify=simplify,
            min_area=min_area,
            decimals=decimals,
            smooth_passes=smooth_passes,
            max_path_bytes=max_path_bytes,
        )

    paths: list[str] = []
    overlap = 1
    step = max(32, tile_size)

    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            x1 = min(w, x0 + step)
            y1 = min(h, y0 + step)
            sx0 = max(0, x0 - overlap)
            sy0 = max(0, y0 - overlap)
            sx1 = min(w, x1 + overlap)
            sy1 = min(h, y1 + overlap)
            crop = binary[sy0:sy1, sx0:sx1]
            if crop.size == 0 or crop.max() == 0:
                continue
            sub = paths_from_binary_no_tile(
                crop,
                simplify=simplify,
                min_area=min_area,
                decimals=decimals,
                smooth_passes=smooth_passes,
                max_path_bytes=max_path_bytes,
                offset_x=sx0,
                offset_y=sy0,
            )
            paths.extend(sub)

    return paths


def dominant_color(palette: list[tuple[int, int, int]], labels: np.ndarray) -> tuple[int, int, int]:
    if not palette:
        return (0, 0, 0)
    best_i = 0
    best_count = -1
    for i in range(len(palette)):
        cnt = int((labels == i).sum())
        if cnt > best_count:
            best_count = cnt
            best_i = i
    return palette[best_i]


def darkest_palette_color(palette: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    if not palette:
        return (15, 18, 26)
    return min(palette, key=lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def add_group(parts: list[str], fill: str, paths: list[str]) -> tuple[int, int, int]:
    if not paths:
        return 0, 0, 0
    parts.append(f'<g fill="{fill}">')
    max_size = 0
    for d in paths:
        max_size = max(max_size, path_tag_size(d))
        parts.append(f'<path d="{d}"/>')
    parts.append("</g>")
    return 1, len(paths), max_size


def make_line_mask(work_img: Image.Image, visible_mask: np.ndarray, threshold: int, min_alpha: int) -> np.ndarray:
    arr = np.array(work_img.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    dark = ((gray <= threshold) & (alpha >= min_alpha) & (visible_mask > 0)).astype(np.uint8) * 255

    # 細線が切れすぎない程度に接続。太りすぎを避けるため控えめ。
    kernel = np.ones((2, 2), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)
    return dark


def build_svg_layered(
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    visible_mask: np.ndarray,
    work_img: Image.Image,
    width: int,
    height: int,
    simplify: float,
    min_area: int,
    decimals: int,
    smooth_passes: int,
    white_bg: bool,
    use_underpaint: bool,
    overlap_px: int,
    max_path_bytes: int,
    tile_size: int,
    line_layer: bool,
    line_threshold: int,
):
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]

    group_count = 0
    path_count = 0
    max_path_size = 0

    if white_bg:
        d = f"M0 0H{width}V{height}H0Z"
        parts.append(f'<path fill="#fff" d="{d}"/>')
        path_count += 1
        max_path_size = max(max_path_size, path_tag_size(d))

    if use_underpaint and visible_mask is not None and visible_mask.max() > 0:
        base_binary = connected_clean(visible_mask > 0, min_area=max(1, min_area))
        paths = paths_from_binary_tiled(
            base_binary,
            simplify=max(0.12, simplify * 1.15),
            min_area=max(1, min_area),
            decimals=decimals,
            smooth_passes=max(0, smooth_passes - 1),
            max_path_bytes=max_path_bytes,
            tile_size=tile_size,
        )
        r, g, b = dominant_color(palette, labels)
        g_count, p_count, mx = add_group(parts, short_hex(r, g, b), paths)
        group_count += g_count
        path_count += p_count
        max_path_size = max(max_path_size, mx)

    order = sorted(
        [(int((labels == i).sum()), i) for i in range(len(palette))],
        reverse=True,
    )

    kernel = None
    if overlap_px > 0:
        k = overlap_px * 2 + 1
        kernel = np.ones((k, k), np.uint8)
        visible_binary = (visible_mask > 0).astype(np.uint8) * 255
        visible_binary = cv2.dilate(visible_binary, kernel, iterations=1)
    else:
        visible_binary = (visible_mask > 0).astype(np.uint8) * 255

    for _, color_index in order:
        mask = labels == color_index
        if int(mask.sum()) < min_area:
            continue

        binary = connected_clean(mask, min_area=max(1, min_area))

        if kernel is not None:
            binary = cv2.dilate(binary, kernel, iterations=1)
            binary = cv2.bitwise_and(binary, visible_binary)

        if binary.max() == 0:
            continue

        paths = paths_from_binary_tiled(
            binary,
            simplify=simplify,
            min_area=max(1, min_area),
            decimals=decimals,
            smooth_passes=smooth_passes,
            max_path_bytes=max_path_bytes,
            tile_size=tile_size,
        )

        if not paths:
            continue

        r, g, b = palette[color_index]
        g_count, p_count, mx = add_group(parts, short_hex(r, g, b), paths)
        group_count += g_count
        path_count += p_count
        max_path_size = max(max_path_size, mx)

    # 線画を最後に重ねる。細い黒線や目・髪の輪郭の欠落を補う。
    if line_layer:
        line_binary = make_line_mask(work_img, visible_mask, threshold=line_threshold, min_alpha=8)
        line_binary = connected_clean(line_binary, min_area=max(1, min_area))
        if line_binary.max() > 0:
            paths = paths_from_binary_tiled(
                line_binary,
                simplify=max(0.18, simplify * 0.85),
                min_area=max(1, min_area),
                decimals=decimals,
                smooth_passes=max(0, smooth_passes - 1),
                max_path_bytes=max_path_bytes,
                tile_size=tile_size,
            )
            r, g, b = darkest_palette_color(palette)
            g_count, p_count, mx = add_group(parts, short_hex(r, g, b), paths)
            group_count += g_count
            path_count += p_count
            max_path_size = max(max_path_size, mx)

    parts.append("</svg>")
    svg = clean_svg("".join(parts))
    return svg, group_count, path_count, max_path_size


def make_reference_image(original: Image.Image, quant_preview: Image.Image) -> bytes:
    def fit_for_preview(im: Image.Image) -> Image.Image:
        im = im.convert("RGBA")
        if im.height > 900:
            scale = 900 / im.height
            im = im.resize((int(im.width * scale), 900), Image.Resampling.LANCZOS)
        return im

    a = paste_on_checkerboard(fit_for_preview(original))
    b = paste_on_checkerboard(fit_for_preview(quant_preview))

    margin = 16
    gap = 24
    title_h = 36
    w = a.width + b.width + gap + margin * 2
    h = max(a.height, b.height) + title_h + margin * 2

    canvas = Image.new("RGBA", (w, h), (245, 245, 245, 255))
    dr = ImageDraw.Draw(canvas)
    dr.text((margin, 10), "Original", fill=(0, 0, 0))
    dr.text((margin + a.width + gap, 10), "Color reference", fill=(0, 0, 0))
    canvas.alpha_composite(a, (margin, title_h + margin))
    canvas.alpha_composite(b, (margin + a.width + gap, title_h + margin))
    return png_bytes(canvas)


def convert_once(
    img: Image.Image,
    colors: int,
    side: int,
    simplify: float,
    min_area: int,
    decimals: int,
    smooth_passes: int,
    alpha_threshold: int,
    enhance: bool,
    smooth_color: bool,
    white_bg: bool,
    use_underpaint: bool,
    overlap_px: int,
    max_path_bytes: int,
    tile_size: int,
    quant_mode: str,
    line_layer: bool,
    line_threshold: int,
):
    work = fit_image(img, side=side, enhance=enhance, smooth_color=smooth_color)

    if quant_mode == "高精度":
        labels, palette, visible_mask, processed_img = quantize_lab_kmeans(
            work,
            n_colors=colors,
            alpha_threshold=alpha_threshold,
            white_bg=white_bg,
        )
    else:
        labels, palette, visible_mask, processed_img = quantize_pillow(
            work,
            n_colors=colors,
            alpha_threshold=alpha_threshold,
            white_bg=white_bg,
        )

    quant_preview = labels_to_preview(labels, palette, white_bg)

    svg, groups, paths, max_path_size = build_svg_layered(
        labels=labels,
        palette=palette,
        visible_mask=visible_mask,
        work_img=processed_img,
        width=work.width,
        height=work.height,
        simplify=simplify,
        min_area=min_area,
        decimals=decimals,
        smooth_passes=smooth_passes,
        white_bg=white_bg,
        use_underpaint=use_underpaint,
        overlap_px=overlap_px,
        max_path_bytes=max_path_bytes,
        tile_size=tile_size,
        line_layer=line_layer,
        line_threshold=line_threshold,
    )

    return {
        "svg": svg,
        "size": len(svg.encode("utf-8")),
        "reference": make_reference_image(work, quant_preview),
        "colors": len(palette),
        "groups": groups,
        "paths": paths,
        "max_path_size": max_path_size,
        "width": work.width,
        "height": work.height,
    }


def convert_image(img: Image.Image, cfg: dict):
    target = min(int(cfg["target"]), HARD_LIMIT)

    colors = int(cfg["colors"])
    side = int(cfg["side"])
    simplify = float(cfg["simp"])
    min_area = int(cfg["area"])
    decimals = int(cfg["dec"])
    attempts = int(cfg["attempts"])

    best = None

    for i in range(attempts):
        result = convert_once(
            img=img,
            colors=max(2, colors),
            side=max(128, side),
            simplify=max(0.03, simplify),
            min_area=max(1, min_area),
            decimals=max(0, decimals),
            smooth_passes=max(0, int(cfg["smooth"])),
            alpha_threshold=int(cfg["alpha"]),
            enhance=bool(cfg["enhance"]),
            smooth_color=bool(cfg["smooth_color"]),
            white_bg=bool(cfg["white_bg"]),
            use_underpaint=bool(cfg["underpaint"]),
            overlap_px=int(cfg["overlap"]),
            max_path_bytes=int(cfg["max_path_kb"] * 1024),
            tile_size=int(cfg["tile"]),
            quant_mode=str(cfg["quant"]),
            line_layer=bool(cfg["line"]),
            line_threshold=int(cfg["line_threshold"]),
        )
        result["attempts"] = i + 1

        if best is None:
            best = result
        else:
            if result["size"] <= target and best["size"] <= target:
                # SVGOMG前提なので、目標内なら情報量が多い方を優先。
                if result["size"] > best["size"]:
                    best = result
            elif result["size"] <= target < best["size"]:
                best = result
            elif result["size"] > target and best["size"] > target:
                if result["size"] < best["size"]:
                    best = result

        if result["size"] <= target:
            return result

        # サイズ超過時のみ、画質低下を最小限にして軽量化。
        colors = max(16, int(colors * 0.88))
        side = max(700, int(side * 0.94))
        simplify = min(3.0, simplify * 1.14)
        if i >= 1:
            min_area = min(10, int(min_area * 1.2 + 1))
        if i >= 1:
            decimals = 0

    return best


def data_url_svg(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return "data:image/svg+xml;base64," + encoded


def target_from_choice(choice: str, custom_kb: int) -> int:
    mapping = {
        "2MB 絶対上限": 2 * 1024 * 1024,
        "1.8MB SVGOMG前提": int(1.8 * 1024 * 1024),
        "1.5MB SVGOMG前提": int(1.5 * 1024 * 1024),
        "1MB": 1024 * 1024,
        "500KB": 500 * 1024,
    }
    if choice == "カスタムKB":
        return max(1, min(2048, int(custom_kb))) * 1024
    return mapping.get(choice, 2 * 1024 * 1024)


# ---------------- UI ----------------
st.set_page_config(page_title="GT7 SVG Vectorizer Split Safe", page_icon="🏁", layout="wide")

st.title("🏁 GT7 SVG Vectorizer")
st.caption("高精度トレース + 1path 15KB未満を狙う分割安全版です。SVGOMG前提の出力に寄せています。")

with st.expander("今回の改良内容", expanded=True):
    st.markdown(
        """
- 色ごとの巨大な `path` を作らず、**同じ色を `<g fill=...>` にまとめて、中の `path` を細かく分割**します。  
- 各 `<path d=.../>` は指定KB未満を狙います。初期値は **12KB** なので、15KB制限の分割サイトに通しやすいです。  
- 色解析は **Lab色空間 k-means** を使えるようにし、前より元画像の色分けに近づけています。  
- 輪郭は `CHAIN_APPROX_NONE` ベースで拾い、必要に応じて軽く滑らかにします。  
- 細い黒線・目・髪の輪郭を残しやすくするため、**線画レイヤー**を最後に重ねられます。  
"""
    )

uploaded = st.file_uploader("画像を選択", type=["png", "jpg", "jpeg", "webp"])

with st.sidebar:
    st.header("設定")

    preset_name = st.selectbox(
        "品質プリセット",
        list(PRESETS.keys()) + ["手動設定"],
        index=1,
    )
    base = PRESETS.get(preset_name, PRESETS["標準 / 30秒〜1分"]).copy()

    size_choice = st.selectbox(
        "目標サイズ",
        ["2MB 絶対上限", "1.8MB SVGOMG前提", "1.5MB SVGOMG前提", "1MB", "500KB", "カスタムKB"],
        index=0,
    )
    custom_kb = st.number_input("カスタムKB", min_value=1, max_value=2048, value=2048)

    st.divider()
    st.subheader("精度")

    base["quant"] = st.selectbox(
        "色解析方式",
        ["高精度", "高速"],
        index=0 if base["quant"] == "高精度" else 1,
    )
    enhance = st.toggle("低画質画像を軽く補正", True)
    smooth_color = st.toggle("色面をなめらかにしてからトレース", True)
    white_bg = st.toggle("透明部分を白背景にする", False)

    base["colors"] = st.slider("色数", 2, 192, int(base["colors"]))
    base["side"] = st.slider("処理サイズ（長辺px）", 256, 2000, int(base["side"]), step=16)
    base["simp"] = st.slider("パス簡略化（小さいほど精密）", 0.05, 4.0, float(base["simp"]), step=0.05)
    base["smooth"] = st.slider("輪郭の滑らかさ", 0, 3, int(base["smooth"]))
    base["area"] = st.slider("小さい形状の削除", 1, 40, int(base["area"]))
    base["dec"] = st.slider("座標の小数桁", 0, 2, int(base["dec"]))
    base["alpha"] = st.slider("透明判定", 0, 255, 16)

    st.divider()
    st.subheader("隙間対策・線画")

    underpaint = st.toggle("下塗りシルエットを入れる", True)
    base["overlap"] = st.slider("色境界の重ね描きpx", 0, 2, int(base["overlap"]))
    base["line"] = st.toggle("線画レイヤーを追加", bool(base["line"]))
    base["line_threshold"] = st.slider("線画として拾う暗さ", 20, 140, 74)

    st.divider()
    st.subheader("15KB分割対策")

    base["max_path_kb"] = st.slider("1path最大KB", 4.0, 14.5, 12.0, step=0.5)
    base["tile"] = st.slider("分割タイルサイズpx", 96, 512, int(base["tile"]), step=32)
    base["attempts"] = st.slider("自動軽量化の試行回数", 1, 4, int(base["attempts"]))

    base["target"] = target_from_choice(size_choice, custom_kb)
    base["enhance"] = enhance
    base["smooth_color"] = smooth_color
    base["white_bg"] = white_bg
    base["underpaint"] = underpaint

    st.divider()
    st.markdown(
        """
**おすすめ**  
まずは **標準 / 30秒〜1分**。粗い場合は以下を上げます。  
- 色数: 128  
- 処理サイズ: 1600  
- パス簡略化: 0.30〜0.40  
- 輪郭の滑らかさ: 1〜2  

SVGOMGでは **Merge paths はOFF推奨** です。ONにすると、分割したpathがまた巨大化することがあります。
"""
    )

if uploaded is None:
    st.info("画像をアップロードしてください。")
    st.stop()

try:
    original_img = Image.open(uploaded).convert("RGBA")
except Exception as e:
    st.error(f"画像を開けませんでした: {e}")
    st.stop()

if st.button("SVGへ変換", type="primary", use_container_width=True):
    start = time.time()
    with st.spinner("変換中です…"):
        st.session_state.result = convert_image(original_img, base)
        st.session_state.elapsed = time.time() - start
        st.session_state.filename = uploaded.name
        st.session_state.used_settings = base.copy()

if "result" not in st.session_state:
    st.subheader("元画像")
    st.image(paste_on_checkerboard(original_img), use_container_width=True)
    st.stop()

result = st.session_state.result
used = st.session_state.used_settings

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("SVGサイズ", f"{result['size']:,} bytes")
m2.metric("目標サイズ", f"{used['target']:,} bytes")
m3.metric("色数", str(result["colors"]))
m4.metric("グループ", str(result["groups"]))
m5.metric("path数", str(result["paths"]))
m6.metric("変換時間", f"{st.session_state.elapsed:.1f} 秒")

m7, m8 = st.columns(2)
m7.metric("最大pathサイズ", f"{result['max_path_size']:,} bytes")
m8.metric("処理後サイズ", f"{result['width']} × {result['height']} px")

if result["size"] <= used["target"]:
    st.success("目標サイズ内です。SVGOMGに通すとさらに軽くなる可能性があります。")
else:
    st.warning("目標サイズを超えています。色数・処理サイズを少し下げるか、パス簡略化を上げてください。")

if result["max_path_size"] <= int(used["max_path_kb"] * 1024):
    st.success("各pathは指定した安全上限内です。")
else:
    st.warning("一部のpathが安全上限を超えています。分割タイルサイズを小さくするか、1path最大KBを少し下げて再変換してください。")

if result["size"] > HARD_LIMIT:
    st.error("2MBの絶対上限を超えています。SVGOMG前に設定を軽くしてください。")

if st.session_state.elapsed > 60:
    st.warning("1分を超えました。処理サイズ・色数・試行回数を下げると速くなります。")

svg_url = data_url_svg(result["svg"])
orig_png_url = "data:image/png;base64," + base64.b64encode(png_bytes(paste_on_checkerboard(original_img))).decode("ascii")

st.subheader("実データSVGプレビュー")
components.html(
    f"""
    <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap;">
      <div style="flex:1;min-width:260px;">
        <div style="font-weight:bold;margin-bottom:8px;">Original</div>
        <div style="border:1px solid #ddd;border-radius:12px;padding:12px;background:#eee;text-align:center;">
          <img src="{orig_png_url}" style="max-width:100%;height:auto;">
        </div>
      </div>
      <div style="flex:1;min-width:260px;">
        <div style="font-weight:bold;margin-bottom:8px;">Actual SVG</div>
        <div style="border:1px solid #ddd;border-radius:12px;padding:12px;background:#eee;text-align:center;">
          <img src="{svg_url}" style="max-width:100%;height:auto;">
        </div>
      </div>
    </div>
    """,
    height=650,
    scrolling=True,
)

with st.expander("参考用：色分けプレビューPNG"):
    st.markdown("これは実際のSVG描画ではなく、色分けの確認用です。実データは上の Actual SVG を見てください。")
    st.image(result["reference"], use_container_width=True)

base_name = re.sub(r"\.[^.]+$", "", st.session_state.filename)
base_name = re.sub(r"[^A-Za-z0-9_-]+", "_", base_name).strip("_") or "converted"

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "SVGを保存",
        data=result["svg"].encode("utf-8"),
        file_name=f"{base_name}_split_safe.svg",
        mime="image/svg+xml",
        use_container_width=True,
    )
with d2:
    st.download_button(
        "参考PNGを保存",
        data=result["reference"],
        file_name=f"{base_name}_reference.png",
        mime="image/png",
        use_container_width=True,
    )

with st.expander("SVGコードを表示 / コピー"):
    st.code(result["svg"], language="xml")

with st.expander("SVGOMGに通す時のおすすめ設定"):
    st.markdown(
        """
このアプリは、分割サイトで扱いやすいように `path` を細かく分け、同じ色を `<g fill=...>` でまとめています。  
SVGOMGで以下を推奨します。

- **Prettify markup**：OFF
- **Remove metadata**：ON
- **Remove comments**：ON
- **Collapse useless groups**：ONでもよいが、色グループを残したいならOFF
- **Convert colors**：ON
- **Round/rewrite paths**：ON
- **Merge paths**：**OFF推奨**
- **Remove viewBox**：OFF

`Merge paths` をONにすると、せっかく15KB未満に分けたpathが同色で結合され、再び巨大なpathになる可能性があります。
"""
    )
