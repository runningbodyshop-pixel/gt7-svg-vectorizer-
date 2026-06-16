import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


def build_svg_from_image(
    image: Image.Image,
    target_w: int = 1000,
    k_colors: int = 20,
    eps: float = 0.9,
    min_area: int = 8,
    hole_min_area: int = 24,
    alpha_thr: int = 25,
    sample_n: int = 120_000,
    seed: int = 9,
):
    """
    画像をレイヤー分けSVGに変換する簡易ベクター化エンジン。

    方針:
    - 画像を透明部分で自動クロップ
    - 色を少数に統合
    - 色面ごとに輪郭抽出
    - 顔、手、髪、服、猫、シアン、線画などのSVG groupに分類
    - 容量を抑えるため path を簡略化
    """

    cv2.setRNGSeed(int(seed))

    im = image.convert("RGBA")
    arr = np.array(im)
    alpha = arr[:, :, 3]

    if not np.any(alpha > alpha_thr):
        raise ValueError("透明でない部分が見つかりません。")

    # 透明部分を除いて自動クロップ
    ys, xs = np.where(alpha > alpha_thr)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1

    margin = 6
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(arr.shape[1], x1 + margin)
    y1 = min(arr.shape[0], y1 + margin)

    crop = arr[y0:y1, x0:x1]
    h0, w0 = crop.shape[:2]
    target_h = max(1, int(round(h0 * target_w / w0)))

    resized = Image.fromarray(crop, "RGBA").resize(
        (target_w, target_h),
        Image.Resampling.LANCZOS,
    )

    arr2 = np.array(resized)
    a = arr2[:, :, 3]
    mask = a > alpha_thr
    rgb_raw = arr2[:, :, :3]

    # 白背景・薄い外周背景を削る処理
    neutral_span = np.max(rgb_raw, axis=2) - np.min(rgb_raw, axis=2)
    near_bg = (
        (a > alpha_thr)
        & (rgb_raw[:, :, 0] > 225)
        & (rgb_raw[:, :, 1] > 225)
        & (rgb_raw[:, :, 2] > 225)
        & (neutral_span < 10)
    )

    num, cc, stats, _ = cv2.connectedComponentsWithStats(
        near_bg.astype(np.uint8) * 255,
        8,
    )

    remove_bg = np.zeros_like(mask)

    for j in range(1, num):
        x, y, w, h, area = stats[j]
        touches_edge = (
            x == 0
            or y == 0
            or x + w >= target_w
            or y + h >= target_h
        )
        if touches_edge and area > 500:
            remove_bg |= cc == j

    mask &= ~remove_bg

    # 色面を少し滑らかにしてから減色
    rgb = cv2.medianBlur(rgb_raw.copy(), 3)
    pix = rgb[mask].astype(np.float32)

    if pix.size == 0:
        raise ValueError("抽出対象がありません。背景除去設定を弱めてください。")

    rng = np.random.default_rng(int(seed))

    if len(pix) > sample_n:
        pix_sample = pix[rng.choice(len(pix), sample_n, replace=False)]
    else:
        pix_sample = pix

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        60,
        0.4,
    )

    _, _, centers = cv2.kmeans(
        pix_sample,
        int(k_colors),
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = np.clip(np.round(centers), 0, 255).astype(np.uint8)

    pix_all = rgb.reshape(-1, 3).astype(np.int32)
    centers_i = centers.astype(np.int32)

    dist = ((pix_all[:, None, :] - centers_i[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(dist, axis=1).reshape(target_h, target_w)
    labels[~mask] = -1

    areas = {i: int(np.sum(labels == i)) for i in range(len(centers))}

    def lum(c):
        r, g, b = [int(v) for v in c]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def sat(c):
        return int(max(c) - min(c))

    def color_type(c):
        r, g, b = [int(v) for v in c]
        l = lum(c)
        s = sat(c)

        if r > 240 and g > 215 and b > 200 and (r - b) > 12:
            return "skin", 45

        if l > 242:
            return "white", 12

        if l > 195 and s < 35:
            return "jacket_white", 20

        if l > 120 and s < 45:
            return "jacket_shadow", 25

        if b > 135 and g > 130 and r < 160 and (g - r) > 45:
            return "cyan", 80

        if l < 28:
            return "black", 95

        if l < 75 and b > 45 and (b - r) > 5:
            return "cat_darkblue", 38

        if l < 90 and abs(r - g) < 35 and abs(g - b) < 35:
            return "dark", 60

        return "mid", 50

    groups_meta = []

    for i, c in enumerate(centers):
        typ, order = color_type(c)
        groups_meta.append(
            (
                order,
                i,
                typ,
                tuple(int(v) for v in c),
                areas[i],
                lum(c),
            )
        )

    groups_meta.sort(key=lambda t: (t[0], t[5]))

    layer_order = [
        "jacket_base",
        "jacket_shadows",
        "left_cat",
        "right_cat",
        "skin_face",
        "skin_right_hand",
        "skin_left_hand",
        "skin_details",
        "hair",
        "shirt_pants_dark",
        "cyan_eyes",
        "cyan_cats",
        "cyan_suit",
        "top_linework",
    ]

    buckets = {name: {} for name in layer_order}

    def classify(typ, bbox):
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2

        if typ in ("jacket_white", "white"):
            return "jacket_base"

        if typ in ("jacket_shadow", "mid"):
            if cx < target_w * 0.34 and cy < target_h * 0.55:
                return "left_cat"
            if cx > target_w * 0.70 and cy > target_h * 0.70:
                return "right_cat"
            if cy < target_h * 0.47 and target_w * 0.30 < cx < target_w * 0.72:
                return "hair"
            return "jacket_shadows"

        if typ == "skin":
            if cx > target_w * 0.55 and cy < target_h * 0.62:
                return "skin_right_hand"
            if cx < target_w * 0.32 and cy > target_h * 0.70:
                return "skin_left_hand"
            if target_w * 0.35 < cx < target_w * 0.70 and cy < target_h * 0.55:
                return "skin_face"
            return "skin_details"

        if typ == "cat_darkblue":
            if cx < target_w * 0.38 and cy < target_h * 0.62:
                return "left_cat"
            if cx > target_w * 0.70 and cy > target_h * 0.64:
                return "right_cat"
            if cy < target_h * 0.45:
                return "hair"
            return "shirt_pants_dark"

        if typ == "dark":
            if cy < target_h * 0.48 and target_w * 0.30 < cx < target_w * 0.75:
                return "hair"
            if cx < target_w * 0.34 and cy < target_h * 0.58:
                return "left_cat"
            return "shirt_pants_dark"

        if typ == "cyan":
            if target_w * 0.36 < cx < target_w * 0.67 and cy < target_h * 0.45:
                return "cyan_eyes"
            if cx < target_w * 0.36 or (cx > target_w * 0.70 and cy > target_h * 0.64):
                return "cyan_cats"
            return "cyan_suit"

        if typ == "black":
            return "top_linework"

        return "jacket_shadows"

    for _, i, typ, c, area, _ in groups_meta:
        if area < min_area:
            continue

        one = (labels == i).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(
            one,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if hierarchy is None:
            continue

        hier = hierarchy[0]
        fill = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

        for idx, cnt in enumerate(contours):
            # 親輪郭だけ処理
            if hier[idx][3] != -1:
                continue

            area = cv2.contourArea(cnt)

            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            layer = classify(typ, (x, y, w, h))

            selected = [idx]
            child = hier[idx][2]

            # 穴抜き用の子輪郭
            while child != -1:
                if cv2.contourArea(contours[child]) >= hole_min_area:
                    selected.append(child)
                child = hier[child][0]

            subpaths = []

            for j in sorted(selected, key=lambda t: -cv2.contourArea(contours[t])):
                area_j = cv2.contourArea(contours[j])

                e = eps

                if area_j > 50_000:
                    e = eps * 2.0
                elif area_j > 15_000:
                    e = eps * 1.55
                elif area_j > 5_000:
                    e = eps * 1.25
                elif area_j < 150:
                    e = max(0.7, eps * 0.8)

                approx = cv2.approxPolyDP(contours[j], e, True)

                if len(approx) < 3:
                    continue

                pts = approx.reshape(-1, 2)

                d = f"M{pts[0, 0]},{pts[0, 1]}"
                d += "".join(f"L{x},{y}" for x, y in pts[1:])
                d += "Z"

                subpaths.append(d)

            if subpaths:
                buckets[layer].setdefault(fill, []).append("".join(subpaths))

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {target_w} {target_h}" shape-rendering="geometricPrecision">',
        "<title>layered_parts_svg</title>",
    ]

    layer_counts = {}

    for layer in layer_order:
        if not buckets[layer]:
            continue

        elements = []
        comp_count = 0

        for fill, d_list in buckets[layer].items():
            comp_count += len(d_list)
            elements.append(
                f'<path fill="{fill}" d="{"".join(d_list)}"/>'
            )

        layer_counts[layer] = comp_count
        pieces.append(
            f'<g id="{layer}" fill-rule="evenodd">{"".join(elements)}</g>'
        )

    pieces.append("</svg>")

    svg = "".join(pieces)

    meta = {
        "bytes": len(svg.encode("utf-8")),
        "viewbox": f"0 0 {target_w} {target_h}",
        "layers": layer_counts,
        "colors": k_colors,
    }

    return svg, meta


def convert_with_limit(image: Image.Image, max_bytes: int, **settings):
    """
    目標容量以下になるまで自動で軽量化する。
    """

    target_w = int(settings.pop("target_w"))
    k_colors = int(settings.pop("k_colors"))
    eps = float(settings.pop("eps"))
    min_area = int(settings.pop("min_area"))
    hole_min_area = int(settings.pop("hole_min_area"))
    seed = int(settings.pop("seed"))

    trials = []

    for step in range(8):
        trial = {
            "target_w": max(540, int(target_w * (0.94 ** step))),
            "k_colors": max(10, k_colors - step // 2),
            "eps": eps + step * 0.25,
            "min_area": min_area + step * 4,
            "hole_min_area": hole_min_area + step * 8,
            "seed": seed,
        }

        svg, meta = build_svg_from_image(image, **trial)
        trials.append((svg, meta, trial))

        if meta["bytes"] <= max_bytes:
            meta["settings"] = trial
            meta["auto_reduced"] = step > 0
            return svg, meta

    svg, meta, trial = min(trials, key=lambda x: x[1]["bytes"])
    meta["settings"] = trial
    meta["auto_reduced"] = True

    return svg, meta


st.set_page_config(
    page_title="Layered SVG Converter",
    layout="wide",
)

st.title("Layered SVG Converter / 100KB以下向け")
st.caption(
    "画像を、顔・髪・手・服・猫・水色アクセント・線画などのSVGレイヤーに分けて変換します。"
)

uploaded = st.file_uploader(
    "変換したい画像をアップロード",
    type=["png", "jpg", "jpeg", "webp"],
)

with st.sidebar:
    st.header("設定")

    max_kb = st.number_input(
        "目標容量 KB",
        min_value=10,
        max_value=500,
        value=100,
        step=5,
    )

    target_w = st.slider(
        "内部SVG幅",
        min_value=540,
        max_value=1400,
        value=1000,
        step=20,
    )

    k_colors = st.slider(
        "色数",
        min_value=8,
        max_value=28,
        value=20,
        step=1,
    )

    eps = st.slider(
        "曲線・形状の簡略化",
        min_value=0.4,
        max_value=3.0,
        value=0.9,
        step=0.1,
    )

    min_area = st.slider(
        "小さいゴミの除去",
        min_value=1,
        max_value=80,
        value=8,
        step=1,
    )

    hole_min_area = st.slider(
        "穴抜きの最小面積",
        min_value=1,
        max_value=120,
        value=24,
        step=1,
    )

    seed = st.number_input(
        "固定シード",
        value=9,
        step=1,
    )

st.markdown(
    """
### 使い方
1. PNG / JPG / WEBPをアップロード  
2. 目標容量を100KBにする  
3. SVGを生成  
4. 必要ならSVGをダウンロードして、Inkscape / Photopea / SVGOMGで調整  
"""
)

if uploaded is None:
    st.info("画像をアップロードしてください。")
    st.stop()

image = Image.open(uploaded)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("入力画像")
    st.image(image, use_container_width=True)

try:
    with st.spinner("SVGを生成中..."):
        svg_text, meta = convert_with_limit(
            image,
            max_bytes=int(max_kb * 1024),
            target_w=target_w,
            k_colors=k_colors,
            eps=eps,
            min_area=min_area,
            hole_min_area=hole_min_area,
            seed=seed,
        )

    with col2:
        st.subheader("SVGプレビュー")
        components.html(
            f"""
            <div style="background:#222;padding:12px;border-radius:10px;">
              <div style="
                background:#000;
                display:flex;
                align-items:center;
                justify-content:center;
                overflow:auto;
                max-height:75vh;
              ">
                {svg_text}
              </div>
            </div>
            """,
            height=700,
            scrolling=True,
        )

    st.success(
        f"生成完了: {meta['bytes'] / 1024:.1f} KB / viewBox {meta['viewbox']}"
    )

    if meta["bytes"] <= int(max_kb * 1024):
        st.info("指定容量以下に収まっています。")
    else:
        st.warning(
            "指定容量を少し超えています。内部SVG幅・色数を下げるか、簡略化を上げてください。"
        )

    if meta.get("auto_reduced"):
        st.warning(
            "目標容量に合わせて、内部幅・色数・簡略化を自動調整しました。"
        )

    st.write("レイヤー別パーツ数")
    st.json(meta["layers"])

    st.write("実際に使われた設定")
    st.json(meta["settings"])

    st.download_button(
        "SVGをダウンロード",
        data=svg_text.encode("utf-8"),
        file_name="layered_parts_under_limit.svg",
        mime="image/svg+xml",
    )

    with st.expander("SVGテキストを表示 / コピペ用"):
        st.code(svg_text, language="xml")

except Exception as e:
    st.error(f"変換に失敗しました: {e}")