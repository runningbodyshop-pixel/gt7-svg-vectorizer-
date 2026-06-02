# GT7 SVG Vectorizer - smartphone copy/paste edition
from __future__ import annotations

import base64, html, io, re, time
from typing import Optional
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from skimage import measure

GT7_LIMIT = 15 * 1024
HARD_LIMIT = 2 * 1024 * 1024

PRESETS = {
    "GT7最小 / 15KB狙い": dict(colors=8, side=360, simp=2.6, area=18, smooth=0, dec=0),
    "低品質 / 軽量": dict(colors=12, side=520, simp=2.0, area=12, smooth=0, dec=0),
    "中品質 / バランス": dict(colors=24, side=720, simp=1.25, area=7, smooth=0, dec=1),
    "高品質 / 大きめ": dict(colors=48, side=960, simp=0.75, area=3, smooth=1, dec=1),
}

def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO(); img.save(b, "PNG", optimize=True); return b.getvalue()

def safe_svg(s: str) -> str:
    for tag in ["script","foreignObject","image","text","filter","mask","clipPath","pattern","style","defs"]:
        s = re.sub(rf"<\s*{tag}\b.*?<\s*/\s*{tag}\s*>", "", s, flags=re.I|re.S)
        s = re.sub(rf"<\s*{tag}\b[^>]*/\s*>", "", s, flags=re.I|re.S)
    return re.sub(r"\s+", " ", s).replace("> <", "><").strip()

def fit_image(img: Image.Image, side: int, enhance: bool) -> Image.Image:
    img = img.convert("RGBA")
    if enhance:
        rgb = ImageOps.autocontrast(img.convert("RGB"))
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=135, threshold=3))
        img = Image.merge("RGBA", (*rgb.split(), img.getchannel("A")))
        if max(img.size) < 512:
            sc = min(2.0, 512 / max(img.size))
            img = img.resize((int(img.width*sc), int(img.height*sc)), Image.Resampling.LANCZOS)
    if max(img.size) > side:
        sc = side / max(img.size)
        img = img.resize((max(1, int(img.width*sc)), max(1, int(img.height*sc))), Image.Resampling.LANCZOS)
    return img

def quantize(img: Image.Image, n: int, alpha_th: int, white_bg: bool):
    img = img.convert("RGBA")
    a = np.array(img.getchannel("A"))
    visible = a >= alpha_th
    if white_bg:
        bg = Image.new("RGBA", img.size, (255,255,255,255))
        img = Image.alpha_composite(bg, img); visible[:] = True
    q = img.convert("RGB").quantize(colors=int(n), method=Image.Quantize.MEDIANCUT)
    labels = np.array(q, dtype=np.int32); labels[~visible] = -1
    pal = q.getpalette()[:768]
    used = sorted(int(x) for x in np.unique(labels) if int(x) >= 0)
    out = np.full_like(labels, -1)
    colors = []
    for new, old in enumerate(used):
        out[labels == old] = new
        colors.append(tuple(pal[old*3:old*3+3]))
    prev = np.zeros((out.shape[0], out.shape[1], 4), dtype=np.uint8)
    for i, c in enumerate(colors):
        m = out == i; prev[m, :3] = c; prev[m, 3] = 255
    if white_bg: prev[:, :, 3] = 255
    return out, colors, Image.fromarray(prev, "RGBA")

def area(p: np.ndarray) -> float:
    if len(p) < 3: return 0.0
    x, y = p[:,0], p[:,1]
    return float(abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))) / 2)

def rdp(p: np.ndarray, eps: float) -> np.ndarray:
    if len(p) < 3 or eps <= 0: return p
    s, e = p[0], p[-1]; line = e - s; ln = float(np.linalg.norm(line))
    if ln == 0:
        d = np.linalg.norm(p - s, axis=1)
    else:
        d = abs(line[0]*(s[1]-p[:,1]) - line[1]*(s[0]-p[:,0])) / ln
    i = int(np.argmax(d))
    if float(d[i]) > eps:
        return np.vstack((rdp(p[:i+1], eps)[:-1], rdp(p[i:], eps)))
    return np.vstack((s, e))

def chaikin(p: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0 or len(p) < 4: return p
    for _ in range(steps):
        q = []
        for i in range(len(p)):
            a, b = p[i], p[(i+1) % len(p)]
            q += [0.75*a + 0.25*b, 0.25*a + 0.75*b]
        p = np.array(q, dtype=np.float32)
        if len(p) > 7000: break
    return p

def fmt(x: float, dec: int) -> str:
    if dec <= 0: return str(int(round(x)))
    return f"{x:.{dec}f}".rstrip("0").rstrip(".").replace("-0", "0")

def remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1 or not mask.any(): return mask
    lab = measure.label(mask, connectivity=1)
    if lab.max() == 0: return mask
    cnt = np.bincount(lab.ravel()); keep = cnt >= min_size; keep[0] = False
    return keep[lab]

def contour_path(c: np.ndarray, simp: float, min_area: int, smooth: int, dec: int) -> Optional[str]:
    p = np.stack([c[:,1], c[:,0]], axis=1).astype(np.float32)
    if len(p) < 3: return None
    if np.linalg.norm(p[0] - p[-1]) > 0.01: p = np.vstack([p, p[0]])
    if area(p) < min_area: return None
    p = rdp(p, simp)
    if len(p) < 3: return None
    if np.linalg.norm(p[0] - p[-1]) < 0.01: p = p[:-1]
    p = chaikin(p, smooth); p = rdp(p, max(0.05, simp*0.65))
    if len(p) < 3: return None
    d = [f"M{fmt(p[0,0],dec)} {fmt(p[0,1],dec)}"]
    d += [f"L{fmt(x,dec)} {fmt(y,dec)}" for x, y in p[1:]]
    d.append("Z")
    return "".join(d)

def build_svg(labels, colors, w, h, simp, min_area, smooth, dec, white_bg):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    paths = 0
    if white_bg:
        parts.append(f'<path fill="#fff" d="M0 0H{w}V{h}H0Z"/>'); paths += 1
    order = sorted([(int((labels == i).sum()), i) for i in range(len(colors))], reverse=True)
    for _, i in order:
        mask = labels == i
        if int(mask.sum()) < min_area: continue
        mask = remove_small(mask, max(2, min_area // 2))
        pad = np.pad(mask.astype(np.uint8), 1)
        sub = []
        for c in measure.find_contours(pad, 0.5, fully_connected="high"):
            d = contour_path(c - 1.0, simp, min_area, smooth, dec)
            if d: sub.append(d)
        if not sub: continue
        r, g, b = colors[i]
        parts.append(f'<path fill="#{r:02x}{g:02x}{b:02x}" fill-rule="evenodd" d="{html.escape("".join(sub), quote=True)}"/>')
        paths += 1
    parts.append("</svg>")
    return safe_svg("".join(parts)), paths

def compare_png(orig: Image.Image, prev: Image.Image) -> bytes:
    def fit(im):
        im = im.convert("RGBA")
        if im.height > 720:
            sc = 720 / im.height
            im = im.resize((int(im.width*sc), 720), Image.Resampling.LANCZOS)
        return im
    a, b = fit(orig), fit(prev)
    canvas = Image.new("RGBA", (a.width + b.width + 24, max(a.height, b.height) + 54), (245,245,245,255))
    dr = ImageDraw.Draw(canvas); dr.text((10,8), "Original", fill=(0,0,0)); dr.text((a.width+22,8), "Vector preview", fill=(0,0,0))
    canvas.alpha_composite(a, (8,42)); canvas.alpha_composite(b, (a.width+16,42))
    return png_bytes(canvas)

def convert_once(img, colors, side, simp, min_area, smooth, dec, alpha, enhance, white_bg):
    work = fit_image(img, side, enhance)
    labels, pal, prev = quantize(work, colors, alpha, white_bg)
    svg, paths = build_svg(labels, pal, work.width, work.height, simp, min_area, smooth, dec, white_bg)
    return dict(svg=svg, size=len(svg.encode()), prev=prev, compare=compare_png(work, prev), colors=len(pal), paths=paths, w=work.width, h=work.height)

def convert(img, cfg):
    best = None
    colors, side, simp, min_area, smooth, dec = cfg["colors"], cfg["side"], cfg["simp"], cfg["area"], cfg["smooth"], cfg["dec"]
    target = min(int(cfg["target"]), HARD_LIMIT)
    loops = 34 if cfg["auto"] else 1
    for i in range(loops):
        r = convert_once(img, max(2,int(colors)), max(96,int(side)), max(0.2,float(simp)), max(1,int(min_area)), max(0,int(smooth)), max(0,int(dec)), cfg["alpha"], cfg["enhance"], cfg["white_bg"])
        r["attempts"] = i + 1
        if best is None or r["size"] < best["size"]: best = r
        if r["size"] <= target: return r
        if i % 2 == 0:
            simp *= 1.35; min_area = int(min_area * 1.35 + 1)
        else:
            colors *= 0.82; side *= 0.92
        if r["size"] > target * 2.5: smooth = 0
        if r["size"] > target * 4: dec = 0
    return best

def data_url(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

def target_from(choice: str, custom_kb: int) -> int:
    return {"GT7厳格 15KB": GT7_LIMIT, "100KB": 100*1024, "500KB": 500*1024, "2MB 絶対上限": HARD_LIMIT}.get(choice, max(1, min(2048, int(custom_kb))) * 1024)

st.set_page_config(page_title="GT7 SVG Vectorizer", page_icon="🏁", layout="wide")
st.title("🏁 GT7 SVG Vectorizer")
st.caption("スマホだけで使えるGT7向けSVG変換ツール。SVG内に画像・文字・CSS・フィルターは入れません。")

with st.expander("使い方", expanded=False):
    st.markdown("1. 画像をアップロード → 2. **SVGへ変換** → 3. SVGを保存。GT7用は基本的に15KB以下を狙ってください。")

up = st.file_uploader("画像を選択", type=["png", "jpg", "jpeg", "webp"])
with st.sidebar:
    st.header("設定")
    preset = st.selectbox("品質", list(PRESETS.keys()) + ["手動設定"], 0)
    base = PRESETS.get(preset, PRESETS["中品質 / バランス"]).copy()
    t_choice = st.selectbox("目標サイズ", ["GT7厳格 15KB", "100KB", "500KB", "2MB 絶対上限", "カスタムKB"], 0)
    custom = st.number_input("カスタムKB", 1, 2048, 15)
    auto = st.toggle("目標サイズまで自動軽量化", True)
    enhance = st.toggle("低画質を軽く補正", True)
    white_bg = st.toggle("透明部分を白にする", False)
    st.divider()
    base["colors"] = st.slider("色数", 2, 96, int(base["colors"]))
    base["side"] = st.slider("処理サイズ：長辺px", 96, 1400, int(base["side"]), 16)
    base["simp"] = st.slider("パス軽量化", 0.2, 6.0, float(base["simp"]), 0.05)
    base["area"] = st.slider("小さい形状を削除", 1, 80, int(base["area"]))
    base["smooth"] = st.slider("角の平滑化", 0, 2, int(base["smooth"]))
    base["alpha"] = st.slider("透明判定", 0, 255, 16)
    base["dec"] = st.slider("座標の小数桁", 0, 2, int(base["dec"]))
    base["target"] = target_from(t_choice, custom)
    base["auto"] = auto; base["enhance"] = enhance; base["white_bg"] = white_bg

if up is None:
    st.info("画像をアップロードしてください。")
    st.stop()

try:
    img = Image.open(up).convert("RGBA")
except Exception as e:
    st.error(f"画像を開けませんでした: {e}"); st.stop()

if st.button("SVGへ変換", type="primary", use_container_width=True):
    t = time.time()
    with st.spinner("変換中..."):
        st.session_state.result = convert(img, base)
        st.session_state.elapsed = time.time() - t
        st.session_state.name = up.name

if "result" not in st.session_state:
    st.image(img, caption="元画像", use_container_width=True); st.stop()

r = st.session_state.result
cols = st.columns(5)
cols[0].metric("SVGサイズ", f"{r['size']:,} bytes")
cols[1].metric("目標", f"{base['target']:,} bytes")
cols[2].metric("色数", str(r["colors"]))
cols[3].metric("path数", str(r["paths"]))
cols[4].metric("変換", f"{st.session_state.elapsed:.1f}秒")

if r["size"] <= base["target"]:
    st.success("目標サイズ内です。")
else:
    st.warning("目標サイズを超えています。色数・処理サイズを下げるか、パス軽量化を上げてください。")
if r["size"] > HARD_LIMIT:
    st.error("2MB絶対上限を超えています。このSVGは使わないでください。")

left, right = st.columns(2)
with left:
    st.subheader("元画像"); st.image(img, use_container_width=True)
with right:
    st.subheader("SVGプレビュー")
    components.html(f'<div style="background:#f5f5f5;border:1px solid #ddd;border-radius:10px;padding:12px;text-align:center"><img src="{data_url(r["svg"])}" style="max-width:100%;height:auto"></div>', height=520, scrolling=True)

st.subheader("比較プレビュー")
st.image(r["compare"], use_container_width=True)

name = re.sub(r"\.[^.]+$", "", st.session_state.name)
name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "converted"
a, b = st.columns(2)
a.download_button("SVGを保存", r["svg"].encode(), f"{name}_gt7.svg", "image/svg+xml", use_container_width=True)
b.download_button("比較PNGを保存", r["compare"], f"{name}_compare.png", "image/png", use_container_width=True)
with st.expander("SVGコードを表示 / コピー"):
    st.code(r["svg"], language="xml")