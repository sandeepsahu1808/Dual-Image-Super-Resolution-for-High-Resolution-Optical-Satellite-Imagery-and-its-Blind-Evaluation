"""ISRO PS-12 Dual Image Super Resolution — Streamlit demo."""

from __future__ import annotations

import os
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.data.preprocessing import preprocess_triplet
from src.evaluation.blind_metrics import compute_brisque, compute_niqe
from src.evaluation.full_reference import evaluate_scene
from src.models.srcnn import build_srcnn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "best_model.h5")
LR_SIZE = (128, 128)
HR_SIZE = (384, 384)

SPACE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Mono:wght@400;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #000008 !important;
    color: #e8f4ff !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: radial-gradient(ellipse at 20% 20%, #0a0a2e 0%, #000008 50%, #000008 100%);
}

/* Twinkling stars */
.stars {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.star {
    position: absolute;
    width: 2px; height: 2px;
    background: #fff;
    border-radius: 50%;
    animation: twinkle 3s infinite ease-in-out;
    opacity: 0.3;
}
@keyframes twinkle {
    0%, 100% { opacity: 0.2; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.4); }
}

/* Corner planets */
.planet-tl, .planet-br {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(1px);
}
.planet-tl {
    top: -60px; left: -60px;
    width: 180px; height: 180px;
    background: radial-gradient(circle at 35% 35%, #4a5568, #1a1a3e 60%, transparent 70%);
    box-shadow: 0 0 40px rgba(0, 212, 255, 0.15);
}
.planet-br {
    bottom: -80px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle at 40% 40%, #FF6B00 0%, #3d1a00 50%, transparent 72%);
    box-shadow: 0 0 60px rgba(255, 107, 0, 0.2);
    animation: planet-glow 6s ease-in-out infinite;
}
@keyframes planet-glow {
    0%, 100% { box-shadow: 0 0 40px rgba(255, 107, 0, 0.15); }
    50% { box-shadow: 0 0 80px rgba(255, 107, 0, 0.35); }
}

.block-container {
    position: relative;
    z-index: 1;
    max-width: 1200px;
    padding-top: 1rem;
}

h1, h2, h3, .orbitron {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 0.06em;
}

.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(1.4rem, 4vw, 2.4rem);
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #FF6B00, #00D4FF, #FF6B00);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    margin-bottom: 0.25rem;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.hero-rocket {
    display: inline-block;
    font-size: 2rem;
    animation: rocket-float 2.5s ease-in-out infinite;
    filter: drop-shadow(0 0 12px rgba(255, 107, 0, 0.8));
}
@keyframes rocket-float {
    0%, 100% { transform: translateY(0) rotate(-8deg); }
    50% { transform: translateY(-14px) rotate(8deg); }
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    text-align: center;
    color: #00D4FF;
    font-size: 0.95rem;
    opacity: 0.9;
    margin-bottom: 2rem;
}

.glass-card {
    background: rgba(10, 15, 40, 0.55);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 107, 0, 0.45);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 0 24px rgba(255, 107, 0, 0.12), inset 0 0 30px rgba(0, 212, 255, 0.03);
    margin-bottom: 1rem;
}

.upload-label {
    font-family: 'Orbitron', sans-serif;
    color: #00D4FF;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}

.panel-caption {
    font-family: 'Space Mono', monospace;
    color: #00D4FF;
    font-size: 0.75rem;
    text-align: center;
    margin-top: 0.5rem;
}

.cockpit-title {
    font-family: 'Orbitron', sans-serif;
    color: #FF6B00;
    font-size: 0.9rem;
    margin-bottom: 1rem;
    text-shadow: 0 0 10px rgba(255, 107, 0, 0.5);
}

.readout {
    font-family: 'Space Mono', monospace;
    margin: 0.75rem 0;
    padding: 0.6rem 0.8rem;
    background: rgba(0, 0, 20, 0.6);
    border-left: 3px solid #00D4FF;
    border-radius: 0 8px 8px 0;
}
.readout-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #00D4FF;
    text-shadow: 0 0 12px rgba(0, 212, 255, 0.6);
    line-height: 1.2;
}
.readout-unit {
    font-size: 0.7rem;
    color: #FF6B00;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.readout-hint {
    font-size: 0.72rem;
    color: #8899aa;
    margin-top: 0.25rem;
}

/* Glowing launch button */
div.stButton > button[kind="primary"],
div.stButton > button[kind="primary"] {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.08em !important;
    background: linear-gradient(135deg, #FF6B00, #ff9533) !important;
    color: #000008 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    box-shadow: 0 0 20px rgba(255, 107, 0, 0.5) !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[kind="primary"]:focus,
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 40px rgba(255, 107, 0, 0.9), 0 0 60px rgba(255, 107, 0, 0.4) !important;
    transform: scale(1.02);
    animation: pulse-orange 1.2s ease-in-out infinite !important;
}
@keyframes pulse-orange {
    0%, 100% { box-shadow: 0 0 20px rgba(255, 107, 0, 0.5); }
    50% { box-shadow: 0 0 45px rgba(255, 107, 0, 1); }
}

[data-testid="stFileUploader"] {
    background: rgba(10, 15, 40, 0.4);
    border: 1px dashed rgba(0, 212, 255, 0.4);
    border-radius: 12px;
    padding: 0.5rem;
}

[data-testid="stDownloadButton"] button {
    font-family: 'Orbitron', sans-serif !important;
    background: transparent !important;
    border: 1px solid #00D4FF !important;
    color: #00D4FF !important;
    border-radius: 10px !important;
}
[data-testid="stDownloadButton"] button:hover {
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.5) !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
"""


def inject_stars_html(n_stars: int = 80) -> str:
    stars = []
    for i in range(n_stars):
        top = (i * 17 + 3) % 100
        left = (i * 23 + 7) % 100
        delay = (i % 5) * 0.4
        size = 1 + (i % 3)
        stars.append(
            f'<div class="star" style="top:{top}%;left:{left}%;'
            f'width:{size}px;height:{size}px;animation-delay:{delay}s;"></div>'
        )
    return (
        '<div class="stars" aria-hidden="true">'
        + "".join(stars)
        + '<div class="planet-tl"></div><div class="planet-br"></div>'
        + "</div>"
    )


def inject_theme() -> None:
    st.markdown(SPACE_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="stars" aria-hidden="true">'
        + "".join(
            f'<div class="star" style="top:{(i*17+3)%100}%;left:{(i*23+7)%100}%;'
            f'animation-delay:{(i%5)*0.4}s;width:{1+(i%3)}px;height:{1+(i%3)}px;"></div>'
            for i in range(80)
        )
        + '<div class="planet-tl"></div><div class="planet-br"></div>'
        + "</div>",
        unsafe_allow_html=True,
    )


def load_and_display(uploaded_file, target_size=(128, 128)):
    """Load any PNG (8-bit or 16-bit) and return display-ready uint8 array."""
    uploaded_file.seek(0)
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode uploaded image.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_display = (img * 255).astype(np.uint8)
    return img, img_display


@st.cache_resource(show_spinner="Loading SRCNN weights…")
def load_sr_model():
    model = build_srcnn(input_shape=(384, 384, 2))
    if os.path.isfile(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)
        loaded = True
    else:
        loaded = False
    return model, loaded


def render_readout(label: str, value: str, unit: str, hint: str = "") -> None:
    hint_html = f'<div class="readout-hint">{hint}</div>' if hint else ""
    st.markdown(
        f"""
        <div class="readout">
            <div class="readout-unit">{label}</div>
            <div class="readout-value">{value}<span style="font-size:0.9rem;margin-left:6px;">{unit}</span></div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="ISRO PS-12 | Dual Image SR",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_theme()

    st.markdown(
        """
        <div style="text-align:center;margin:1rem 0 0.5rem;">
            <span class="hero-rocket">🚀</span>
            <div class="hero-title">ISRO BAH 2025 | PS-12 | Dual Image Super Resolution</div>
            <div class="hero-sub">Transforming Low-Resolution Satellite Imagery into Stellar Clarity</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, weights_loaded = load_sr_model()
    if not weights_loaded:
        st.warning(
            f"Weights not found at `{WEIGHTS_PATH}`. "
            "Run training first — inference uses untrained weights."
        )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    up1, up2 = st.columns(2)
    with up1:
        st.markdown(
            '<p class="upload-label">🛰️ LR Frame 1</p>',
            unsafe_allow_html=True,
        )
        lr1_file = st.file_uploader(
            "LR Frame 1",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="lr1",
            label_visibility="collapsed",
        )
    with up2:
        st.markdown(
            '<p class="upload-label">🛰️ LR Frame 2</p>',
            unsafe_allow_html=True,
        )
        lr2_file = st.file_uploader(
            "LR Frame 2",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="lr2",
            label_visibility="collapsed",
        )

    st.markdown(
        '<p class="upload-label" style="margin-top:1rem;">🌕 Ground Truth HR (optional)</p>',
        unsafe_allow_html=True,
    )
    hr_file = st.file_uploader(
        "HR ground truth",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="hr",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    launch = st.button("LAUNCH SUPER RESOLUTION 🚀", type="primary", use_container_width=True)

    if not launch:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center;color:#8899aa;font-family:'Space Mono',monospace;">
            Upload two 128×128 (or resizable) LR frames and hit launch to fuse them into a 384×384 SR image.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if not lr1_file or not lr2_file:
        st.error("Please upload both LR Frame 1 and LR Frame 2 before launch.")
        return

    lr1_file.seek(0)
    lr2_file.seek(0)
    with st.spinner("Running super-resolution inference…"):
        lr1, lr1_display = load_and_display(lr1_file, LR_SIZE)
        lr2, lr2_display = load_and_display(lr2_file, LR_SIZE)
        hr_dummy = np.zeros(HR_SIZE, dtype=np.float32)
        x, _ = preprocess_triplet(lr1, lr2, hr_dummy, augment=False)
        sr = model.predict(x[np.newaxis, ...], verbose=0)[0].squeeze()
        sr = np.clip(sr, 0.0, 1.0).astype(np.float32)
        sr_metrics = sr[:, :, np.newaxis]
        sr = (sr - sr.min()) / (sr.max() - sr.min() + 1e-8)
        sr_display = (sr * 255).astype(np.uint8)

    hr_display = None
    hr_for_metrics = None
    if hr_file is not None:
        hr_file.seek(0)
        hr, hr_display = load_and_display(hr_file, HR_SIZE)
        hr_for_metrics = hr[:, :, np.newaxis].astype(np.float32)

    st.markdown("### Mission Results")
    if hr_for_metrics is not None:
        img1, img2, img3, img4 = st.columns(4)
    else:
        img1, img2, img3 = st.columns(3)
        img4 = None

    with img1:
        st.image(lr1_display)
        st.markdown('<p class="panel-caption">LR Frame 1 · 128×128</p>', unsafe_allow_html=True)
    with img2:
        st.image(lr2_display)
        st.markdown('<p class="panel-caption">LR Frame 2 · 128×128</p>', unsafe_allow_html=True)
    with img3:
        st.image(sr_display)
        st.markdown(
            '<p class="panel-caption">Super-Resolved · 128×128 → 384×384</p>',
            unsafe_allow_html=True,
        )
    if img4 is not None:
        with img4:
            st.image(hr_display)
            st.markdown('<p class="panel-caption">Ground Truth HR · 384×384</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    met_left, met_right = st.columns(2)

    with met_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="cockpit-title">🛰️ Full-Reference Metrics</p>', unsafe_allow_html=True)
        if hr_for_metrics is not None:
            fr = evaluate_scene(hr_for_metrics, sr_metrics)
            render_readout("PSNR", f"{fr['psnr']:.2f}", "dB", "Higher is better")
            render_readout("SSIM", f"{fr['ssim']:.4f}", "", "Higher is better (max 1)")
            render_readout("MSE", f"{fr['mse']:.2e}", "", "Lower is better")
            render_readout("RMSE", f"{fr['rmse']:.4f}", "", "Lower is better")
        else:
            st.markdown(
                '<p style="font-family:\'Space Mono\',monospace;color:#8899aa;font-size:0.85rem;">'
                "Upload HR ground truth to enable PSNR, SSIM, MSE, and RMSE.</p>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with met_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="cockpit-title">🌌 Blind Quality Assessment</p>', unsafe_allow_html=True)
        with st.spinner("Computing NIQE & BRISQUE…"):
            niqe_val = compute_niqe(sr_metrics)
            brisque_val = compute_brisque(sr_metrics)
        render_readout("NIQE", f"{niqe_val:.3f}", "", "Naturalness — lower is better")
        render_readout("BRISQUE", f"{brisque_val:.2f}", "", "Spatial quality — lower is better")
        st.markdown("</div>", unsafe_allow_html=True)

    sr_pil = Image.fromarray(sr_display, mode="L")
    buf = BytesIO()
    sr_pil.save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download Super-Resolved Image",
        data=buf.getvalue(),
        file_name="super_resolved_isro_ps12.png",
        mime="image/png",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
