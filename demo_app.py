import io
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from pipeline import run_pipeline, TIER_COLORS

st.set_page_config(
    page_title="SmartSat — Signal Faible",
    page_icon="🛰️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# THÈME ESPACE / SATELLITE
# ---------------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600&display=swap');

:root {
    --space-0: #02030a;
    --space-1: #070b1f;
    --space-2: #0d1433;
    --panel: rgba(14, 22, 54, 0.62);
    --line: rgba(120, 170, 255, 0.22);
    --cyan: #4de1ff;
    --violet: #a78bfa;
    --amber: #ffb547;
    --text: #dbe6ff;
    --muted: #8ea2d6;
}

html, body, [class*="css"], .stApp {
    font-family: 'Rajdhani', sans-serif;
    color: var(--text);
}

/* Fond : dégradé nébuleuse + étoiles */
.stApp {
    background:
        radial-gradient(ellipse at 15% 10%, rgba(88, 62, 190, 0.35) 0%, transparent 45%),
        radial-gradient(ellipse at 85% 20%, rgba(0, 150, 200, 0.25) 0%, transparent 40%),
        radial-gradient(ellipse at 50% 100%, rgba(30, 40, 120, 0.45) 0%, transparent 55%),
        linear-gradient(180deg, var(--space-0), var(--space-1) 60%, var(--space-2));
    background-attachment: fixed;
}
.stApp::before {
    content: "";
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        radial-gradient(1.5px 1.5px at 20px 30px, #fff, transparent),
        radial-gradient(1px 1px at 90px 120px, #cfe3ff, transparent),
        radial-gradient(1.5px 1.5px at 160px 60px, #fff, transparent),
        radial-gradient(1px 1px at 230px 180px, #ffe9c4, transparent),
        radial-gradient(1px 1px at 310px 90px, #fff, transparent),
        radial-gradient(1.5px 1.5px at 380px 210px, #b9d4ff, transparent);
    background-size: 420px 260px;
    opacity: 0.55;
    animation: twinkle 6s ease-in-out infinite alternate;
}
@keyframes twinkle { from { opacity: 0.30; } to { opacity: 0.75; } }

header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1250px; position: relative; z-index: 1; }

/* Titres */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 0.06em;
    color: #eaf1ff !important;
}
h2, h3 { font-size: 1.15rem !important; text-transform: uppercase; }
h2::before, h3::before { content: "◈ "; color: var(--cyan); }

/* Hero */
.hero {
    position: relative; overflow: hidden;
    display: flex; align-items: center; justify-content: space-between; gap: 2rem;
    padding: 2rem 2.4rem; margin-bottom: 1.4rem;
    border: 1px solid var(--line); border-radius: 22px;
    background:
        linear-gradient(120deg, rgba(20, 30, 80, 0.75), rgba(8, 12, 34, 0.75));
    box-shadow: 0 0 40px rgba(77, 225, 255, 0.08), inset 0 0 60px rgba(120, 90, 255, 0.07);
    backdrop-filter: blur(6px);
}
.hero-tag {
    font-family: 'Orbitron', sans-serif; font-size: 0.72rem; letter-spacing: 0.35em;
    color: var(--cyan); margin-bottom: 0.6rem;
}
.hero h1 {
    margin: 0 0 0.5rem 0; font-size: 2.5rem; font-weight: 900;
    background: linear-gradient(90deg, #ffffff, var(--cyan) 55%, var(--violet));
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { margin: 0; color: var(--muted); font-size: 1.15rem; max-width: 560px; }
.hero-badges { margin-top: 1rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }
.badge {
    font-family: 'Orbitron', sans-serif; font-size: 0.68rem; letter-spacing: 0.12em;
    padding: 0.35rem 0.8rem; border-radius: 999px;
    border: 1px solid var(--line); background: rgba(77, 225, 255, 0.07); color: var(--cyan);
}

/* Planète + orbite */
.orbit-wrap { position: relative; width: 190px; height: 190px; flex-shrink: 0; }
.planet {
    position: absolute; top: 50%; left: 50%; width: 84px; height: 84px;
    margin: -42px 0 0 -42px; border-radius: 50%;
    background:
        radial-gradient(circle at 32% 30%, #6fd6ff 0%, #2a6fd1 38%, #0b1f5e 75%, #050b26 100%);
    box-shadow: 0 0 35px rgba(77, 180, 255, 0.55), inset -14px -10px 24px rgba(0, 0, 10, 0.7);
}
.ring {
    position: absolute; inset: 8px; border-radius: 50%;
    border: 1px dashed rgba(77, 225, 255, 0.45);
    animation: spin 14s linear infinite;
}
.ring.r2 { inset: -6px; border-color: rgba(167, 139, 250, 0.35); animation-duration: 22s; animation-direction: reverse; }
.sat {
    position: absolute; top: -9px; left: 50%; margin-left: -9px;
    font-size: 18px; filter: drop-shadow(0 0 6px var(--cyan));
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(5, 8, 26, 0.96), rgba(10, 16, 44, 0.96));
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] h2 { font-size: 0.95rem !important; }
.sidebar-brand {
    font-family: 'Orbitron', sans-serif; letter-spacing: 0.2em; font-size: 0.7rem;
    color: var(--cyan); padding-bottom: 0.6rem; border-bottom: 1px solid var(--line);
    margin-bottom: 0.8rem;
}

/* Cartes métriques */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0 1.2rem; }
@media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }
.metric-card {
    position: relative; padding: 1.1rem 1.2rem; border-radius: 16px;
    border: 1px solid var(--line); background: var(--panel);
    backdrop-filter: blur(6px); overflow: hidden;
}
.metric-card::after {
    content: ""; position: absolute; left: 0; top: 0; height: 2px; width: 100%;
    background: linear-gradient(90deg, var(--cyan), var(--violet), transparent);
}
.metric-label {
    font-family: 'Orbitron', sans-serif; font-size: 0.64rem; letter-spacing: 0.18em;
    color: var(--muted); text-transform: uppercase;
}
.metric-value { font-family: 'Orbitron', sans-serif; font-size: 1.45rem; font-weight: 700; color: #fff; margin-top: 0.35rem; }
.metric-unit { font-size: 0.8rem; color: var(--muted); margin-left: 0.3rem; font-family: 'Rajdhani', sans-serif; }

/* Légende des niveaux */
.legend { display: flex; gap: 0.7rem; flex-wrap: wrap; margin: 0.4rem 0 0.8rem; }
.legend-item {
    display: flex; align-items: center; gap: 0.5rem; padding: 0.3rem 0.8rem;
    border: 1px solid var(--line); border-radius: 999px; background: var(--panel);
    font-size: 0.95rem;
}
.dot { width: 10px; height: 10px; border-radius: 50%; box-shadow: 0 0 8px currentColor; }

/* Bandeau de section "télémétrie" */
.telemetry {
    font-family: 'Orbitron', sans-serif; font-size: 0.68rem; letter-spacing: 0.25em;
    color: var(--amber); margin: 0.2rem 0 0.3rem;
}

/* Widgets Streamlit */
.stButton > button[kind="primary"] {
    font-family: 'Orbitron', sans-serif; letter-spacing: 0.14em; text-transform: uppercase;
    border: 1px solid var(--cyan); border-radius: 12px; padding: 0.7rem 1.6rem;
    background: linear-gradient(90deg, #0b6e99, #5b3fd1); color: #fff;
    box-shadow: 0 0 22px rgba(77, 225, 255, 0.35);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px); box-shadow: 0 0 34px rgba(77, 225, 255, 0.6);
}
[data-testid="stFileUploader"] section {
    border: 1.5px dashed rgba(77, 225, 255, 0.5); border-radius: 16px;
    background: rgba(10, 18, 50, 0.55);
}
[data-testid="stImage"] img { border-radius: 14px; border: 1px solid var(--line); }
[data-testid="stExpander"] {
    border: 1px solid var(--line); border-radius: 14px; background: var(--panel);
}
[data-testid="stExpander"] summary { font-weight: 600; }
.stProgress > div > div > div { background: linear-gradient(90deg, var(--cyan), var(--violet)); }
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--line); background: rgba(14, 22, 54, 0.7); }
[data-testid="stCaptionContainer"] { color: var(--muted); }
hr { border-color: var(--line); }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div>
            <div class="hero-tag">▲ MISSION CONTROL · OBSERVATION DE LA TERRE</div>
            <h1>SmartSat — Signal Faible</h1>
            <p>Détection des navires depuis l'orbite, suppression des doublons
            et transmission adaptée au niveau de confiance.</p>
            <div class="hero-badges">
                <span class="badge">YOLO</span>
                <span class="badge">TUILAGE</span>
                <span class="badge">DÉDOUBLONNAGE</span>
                <span class="badge">LIAISON DESCENDANTE</span>
            </div>
        </div>
        <div class="orbit-wrap">
            <div class="ring r2"></div>
            <div class="ring"><span class="sat">🛰️</span></div>
            <div class="planet"></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
default_weights = ROOT / "detection" / "models" / "best_v2.pt"

if not default_weights.exists():
    default_weights = ROOT / "detection" / "models" / "best.pt"

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🛰️ SMARTSAT · CONSOLE</div>', unsafe_allow_html=True)
    st.header("Paramètres")
    weights = st.text_input("Modèle YOLO", str(default_weights))
    tile_size = st.selectbox("Taille des tuiles", [640, 768], index=1)
    overlap = st.slider("Recouvrement", 0.0, 0.5, 0.2, 0.05)
    confidence = st.slider("Confiance minimale", 0.05, 0.95, 0.25, 0.05)

    st.markdown("---")
    st.markdown("**Niveaux de confiance**")
    legend_html = "".join(
        f'<div class="legend-item"><span class="dot" '
        f'style="background:{color};color:{color}"></span>{tier}</div>'
        for tier, color in TIER_COLORS.items()
    )
    st.markdown(f'<div class="legend" style="flex-direction:column">{legend_html}</div>',
                unsafe_allow_html=True)


def metric_card(label: str, value: str, unit: str = "") -> str:
    unit_html = f'<span class="metric-unit">{unit}</span>' if unit else ""
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}{unit_html}</div></div>'
    )


st.markdown('<div class="telemetry">// ÉTAPE 01 — ACQUISITION</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Choisir une image satellite",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.subheader("Image d'entrée")
    st.image(image, caption=uploaded.name, use_container_width=True)

    if st.button("🚀 Lancer l'analyse", type="primary"):
        if not Path(weights).exists():
            st.error(f"Modèle introuvable : {weights}")
            st.stop()

        from detection.detect import ShipDetector

        detector = ShipDetector(
            weights,
            conf_threshold=confidence,
        )

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / uploaded.name
            image.save(image_path)

            with st.spinner("📡 Balayage de la zone en cours..."):
                result = run_pipeline(
                    str(image_path),
                    weights,
                    tile_size,
                    overlap,
                    detector=detector,
                )

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        crops = []
        payload_bytes = 0

        for index, det in enumerate(result["detections"], 1):
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]

            color = TIER_COLORS[det["tier"]]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw.text(
                (max(0, x1), max(0, y1 - 15)),
                f"Navire {index} | {det['confidence']:.2f}",
                fill=color,
            )

            margin_x = max(1, int((x2 - x1) * 0.2))
            margin_y = max(1, int((y2 - y1) * 0.2))

            crop = image.crop((
                max(0, x1 - margin_x),
                max(0, y1 - margin_y),
                min(image.width, x2 + margin_x),
                min(image.height, y2 + margin_y),
            ))

            crop_buffer = io.BytesIO()
            transmitted_crop = None
            crop_size = 0

            if det["tier"] == "uncertain":
                transmitted_crop = crop.copy()
                transmitted_crop.thumbnail((64, 64))
                transmitted_crop.save(crop_buffer, format="JPEG", quality=60)
                crop_size = len(crop_buffer.getvalue())

            elif det["tier"] == "highly_uncertain":
                transmitted_crop = crop
                transmitted_crop.save(crop_buffer, format="JPEG", quality=90)
                crop_size = len(crop_buffer.getvalue())

            size = 64 + crop_size
            payload_bytes += size

            crops.append((index, det, crop, transmitted_crop, size, crop_size))

        st.markdown('<div class="telemetry">// ÉTAPE 02 — DÉTECTION</div>', unsafe_allow_html=True)
        st.subheader("Résultat de détection")
        st.image(annotated, use_container_width=True)

        jpeg_bytes = len(uploaded.getvalue())
        raw_bytes = image.width * image.height * 3

        st.markdown('<div class="telemetry">// ÉTAPE 03 — TÉLÉMÉTRIE</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-grid">'
            + metric_card("Navires détectés", f"{len(crops)}")
            + metric_card("Tuiles", f"{result['num_tiles']}")
            + metric_card("Image JPEG", f"{jpeg_bytes:,}", "octets")
            + metric_card("Payload mesuré", f"{payload_bytes:,}", "octets")
            + "</div>",
            unsafe_allow_html=True,
        )

        if jpeg_bytes > 0:
            reduction = (1 - payload_bytes / jpeg_bytes) * 100
            st.progress(
                min(1.0, payload_bytes / jpeg_bytes),
                text=(
                    f"Payload / JPEG : "
                    f"{payload_bytes / jpeg_bytes * 100:.2f}% "
                    f"— réduction : {reduction:.2f}%"
                ),
            )

        st.caption(
            f"Taille RGB non compressée : {raw_bytes:,} octets. "
            "Le payload de cette démonstration comprend 64 octets "
            "de métadonnées par navire et les recadrages JPEG "
            "lorsque le niveau de confiance le nécessite."
        )

        st.subheader("Recadrages sélectionnés")

        if not crops:
            st.info("Aucun navire détecté avec ce seuil.")

        for index, det, crop, sent_crop, size, crop_size in crops:
            with st.expander(
                f"🚢 Navire {index} — confiance {det['confidence']:.3f} "
                f"— {det['tier']} — {size:,} octets"
            ):
                left, right = st.columns(2)

                left.image(
                    crop,
                    caption="Recadrage candidat",
                    use_container_width=True,
                )

                if sent_crop is None:
                    right.info(
                        "Métadonnées uniquement : aucun recadrage transmis."
                    )
                else:
                    right.image(
                        sent_crop,
                        caption=f"Recadrage transmis : {crop_size:,} octets",
                        use_container_width=True,
                    )

                st.write("Boîte :", det["bbox_xyxy"])

        st.subheader("Résumé technique")
        st.json({
            "tile_size": tile_size,
            "num_tiles": result["num_tiles"],
            "raw_detections": result["raw_detections"],
            "deduped_detections": result["deduped_detections"],
            "payload_bytes": payload_bytes,
            "input_jpeg_bytes": jpeg_bytes,
            "input_rgb_uncompressed_bytes": raw_bytes,
            "stage_times_s": result["stage_times_s"],
        })

else:
    st.info("🌍 Charge une image JPG ou PNG pour commencer la mission.")