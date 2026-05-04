from __future__ import annotations
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from image_analysis import analyse_crack_image

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "crack_growth_data.csv"
FEATURE_COLUMNS = ["crack_length_mm", "stress_intensity", "load_cycles"]
FEATURE_LABELS = {
    "crack_length_mm": "Crack length",
    "stress_intensity": "Stress intensity",
    "load_cycles": "Load cycles",
}

st.set_page_config(
    page_title="Structural Failure Risk Predictor",
    page_icon=":gear:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None
if "image_result" not in st.session_state:
    st.session_state.image_result = None
if "prefill_crack" not in st.session_state:
    st.session_state.prefill_crack = 10.0
if "prefill_stress" not in st.session_state:
    st.session_state.prefill_stress = 35.0
if "prefill_cycles" not in st.session_state:
    st.session_state.prefill_cycles = 250_000
if "carousel_index" not in st.session_state:
    st.session_state.carousel_index = 0

CUSTOM_CSS = """
<style>
:root {
    --bg:          #f8fafc;
    --surface:     #ffffff;
    --surface2:    #f1f5f9;
    --text:        #0f172a;
    --text-soft:   #475569;
    --muted:       #64748b;
    --border:      #e2e8f0;
    --primary:     #1e293b;
    --accent:      #f59e0b;
    --success:     #16a34a;
    --warning:     #d97706;
    --danger:      #dc2626;
    --blue:        #2563eb;
    --radius:      8px;
    --shadow-sm:   0 1px 2px rgba(0,0,0,0.05);
    --shadow-md:   0 4px 6px -1px rgba(0,0,0,0.1);
    --shadow-lg:   0 10px 15px -3px rgba(0,0,0,0.1);
    --transition:  all 0.3s ease;
}

/* Base Styles */
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 1rem; max-width: 1400px; }

/* Scroll Reveal */
.reveal {
    opacity: 0;
    transform: translateY(20px);
    transition: var(--transition);
}
.reveal.active {
    opacity: 1;
    transform: translateY(0);
}

/* Parallax Hero */
.parallax-container {
    position: relative;
    height: 220px;
    overflow: hidden;
    border-radius: var(--radius);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0f172a;
    box-shadow: var(--shadow-md);
}
.parallax-bg {
    position: absolute;
    top: -20%; left: 0; width: 100%; height: 140%;
    background-image: url('https://images.unsplash.com/photo-1581092160562-40aa08e78837?auto=format&fit=crop&q=80&w=2070');
    background-size: cover;
    background-position: center;
    opacity: 0.35;
    z-index: 0;
}
.parallax-content {
    position: relative;
    z-index: 1;
    text-align: center;
    color: white;
}

/* Carousel */
.carousel-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    text-align: center;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: var(--shadow-sm);
}

/* KPI Cards */
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    height: 100%;
    transition: var(--transition);
    box-shadow: var(--shadow-sm);
}
.kpi-card:hover {
    border-color: var(--blue);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
.kpi-label { font-size: 0.75rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { font-size: 1.75rem; font-weight: 800; color: var(--primary); margin: 0.25rem 0; }

/* Risk Badges */
.risk-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
}
.risk-low { background: #dcfce7; color: #166534; }
.risk-mod { background: #fef3c7; color: #92400e; }
.risk-high { background: #fee2e2; color: #991b1b; }

/* Photo Analysis */
.photo-findings {
    background: var(--surface2);
    border-radius: var(--radius);
    padding: 1rem;
    border-left: 4px solid var(--blue);
    font-size: 0.9rem;
}
</style>

<script>
function reveal() {
  var reveals = document.querySelectorAll(".reveal");
  for (var i = 0; i < reveals.length; i++) {
    var windowHeight = window.innerHeight;
    var elementTop = reveals[i].getBoundingClientRect().top;
    if (elementTop < windowHeight - 50) {
      reveals[i].classList.add("active");
    }
  }
}
window.addEventListener("scroll", reveal);
window.addEventListener("scroll", function() {
  const parallax = document.querySelector(".parallax-bg");
  if (parallax) {
    let scrollPosition = window.pageYOffset;
    parallax.style.transform = 'translateY(' + scrollPosition * 0.15 + 'px)';
  }
});
setTimeout(reveal, 100);
</script>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Data & Model -----------------------------------------------------
@st.cache_data
def load_dataset(path: Path):
    if not path.exists():
        from data.generate_data import generate_crack_growth_data
        df = generate_crack_growth_data()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)

def train_model(path: Path):
    data = load_dataset(path)
    x = data[FEATURE_COLUMNS]
    y = data["failure"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions).astype(int).tolist()
    return model, float(accuracy), matrix, int(len(data)), int(len(x_test))

model, accuracy, matrix, total_rows, test_rows = train_model(DATASET_PATH)

# ---------- Sidebar ----------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Inspection")
    crack_size = st.number_input("Crack size (mm)", 0.0, 250.0, float(st.session_state.prefill_crack), step=0.1)
    stress = st.number_input("Stress intensity", 0.0, 250.0, float(st.session_state.prefill_stress), step=0.5)
    cycles = st.number_input("Load cycles", 0, 10_000_000, int(st.session_state.prefill_cycles), step=1000)
    st.markdown("---")
    analyze_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ---------- Main UI ----------------------------------------------------------

# 1. Parallax Hero (Reduced height)
st.markdown(
    """
    <div class="parallax-container">
        <div class="parallax-bg"></div>
        <div class="parallax-content">
            <h1 style="font-size: 2.5rem; margin: 0;">Structural Failure Analysis</h1>
            <p style="font-size: 1.1rem; opacity: 0.9;">Physics-Informed Predictive Maintenance</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Materials & KPIs (Combined row to save space)
top_col1, top_col2 = st.columns([1, 2], gap="medium")

with top_col1:
    st.markdown("##### 🏗️ Material Profiles")
    materials = [
        {"name": "Aluminum 7075-T6", "desc": "Aerospace alloy, fatigue sensitive.", "toughness": "29.0 MPa√m", "icon": "✈️"},
        {"name": "A36 Carbon Steel", "desc": "Structural steel, ductile.", "toughness": "52.0 MPa√m", "icon": "🏗️"},
        {"name": "Ti-6Al-4V Titanium", "desc": "Strong, corrosion resistant.", "toughness": "66.0 MPa√m", "icon": "🚀"},
    ]
    
    m_prev, m_next = st.columns(2)
    if m_prev.button("◀ Prev", use_container_width=True):
        st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(materials)
    if m_next.button("Next ▶", use_container_width=True):
        st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(materials)
    
    cur = materials[st.session_state.carousel_index]
    st.markdown(
        f"""
        <div class="carousel-card">
            <div style="font-size: 2rem;">{cur['icon']}</div>
            <div style="font-weight: 700; color: var(--primary);">{cur['name']}</div>
            <div style="font-size: 0.85rem; color: var(--muted); margin: 0.25rem 0;">{cur['desc']}</div>
            <div style="font-size: 0.8rem; font-weight: 600; color: var(--blue);">K<sub>IC</sub>: {cur['toughness']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_col2:
    st.markdown("##### 📊 Live Risk Status")
    current_inputs = (crack_size, stress, cycles)
    if analyze_clicked:
        input_row = pd.DataFrame([{"crack_length_mm": crack_size, "stress_intensity": stress, "load_cycles": cycles}])
        prob = float(model.predict_proba(input_row)[0, 1])
        st.session_state.result = {"probability": prob, "crack": crack_size, "stress": stress, "cycles": cycles}
        st.session_state.last_inputs = current_inputs

    res = st.session_state.result
    prob = res["probability"] if res else 0.0
    risk_label = "High" if prob >= 0.7 else "Moderate" if prob >= 0.35 else "Low"
    risk_class = "risk-high" if prob >= 0.7 else "risk-mod" if prob >= 0.35 else "risk-low"

    k_col1, k_col2, k_col3 = st.columns(3)
    with k_col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Probability</div><div class="kpi-value">{prob*100:.1f}%</div></div>', unsafe_allow_html=True)
    with k_col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Risk Level</div><div style="margin-top:0.5rem;"><span class="risk-badge {risk_class}">{risk_label}</span></div></div>', unsafe_allow_html=True)
    with k_col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Accuracy</div><div class="kpi-value">{accuracy*100:.1f}%</div></div>', unsafe_allow_html=True)

# 3. Tabs for Deep Analysis & Photo Analysis
st.markdown("---")
tab_diag, tab_photo, tab_about = st.tabs(["📈 Diagnostics", "📷 Photo Analysis", "ℹ️ About"])

with tab_diag:
    st.markdown('<div class="reveal">', unsafe_allow_html=True)
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown("**Feature Importance**")
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
        st.bar_chart(importances)
    with d_col2:
        st.markdown("**Confusion Matrix**")
        st.table(pd.DataFrame(matrix, index=["Actual Safe", "Actual Failed"], columns=["Pred Safe", "Pred Failed"]))
    st.markdown('</div><script>reveal();</script>', unsafe_allow_html=True)

with tab_photo:
    st.markdown('<div class="reveal">', unsafe_allow_html=True)
    p_left, p_right = st.columns([1, 1], gap="large")
    with p_left:
        up_file = st.file_uploader("Upload inspection photo", type=["jpg", "png", "jpeg"])
        scale_ref = st.text_input("Scale reference (optional)", placeholder="e.g. 50mm ruler")
        if st.button("🔍 Analyze Photo", disabled=(up_file is None), use_container_width=True, type="primary"):
            with st.spinner("AI analyzing..."):
                try:
                    ir = analyse_crack_image(up_file.read(), mime_type=up_file.type, scale_reference=scale_ref)
                    st.session_state.image_result = ir
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with p_right:
        ir = st.session_state.image_result
        if ir:
            st.markdown(f"**Crack Detected:** {'✅ Yes' if ir.get('crack_detected') else '❌ No'}")
            st.markdown(f"**Type:** {ir.get('crack_type')}")
            st.markdown(f"**Severity:** {ir.get('severity').title()}")
            st.markdown(f'<div class="photo-findings"><b>Findings:</b><br>{ir.get("findings")}</div>', unsafe_allow_html=True)
            if st.button("📥 Load estimates into predictor"):
                nest = ir.get("numeric_estimates", {})
                if nest.get("crack_length_mm"): st.session_state.prefill_crack = float(nest["crack_length_mm"])
                if nest.get("stress_intensity"): st.session_state.prefill_stress = float(nest["stress_intensity"])
                st.rerun()
        else:
            st.info("Upload a photo to see AI-driven crack analysis.")
    st.markdown('</div><script>reveal();</script>', unsafe_allow_html=True)

with tab_about:
    st.markdown(
        """
        <div class="reveal">
            <h4>Methodology</h4>
            <p>This tool combines fracture mechanics (Paris' Law) with Random Forest classification to predict the likelihood of structural failure.</p>
            <ul>
                <li><b>Paris' Law:</b> Models crack growth rate under cyclic loading.</li>
                <li><b>Random Forest:</b> Provides robust classification across varied material profiles.</li>
                <li><b>Computer Vision:</b> Leverages GPT-5 Vision for preliminary inspection triage.</li>
            </ul>
        </div>
        <script>reveal();</script>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
