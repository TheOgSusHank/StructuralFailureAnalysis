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
    --bg:          #f6f8fb;
    --surface:     #ffffff;
    --surface2:    #f0f4f9;
    --text:        #111827;
    --text-soft:   #374151;
    --muted:       #4b5563;
    --border:      #d8dee8;
    --primary:     #17324d;
    --accent:      #f59e0b;
    --success:     #15803d;
    --warning:     #b45309;
    --danger:      #b91c1c;
    --blue:        #1d4ed8;
    --radius:      12px;
    --shadow-sm:   0 1px 3px rgba(15,23,42,0.07);
    --shadow-md:   0 6px 20px rgba(15,23,42,0.12);
    --shadow-lg:   0 12px 32px rgba(15,23,42,0.18);
    --transition:  all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.stApp { background: var(--bg); color: var(--text); overflow-x: hidden; }

/* ── Scroll Animations ── */
.reveal {
    opacity: 0;
    transform: translateY(40px);
    transition: all 0.8s ease-out;
}
.reveal.active {
    opacity: 1;
    transform: translateY(0);
}

/* ── Parallax ── */
.parallax-container {
    position: relative;
    height: 350px;
    overflow: hidden;
    border-radius: var(--radius);
    margin-bottom: 2.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #17324d;
    box-shadow: var(--shadow-lg);
}
.parallax-bg {
    position: absolute;
    top: -10%; left: 0; width: 100%; height: 130%;
    background-image: url('https://images.unsplash.com/photo-1581092160562-40aa08e78837?auto=format&fit=crop&q=80&w=2070');
    background-size: cover;
    background-position: center;
    opacity: 0.4;
    z-index: 0;
    transition: transform 0.1s ease-out;
}
.parallax-content {
    position: relative;
    z-index: 1;
    text-align: center;
    color: white;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}

/* ── Carousel ── */
.carousel-container {
    position: relative;
    background: var(--surface);
    border-radius: var(--radius);
    padding: 2.5rem;
    box-shadow: var(--shadow-md);
    margin-bottom: 2.5rem;
    min-height: 280px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    border: 1px solid var(--border);
}
.carousel-item {
    animation: slideIn 0.6s cubic-bezier(0.23, 1, 0.32, 1);
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(30px); }
    to { opacity: 1; transform: translateX(0); }
}

/* ── Enhanced Cards ── */
.sfp-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem;
    transition: var(--transition);
    height: 100%;
}
.sfp-card:hover {
    transform: translateY(-8px);
    box-shadow: var(--shadow-lg);
    border-color: var(--blue);
}
.sfp-risk {
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
    display: inline-block;
}
.sfp-risk-low { background: #dcfce7; color: #15803d; }
.sfp-risk-mod { background: #fef3c7; color: #b45309; }
.sfp-risk-high { background: #fee2e2; color: #b91c1c; }
</style>

<script>
function reveal() {
  var reveals = document.querySelectorAll(".reveal");
  for (var i = 0; i < reveals.length; i++) {
    var windowHeight = window.innerHeight;
    var elementTop = reveals[i].getBoundingClientRect().top;
    var elementVisible = 100;
    if (elementTop < windowHeight - elementVisible) {
      reveals[i].classList.add("active");
    }
  }
}
window.addEventListener("scroll", reveal);

window.addEventListener("scroll", function() {
  const parallax = document.querySelector(".parallax-bg");
  if (parallax) {
    let scrollPosition = window.pageYOffset;
    parallax.style.transform = 'translateY(' + scrollPosition * 0.2 + 'px)';
  }
});

// Initial call
setTimeout(reveal, 100);
</script>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Data Loading & Model ---------------------------------------------
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
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y,
    )
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions).astype(int).tolist()
    return model, float(accuracy), matrix, int(len(data)), int(len(x_test))

model, accuracy, matrix, total_rows, test_rows = train_model(DATASET_PATH)

# ---------- Sidebar ----------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")
    st.subheader("Inspection Parameters")
    crack_size = st.number_input("Crack size (mm)", 0.0, 250.0, float(st.session_state.prefill_crack), step=0.1)
    stress = st.number_input("Stress intensity (MPa√m)", 0.0, 250.0, float(st.session_state.prefill_stress), step=0.5)
    cycles = st.number_input("Load cycles", 0, 10_000_000, int(st.session_state.prefill_cycles), step=1000)
    st.markdown("---")
    analyze_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ---------- Main UI ----------------------------------------------------------

# 1. Parallax Hero
st.markdown(
    """
    <div class="parallax-container">
        <div class="parallax-bg"></div>
        <div class="parallax-content">
            <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">Structural Failure Analysis</h1>
            <p style="font-size: 1.4rem; opacity: 0.9; font-weight: 300;">Physics-Informed Machine Learning for Predictive Maintenance</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Interactive Carousel for Material Profiles
st.markdown("### 🏗️ Material Integrity Profiles")
materials = [
    {"name": "Aluminum 7075-T6", "desc": "High strength aerospace alloy, but sensitive to fatigue crack growth.", "toughness": "29.0 MPa√m", "icon": "✈️"},
    {"name": "A36 Carbon Steel", "desc": "Standard structural steel with high ductility and moderate toughness.", "toughness": "52.0 MPa√m", "icon": "🏗️"},
    {"name": "Ti-6Al-4V Titanium", "desc": "Excellent strength and corrosion resistance, used in critical components.", "toughness": "66.0 MPa√m", "icon": "🚀"},
    {"name": "316 Stainless Steel", "desc": "High toughness and resistance to environmental degradation.", "toughness": "82.0 MPa√m", "icon": "🌊"},
]

c_prev, c_main, c_next = st.columns([1, 6, 1])
with c_prev:
    st.write("") # vertical spacer
    st.write("")
    if st.button("◀", key="prev_mat", use_container_width=True):
        st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(materials)
with c_next:
    st.write("") # vertical spacer
    st.write("")
    if st.button("▶", key="next_mat", use_container_width=True):
        st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(materials)

current_material = materials[st.session_state.carousel_index]
with c_main:
    st.markdown(
        f"""
        <div class="carousel-container">
            <div class="carousel-item">
                <div style="font-size: 3rem; margin-bottom: 1rem;">{current_material['icon']}</div>
                <h2 style="color: var(--primary); margin-top: 0;">{current_material['name']}</h2>
                <p style="font-size: 1.2rem; color: var(--text-soft); max-width: 600px; margin: 0 auto 1.5rem;">{current_material['desc']}</p>
                <div style="background: var(--surface2); padding: 0.7rem 1.5rem; border-radius: 30px; display: inline-block; border: 1px solid var(--border);">
                    <strong style="color: var(--primary);">Critical Fracture Toughness (K<sub>IC</sub>):</strong> {current_material['toughness']}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 3. Scroll-Triggered KPI Cards
st.markdown("### 📊 Live Risk Assessment")
current_inputs = (crack_size, stress, cycles)

if analyze_clicked:
    input_row = pd.DataFrame([{"crack_length_mm": crack_size, "stress_intensity": stress, "load_cycles": cycles}])
    # Add dummy material for model compatibility if needed, though original model seems to use only 3 features
    # Based on earlier code, FEATURE_COLUMNS = ["crack_length_mm", "stress_intensity", "load_cycles"]
    prob = float(model.predict_proba(input_row)[0, 1])
    st.session_state.result = {"probability": prob, "crack": crack_size, "stress": stress, "cycles": cycles}
    st.session_state.last_inputs = current_inputs

res = st.session_state.result
prob = res["probability"] if res else 0.0
risk_label = "High Risk" if prob >= 0.7 else "Moderate Risk" if prob >= 0.35 else "Low Risk"
risk_class = "sfp-risk-high" if prob >= 0.7 else "sfp-risk-mod" if prob >= 0.35 else "sfp-risk-low"

st.markdown(
    f"""
    <div class="reveal">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 2rem; margin-bottom: 3rem;">
            <div class="sfp-card">
                <p style="text-transform: uppercase; font-size: 0.85rem; font-weight: 700; color: var(--muted); margin-bottom: 1rem;">Failure Probability</p>
                <h1 style="margin: 0; color: var(--primary); font-size: 3.5rem;">{prob*100:.1f}%</h1>
                <p style="color: var(--muted); font-size: 0.9rem; margin-top: 1rem;">Based on current fatigue loading</p>
            </div>
            <div class="sfp-card">
                <p style="text-transform: uppercase; font-size: 0.85rem; font-weight: 700; color: var(--muted); margin-bottom: 1rem;">Risk Classification</p>
                <div style="margin: 1rem 0;"><span class="sfp-risk {risk_class}" style="font-size: 1.5rem;">{risk_label}</span></div>
                <p style="color: var(--muted); font-size: 0.9rem; margin-top: 1rem;">Safety threshold compliance</p>
            </div>
            <div class="sfp-card">
                <p style="text-transform: uppercase; font-size: 0.85rem; font-weight: 700; color: var(--muted); margin-bottom: 1rem;">Model Reliability</p>
                <h1 style="margin: 0; color: var(--success); font-size: 3.5rem;">{accuracy*100:.1f}%</h1>
                <p style="color: var(--muted); font-size: 0.9rem; margin-top: 1rem;">Validated on {test_rows} test samples</p>
            </div>
        </div>
    </div>
    <script>reveal();</script>
    """,
    unsafe_allow_html=True,
)

# Additional Scrollable Content
st.markdown("### 📈 Technical Insights")
t_col1, t_col2 = st.columns(2)

with t_col1:
    st.markdown(
        """
        <div class="reveal">
            <div class="sfp-card">
                <h3>Feature Importance</h3>
                <p>Analyzing which factors contribute most to structural failure.</p>
            </div>
        </div>
        <script>reveal();</script>
        """,
        unsafe_allow_html=True,
    )
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    st.bar_chart(importances)

with t_col2:
    st.markdown(
        """
        <div class="reveal">
            <div class="sfp-card">
                <h3>Model Diagnostics</h3>
                <p>Confusion matrix showing classification performance.</p>
            </div>
        </div>
        <script>reveal();</script>
        """,
        unsafe_allow_html=True,
    )
    st.table(pd.DataFrame(matrix, index=["Actual Safe", "Actual Failed"], columns=["Pred Safe", "Pred Failed"]))

st.markdown("<div style='height: 200px;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="reveal" style="text-align: center; padding: 3rem; background: var(--surface2); border-radius: var(--radius);">
        <h2>Ready to improve safety?</h2>
        <p>Integrate this model into your maintenance workflow for real-time monitoring.</p>
    </div>
    <script>reveal();</script>
    """,
    unsafe_allow_html=True,
)
