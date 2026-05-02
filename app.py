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

from data.generate_data import generate_crack_growth_data
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
    st.session_state.result = None        # dict once analysed, None before
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None   # tuple(crack, stress, cycles) used last run
if "image_result" not in st.session_state:
    st.session_state.image_result = None  # dict from vision API, None before
if "prefill_crack" not in st.session_state:
    st.session_state.prefill_crack = 10.0
if "prefill_stress" not in st.session_state:
    st.session_state.prefill_stress = 35.0
if "prefill_cycles" not in st.session_state:
    st.session_state.prefill_cycles = 250_000


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
    --radius:      10px;
    --shadow-sm:   0 1px 3px rgba(15,23,42,0.07);
    --shadow-md:   0 6px 20px rgba(15,23,42,0.12);
    --shadow-lg:   0 12px 32px rgba(15,23,42,0.18);
    --transition:  all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* ── Base ── */
.stApp { background: var(--bg); color: var(--text); }
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}
.block-container p, .block-container li,
.block-container label, .block-container span,
.block-container div { color: inherit; }

/* ── Hero ── */
.sfp-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    padding: 1.4rem 1.6rem;
    border-radius: var(--radius);
    background: linear-gradient(135deg, #17324d 0%, #276749 100%);
    color: #fff;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-md);
    transition: var(--transition);
}
.sfp-hero:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}
.sfp-hero h1 {
    font-size: 1.65rem;
    font-weight: 700;
    margin: 0;
    color: #fff;
    line-height: 1.2;
}
.sfp-hero p {
    margin: 0.4rem 0 0;
    color: #eef2f7;
    font-size: 0.96rem;
    max-width: 68ch;
    line-height: 1.5;
}
.sfp-hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    background: rgba(245,158,11,0.18);
    color: #fde68a;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    border: 1px solid rgba(253,230,138,0.45);
    margin-bottom: 0.45rem;
}

/* ── Force equal-height KPI columns ── */
[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="column"] {
    display: flex;
    flex-direction: column;
}
[data-testid="stHorizontalBlock"] > [data-testid="column"] > [data-testid="stVerticalBlock"] {
    flex: 1;
    display: flex;
    flex-direction: column;
}
[data-testid="stHorizontalBlock"] > [data-testid="column"] > [data-testid="stVerticalBlock"] > [data-testid="stMarkdownContainer"] {
    flex: 1;
    display: flex;
    flex-direction: column;
}

/* ── KPI cards ── */
.sfp-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    padding: 1.4rem 1.4rem 1.2rem;
    height: 148px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    color: var(--text);
    transition: var(--transition);
    cursor: default;
    box-sizing: border-box;
}
.sfp-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: var(--shadow-lg);
    border-color: #b8c8da;
}
.sfp-card-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-soft);
    margin: 0 0 0.5rem;
    line-height: 1.25;
}
.sfp-card-value {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    color: var(--primary) !important;
    margin: 0 !important;
    line-height: 1 !important;
}
.sfp-card-value--badge {
    display: flex !important;
    align-items: center !important;
    font-size: 1rem !important;
}
.sfp-card-sub {
    font-size: 0.8rem;
    color: var(--muted);
    margin: 0;
    padding-top: 0.6rem;
    line-height: 1.3;
    border-top: 1px solid var(--border);
}

/* ── Risk badge ── */
.sfp-risk {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.42rem 0.85rem;
    border-radius: 999px;
    font-size: 0.86rem;
    font-weight: 700;
    white-space: nowrap;
    transition: var(--transition);
}
.sfp-risk:hover { transform: scale(1.06); }
.sfp-risk-dot {
    width: 8px; height: 8px;
    flex: 0 0 8px;
    border-radius: 50%;
    background: currentColor;
}
.sfp-risk-low  { color: var(--success); background: rgba(21,128,61,0.10);  border: 1px solid rgba(21,128,61,0.28); }
.sfp-risk-mod  { color: var(--warning); background: rgba(180,83,9,0.12);   border: 1px solid rgba(180,83,9,0.30); }
.sfp-risk-high { color: var(--danger);  background: rgba(185,28,28,0.10);  border: 1px solid rgba(185,28,28,0.32); }

/* ── Section headings ── */
.sfp-section-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.45rem;
    line-height: 1.25;
}
.sfp-section-sub {
    font-size: 0.92rem;
    color: var(--text-soft);
    margin: 0 0 1rem;
    line-height: 1.45;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--border);
    color: var(--text);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--text) !important; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: var(--text-soft) !important; }
section[data-testid="stSidebar"] input { color: var(--text) !important; background: #fff !important; }
section[data-testid="stSidebar"] [data-baseweb="input"] { border-color: #cbd5e1; }

.sfp-thresholds {
    font-size: 0.86rem;
    color: var(--text);
    line-height: 1.65;
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 0.9rem;
    transition: var(--transition);
}
.sfp-thresholds:hover {
    transform: scale(1.02);
    box-shadow: var(--shadow-sm);
}
.sfp-thresholds span { font-weight: 700; }

/* ── Input chips ── */
.sfp-chip-row { display: grid; grid-template-columns: 1fr; gap: 0.6rem; margin-top: 0.25rem; }
.sfp-chip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0.65rem 0.75rem;
    border-radius: 8px;
    background: #f8fafc;
    border: 1px solid var(--border);
    font-size: 0.9rem;
    color: var(--text);
    line-height: 1.25;
    transition: var(--transition);
    cursor: default;
}
.sfp-chip:hover {
    transform: translateX(4px) scale(1.01);
    box-shadow: var(--shadow-sm);
    border-color: #b8c8da;
    background: #eef4fb;
}
.sfp-chip strong { color: var(--primary); font-weight: 750; text-align: right; white-space: nowrap; }

.sfp-action-title { font-weight: 700; color: var(--text); margin: 1.15rem 0 0.35rem; }

/* ── Gauge readout ── */
.sfp-readout {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 1rem 0;
    height: 100%;
}
.sfp-gauge-value {
    font-size: 5rem !important;
    font-weight: 800 !important;
    color: var(--primary) !important;
    line-height: 1 !important;
    margin: 0 !important;
    letter-spacing: -2px !important;
}
.sfp-gauge-value .sfp-gauge-pct {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: var(--text-soft) !important;
    vertical-align: baseline !important;
    letter-spacing: 0 !important;
    margin-left: 4px !important;
}
.sfp-gauge-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-soft);
    margin: 0.5rem 0 0;
    font-weight: 700;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-soft);
    font-weight: 700;
    padding-left: 0.75rem;
    padding-right: 0.75rem;
    transition: var(--transition);
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--primary) !important;
    transform: translateY(-1px);
}
.stTabs [aria-selected="true"] { color: var(--primary) !important; }

/* ── Streamlit containers + alerts ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border-color: var(--border) !important;
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}
[data-testid="stAlert"],
[data-testid="stAlert"] * { color: var(--text) !important; }
[data-testid="stDataFrame"] { color: var(--text); }

/* ── Buttons ── */
.stButton > button {
    transition: var(--transition) !important;
    border-radius: 8px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: var(--shadow-md) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
}

/* ── Number inputs ── */
[data-testid="stNumberInput"] {
    transition: var(--transition);
}
[data-testid="stNumberInput"]:hover {
    transform: scale(1.01);
}

/* ── Metric tiles (model performance) ── */
.sfp-metric-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.1rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
    cursor: default;
}
.sfp-metric-tile:hover {
    transform: translateY(-4px) scale(1.03);
    box-shadow: var(--shadow-lg);
    border-color: #b8c8da;
}
.sfp-metric-val {
    font-size: 1.85rem;
    font-weight: 750;
    color: var(--primary);
    margin: 0;
    line-height: 1.1;
}
.sfp-metric-lbl {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    color: var(--text-soft);
    margin-top: 0.3rem;
}
.sfp-metric-sub {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 0.3rem;
    line-height: 1.35;
}

/* ── Plot containers ── */
[data-testid="stImage"], .stPyplot {
    transition: var(--transition);
}
[data-testid="stImage"]:hover, .stPyplot > *:hover {
    transform: scale(1.01);
}

/* ── Footer ── */
.sfp-footnote {
    font-size: 0.86rem;
    color: var(--text-soft);
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    line-height: 1.5;
}
.sfp-footnote code { color: var(--primary); }

header[data-testid="stHeader"] { background: transparent; }

/* ── Analysis states ── */
@keyframes thinking-pulse {
    0%, 100% { opacity: 0.4; transform: scaleY(0.6); }
    50%       { opacity: 1;   transform: scaleY(1); }
}
.sfp-pending-value {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    color: #cbd5e1 !important;
    margin: 0 !important;
    line-height: 1 !important;
    letter-spacing: 1px;
}
.sfp-stale-note {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--warning);
    background: rgba(180,83,9,0.08);
    border: 1px solid rgba(180,83,9,0.25);
    border-radius: 6px;
    padding: 0.2rem 0.5rem;
    margin-top: 0.35rem;
}
.sfp-analyze-hint {
    font-size: 0.8rem;
    color: var(--text-soft);
    font-style: italic;
    margin-top: 0.3rem;
}
.sfp-thinking-bars {
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 2.2rem;
    margin: 0.3rem 0;
}
.sfp-thinking-bar {
    width: 7px;
    background: var(--primary);
    border-radius: 3px;
    animation: thinking-pulse 1.1s ease-in-out infinite;
}
.sfp-thinking-bar:nth-child(2) { animation-delay: 0.18s; }
.sfp-thinking-bar:nth-child(3) { animation-delay: 0.36s; }
.sfp-thinking-bar:nth-child(4) { animation-delay: 0.54s; }
.sfp-thinking-bar:nth-child(5) { animation-delay: 0.72s; }

@media (max-width: 760px) {
    .block-container { padding-top: 1rem; }
    .sfp-hero { padding: 1.15rem; }
    .sfp-hero h1 { font-size: 1.35rem; }
    .sfp-card { min-height: 118px; }
    .sfp-gauge-value { font-size: 2.5rem; }
}

/* ── Photo analysis ── */
.sfp-sev-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.85rem;
    border-radius: 999px;
    font-size: 0.86rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    transition: var(--transition);
    cursor: default;
}
.sfp-sev-badge:hover { transform: scale(1.06); }
.sfp-sev-low      { color: var(--success); background: rgba(21,128,61,0.10);  border: 1px solid rgba(21,128,61,0.30); }
.sfp-sev-moderate { color: var(--warning); background: rgba(180,83,9,0.12);   border: 1px solid rgba(180,83,9,0.30); }
.sfp-sev-high     { color: var(--danger);  background: rgba(185,28,28,0.10);  border: 1px solid rgba(185,28,28,0.32); }
.sfp-sev-critical {
    color: #7c0000;
    background: rgba(124,0,0,0.10);
    border: 1px solid rgba(124,0,0,0.35);
    animation: critical-pulse 1.8s ease-in-out infinite;
}
@keyframes critical-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(185,28,28,0); }
    50%       { box-shadow: 0 0 0 6px rgba(185,28,28,0.18); }
}
.sfp-conf-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.22rem 0.55rem;
    border-radius: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text-soft);
    transition: var(--transition);
}
.sfp-conf-badge:hover { transform: scale(1.04); }
.sfp-detail-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.55rem;
    margin: 0.9rem 0;
}
.sfp-detail-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
    transition: var(--transition);
    cursor: default;
}
.sfp-detail-item:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: var(--shadow-sm);
    border-color: #b8c8da;
}
.sfp-detail-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: var(--text-soft);
    margin: 0 0 0.2rem;
}
.sfp-detail-val {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--primary);
    margin: 0;
}
.sfp-findings-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem;
    margin: 0.9rem 0;
    font-size: 0.93rem;
    color: var(--text);
    line-height: 1.55;
    transition: var(--transition);
}
.sfp-findings-box:hover { box-shadow: var(--shadow-sm); }
.sfp-no-crack {
    text-align: center;
    padding: 2.5rem 1rem;
    color: var(--text-soft);
}
.sfp-no-crack-icon { font-size: 2.8rem; margin-bottom: 0.5rem; }
.sfp-upload-hint {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2.5rem 1rem;
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    text-align: center;
    color: var(--text-soft);
    font-size: 0.9rem;
    background: var(--surface2);
    transition: var(--transition);
    cursor: default;
}
.sfp-upload-hint:hover { border-color: #94a3b8; background: #eef4fb; }
.sfp-upload-hint-icon { font-size: 2.2rem; margin-bottom: 0.5rem; opacity: 0.6; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- Data + Model -----------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        return generate_crack_growth_data()
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
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


def risk_label(p: float) -> str:
    if p >= 0.70: return "High risk"
    if p >= 0.35: return "Moderate risk"
    return "Low risk"

def risk_class(p: float) -> str:
    if p >= 0.70: return "sfp-risk-high"
    if p >= 0.35: return "sfp-risk-mod"
    return "sfp-risk-low"

def format_cycles(cycles: int) -> str:
    if cycles >= 1_000_000: return f"{cycles / 1_000_000:.2f}M"
    if cycles >= 1_000: return f"{cycles / 1_000:.1f}k"
    return str(cycles)


# ---------- Charts -----------------------------------------------------------

def render_gauge(probability: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 2.9), subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    theta_start, theta_end = np.pi, 0.0
    n_segments = 180
    theta = np.linspace(theta_start, theta_end, n_segments)
    cmap = LinearSegmentedColormap.from_list(
        "risk", ["#15803d", "#facc15", "#f59e0b", "#b91c1c"], N=n_segments
    )
    width = np.pi / n_segments
    bottom, height = 0.78, 0.22
    colors = [cmap(i / (n_segments - 1)) for i in range(n_segments)]
    ax.bar(theta, height, width=width, bottom=bottom, color=colors, edgecolor="none", align="edge")

    needle_theta = np.pi * (1.0 - float(np.clip(probability, 0.0, 1.0)))
    ax.plot([needle_theta, needle_theta], [0.0, 0.92], color="#111827", linewidth=2.4, solid_capstyle="round")
    ax.scatter([needle_theta], [0.0], s=80,  color="#111827", zorder=5)
    ax.scatter([needle_theta], [0.0], s=28,  color="#ffffff",  zorder=6)

    for frac, text in [(0.0, "0%"), (0.5, "50%"), (1.0, "100%")]:
        t = np.pi * (1.0 - frac)
        ax.text(t, 1.18, text, ha="center", va="center", fontsize=9, color="#374151")

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0, 1.3)
    ax.set_thetalim(0, np.pi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    fig.tight_layout(pad=0.2)
    return fig


def render_confusion_matrix(matrix) -> plt.Figure:
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    fig.patch.set_alpha(0)

    cmap = LinearSegmentedColormap.from_list("steel", ["#eef2f7", "#17324d"])
    im = ax.imshow(arr, cmap=cmap, aspect="auto")

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(["Predicted safe", "Predicted failed"], fontsize=10, color="#111827")
    ax.set_yticklabels(["Actual safe", "Actual failed"],       fontsize=10, color="#111827")
    ax.tick_params(length=0)

    vmax = arr.max() if arr.size else 1
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = int(arr[i, j])
            color = "#ffffff" if v > vmax * 0.55 else "#111827"
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=14, fontweight="bold", color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8, colors="#374151")
    fig.tight_layout()
    return fig


def render_feature_importance(model) -> plt.Figure:
    importances = model.feature_importances_
    labels = [FEATURE_LABELS[c] for c in FEATURE_COLUMNS]
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.barh(np.array(labels)[order], importances[order],
            color="#17324d", edgecolor="none", height=0.55)
    for i, v in enumerate(importances[order]):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9, color="#111827")

    ax.set_xlim(0, max(importances) * 1.18)
    ax.set_xlabel("Relative importance", fontsize=9, color="#374151")
    ax.tick_params(colors="#111827", labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#cbd5e1")

    fig.tight_layout()
    return fig


# ---------- Layout -----------------------------------------------------------

model, accuracy, matrix, total_rows, test_rows = train_model(DATASET_PATH)

# Sidebar inputs
with st.sidebar:
    st.markdown("### Inspection Inputs")
    st.caption("Adjust component measurements to predict failure risk.")

    crack_size = st.number_input(
        "Crack size (mm)",
        min_value=0.0, max_value=250.0,
        value=float(st.session_state.prefill_crack), step=0.5,
        help="Measured surface crack length in millimeters.",
        key="sb_crack",
    )
    stress = st.number_input(
        "Stress intensity",
        min_value=0.0, max_value=250.0,
        value=float(st.session_state.prefill_stress), step=1.0,
        help="Stress intensity factor in MPa·√m.",
        key="sb_stress",
    )
    cycles = st.number_input(
        "Load cycles",
        min_value=0, max_value=10_000_000,
        value=int(st.session_state.prefill_cycles), step=10_000,
        help="Cumulative number of load cycles experienced by the component.",
        key="sb_cycles",
    )

    st.markdown("---")

    # ── Analyze button ──────────────────────────────────────────────────────
    analyze_clicked = st.button(
        "⚙ Analyze Component",
        use_container_width=True,
        type="primary",
    )

    st.markdown("---")
    st.markdown("**Risk thresholds**")
    st.markdown(
        "<div class='sfp-thresholds'>"
        "<span style='color:#15803d'>&#9679;</span> Low risk: under 35%<br>"
        "<span style='color:#b45309'>&#9679;</span> Moderate risk: 35–70%<br>"
        "<span style='color:#b91c1c'>&#9679;</span> High risk: 70% or higher"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Run analysis with animated delay ─────────────────────────────────────────
current_inputs = (crack_size, stress, cycles)

if analyze_clicked:
    delay = random.uniform(3, 15)
    steps = 60
    step_time = delay / steps

    progress_placeholder = st.empty()
    with progress_placeholder.container():
        st.markdown(
            """
            <div style="background:#ffffff;border:1px solid #d8dee8;border-radius:10px;
                        padding:1.2rem 1.5rem;margin-bottom:1rem;
                        box-shadow:0 1px 3px rgba(15,23,42,0.07);">
                <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.5px;color:#374151;margin-bottom:0.6rem;">
                    ⚙ Running AI analysis&hellip;
                </div>
                <div style="font-size:0.85rem;color:#4b5563;margin-bottom:0.9rem;">
                    Evaluating crack propagation, stress intensity factor,
                    and fatigue cycle data against the trained model.
                </div>
            """,
            unsafe_allow_html=True,
        )
        bar = st.progress(0, text="Initialising model inference…")
        st.markdown("</div>", unsafe_allow_html=True)

    phases = [
        (0.15, "Loading feature vectors…"),
        (0.35, "Traversing decision trees…"),
        (0.55, "Aggregating ensemble votes…"),
        (0.75, "Calibrating probability estimate…"),
        (0.92, "Finalising risk classification…"),
        (1.00, "Analysis complete."),
    ]
    phase_idx = 0
    for i in range(steps + 1):
        frac = i / steps
        if phase_idx < len(phases) and frac >= phases[phase_idx][0]:
            bar.progress(frac, text=phases[phase_idx][1])
            phase_idx += 1
        else:
            bar.progress(frac)
        time.sleep(step_time)

    # Compute actual result
    input_row = pd.DataFrame([{
        "crack_length_mm": crack_size,
        "stress_intensity": stress,
        "load_cycles": cycles,
    }])
    prob = float(model.predict_proba(input_row)[0, 1])
    st.session_state.result = {
        "probability": prob,
        "crack": crack_size,
        "stress": stress,
        "cycles": cycles,
    }
    st.session_state.last_inputs = current_inputs
    progress_placeholder.empty()
    st.rerun()

# ── Derive display values from session state ──────────────────────────────────
res = st.session_state.result
has_result = res is not None
inputs_changed = has_result and (st.session_state.last_inputs != current_inputs)

if has_result:
    failure_probability = res["probability"]
    risk_text = risk_label(failure_probability)
    risk_css  = risk_class(failure_probability)
else:
    failure_probability = None
    risk_text = "—"
    risk_css  = "sfp-risk-low"

# Hero
st.markdown(
    """
    <div class="sfp-hero">
        <div>
            <span class="sfp-hero-badge">Predictive Maintenance</span>
            <h1>Structural Failure Risk Predictor</h1>
            <p>Estimate failure probability for fatigue-loaded components using crack length,
            stress intensity, and accumulated load cycles. Powered by a Random Forest trained
            on Paris&rsquo; Law synthetic data.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI strip ─────────────────────────────────────────────────────────────────
stale_badge = (
    '<div class="sfp-stale-note">⚠ Inputs changed — re-analyze</div>'
    if inputs_changed else ""
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="medium")
with kpi1:
    if has_result:
        st.markdown(
            f"""<div class="sfp-card">
                <p class="sfp-card-title">Failure probability</p>
                <p class="sfp-card-value">{failure_probability * 100:.1f}%</p>
                <p class="sfp-card-sub">Last analysis result{' · inputs updated' if inputs_changed else ''}</p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="sfp-card">
                <p class="sfp-card-title">Failure probability</p>
                <p class="sfp-pending-value">— %</p>
                <p class="sfp-analyze-hint">Run analysis to see result</p>
            </div>""",
            unsafe_allow_html=True,
        )
with kpi2:
    if has_result:
        st.markdown(
            f"""<div class="sfp-card">
                <p class="sfp-card-title">Risk classification</p>
                <p class="sfp-card-value sfp-card-value--badge">
                    <span class="sfp-risk {risk_css}">
                        <span class="sfp-risk-dot"></span>{risk_text}
                    </span>
                </p>
                <p class="sfp-card-sub">Threshold-based label</p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="sfp-card">
                <p class="sfp-card-title">Risk classification</p>
                <p class="sfp-pending-value">—</p>
                <p class="sfp-analyze-hint">Awaiting analysis</p>
            </div>""",
            unsafe_allow_html=True,
        )
with kpi3:
    st.markdown(
        f"""<div class="sfp-card">
            <p class="sfp-card-title">Model accuracy</p>
            <p class="sfp-card-value">{accuracy * 100:.1f}%</p>
            <p class="sfp-card-sub">On {test_rows:,} held-out samples</p>
        </div>""",
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        f"""<div class="sfp-card">
            <p class="sfp-card-title">Training rows</p>
            <p class="sfp-card-value">{total_rows:,}</p>
            <p class="sfp-card-sub">Random Forest · 200 trees</p>
        </div>""",
        unsafe_allow_html=True,
    )

if inputs_changed:
    st.markdown(stale_badge, unsafe_allow_html=True)

st.write("")

# Tabs
tab_pred, tab_photo, tab_model, tab_about = st.tabs(
    ["Prediction", "📷 Photo Analysis", "Model Performance", "About"]
)

# ── PREDICTION ──────────────────────────────────────────────────────────────
with tab_pred:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        with st.container(border=True):
            st.markdown('<p class="sfp-section-title">Risk gauge</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="sfp-section-sub">Failure probability for the current inspection inputs.</p>',
                unsafe_allow_html=True,
            )
            gauge_left, gauge_right = st.columns([1.4, 1])
            with gauge_left:
                gauge_val = failure_probability if has_result else 0.0
                st.pyplot(render_gauge(gauge_val), use_container_width=True)
            with gauge_right:
                if has_result:
                    st.markdown(
                        f"""<div class="sfp-readout">
                            <p class="sfp-gauge-value">{failure_probability * 100:.1f}<span class="sfp-gauge-pct">%</span></p>
                            <p class="sfp-gauge-label">Failure probability</p>
                            <div style="margin-top:1rem;">
                                <span class="sfp-risk {risk_css}">
                                    <span class="sfp-risk-dot"></span>{risk_text}
                                </span>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """<div class="sfp-readout">
                            <div class="sfp-thinking-bars">
                                <div class="sfp-thinking-bar" style="height:60%"></div>
                                <div class="sfp-thinking-bar" style="height:100%"></div>
                                <div class="sfp-thinking-bar" style="height:75%"></div>
                                <div class="sfp-thinking-bar" style="height:90%"></div>
                                <div class="sfp-thinking-bar" style="height:50%"></div>
                            </div>
                            <p class="sfp-gauge-label">Awaiting analysis</p>
                            <p class="sfp-analyze-hint" style="margin-top:0.4rem;">
                                Click &ldquo;Analyze Component&rdquo; in the sidebar
                            </p>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    with right:
        with st.container(border=True):
            st.markdown('<p class="sfp-section-title">Inspection summary</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="sfp-section-sub">Snapshot of the inputs feeding the prediction.</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<div class="sfp-chip-row">
                    <div class="sfp-chip"><span>Crack length</span><strong>{crack_size:.1f} mm</strong></div>
                    <div class="sfp-chip"><span>Stress intensity</span><strong>{stress:.1f}</strong></div>
                    <div class="sfp-chip"><span>Load cycles</span><strong>{format_cycles(int(cycles))}</strong></div>
                </div>""",
                unsafe_allow_html=True,
            )

            st.markdown('<p class="sfp-action-title">Recommended action</p>', unsafe_allow_html=True)
            if not has_result:
                st.info(
                    "Set your inspection values in the sidebar and click **Analyze Component** to get a risk assessment.",
                    icon=":material/info:",
                )
            elif failure_probability >= 0.70:
                st.error(
                    "Immediate action required. Remove the component from service and perform "
                    "non-destructive evaluation before further loading.",
                    icon=":material/warning:",
                )
            elif failure_probability >= 0.35:
                st.warning(
                    "Schedule a near-term inspection. Consider reducing duty cycle and tracking "
                    "crack growth rate.",
                    icon=":material/error:",
                )
            else:
                st.success(
                    "Component appears within safe operating bounds. Continue with the regular "
                    "inspection interval.",
                    icon=":material/check_circle:",
                )

# ── PHOTO ANALYSIS ───────────────────────────────────────────────────────────
with tab_photo:
    st.markdown('<p class="sfp-section-title">AI-powered crack photo analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sfp-section-sub">Upload a photograph of the component and the AI will characterise '
        'any visible cracks, estimate measurements, and optionally load the findings into the predictor.</p>',
        unsafe_allow_html=True,
    )

    photo_left, photo_right = st.columns([1, 1.1], gap="large")

    with photo_left:
        with st.container(border=True):
            st.markdown("**Upload photo**")
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
                help="JPEG, PNG or WebP — maximum 20 MB.",
            )

            scale_ref = None
            with st.expander("Add scale reference (optional but improves accuracy)"):
                scale_ref = st.text_input(
                    "Describe a scale reference visible in the photo",
                    placeholder='e.g. "Ruler alongside crack shows 50 mm total"',
                    help="If there is a ruler, coin, or object of known size in the photo, describe it here.",
                )

            analyze_photo_btn = st.button(
                "🔍 Analyse Photo",
                disabled=(uploaded_file is None),
                use_container_width=True,
                type="primary",
            )

            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded photo", use_container_width=True)

    with photo_right:
        with st.container(border=True):
            if analyze_photo_btn and uploaded_file is not None:
                mime = uploaded_file.type or "image/jpeg"
                img_bytes = uploaded_file.read()
                with st.spinner("Analysing photo with AI vision…"):
                    try:
                        ir = analyse_crack_image(
                            img_bytes,
                            mime_type=mime,
                            scale_reference=scale_ref if scale_ref else None,
                        )
                        st.session_state.image_result = ir
                    except Exception as exc:
                        err_msg = str(exc)
                        if "FREE_CLOUD_BUDGET_EXCEEDED" in err_msg:
                            st.error("Your Replit AI cloud budget has been exceeded. Please check your account.")
                        else:
                            st.error(f"Analysis failed: {err_msg}")
                        st.session_state.image_result = None

            ir = st.session_state.image_result

            if ir is None:
                st.markdown(
                    """<div class="sfp-upload-hint">
                        <div class="sfp-upload-hint-icon">🔬</div>
                        <strong>Upload a photo and click Analyse</strong><br>
                        <span style="font-size:0.82rem;margin-top:0.3rem;display:block;">
                        Results will appear here — crack type, dimensions, severity, and recommended action.
                        </span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                detected = ir.get("crack_detected", False)
                severity = ir.get("severity", "low").lower()
                confidence = ir.get("confidence", "low").lower()

                sev_class_map = {
                    "low": "sfp-sev-low",
                    "moderate": "sfp-sev-moderate",
                    "high": "sfp-sev-high",
                    "critical": "sfp-sev-critical",
                }
                sev_icon_map = {
                    "low": "✅", "moderate": "⚠️", "high": "🔴", "critical": "🚨"
                }
                sev_css = sev_class_map.get(severity, "sfp-sev-low")
                sev_icon = sev_icon_map.get(severity, "✅")

                if not detected:
                    st.markdown(
                        """<div class="sfp-no-crack">
                            <div class="sfp-no-crack-icon">🔍</div>
                            <strong>No crack detected</strong><br>
                            <span style="font-size:0.85rem;">
                            The AI found no visible cracks in this image.
                            Try a clearer photo or adjust the lighting.
                            </span>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    # Header row: severity + confidence
                    crack_type = ir.get("crack_type", "unknown").title()
                    st.markdown(
                        f"""<div style="display:flex;align-items:center;gap:0.65rem;flex-wrap:wrap;margin-bottom:0.5rem;">
                            <span class="sfp-sev-badge {sev_css}">{sev_icon} {severity.title()} severity</span>
                            <span class="sfp-conf-badge">Confidence: {confidence.title()}</span>
                            <span style="font-size:0.9rem;font-weight:700;color:var(--primary);">{crack_type} crack</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                    # Findings
                    findings = ir.get("findings", "")
                    if findings:
                        st.markdown(
                            f'<div class="sfp-findings-box">{findings}</div>',
                            unsafe_allow_html=True,
                        )

                    # Detail grid
                    details = [
                        ("Estimated length", ir.get("crack_length_estimate", "—")),
                        ("Estimated width",  ir.get("crack_width_estimate",  "—")),
                        ("Orientation",      ir.get("orientation",           "—").title()),
                        ("Surface",          ir.get("surface_condition",     "—").title()),
                    ]
                    grid_html = '<div class="sfp-detail-grid">'
                    for label, val in details:
                        grid_html += (
                            f'<div class="sfp-detail-item">'
                            f'<p class="sfp-detail-label">{label}</p>'
                            f'<p class="sfp-detail-val">{val}</p>'
                            f'</div>'
                        )
                    grid_html += '</div>'
                    st.markdown(grid_html, unsafe_allow_html=True)

                    # Recommended action
                    action = ir.get("recommended_action", "")
                    if action:
                        if severity == "critical":
                            st.error(action, icon=":material/warning:")
                        elif severity == "high":
                            st.error(action, icon=":material/warning:")
                        elif severity == "moderate":
                            st.warning(action, icon=":material/error:")
                        else:
                            st.success(action, icon=":material/check_circle:")

                    # Load into predictor
                    num = ir.get("numeric_estimates", {})
                    has_crack_mm    = num.get("crack_length_mm") is not None
                    has_stress      = num.get("stress_intensity") is not None

                    if has_crack_mm or has_stress:
                        st.markdown("---")
                        st.markdown(
                            "**AI-estimated measurements** — load these values into the predictor "
                            "and run a full analysis:"
                        )
                        est_crack  = num.get("crack_length_mm")  or st.session_state.prefill_crack
                        est_stress = num.get("stress_intensity") or st.session_state.prefill_stress

                        col_e1, col_e2 = st.columns(2)
                        with col_e1:
                            if has_crack_mm:
                                st.markdown(
                                    f'<div class="sfp-detail-item"><p class="sfp-detail-label">Crack length</p>'
                                    f'<p class="sfp-detail-val">{est_crack:.1f} mm</p></div>',
                                    unsafe_allow_html=True,
                                )
                        with col_e2:
                            if has_stress:
                                st.markdown(
                                    f'<div class="sfp-detail-item"><p class="sfp-detail-label">Stress intensity</p>'
                                    f'<p class="sfp-detail-val">{est_stress:.1f} MPa·√m</p></div>',
                                    unsafe_allow_html=True,
                                )

                        if st.button(
                            "⚙ Load into Predictor",
                            use_container_width=True,
                            help="Copies AI-estimated values to the sidebar inputs on the Prediction tab.",
                        ):
                            if has_crack_mm:
                                st.session_state.prefill_crack = float(est_crack)
                            if has_stress:
                                st.session_state.prefill_stress = float(est_stress)
                            # Clear widget keys so they pick up the new defaults on rerun
                            for k in ("sb_crack", "sb_stress", "sb_cycles"):
                                if k in st.session_state:
                                    del st.session_state[k]
                            st.rerun()


# ── MODEL PERFORMANCE ────────────────────────────────────────────────────────
with tab_model:
    flat = [v for row in matrix for v in row]
    tn, fp, fn, tp = flat if len(flat) == 4 else (0, 0, 0, 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    st.markdown('<p class="sfp-section-title">Model performance</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sfp-section-sub">Evaluated on {test_rows:,} held-out samples '
        f'({total_rows:,} total rows).</p>',
        unsafe_allow_html=True,
    )

    perf_a, perf_b, perf_c, perf_d = st.columns(4, gap="medium")
    for col, label, val, sub in [
        (perf_a, "Accuracy",  accuracy,  f"On {test_rows:,} samples"),
        (perf_b, "Precision", precision, "True failures / predicted"),
        (perf_c, "Recall",    recall,    "Failures correctly caught"),
        (perf_d, "F1 Score",  f1,        "Balanced precision & recall"),
    ]:
        with col:
            st.markdown(
                f"""<div class="sfp-metric-tile">
                    <p class="sfp-metric-val">{val * 100:.1f}%</p>
                    <div class="sfp-metric-lbl">{label}</div>
                    <div class="sfp-metric-sub">{sub}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    cm_col, fi_col = st.columns([1, 1], gap="large")
    with cm_col:
        st.markdown("**Confusion matrix**")
        st.pyplot(render_confusion_matrix(matrix), use_container_width=True)
    with fi_col:
        st.markdown("**Feature importance**")
        st.pyplot(render_feature_importance(model), use_container_width=True)

    st.write("")
    with st.expander("View raw confusion matrix table"):
        confusion_df = pd.DataFrame(
            matrix,
            index=["Actual safe", "Actual failed"],
            columns=["Predicted safe", "Predicted failed"],
        )
        st.dataframe(confusion_df, use_container_width=True)

# ── ABOUT ────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown('<p class="sfp-section-title">About this model</p>', unsafe_allow_html=True)
    st.markdown(
        """
        This dashboard predicts the probability that a fatigue-loaded structural component will fail,
        given three inspection inputs:

        - **Crack length (mm)** — measured surface crack size.
        - **Stress intensity** — stress intensity factor in MPa&middot;&radic;m.
        - **Load cycles** — cumulative number of load cycles applied.

        The classifier is a `RandomForestClassifier` (200 trees, balanced class weights) trained on
        synthetic crack growth samples generated from **Paris' Law**:
        """
    )
    st.latex(r"\frac{da}{dN} = C \cdot (\Delta K)^{m}")
    st.markdown(
        """
        Risk thresholds are applied to the predicted failure probability:

        | Probability | Label |
        | --- | --- |
        | &lt; 35% | Low risk |
        | 35% &ndash; 70% | Moderate risk |
        | &ge; 70% | High risk |
        """
    )
    if DATASET_PATH.exists():
        st.markdown(
            f'<p class="sfp-footnote">Training data loaded from <code>{DATASET_PATH.relative_to(PROJECT_ROOT)}</code>. '
            "Replace the synthetic dataset with validated inspection data before using these predictions "
            "for engineering decisions.</p>",
            unsafe_allow_html=True,
        )
