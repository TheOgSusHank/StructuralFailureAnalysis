from __future__ import annotations

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


# ---------- Styling ----------------------------------------------------------

CUSTOM_CSS = """
<style>
:root {
    --sfp-bg: #f6f8fb;
    --sfp-surface: #ffffff;
    --sfp-text: #111827;
    --sfp-text-soft: #374151;
    --sfp-muted: #4b5563;
    --sfp-border: #d8dee8;
    --sfp-primary: #17324d;
    --sfp-accent: #f59e0b;
    --sfp-success: #15803d;
    --sfp-warning: #b45309;
    --sfp-danger: #b91c1c;
}

.stApp {
    background: var(--sfp-bg);
    color: var(--sfp-text);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

.block-container p,
.block-container li,
.block-container label,
.block-container span,
.block-container div {
    color: inherit;
}

/* Hero header */
.sfp-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    padding: 1.35rem 1.5rem;
    border-radius: 8px;
    background: linear-gradient(135deg, #17324d 0%, #276749 100%);
    color: #ffffff;
    margin-bottom: 1.25rem;
    box-shadow: 0 10px 26px rgba(17, 24, 39, 0.13);
}
.sfp-hero h1 {
    font-size: 1.65rem;
    font-weight: 700;
    margin: 0;
    color: #ffffff;
    letter-spacing: 0;
    line-height: 1.2;
}
.sfp-hero p {
    margin: 0.4rem 0 0 0;
    color: #eef2f7;
    font-size: 0.96rem;
    max-width: 68ch;
    line-height: 1.5;
}
.sfp-hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(245, 158, 11, 0.18);
    color: #fde68a;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
    border: 1px solid rgba(253, 230, 138, 0.45);
    margin-bottom: 0.45rem;
}

/* Panel and card surfaces */
.sfp-panel,
.sfp-card {
    background: var(--sfp-surface);
    border: 1px solid var(--sfp-border);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}
.sfp-panel {
    padding: 1.25rem;
    min-height: 360px;
    color: var(--sfp-text);
}
.sfp-card {
    padding: 1.15rem 1.2rem;
    min-height: 132px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    color: var(--sfp-text);
}
.sfp-card-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0;
    color: var(--sfp-text-soft);
    margin: 0 0 0.45rem 0;
    line-height: 1.25;
}
.sfp-card-value {
    font-size: 1.9rem;
    font-weight: 750;
    color: var(--sfp-primary);
    margin: 0;
    letter-spacing: 0;
    line-height: 1.1;
}
.sfp-card-sub {
    font-size: 0.86rem;
    color: var(--sfp-muted);
    margin-top: 0.45rem;
    line-height: 1.35;
}

/* Risk badge */
.sfp-risk {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.42rem 0.8rem;
    border-radius: 999px;
    font-size: 0.86rem;
    font-weight: 700;
    letter-spacing: 0;
    line-height: 1.1;
    white-space: nowrap;
}
.sfp-risk-dot {
    width: 8px;
    height: 8px;
    flex: 0 0 8px;
    border-radius: 50%;
    background: currentColor;
}
.sfp-risk-low {
    color: var(--sfp-success);
    background: rgba(21, 128, 61, 0.1);
    border: 1px solid rgba(21, 128, 61, 0.28);
}
.sfp-risk-mod {
    color: var(--sfp-warning);
    background: rgba(180, 83, 9, 0.12);
    border: 1px solid rgba(180, 83, 9, 0.3);
}
.sfp-risk-high {
    color: var(--sfp-danger);
    background: rgba(185, 28, 28, 0.1);
    border: 1px solid rgba(185, 28, 28, 0.32);
}

/* Section heading */
.sfp-section-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: var(--sfp-text);
    margin: 0 0 0.45rem 0;
    letter-spacing: 0;
    line-height: 1.25;
}
.sfp-section-sub {
    font-size: 0.92rem;
    color: var(--sfp-text-soft);
    margin: 0 0 1rem 0;
    line-height: 1.45;
}

/* Sidebar tweaks */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--sfp-border);
    color: var(--sfp-text);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.4rem;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--sfp-text) !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: var(--sfp-text-soft) !important;
}
section[data-testid="stSidebar"] input {
    color: var(--sfp-text) !important;
    background: #ffffff !important;
}
section[data-testid="stSidebar"] [data-baseweb="input"] {
    border-color: #cbd5e1;
}
.sfp-thresholds {
    font-size: 0.86rem;
    color: var(--sfp-text);
    line-height: 1.65;
    background: #f8fafc;
    border: 1px solid var(--sfp-border);
    border-radius: 8px;
    padding: 0.8rem 0.9rem;
}
.sfp-thresholds span {
    font-weight: 700;
}

/* Input summary chips */
.sfp-chip-row {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.6rem;
    margin-top: 0.25rem;
}
.sfp-chip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0.65rem 0.75rem;
    border-radius: 8px;
    background: #f8fafc;
    border: 1px solid var(--sfp-border);
    font-size: 0.9rem;
    color: var(--sfp-text);
    line-height: 1.25;
}
.sfp-chip strong {
    color: var(--sfp-primary);
    font-weight: 750;
    text-align: right;
    white-space: nowrap;
}
.sfp-action-title {
    font-weight: 700;
    color: var(--sfp-text);
    margin: 1.15rem 0 0.35rem 0;
}

/* Gauge readout */
.sfp-readout {
    padding-top: 0.5rem;
}
.sfp-gauge-value {
    font-size: 3.1rem;
    font-weight: 750;
    color: var(--sfp-primary);
    line-height: 1;
    letter-spacing: 0;
    margin: 0;
}
.sfp-gauge-label {
    font-size: 0.84rem;
    text-transform: uppercase;
    letter-spacing: 0;
    color: var(--sfp-text-soft);
    margin: 0.45rem 0 0 0;
    font-weight: 700;
}

/* Streamlit components */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    border-bottom: 1px solid var(--sfp-border);
}
.stTabs [data-baseweb="tab"] {
    color: var(--sfp-text-soft);
    font-weight: 700;
    padding-left: 0.75rem;
    padding-right: 0.75rem;
}
.stTabs [aria-selected="true"] {
    color: var(--sfp-primary) !important;
}
[data-testid="stAlert"],
[data-testid="stAlert"] * {
    color: var(--sfp-text) !important;
}
[data-testid="stDataFrame"] {
    color: var(--sfp-text);
}

/* Footer note */
.sfp-footnote {
    font-size: 0.86rem;
    color: var(--sfp-text-soft);
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--sfp-border);
    line-height: 1.5;
}
.sfp-footnote code {
    color: var(--sfp-primary);
}

@media (max-width: 760px) {
    .block-container {
        padding-top: 1rem;
    }
    .sfp-hero {
        padding: 1.15rem;
    }
    .sfp-hero h1 {
        font-size: 1.35rem;
    }
    .sfp-panel {
        min-height: auto;
        padding: 1rem;
    }
    .sfp-card {
        min-height: 118px;
    }
    .sfp-gauge-value {
        font-size: 2.5rem;
    }
}

/* Hide the default Streamlit header padding bumps */
header[data-testid="stHeader"] {
    background: transparent;
}
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
def train_model(path: Path) -> tuple[RandomForestClassifier, float, list[list[int]], int, int]:
    data = load_dataset(path)
    x = data[FEATURE_COLUMNS]
    y = data["failure"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions).astype(int).tolist()
    return model, float(accuracy), matrix, int(len(data)), int(len(x_test))


def risk_label(probability: float) -> str:
    if probability >= 0.70:
        return "High risk"
    if probability >= 0.35:
        return "Moderate risk"
    return "Low risk"


def risk_class(probability: float) -> str:
    if probability >= 0.70:
        return "sfp-risk-high"
    if probability >= 0.35:
        return "sfp-risk-mod"
    return "sfp-risk-low"


def format_cycles(cycles: int) -> str:
    if cycles >= 1_000_000:
        return f"{cycles / 1_000_000:.2f}M"
    if cycles >= 1_000:
        return f"{cycles / 1_000:.1f}k"
    return str(cycles)


# ---------- Charts -----------------------------------------------------------

def render_gauge(probability: float) -> plt.Figure:
    """Half-circle gauge showing risk probability."""
    fig, ax = plt.subplots(figsize=(5.2, 2.9), subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # Gauge spans 180 degrees from pi (left) to 0 (right)
    theta_start, theta_end = np.pi, 0.0
    n_segments = 180
    theta = np.linspace(theta_start, theta_end, n_segments)

    # Color band (green -> amber -> red)
    cmap = LinearSegmentedColormap.from_list(
        "risk", ["#15803d", "#facc15", "#f59e0b", "#b91c1c"], N=n_segments
    )
    width = np.pi / n_segments
    bottom = 0.78
    height = 0.22
    colors = [cmap(i / (n_segments - 1)) for i in range(n_segments)]
    ax.bar(theta, height, width=width, bottom=bottom, color=colors, edgecolor="none", align="edge")

    # Needle
    needle_theta = np.pi * (1.0 - float(np.clip(probability, 0.0, 1.0)))
    ax.plot([needle_theta, needle_theta], [0.0, 0.92], color="#111827", linewidth=2.4, solid_capstyle="round")
    ax.scatter([needle_theta], [0.0], s=80, color="#111827", zorder=5)
    ax.scatter([needle_theta], [0.0], s=28, color="#ffffff", zorder=6)

    # Tick labels at 0%, 50%, 100%
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


def render_confusion_matrix(matrix: list[list[int]]) -> plt.Figure:
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    fig.patch.set_alpha(0)

    cmap = LinearSegmentedColormap.from_list("steel", ["#eef2f7", "#17324d"])
    im = ax.imshow(arr, cmap=cmap, aspect="auto")

    labels_x = ["Predicted safe", "Predicted failed"]
    labels_y = ["Actual safe", "Actual failed"]
    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(labels_x, fontsize=10, color="#111827")
    ax.set_yticklabels(labels_y, fontsize=10, color="#111827")
    ax.tick_params(length=0)

    vmax = arr.max() if arr.size else 1
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = int(arr[i, j])
            color = "#ffffff" if value > vmax * 0.55 else "#111827"
            ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=14, fontweight="bold", color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8, colors="#374151")
    fig.tight_layout()
    return fig


def render_feature_importance(model: RandomForestClassifier) -> plt.Figure:
    importances = model.feature_importances_
    labels = [FEATURE_LABELS[c] for c in FEATURE_COLUMNS]
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.barh(
        np.array(labels)[order],
        importances[order],
        color="#17324d",
        edgecolor="none",
        height=0.55,
    )
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
        min_value=0.0,
        max_value=250.0,
        value=10.0,
        step=0.5,
        help="Measured surface crack length in millimeters.",
    )
    stress = st.number_input(
        "Stress intensity",
        min_value=0.0,
        max_value=250.0,
        value=35.0,
        step=1.0,
        help="Stress intensity factor in MPa\u00b7\u221am.",
    )
    cycles = st.number_input(
        "Load cycles",
        min_value=0,
        max_value=10_000_000,
        value=250_000,
        step=10_000,
        help="Cumulative number of load cycles experienced by the component.",
    )

    st.markdown("---")
    st.markdown("**Risk thresholds**")
    st.markdown(
        "<div class='sfp-thresholds'>"
        "<span style='color:#15803d'>&#9679;</span> Low risk: under 35%<br>"
        "<span style='color:#b45309'>&#9679;</span> Moderate risk: 35-70%<br>"
        "<span style='color:#b91c1c'>&#9679;</span> High risk: 70% or higher"
        "</div>",
        unsafe_allow_html=True,
    )

# Hero
st.markdown(
    """
    <div class="sfp-hero">
        <div>
            <span class="sfp-hero-badge">Predictive Maintenance</span>
            <h1>Structural Failure Risk Predictor</h1>
            <p>Estimate failure probability for fatigue-loaded components using crack length, stress intensity,
            and accumulated load cycles. Powered by a Random Forest trained on Paris&rsquo; Law synthetic data.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPI strip
input_row = pd.DataFrame(
    [
        {
            "crack_length_mm": crack_size,
            "stress_intensity": stress,
            "load_cycles": cycles,
        }
    ]
)
failure_probability = float(model.predict_proba(input_row)[0, 1])
risk_text = risk_label(failure_probability)
risk_css = risk_class(failure_probability)

kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="medium")
with kpi1:
    st.markdown(
        f"""
        <div class="sfp-card">
            <div>
                <p class="sfp-card-title">Failure probability</p>
                <p class="sfp-card-value">{failure_probability * 100:.1f}%</p>
            </div>
            <p class="sfp-card-sub">Current input scenario</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        f"""
        <div class="sfp-card">
            <div>
                <p class="sfp-card-title">Risk classification</p>
                <p class="sfp-card-value" style="font-size:1.2rem;">
                    <span class="sfp-risk {risk_css}">
                        <span class="sfp-risk-dot"></span>{risk_text}
                    </span>
                </p>
            </div>
            <p class="sfp-card-sub">Threshold-based label</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        f"""
        <div class="sfp-card">
            <div>
                <p class="sfp-card-title">Model accuracy</p>
                <p class="sfp-card-value">{accuracy * 100:.1f}%</p>
            </div>
            <p class="sfp-card-sub">On {test_rows:,} held-out samples</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        f"""
        <div class="sfp-card">
            <div>
                <p class="sfp-card-title">Training rows</p>
                <p class="sfp-card-value">{total_rows:,}</p>
            </div>
            <p class="sfp-card-sub">Random Forest, 200 trees</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Tabs: Prediction / Model Performance / About
tab_pred, tab_model, tab_about = st.tabs(["Prediction", "Model Performance", "About"])

with tab_pred:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown(
            """
            <div class="sfp-panel">
                <p class="sfp-section-title">Risk gauge</p>
                <p class="sfp-section-sub">Probability of failure for the current inspection inputs.</p>
            """,
            unsafe_allow_html=True,
        )
        gauge_col_left, gauge_col_right = st.columns([1.4, 1])
        with gauge_col_left:
            st.pyplot(render_gauge(failure_probability), use_container_width=True)
        with gauge_col_right:
            st.markdown(
                f"""
                <div class="sfp-readout">
                    <p class="sfp-gauge-value">{failure_probability * 100:.1f}<span style="font-size:1.6rem;color:#374151;">%</span></p>
                    <p class="sfp-gauge-label">Failure probability</p>
                    <div style="margin-top:0.9rem;">
                        <span class="sfp-risk {risk_css}">
                            <span class="sfp-risk-dot"></span>{risk_text}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="sfp-panel">
                <p class="sfp-section-title">Inspection summary</p>
                <p class="sfp-section-sub">Snapshot of the inputs feeding the prediction.</p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="sfp-chip-row">
                <div class="sfp-chip"><span>Crack length</span><strong>{crack_size:.1f} mm</strong></div>
                <div class="sfp-chip"><span>Stress intensity</span><strong>{stress:.1f}</strong></div>
                <div class="sfp-chip"><span>Load cycles</span><strong>{format_cycles(int(cycles))}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<p class="sfp-action-title">Recommended action</p>', unsafe_allow_html=True)
        if failure_probability >= 0.70:
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
        st.markdown("</div>", unsafe_allow_html=True)

with tab_model:
    st.markdown('<p class="sfp-section-title">Model performance</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sfp-section-sub">How the trained Random Forest classifier performs on the held-out test split.</p>',
        unsafe_allow_html=True,
    )

    cm_col, fi_col = st.columns([1, 1], gap="large")
    with cm_col:
        st.markdown("**Confusion matrix**")
        st.pyplot(render_confusion_matrix(matrix), use_container_width=True)
    with fi_col:
        st.markdown("**Feature importance**")
        st.pyplot(render_feature_importance(model), use_container_width=True)

    st.write("")
    perf_a, perf_b, perf_c = st.columns(3, gap="medium")
    flat = [v for row in matrix for v in row]
    tn, fp, fn, tp = flat if len(flat) == 4 else (0, 0, 0, 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    with perf_a:
        st.markdown(
            f"""
            <div class="sfp-card">
                <div>
                    <p class="sfp-card-title">Precision</p>
                    <p class="sfp-card-value">{precision * 100:.1f}%</p>
                </div>
                <p class="sfp-card-sub">True failures among predicted failures</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with perf_b:
        st.markdown(
            f"""
            <div class="sfp-card">
                <div>
                    <p class="sfp-card-title">Recall</p>
                    <p class="sfp-card-value">{recall * 100:.1f}%</p>
                </div>
                <p class="sfp-card-sub">Failures correctly identified</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with perf_c:
        st.markdown(
            f"""
            <div class="sfp-card">
                <div>
                    <p class="sfp-card-title">F1 score</p>
                    <p class="sfp-card-value">{f1 * 100:.1f}%</p>
                </div>
                <p class="sfp-card-sub">Balanced precision &amp; recall</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    with st.expander("View raw confusion matrix table"):
        confusion_df = pd.DataFrame(
            matrix,
            index=["Actual safe", "Actual failed"],
            columns=["Predicted safe", "Predicted failed"],
        )
        st.dataframe(confusion_df, use_container_width=True)

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
            "Replace the synthetic dataset with validated inspection data before using these predictions for "
            "engineering decisions.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="sfp-footnote">Training data was generated at startup from Paris&rsquo; Law. '
            "Replace the synthetic dataset with validated inspection data before using these predictions for "
            "engineering decisions.</p>",
            unsafe_allow_html=True,
        )
