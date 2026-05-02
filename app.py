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

MISSIONS = [
    {
        "name": "Tutorial: Baby Crack",
        "icon": "🐣",
        "desc": "A tiny surface crack, low stress, fresh component.",
        "crack": 2.0, "stress": 10.0, "cycles": 50_000,
        "target": "low",
    },
    {
        "name": "Mission 1: Routine Check",
        "icon": "🔧",
        "desc": "Standard inspection on an aging bridge support.",
        "crack": 8.0, "stress": 30.0, "cycles": 200_000,
        "target": "low",
    },
    {
        "name": "Mission 2: Elevated Risk",
        "icon": "⚠️",
        "desc": "A turbine blade showing moderate fatigue signs.",
        "crack": 22.0, "stress": 65.0, "cycles": 800_000,
        "target": "moderate",
    },
    {
        "name": "Mission 3: Critical Infrastructure",
        "icon": "🏗️",
        "desc": "A suspension cable with a worrying crack under heavy load.",
        "crack": 55.0, "stress": 120.0, "cycles": 2_500_000,
        "target": "moderate",
    },
    {
        "name": "Mission 4: Code Red",
        "icon": "🚨",
        "desc": "A pressure vessel showing signs of imminent failure.",
        "crack": 110.0, "stress": 190.0, "cycles": 7_000_000,
        "target": "high",
    },
    {
        "name": "BOSS: Nuclear Reactor",
        "icon": "☢️",
        "desc": "Maximum stress. Maximum stakes. Can you predict it?",
        "crack": 180.0, "stress": 240.0, "cycles": 9_500_000,
        "target": "high",
    },
]

ACHIEVEMENTS = [
    {"id": "first_predict", "icon": "🎯", "name": "First Blood", "desc": "Make your first prediction"},
    {"id": "low_master", "icon": "🟢", "name": "Safety Officer", "desc": "Correctly identify 3 Low risk scenarios"},
    {"id": "high_master", "icon": "🔴", "name": "Danger Detector", "desc": "Correctly identify a High risk scenario"},
    {"id": "all_missions", "icon": "🏆", "name": "Mission Complete", "desc": "Complete all 6 missions"},
    {"id": "perfect_score", "icon": "⭐", "name": "Ace Engineer", "desc": "Score 100 XP in a single session"},
    {"id": "mod_master", "icon": "🟡", "name": "Edge Walker", "desc": "Correctly identify a Moderate risk scenario"},
]

st.set_page_config(
    page_title="FailureQuest — Structural Risk Simulator",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #1c2330;
    --border: #30363d;
    --text: #e6edf3;
    --text-soft: #8b949e;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
    --blue: #58a6ff;
    --purple: #bc8cff;
    --orange: #f0883e;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 1300px;
}

/* ---- Animated hero ---- */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pulse-border {
    0%, 100% { box-shadow: 0 0 0 0 rgba(63,185,80,0.4); }
    50%       { box-shadow: 0 0 0 8px rgba(63,185,80,0); }
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(248,81,73,0.5); }
    50%       { box-shadow: 0 0 0 12px rgba(248,81,73,0); }
}
@keyframes pulse-yellow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(210,153,34,0.5); }
    50%       { box-shadow: 0 0 0 10px rgba(210,153,34,0); }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-6px); }
}
@keyframes spin-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}

.hero-wrapper {
    background: linear-gradient(135deg, #0d1117, #161b22, #1c2330, #0d1117);
    background-size: 400% 400%;
    animation: gradientShift 8s ease infinite;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(88,166,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-title span {
    background: linear-gradient(90deg, #58a6ff, #3fb950, #d29922, #58a6ff);
    background-size: 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
}
.hero-sub {
    color: var(--text-soft);
    font-size: 1rem;
    margin: 0;
    max-width: 60ch;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.75rem;
    background: rgba(88,166,255,0.12);
    border: 1px solid rgba(88,166,255,0.3);
    border-radius: 999px;
    color: var(--blue);
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ---- XP / Level bar ---- */
.xp-bar-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1rem;
}
.xp-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-soft);
    margin-bottom: 0.45rem;
}
.xp-label span { color: var(--blue); }
.xp-track {
    background: #21262d;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.xp-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #58a6ff, #3fb950);
    transition: width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* ---- Stat cards ---- */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    text-align: center;
    animation: fadeIn 0.4s ease both;
}
.stat-card-icon { font-size: 1.6rem; margin-bottom: 0.2rem; }
.stat-card-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
}
.stat-card-lbl {
    font-size: 0.75rem;
    color: var(--text-soft);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.2rem;
}

/* ---- Risk badge ---- */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0;
    margin-top: 0.5rem;
}
.risk-low  { background: rgba(63,185,80,0.15);  border: 1px solid rgba(63,185,80,0.4);  color: #3fb950; animation: pulse-border 2s ease-in-out infinite; }
.risk-mod  { background: rgba(210,153,34,0.15); border: 1px solid rgba(210,153,34,0.4); color: #d29922; animation: pulse-yellow 2s ease-in-out infinite; }
.risk-high { background: rgba(248,81,73,0.15);  border: 1px solid rgba(248,81,73,0.4);  color: #f85149; animation: pulse-red 1.5s ease-in-out infinite; }

/* ---- Mission cards ---- */
.mission-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.15rem;
    cursor: pointer;
    transition: all 0.2s;
    animation: fadeIn 0.3s ease both;
}
.mission-card:hover {
    border-color: var(--blue);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.mission-card.done {
    border-color: var(--green);
    background: rgba(63,185,80,0.07);
}
.mission-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.2rem 0;
}
.mission-desc {
    font-size: 0.8rem;
    color: var(--text-soft);
    margin: 0;
}

/* ---- Achievement badges ---- */
.ach-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 0.5rem;
}
.ach-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.75rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    border: 1px solid;
    animation: fadeIn 0.4s ease both;
}
.ach-unlocked {
    background: rgba(188,140,255,0.12);
    border-color: rgba(188,140,255,0.4);
    color: var(--purple);
}
.ach-locked {
    background: var(--surface2);
    border-color: var(--border);
    color: var(--text-soft);
    opacity: 0.5;
}

/* ---- Gauge readout ---- */
.gauge-readout {
    text-align: center;
    padding: 1rem 0;
    animation: fadeIn 0.4s ease;
}
.gauge-pct {
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
    margin: 0;
}
.gauge-pct-low  { color: #3fb950; }
.gauge-pct-mod  { color: #d29922; }
.gauge-pct-high { color: #f85149; animation: pulse-red 1.2s ease-in-out infinite; }

/* ---- Sidebar tweaks ---- */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] > div {
    color: var(--text-soft) !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    border-bottom: 1px solid var(--border);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-soft) !important;
    font-weight: 700;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom: 2px solid var(--blue) !important;
}

/* ---- Global overrides ---- */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stAlert"] * { color: var(--text) !important; }
header[data-testid="stHeader"] { background: transparent; }

/* feedback pop */
.feedback-pop {
    text-align: center;
    padding: 1.2rem;
    border-radius: 12px;
    font-size: 1.1rem;
    font-weight: 700;
    animation: fadeIn 0.4s ease;
    margin-top: 0.5rem;
}
.feedback-correct {
    background: rgba(63,185,80,0.15);
    border: 1px solid rgba(63,185,80,0.4);
    color: #3fb950;
}
.feedback-wrong {
    background: rgba(248,81,73,0.12);
    border: 1px solid rgba(248,81,73,0.35);
    color: #f85149;
}
.feedback-almost {
    background: rgba(210,153,34,0.12);
    border: 1px solid rgba(210,153,34,0.35);
    color: #d29922;
}

.gear-spin { display: inline-block; animation: spin-slow 4s linear infinite; }
.float-icon { display: inline-block; animation: float 3s ease-in-out infinite; }

/* score pop */
.score-pop {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--blue);
    text-align: center;
    animation: fadeIn 0.3s ease;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- Session state bootstrap ------------------------------------------

def _init_state():
    defaults = {
        "xp": 0,
        "level": 1,
        "correct_low": 0,
        "correct_high": 0,
        "correct_mod": 0,
        "missions_done": set(),
        "achievements": set(),
        "last_feedback": None,
        "last_xp_gain": 0,
        "total_predictions": 0,
        "active_mission_idx": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def level_from_xp(xp: int) -> int:
    thresholds = [0, 50, 120, 220, 350, 500, 700]
    for i, t in enumerate(reversed(thresholds)):
        if xp >= t:
            return len(thresholds) - i
    return 1

def xp_for_level(lvl: int) -> int:
    thresholds = [0, 50, 120, 220, 350, 500, 700]
    return thresholds[min(lvl - 1, len(thresholds) - 1)]

def xp_for_next(lvl: int) -> int:
    thresholds = [0, 50, 120, 220, 350, 500, 700]
    if lvl >= len(thresholds):
        return thresholds[-1]
    return thresholds[lvl]

LEVEL_TITLES = ["", "Apprentice", "Junior Eng", "Senior Eng", "Lead Eng", "Principal", "Chief Eng", "Legend"]


# ---------- Model -------------------------------------------------------------

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
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=1
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions).astype(int).tolist()
    return model, float(accuracy), matrix, int(len(data)), int(len(x_test))


def risk_label(p: float) -> str:
    if p >= 0.70: return "HIGH RISK"
    if p >= 0.35: return "MODERATE RISK"
    return "LOW RISK"

def risk_class(p: float) -> str:
    if p >= 0.70: return "risk-high"
    if p >= 0.35: return "risk-mod"
    return "risk-low"

def risk_tier(p: float) -> str:
    if p >= 0.70: return "high"
    if p >= 0.35: return "moderate"
    return "low"

def risk_emoji(p: float) -> str:
    if p >= 0.70: return "🚨"
    if p >= 0.35: return "⚠️"
    return "✅"

def gauge_pct_class(p: float) -> str:
    if p >= 0.70: return "gauge-pct-high"
    if p >= 0.35: return "gauge-pct-mod"
    return "gauge-pct-low"

def format_cycles(cycles: int) -> str:
    if cycles >= 1_000_000: return f"{cycles / 1_000_000:.2f}M"
    if cycles >= 1_000: return f"{cycles / 1_000:.0f}k"
    return str(cycles)


# ---------- Charts ------------------------------------------------------------

def render_gauge(probability: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 2.9), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    theta_start, theta_end = np.pi, 0.0
    n_segments = 180
    theta = np.linspace(theta_start, theta_end, n_segments)

    cmap = LinearSegmentedColormap.from_list(
        "risk", ["#3fb950", "#d29922", "#f0883e", "#f85149"], N=n_segments
    )
    width = np.pi / n_segments
    bottom = 0.78
    height = 0.22
    colors = [cmap(i / (n_segments - 1)) for i in range(n_segments)]
    ax.bar(theta, height, width=width, bottom=bottom, color=colors, edgecolor="none", align="edge")

    needle_theta = np.pi * (1.0 - float(np.clip(probability, 0.0, 1.0)))
    needle_color = "#f85149" if probability >= 0.70 else ("#d29922" if probability >= 0.35 else "#3fb950")
    ax.plot([needle_theta, needle_theta], [0.0, 0.92], color=needle_color, linewidth=3, solid_capstyle="round")
    ax.scatter([needle_theta], [0.0], s=120, color=needle_color, zorder=5)
    ax.scatter([needle_theta], [0.0], s=40, color="#0d1117", zorder=6)

    for frac, text in [(0.0, "0%"), (0.5, "50%"), (1.0, "100%")]:
        t = np.pi * (1.0 - frac)
        ax.text(t, 1.18, text, ha="center", va="center", fontsize=9, color="#8b949e")

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
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    cmap = LinearSegmentedColormap.from_list("steel", ["#21262d", "#58a6ff"])
    im = ax.imshow(arr, cmap=cmap, aspect="auto")

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(["Predicted safe", "Predicted failed"], fontsize=10, color="#e6edf3")
    ax.set_yticklabels(["Actual safe", "Actual failed"], fontsize=10, color="#e6edf3")
    ax.tick_params(length=0)

    vmax = arr.max() if arr.size else 1
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = int(arr[i, j])
            color = "#0d1117" if v > vmax * 0.55 else "#e6edf3"
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=14, fontweight="bold", color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8, colors="#8b949e")
    fig.tight_layout()
    return fig


def render_feature_importance(model) -> plt.Figure:
    importances = model.feature_importances_
    labels = [FEATURE_LABELS[c] for c in FEATURE_COLUMNS]
    order = np.argsort(importances)
    bar_colors = ["#58a6ff", "#3fb950", "#d29922"]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    ax.barh(
        np.array(labels)[order],
        importances[order],
        color=[bar_colors[i] for i in order],
        edgecolor="none",
        height=0.55,
    )
    for i, v in enumerate(importances[order]):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9, color="#e6edf3")

    ax.set_xlim(0, max(importances) * 1.2)
    ax.set_xlabel("Relative importance", fontsize=9, color="#8b949e")
    ax.tick_params(colors="#e6edf3", labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#30363d")

    fig.tight_layout()
    return fig


# ---------- XP + Achievement helpers ------------------------------------------

def award_xp(amount: int, reason: str = ""):
    st.session_state.xp += amount
    st.session_state.last_xp_gain = amount
    st.session_state.level = level_from_xp(st.session_state.xp)


def check_achievements():
    unlocked = st.session_state.achievements
    new_ones = []

    if st.session_state.total_predictions >= 1 and "first_predict" not in unlocked:
        unlocked.add("first_predict")
        new_ones.append("first_predict")

    if st.session_state.correct_low >= 3 and "low_master" not in unlocked:
        unlocked.add("low_master")
        new_ones.append("low_master")

    if st.session_state.correct_high >= 1 and "high_master" not in unlocked:
        unlocked.add("high_master")
        new_ones.append("high_master")

    if st.session_state.correct_mod >= 1 and "mod_master" not in unlocked:
        unlocked.add("mod_master")
        new_ones.append("mod_master")

    if len(st.session_state.missions_done) >= len(MISSIONS) and "all_missions" not in unlocked:
        unlocked.add("all_missions")
        new_ones.append("all_missions")

    if st.session_state.xp >= 100 and "perfect_score" not in unlocked:
        unlocked.add("perfect_score")
        new_ones.append("perfect_score")

    return new_ones


# ---------- Load model -------------------------------------------------------

model, accuracy, matrix, total_rows, test_rows = train_model(DATASET_PATH)


# ---------- Sidebar ----------------------------------------------------------

with st.sidebar:
    st.markdown(f"## <span class='gear-spin'>⚙️</span> Engineer HQ", unsafe_allow_html=True)

    lvl = st.session_state.level
    xp = st.session_state.xp
    title = LEVEL_TITLES[min(lvl, len(LEVEL_TITLES) - 1)]
    xp_cur = xp_for_level(lvl)
    xp_nxt = xp_for_next(lvl)
    pct = min(100, int((xp - xp_cur) / max(1, xp_nxt - xp_cur) * 100))

    st.markdown(
        f"""
        <div class="xp-bar-wrap">
            <div class="xp-label">
                <span>Lv.{lvl} — {title}</span>
                <span>{xp} XP</span>
            </div>
            <div class="xp-track">
                <div class="xp-fill" style="width:{pct}%"></div>
            </div>
            <div style="font-size:0.75rem;color:var(--text-soft);margin-top:0.35rem;">
                {xp_nxt - xp} XP to next level
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎮 Mission Select")
    for idx, m in enumerate(MISSIONS):
        done = idx in st.session_state.missions_done
        done_cls = " done" if done else ""
        done_mark = " ✓" if done else ""
        if st.button(
            f"{m['icon']} {m['name']}{done_mark}",
            key=f"mission_{idx}",
            use_container_width=True,
        ):
            st.session_state.active_mission_idx = idx

    st.markdown("---")

    st.markdown("### 📐 Manual Controls")
    st.caption("Or dial in your own values below.")

    crack_size = st.slider(
        "🔩 Crack size (mm)",
        min_value=0.0, max_value=250.0, value=10.0, step=0.5,
    )
    stress = st.slider(
        "⚡ Stress intensity",
        min_value=0.0, max_value=250.0, value=35.0, step=1.0,
    )
    cycles = st.slider(
        "🔄 Load cycles",
        min_value=0, max_value=10_000_000, value=250_000, step=10_000,
        format="%d",
    )

    st.markdown("---")
    st.markdown("**Risk thresholds**")
    st.markdown(
        "<div style='font-size:0.84rem;line-height:1.8;'>"
        "<span style='color:#3fb950'>●</span> <strong style='color:#e6edf3'>Low</strong> — under 35%<br>"
        "<span style='color:#d29922'>●</span> <strong style='color:#e6edf3'>Moderate</strong> — 35–70%<br>"
        "<span style='color:#f85149'>●</span> <strong style='color:#e6edf3'>High</strong> — 70%+"
        "</div>",
        unsafe_allow_html=True,
    )


# If a mission was selected, override the slider values
if st.session_state.active_mission_idx is not None:
    m = MISSIONS[st.session_state.active_mission_idx]
    crack_size = m["crack"]
    stress = m["stress"]
    cycles = m["cycles"]


# ---------- Prediction -------------------------------------------------------

input_row = pd.DataFrame([{
    "crack_length_mm": crack_size,
    "stress_intensity": stress,
    "load_cycles": cycles,
}])
failure_probability = float(model.predict_proba(input_row)[0, 1])
tier = risk_tier(failure_probability)
r_label = risk_label(failure_probability)
r_class = risk_class(failure_probability)
r_emoji = risk_emoji(failure_probability)


# ---------- Hero + XP bar ---------------------------------------------------

st.markdown(
    f"""
    <div class="hero-wrapper">
        <div class="hero-badge">🎮 Predictive Maintenance Simulator</div>
        <div class="hero-title"><span>FailureQuest</span> — Structural Risk Simulator</div>
        <p class="hero-sub">Diagnose components, predict failures, earn XP, and climb the leaderboard.
        Can you keep the plant running?</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stats row
sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    st.markdown(f'<div class="stat-card"><div class="stat-card-icon">⚡</div><p class="stat-card-val">{st.session_state.xp}</p><div class="stat-card-lbl">Total XP</div></div>', unsafe_allow_html=True)
with sc2:
    st.markdown(f'<div class="stat-card"><div class="stat-card-icon">🏅</div><p class="stat-card-val">Lv.{st.session_state.level}</p><div class="stat-card-lbl">{LEVEL_TITLES[min(st.session_state.level, len(LEVEL_TITLES)-1)]}</div></div>', unsafe_allow_html=True)
with sc3:
    st.markdown(f'<div class="stat-card"><div class="stat-card-icon">🎯</div><p class="stat-card-val">{st.session_state.total_predictions}</p><div class="stat-card-lbl">Predictions Made</div></div>', unsafe_allow_html=True)
with sc4:
    st.markdown(f'<div class="stat-card"><div class="stat-card-icon">🏆</div><p class="stat-card-val">{len(st.session_state.missions_done)}/{len(MISSIONS)}</p><div class="stat-card-lbl">Missions Done</div></div>', unsafe_allow_html=True)

st.write("")

# ---------- Tabs -------------------------------------------------------------

tab_pred, tab_challenge, tab_model, tab_about = st.tabs([
    "🎯 Prediction", "🚀 Challenges", "📊 Model Stats", "📖 About"
])


# ===== PREDICTION TAB ========================================================

with tab_pred:

    # Active mission banner
    if st.session_state.active_mission_idx is not None:
        m = MISSIONS[st.session_state.active_mission_idx]
        done_tag = " — <span style='color:#3fb950'>COMPLETED ✓</span>" if st.session_state.active_mission_idx in st.session_state.missions_done else ""
        st.markdown(
            f"""<div style="background:rgba(88,166,255,0.08);border:1px solid rgba(88,166,255,0.3);
            border-radius:10px;padding:0.8rem 1rem;margin-bottom:1rem;font-size:0.92rem;">
            <strong style="color:var(--blue)">{m['icon']} Active Mission: {m['name']}</strong>{done_tag}<br>
            <span style="color:var(--text-soft)">{m['desc']}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Risk Gauge**")
            st.caption("Live failure probability for current inputs.")
            g_left, g_right = st.columns([1.4, 1])
            with g_left:
                st.pyplot(render_gauge(failure_probability), use_container_width=True)
            with g_right:
                st.markdown(
                    f"""
                    <div class="gauge-readout">
                        <p class="gauge-pct {gauge_pct_class(failure_probability)}">{failure_probability * 100:.1f}%</p>
                        <div style="color:var(--text-soft);font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-top:0.4rem;">Failure Probability</div>
                        <div class="risk-badge {r_class}" style="margin-top:0.75rem;">
                            {r_emoji} {r_label}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with right:
        with st.container(border=True):
            st.markdown("**Component Snapshot**")
            st.caption("Your current inspection inputs.")

            st.markdown(
                f"""
                <div style="display:grid;gap:0.5rem;margin-top:0.5rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;
                         background:#21262d;border-radius:8px;padding:0.6rem 0.85rem;">
                        <span style="color:#8b949e;font-size:0.88rem;">🔩 Crack size</span>
                        <strong style="color:#58a6ff">{crack_size:.1f} mm</strong>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;
                         background:#21262d;border-radius:8px;padding:0.6rem 0.85rem;">
                        <span style="color:#8b949e;font-size:0.88rem;">⚡ Stress intensity</span>
                        <strong style="color:#58a6ff">{stress:.1f}</strong>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;
                         background:#21262d;border-radius:8px;padding:0.6rem 0.85rem;">
                        <span style="color:#8b949e;font-size:0.88rem;">🔄 Load cycles</span>
                        <strong style="color:#58a6ff">{format_cycles(int(cycles))}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Recommended Action
        if failure_probability >= 0.70:
            st.error(
                "🚨 **IMMEDIATE ACTION REQUIRED** — Remove from service NOW. Perform non-destructive evaluation before any further loading.",
                icon=None,
            )
        elif failure_probability >= 0.35:
            st.warning(
                "⚠️ **Schedule inspection soon.** Consider reducing duty cycle and monitor crack growth rate closely.",
                icon=None,
            )
        else:
            st.success(
                "✅ **Safe to operate.** Component is within normal bounds. Continue regular inspection schedule.",
                icon=None,
            )

    # Predict + score button
    st.write("")
    b_col, _ = st.columns([1, 2])
    with b_col:
        if st.button("⚡ Submit Prediction & Earn XP", use_container_width=True, type="primary"):
            st.session_state.total_predictions += 1

            xp_earned = 10

            if st.session_state.active_mission_idx is not None:
                m = MISSIONS[st.session_state.active_mission_idx]
                target = m["target"]
                if target == tier:
                    xp_earned = 30
                    st.session_state.last_feedback = ("correct", f"🎉 Perfect call! +{xp_earned} XP — You nailed it!")
                    st.session_state.missions_done.add(st.session_state.active_mission_idx)
                    if tier == "low": st.session_state.correct_low += 1
                    elif tier == "high": st.session_state.correct_high += 1
                    elif tier == "moderate": st.session_state.correct_mod += 1
                elif (target == "moderate" and tier in ("low", "high")) or \
                     (target == "low" and tier == "moderate") or \
                     (target == "high" and tier == "moderate"):
                    xp_earned = 15
                    st.session_state.last_feedback = ("almost", f"😅 Close! Expected {target.upper()} — +{xp_earned} XP for effort")
                else:
                    xp_earned = 5
                    st.session_state.last_feedback = ("wrong", f"❌ Missed it. Expected {target.upper()} risk — +{xp_earned} XP anyway. Try again!")
            else:
                if tier == "low": st.session_state.correct_low += 1
                elif tier == "high": st.session_state.correct_high += 1
                elif tier == "moderate": st.session_state.correct_mod += 1
                st.session_state.last_feedback = ("correct", f"✅ Prediction recorded! +{xp_earned} XP")

            award_xp(xp_earned)
            new_ach = check_achievements()
            if new_ach:
                names = [a["name"] for a in ACHIEVEMENTS if a["id"] in new_ach]
                st.toast(f"🏆 Achievement unlocked: {', '.join(names)}!", icon="🎉")

            st.rerun()

    # Feedback pop
    if st.session_state.last_feedback:
        kind, msg = st.session_state.last_feedback
        cls = "feedback-correct" if kind == "correct" else ("feedback-wrong" if kind == "wrong" else "feedback-almost")
        st.markdown(f'<div class="feedback-pop {cls}">{msg}</div>', unsafe_allow_html=True)

    # Achievements row
    st.write("")
    st.markdown("**Achievements**")
    ach_html = '<div class="ach-row">'
    for a in ACHIEVEMENTS:
        if a["id"] in st.session_state.achievements:
            ach_html += f'<div class="ach-badge ach-unlocked" title="{a["desc"]}">{a["icon"]} {a["name"]}</div>'
        else:
            ach_html += f'<div class="ach-badge ach-locked" title="{a["desc"]}">🔒 {a["name"]}</div>'
    ach_html += '</div>'
    st.markdown(ach_html, unsafe_allow_html=True)


# ===== CHALLENGES TAB ========================================================

with tab_challenge:
    st.markdown("### 🚀 Mission Briefings")
    st.caption("Select a mission from the sidebar to load its scenario, then hit **Submit Prediction** to score XP.")

    for idx, m in enumerate(MISSIONS):
        done = idx in st.session_state.missions_done
        done_cls = " done" if done else ""
        done_icon = "✅" if done else "🔒"

        tier_colors = {"low": "#3fb950", "moderate": "#d29922", "high": "#f85149"}
        tier_labels = {"low": "Low Risk", "moderate": "Moderate Risk", "high": "High Risk"}
        tc = tier_colors[m["target"]]

        with st.container():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(
                    f"""
                    <div class="mission-card{done_cls}">
                        <div class="mission-title">{done_icon} {m['icon']} {m['name']}</div>
                        <div class="mission-desc">{m['desc']}</div>
                        <div style="margin-top:0.5rem;font-size:0.75rem;color:{tc};font-weight:700;">
                            Expected result: {tier_labels[m['target']]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_b:
                reward = 30
                st.markdown(
                    f"""
                    <div style="text-align:center;padding:1rem;background:var(--surface2);
                    border-radius:12px;border:1px solid var(--border);">
                        <div style="font-size:1.4rem">⚡</div>
                        <div style="font-weight:700;color:#58a6ff;font-size:1.1rem">+{reward} XP</div>
                        <div style="font-size:0.75rem;color:var(--text-soft)">Correct call</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if st.button(f"Load {m['name']}", key=f"load_m_{idx}", use_container_width=True):
                st.session_state.active_mission_idx = idx
                st.rerun()


# ===== MODEL STATS TAB =======================================================

with tab_model:
    st.markdown("### 📊 Model Performance")
    st.caption(f"Random Forest · 200 trees · {total_rows:,} training rows · tested on {test_rows:,} samples")

    flat = [v for row in matrix for v in row]
    tn, fp, fn, tp = flat if len(flat) == 4 else (0, 0, 0, 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, icon in [
        (m1, "Accuracy",  accuracy,  "🎯"),
        (m2, "Precision", precision, "🔬"),
        (m3, "Recall",    recall,    "📡"),
        (m4, "F1 Score",  f1,        "⭐"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="stat-card-icon">{icon}</div>'
                f'<p class="stat-card-val">{val*100:.1f}%</p>'
                f'<div class="stat-card-lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    cm_col, fi_col = st.columns([1, 1], gap="large")
    with cm_col:
        st.markdown("**Confusion Matrix**")
        st.pyplot(render_confusion_matrix(matrix), use_container_width=True)
    with fi_col:
        st.markdown("**Feature Importance**")
        st.pyplot(render_feature_importance(model), use_container_width=True)

    st.write("")
    with st.expander("View raw confusion matrix table"):
        confusion_df = pd.DataFrame(
            matrix,
            index=["Actual safe", "Actual failed"],
            columns=["Predicted safe", "Predicted failed"],
        )
        st.dataframe(confusion_df, use_container_width=True)


# ===== ABOUT TAB =============================================================

with tab_about:
    st.markdown("### 📖 About FailureQuest")
    st.markdown(
        """
        **FailureQuest** is a gamified predictive maintenance simulator that teaches you how
        structural components fail under fatigue loading.

        You adjust three real engineering parameters:

        - 🔩 **Crack length (mm)** — measured surface crack size
        - ⚡ **Stress intensity** — stress intensity factor in MPa·√m
        - 🔄 **Load cycles** — cumulative cycles applied to the component

        The AI model uses **Paris' Law** physics to generate training data:
        """
    )
    st.latex(r"\frac{da}{dN} = C \cdot (\Delta K)^{m}")
    st.markdown(
        """
        A `RandomForestClassifier` (200 trees, balanced class weights) is trained on that synthetic data
        and gives you a real-time failure probability.

        **XP System:**
        | Action | XP |
        | --- | --- |
        | Submit any prediction | +10 XP |
        | Correct mission call (exact tier) | +30 XP |
        | Close call (adjacent tier) | +15 XP |
        | Wrong call | +5 XP |

        Complete all 6 missions and unlock **Mission Complete** 🏆
        """
    )
    st.info("⚠️ This uses **synthetic training data**. Do not use for real engineering decisions without validated datasets.", icon=None)
