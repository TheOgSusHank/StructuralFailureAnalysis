from __future__ import annotations
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from image_analysis import analyse_crack_image

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "crack_growth_data.csv"
ASSETS_DIR = PROJECT_ROOT / "assets"
FEATURE_COLUMNS = ["crack_length_mm", "stress_intensity", "load_cycles"]

st.set_page_config(
    page_title="Structural Integrity AI | Next-Gen Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Asset Helpers ────────────────────────────────────────────────────────────
def get_base64_img(path: Path):
    if not path.exists(): return ""
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img1_b64 = get_base64_img(ASSETS_DIR / "Photo1.webp")
img2_b64 = get_base64_img(ASSETS_DIR / "Photo2.png")

# ── Session state ────────────────────────────────────────────────────────────
if "result" not in st.session_state: st.session_state.result = None
if "image_result" not in st.session_state: st.session_state.image_result = None
if "prefill_crack" not in st.session_state: st.session_state.prefill_crack = 10.0
if "prefill_stress" not in st.session_state: st.session_state.prefill_stress = 35.0
if "prefill_cycles" not in st.session_state: st.session_state.prefill_cycles = 250_000

# ── Advanced 3D UI Engine ────────────────────────────────────────────────────
ADVANCED_3D_UI = f"""
<!DOCTYPE html>
<html>
<head>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ width: 100%; height: 100%; }}

#canvas-container {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
    background: #020617;
}}

#content {{
    position: relative;
    z-index: 10;
    background: transparent;
}}

.scroll-spacer {{
    height: 300vh;
    width: 100%;
}}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;900&display=swap');

:root {{
    --primary: #0f172a;
    --accent: #3b82f6;
    --text: #f8fafc;
    --glass: rgba(15, 23, 42, 0.85);
}}

.glass-card {{
    background: var(--glass);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 2.5rem;
    max-width: 900px;
    margin: 0 auto;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    font-family: 'Inter', sans-serif;
    color: var(--text);
}}

.scroll-section {{
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
}}

.hero-title {{
    font-size: 5rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 1rem;
    background: linear-gradient(to bottom right, #fff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.kpi-item {{
    background: rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    text-align: center;
}}

.kpi-label {{ 
    font-size: 0.75rem; 
    color: #94a3b8; 
    text-transform: uppercase; 
    letter-spacing: 0.1em; 
    font-weight: 600;
}}

.kpi-value {{ 
    font-size: 2.5rem; 
    font-weight: 700; 
    color: #fff;
    margin-top: 0.5rem;
}}

h2 {{ font-size: 2rem; margin-bottom: 1.5rem; color: #fff; }}
h3 {{ font-size: 1.5rem; margin: 1.5rem 0 1rem; color: #fff; }}
p {{ color: #cbd5e1; line-height: 1.6; margin-bottom: 1rem; }}

.grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-bottom: 2rem;
}}

@media (max-width: 768px) {{
    .grid-2 {{ grid-template-columns: 1fr; }}
    .hero-title {{ font-size: 3rem; }}
}}

button {{
    background: var(--accent);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 12px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.3s ease;
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    width: 100%;
}}

button:hover {{
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
}}

input, textarea {{
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: white;
    padding: 0.75rem;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    margin-bottom: 1rem;
}}

input::placeholder, textarea::placeholder {{
    color: #64748b;
}}

.status-critical {{
    color: #ef4444;
}}

.status-nominal {{
    color: #10b981;
}}
</style>
</head>
<body>
<div id="canvas-container"></div>
<div id="content">
    <div class="scroll-spacer"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/ScrollTrigger.min.js"></script>

<script>
gsap.registerPlugin(ScrollTrigger);

const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x020617, 1);
container.appendChild(renderer.domElement);

const loader = new THREE.TextureLoader();
const tex1 = loader.load('data:image/webp;base64,{img1_b64}');
const tex2 = loader.load('data:image/png;base64,{img2_b64}');

tex1.magFilter = THREE.LinearFilter;
tex2.magFilter = THREE.LinearFilter;

const geometry = new THREE.PlaneGeometry(16, 9, 64, 64);

const material = new THREE.ShaderMaterial({{
    uniforms: {{
        uProgress: {{ value: 0 }},
        uTex1: {{ value: tex1 }},
        uTex2: {{ value: tex2 }}
    }},
    vertexShader: `
        varying vec2 vUv;
        uniform float uProgress;
        
        void main() {{
            vUv = uv;
            vec3 pos = position;
            
            // Wave distortion based on progress
            float wave = sin(vUv.x * 10.0 - uProgress * 5.0) * 0.3;
            float dist = distance(vUv, vec2(0.5));
            pos.z += wave * uProgress;
            pos.z += sin(dist * 15.0 - uProgress * 8.0) * uProgress * 1.5;
            
            gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }}
    `,
    fragmentShader: `
        varying vec2 vUv;
        uniform sampler2D uTex1;
        uniform sampler2D uTex2;
        uniform float uProgress;
        
        void main() {{
            vec4 t1 = texture2D(uTex1, vUv);
            vec4 t2 = texture2D(uTex2, vUv);
            
            // Smooth transition with noise
            float noise = fract(sin(dot(vUv, vec2(12.9898, 78.233))) * 43758.5453);
            float threshold = uProgress + (noise - 0.5) * 0.3;
            
            float mixFactor = smoothstep(threshold - 0.1, threshold + 0.1, vUv.y);
            
            gl_FragColor = mix(t1, t2, mixFactor);
        }}
    `,
    transparent: true
}});

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);
camera.position.z = 8;

// Scroll animation
gsap.to(material.uniforms.uProgress, {{
    value: 1,
    scrollTrigger: {{
        trigger: "body",
        start: "top top",
        end: "bottom bottom",
        scrub: 0.5,
        onUpdate: (self) => {{
            // Optional: log progress for debugging
        }}
    }}
}});

function animate() {{
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>
"""

# Inject the 3D engine
st.components.v1.html(ADVANCED_3D_UI, height=0, scrolling=False)

# ── Custom Streamlit Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; gap: 2rem; }
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; font-weight: 600 !important; }
.stTabs [aria-selected="true"] { color: #fff !important; }
.stButton > button { background: #3b82f6 !important; color: white !important; border-radius: 12px !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Data & Model ──────────────────────────────────────────────────────
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
    accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, float(accuracy), int(len(data))

model, accuracy, total_rows = train_model(DATASET_PATH)

# ---------- Content Sections (Overlaid on 3D) ──────────────────────────────────

# Section 1: Hero
st.markdown('<div class="scroll-section"><div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
st.markdown(
    """
    <h1 class="hero-title">INTEGRITY</h1>
    <p style="font-size: 1.5rem; color: #94a3b8; font-weight: 300;">
        Next-Generation Structural Failure Prediction
    </p>
    <p style="margin-top: 2rem; color: #64748b; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.3em;">
        ↓ Scroll to witness structural collapse ↓
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown('</div></div>', unsafe_allow_html=True)

# Section 2: Analysis Interface
st.markdown('<div class="scroll-section"><div class="glass-card">', unsafe_allow_html=True)
st.markdown("## ⚙️ Diagnostic Engine")

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("### Input Parameters")
    crack_size = st.number_input("Crack size (mm)", 0.0, 250.0, float(st.session_state.prefill_crack))
    stress = st.number_input("Stress intensity", 0.0, 250.0, float(st.session_state.prefill_stress))
    cycles = st.number_input("Load cycles", 0, 10_000_000, int(st.session_state.prefill_cycles))
    if st.button("🚀 EXECUTE ANALYSIS", use_container_width=True):
        input_row = pd.DataFrame([{"crack_length_mm": crack_size, "stress_intensity": stress, "load_cycles": cycles}])
        prob = float(model.predict_proba(input_row)[0, 1])
        st.session_state.result = {"probability": prob}

with col2:
    st.markdown("### Risk Assessment")
    res = st.session_state.result
    if res:
        prob = res["probability"]
        status_class = "status-critical" if prob > 0.7 else "status-nominal"
        status_text = "CRITICAL" if prob > 0.7 else "NOMINAL"
        st.markdown(
            f"""
            <div class="kpi-item" style="border-color: {'#ef4444' if prob > 0.7 else '#10b981'};">
                <div class="kpi-label">Failure Probability</div>
                <div class="kpi-value" style="color: {'#ef4444' if prob > 0.7 else '#10b981'};">{prob*100:.1f}%</div>
                <div style="margin-top: 1rem; font-weight: 700; color: #fff;" class="{status_class}">
                    STATUS: {status_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Adjust parameters and execute to see live risk probability.")

st.markdown('</div></div>', unsafe_allow_html=True)

# Section 3: Computer Vision
st.markdown('<div class="scroll-section"><div class="glass-card">', unsafe_allow_html=True)
st.markdown("## 📷 Neural Inspection")

up_file = st.file_uploader("Upload inspection scan", type=["jpg", "png", "jpeg"])
if st.button("🔍 SCAN FOR ANOMALIES", disabled=(up_file is None), use_container_width=True):
    with st.spinner("AI analyzing structural patterns..."):
        try:
            ir = analyse_crack_image(up_file.read(), mime_type=up_file.type)
            st.session_state.image_result = ir
        except Exception as e:
            st.error(f"Analysis Error: {e}")

ir = st.session_state.image_result
if ir:
    st.markdown(
        f"""
        <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); padding: 1.5rem; border-radius: 12px; margin-top: 1.5rem;">
            <h4 style="color: #60a5fa; margin-top: 0;">AI Detection Report</h4>
            <p><b>Severity:</b> {ir.get('severity', 'unknown').upper()}</p>
            <p>{ir.get('findings', 'No findings available.')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('</div></div>', unsafe_allow_html=True)

# Section 4: Performance & Model
st.markdown('<div class="scroll-section"><div class="glass-card">', unsafe_allow_html=True)
st.markdown("## 🧬 Core Intelligence")

k1, k2 = st.columns(2)
with k1:
    st.markdown(f'<div class="kpi-item"><div class="kpi-label">Model Accuracy</div><div class="kpi-value">{accuracy*100:.1f}%</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-item"><div class="kpi-label">Training Data</div><div class="kpi-value">{total_rows:,}</div></div>', unsafe_allow_html=True)

st.markdown("#### Methodology")
st.write("Our neural network utilizes a physics-informed Random Forest architecture, trained on Paris' Law growth simulations and real-world failure datasets.")

st.markdown('</div></div>', unsafe_allow_html=True)

# Footer
st.markdown("<div style='height: 20vh;'></div>", unsafe_allow_html=True)
