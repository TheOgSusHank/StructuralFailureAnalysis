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

# ── Advanced UI Engine (Three.js + GSAP + Custom CSS) ────────────────────────
CUSTOM_UI_ENGINE = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;900&display=swap');

:root {{
    --primary: #0f172a;
    --accent: #3b82f6;
    --text: #f8fafc;
    --glass: rgba(15, 23, 42, 0.8);
}}

.stApp {{
    background: #020617;
    font-family: 'Inter', sans-serif;
    color: var(--text);
}}

/* ── 3D Scroll Canvas ── */
#canvas-container {{
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0;
    pointer-events: none;
}}

.scroll-section {{
    position: relative;
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1;
    pointer-events: none;
}}

.glass-card {{
    background: var(--glass);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 2.5rem;
    max-width: 800px;
    width: 90%;
    pointer-events: auto;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    animation: slideUp 1s cubic-bezier(0.23, 1, 0.32, 1);
}}

@keyframes slideUp {{
    from {{ opacity: 0; transform: translateY(40px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.hero-title {{
    font-size: 5rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 1rem;
    background: linear-gradient(to bottom right, #fff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-top: 2rem;
}}

.kpi-item {{
    background: rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}}

.kpi-label {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; }}
.kpi-value {{ font-size: 2.5rem; font-weight: 700; color: #fff; }}

/* ── Custom Streamlit Overrides ── */
[data-testid="stHeader"] {{ background: transparent !important; }}
.stTabs [data-baseweb="tab-list"] {{ background: transparent !important; gap: 2rem; }}
.stTabs [data-baseweb="tab"] {{ color: #94a3b8 !important; font-weight: 600 !important; font-size: 1.1rem !important; }}
.stTabs [aria-selected="true"] {{ color: #fff !important; border-bottom-color: var(--accent) !important; }}
.stButton > button {{
    background: var(--accent) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 700 !important;
    border: none !important;
    transition: all 0.3s ease !important;
}}
.stButton > button:hover {{ transform: scale(1.05); box-shadow: 0 0 20px rgba(59, 130, 246, 0.5); }}

</style>

<div id="canvas-container"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/ScrollTrigger.min.js"></script>

<script>
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const loader = new THREE.TextureLoader();
const tex1 = loader.load('data:image/webp;base64,{img1_b64}');
const tex2 = loader.load('data:image/png;base64,{img2_b64}');

const geometry = new THREE.PlaneGeometry(16, 9, 32, 32);
const material = new THREE.ShaderMaterial({{
    uniforms: {{
        uTime: {{ value: 0 }},
        uProgress: {{ value: 0 }},
        uTex1: {{ value: tex1 }},
        uTex2: {{ value: tex2 }},
        uResolution: {{ value: new THREE.Vector2(window.innerWidth, window.innerHeight) }}
    }},
    vertexShader: `
        varying vec2 vUv;
        uniform float uProgress;
        void main() {{
            vUv = uv;
            vec3 pos = position;
            float dist = distance(uv, vec2(0.5));
            pos.z += sin(dist * 10.0 - uProgress * 10.0) * uProgress * 2.0;
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
            float noise = fract(sin(dot(vUv, vec2(12.9898, 78.233))) * 43758.5453);
            float p = smoothstep(uProgress - 0.1, uProgress + 0.1, vUv.y + noise * 0.2);
            gl_FragColor = mix(t1, t2, 1.0 - p);
        }}
    `,
    transparent: true
}});

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);
camera.position.z = 8;

gsap.registerPlugin(ScrollTrigger);

gsap.to(material.uniforms.uProgress, {{
    value: 1,
    scrollTrigger: {{
        trigger: "body",
        start: "top top",
        end: "bottom bottom",
        scrub: 1
    }}
}});

function animate() {{
    requestAnimationFrame(animate);
    material.uniforms.uTime.value += 0.01;
    renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
"""

st.markdown(CUSTOM_UI_ENGINE, unsafe_allow_html=True)

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
    accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, float(accuracy), int(len(data))

model, accuracy, total_rows = train_model(DATASET_PATH)

# ---------- Scroll Sections --------------------------------------------------

# Section 1: Hero
st.markdown('<div class="scroll-section">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="glass-card" style="text-align: center;">
        <h1 class="hero-title">INTEGRITY</h1>
        <p style="font-size: 1.5rem; color: #94a3b8; font-weight: 300;">
            Next-Generation Structural Failure Prediction.
        </p>
        <p style="margin-top: 2rem; color: #64748b; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.3em;">
            Scroll to Analyze Collapse
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# Section 2: Analysis Interface
st.markdown('<div class="scroll-section">', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## ⚙️ Diagnostic Engine")
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        crack_size = st.number_input("Crack size (mm)", 0.0, 250.0, float(st.session_state.prefill_crack))
        stress = st.number_input("Stress intensity", 0.0, 250.0, float(st.session_state.prefill_stress))
        cycles = st.number_input("Load cycles", 0, 10_000_000, int(st.session_state.prefill_cycles))
        if st.button("🚀 EXECUTE ANALYSIS", use_container_width=True):
            input_row = pd.DataFrame([{"crack_length_mm": crack_size, "stress_intensity": stress, "load_cycles": cycles}])
            prob = float(model.predict_proba(input_row)[0, 1])
            st.session_state.result = {"probability": prob}
    
    with col2:
        res = st.session_state.result
        if res:
            prob = res["probability"]
            st.markdown(
                f"""
                <div class="kpi-item" style="text-align: center; border-color: {'#ef4444' if prob > 0.7 else '#3b82f6'};">
                    <div class="kpi-label">Failure Probability</div>
                    <div class="kpi-value" style="color: {'#ef4444' if prob > 0.7 else '#3b82f6'};">{prob*100:.1f}%</div>
                    <div style="margin-top: 1rem; font-weight: 700; color: #fff;">
                        STATUS: {'CRITICAL' if prob > 0.7 else 'NOMINAL'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Adjust parameters and execute to see live risk probability.")
            
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Section 3: Computer Vision
st.markdown('<div class="scroll-section">', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
            <div class="photo-findings" style="margin-top: 1.5rem; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2);">
                <h4 style="color: #60a5fa; margin-top: 0;">AI Detection Report</h4>
                <p><b>Severity:</b> {ir.get('severity').upper()}</p>
                <p>{ir.get('findings')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Section 4: Performance & Model
st.markdown('<div class="scroll-section">', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 🧬 Core Intelligence")
    
    k_grid1, k_grid2 = st.columns(2)
    with k_grid1:
        st.markdown(f'<div class="kpi-item"><div class="kpi-label">Model Accuracy</div><div class="kpi-value">{accuracy*100:.1f}%</div></div>', unsafe_allow_html=True)
    with k_grid2:
        st.markdown(f'<div class="kpi-item"><div class="kpi-label">Training Data</div><div class="kpi-value">{total_rows:,}</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Methodology")
    st.write("Our neural network utilizes a physics-informed Random Forest architecture, trained on Paris' Law growth simulations and real-world failure datasets.")
    
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Footer Spacer
st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)
