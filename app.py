import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from io import StringIO

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="SmartForm AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS (UI Enhancement)
# -----------------------------
st.markdown("""
<style>
/* Page background + typography */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { font-weight: 700; }
small { opacity: 0.8; }

/* Card containers */
.card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 18px 18px 14px 18px;
  box-shadow: 0px 10px 30px rgba(0,0,0,0.25);
}
.card-title {
  font-size: 0.95rem;
  opacity: 0.85;
  margin-bottom: 0.2rem;
}
.card-value {
  font-size: 2.0rem;
  font-weight: 800;
  margin-top: 0.2rem;
}
.card-sub {
  font-size: 0.9rem;
  opacity: 0.75;
  margin-top: 0.25rem;
}

/* Top header */
.header {
  background: linear-gradient(90deg, rgba(56,189,248,0.18), rgba(99,102,241,0.18));
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 18px 20px;
  margin-bottom: 14px;
}
.header h1 { margin: 0; font-size: 2.0rem; }
.header p { margin: 0.25rem 0 0 0; opacity: 0.85; }

/* Section label */
.section {
  margin-top: 10px;
  margin-bottom: 6px;
  font-size: 1.15rem;
  font-weight: 750;
}

/* Sidebar titles */
[data-testid="stSidebar"] h2 { font-size: 1.2rem; }

/* Hide Streamlit default footer/menu */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="header">
  <h1>🏗 SmartForm AI</h1>
  <p>AI-Driven Formwork Kitting & BoQ Optimization • Repetition Intelligence + Reuse Planning Dashboard</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Project Inputs")

kit_cost = st.sidebar.number_input(
    "Cost per Formwork Kit (₹)",
    min_value=10000, max_value=500000,
    value=50000, step=5000
)

carrying_cost_pct = st.sidebar.slider(
    "Inventory Carrying Cost (% of kit cost)",
    0, 30, 12
)

parallel_workfronts = st.sidebar.number_input(
    "Parallel Workfronts (kits used simultaneously)",
    min_value=1, max_value=10, value=2
)

n_clusters = st.sidebar.slider(
    "AI Clusters (auto-detect repetition groups)",
    2, 6, 4
)

st.sidebar.markdown("---")
use_sample = st.sidebar.toggle("Use sample dataset (30-floor tower)", value=True)

# -----------------------------
# Data Loaders
# -----------------------------
@st.cache_data
def load_sample_data():
    sample_csv = """floor,slab_area,beam_length,column_count,wall_area,floor_type,cycle_days
1,2200,520,60,900,Podium,4
2,2150,510,58,880,Podium,4
3,2180,515,59,890,Podium,4
4,1500,400,36,600,Typical,3
5,1495,395,36,590,Typical,3
6,1510,402,37,605,Typical,3
7,1502,398,36,598,Typical,3
8,1498,397,36,600,Typical,3
9,1505,401,36,602,Typical,3
10,1499,399,36,595,Typical,3
11,1503,400,36,600,Typical,3
12,1501,398,36,598,Typical,3
13,1497,396,36,590,Typical,3
14,1504,402,36,605,Typical,3
15,1500,400,36,600,Typical,3
16,1502,401,36,602,Typical,3
17,1496,395,36,590,Typical,3
18,1503,399,36,598,Typical,3
19,1501,400,36,600,Typical,3
20,1504,402,36,605,Typical,3
21,1498,397,36,595,Typical,3
22,1500,400,36,600,Typical,3
23,1502,401,36,602,Typical,3
24,1499,398,36,598,Typical,3
25,1503,399,36,600,Typical,3
26,1400,370,30,550,Upper_Modified,3
27,1395,365,30,540,Upper_Modified,3
28,1402,372,31,552,Upper_Modified,3
29,1398,368,30,545,Upper_Modified,3
30,900,250,18,300,Terrace,2
"""
    return pd.read_csv(StringIO(sample_csv))

# -----------------------------
# Dataset upload / selection
# -----------------------------
uploaded_file = None
if not use_sample:
    uploaded_file = st.file_uploader("Upload Floor Dataset (CSV)", type=["csv"])

if use_sample:
    df = load_sample_data()
    st.info("ℹ Using sample dataset (30-floor tower). Turn off the toggle to upload your own CSV.")
else:
    if uploaded_file is None:
        st.warning("Upload a CSV to continue, or enable 'Use sample dataset' from the sidebar.")
        st.stop()
    df = pd.read_csv(uploaded_file)
    st.success("✅ Uploaded dataset loaded successfully!")

# Validate columns
required_cols = {"floor", "slab_area", "beam_length", "column_count", "wall_area", "cycle_days"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# Optional floor_type
if "floor_type" not in df.columns:
    df["floor_type"] = "Unknown"

df = df.sort_values("floor").reset_index(drop=True)

# -----------------------------
# AI: Repetition Detection (Clustering)
# -----------------------------
features = df[["slab_area", "beam_length", "column_count", "wall_area"]].copy()

# Normalize for clustering fairness
features = (features - features.mean()) / (features.std() + 1e-9)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(features)

cluster_sizes = df["cluster"].value_counts().sort_index()

# -----------------------------
# Optimization: Kit Estimation (Simple but Practical)
# -----------------------------
# Logic: For each repetition cluster, kits needed depends on how many floors of that type
# and how many can be executed in parallel (workfronts).
kits_per_cluster = np.ceil(cluster_sizes / parallel_workfronts).astype(int)

kits_after = int(kits_per_cluster.sum())
kits_before = int(len(df))  # naive / static assumption

# Costs (including carrying cost overhead)
proc_before = kits_before * kit_cost
proc_after = kits_after * kit_cost

carry_before = proc_before * (carrying_cost_pct / 100)
carry_after = proc_after * (carrying_cost_pct / 100)

total_before = proc_before + carry_before
total_after = proc_after + carry_after

savings = total_before - total_after
savings_pct = (savings / total_before) * 100 if total_before > 0 else 0

# -----------------------------
# KPI ROW (Cards)
# -----------------------------
st.markdown('<div class="section">📌 Executive Summary</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

def kpi_card(title, value, sub=""):
    st.markdown(f"""
    <div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value">{value}</div>
      <div class="card-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

with k1:
    kpi_card("Kits Before (Manual)", f"{kits_before}", "Static BoQ assumption")
with k2:
    kpi_card("Kits After (Optimized)", f"{kits_after}", "AI repetition + reuse planning")
with k3:
    kpi_card("Total Cost Before", f"₹{total_before:,.0f}", f"Including {carrying_cost_pct}% carrying cost")
with k4:
    kpi_card("Estimated Savings", f"{savings_pct:.2f}%", f"≈ ₹{savings:,.0f}")

st.markdown("---")

# -----------------------------
# Main Content (2 columns)
# -----------------------------
left, right = st.columns([1.35, 1])

with left:
    st.markdown('<div class="section">📄 Floor Dataset Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=360)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section">💰 Cost Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    cost_df = pd.DataFrame({
        "Scenario": ["Before", "After"],
        "Total Cost (₹)": [total_before, total_after],
        "Procurement (₹)": [proc_before, proc_after],
        "Carrying (₹)": [carry_before, carry_after]
    }).set_index("Scenario")

    st.bar_chart(cost_df[["Total Cost (₹)"]])
    st.caption("Bar chart compares total estimated cost before vs after optimization.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Repetition Intelligence Section
# -----------------------------
st.markdown('<div class="section">🧠 Repetition Intelligence (AI Clusters)</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

cluster_table = pd.DataFrame({
    "cluster": cluster_sizes.index,
    "floors_in_cluster": cluster_sizes.values,
    "recommended_kits": kits_per_cluster.values
})

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Cluster Summary (Detected repetition groups)")
    st.dataframe(cluster_table, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Floors per Cluster (Repetition level)")
    st.bar_chart(cluster_sizes)
    st.caption("Higher bars mean more repetition → better reuse potential.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Strategic Insight
# -----------------------------
st.markdown("---")
st.markdown('<div class="section">📌 Strategic Insight</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

if savings > 0:
    st.success(
        f"SmartForm AI reduced required kits from {kits_before} to {kits_after}, "
        f"unlocking ~₹{savings:,.0f} savings by minimizing excess inventory and carrying cost."
    )
else:
    st.warning("Limited savings under current constraints. Try changing clusters/workfronts or upload a different dataset.")

st.caption("MVP note: In production, floor features can be extracted from BIM/BoQ and connected to Primavera/MS Project schedules.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Export Section
# -----------------------------
st.markdown('<div class="section">⬇ Export Outputs</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

export_df = df[["floor", "floor_type", "cluster", "cycle_days", "slab_area", "beam_length", "column_count", "wall_area"]].copy()
csv_bytes = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Clustered Floor Plan (CSV)",
    data=csv_bytes,
    file_name="smartform_clustered_plan.csv",
    mime="text/csv"
)

st.markdown('</div>', unsafe_allow_html=True)