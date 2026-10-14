import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Predictive Machine Maintenance using Markov Process
# Uses the project's existing pickle file; the original Markov transition
# probabilities are retained exactly as defined in the notebook.

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- Styling ---------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#eaf1ff; --muted:#9caecb; --panel:rgba(18,31,58,.78); --line:rgba(148,174,220,.17); }
.stApp { background: radial-gradient(ellipse at 15% 0%, #17365c 0%, #0a1428 43%, #070d1b 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0e1a31,#091120); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#eaf1ff; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.4px; }
.hero { padding:27px 30px; border:1px solid rgba(103,171,255,.28); border-radius:24px; background:linear-gradient(120deg,rgba(26,64,106,.85),rgba(19,31,59,.76)); box-shadow:0 18px 60px rgba(0,0,0,.2); margin-bottom:22px; }
.eyebrow { color:#78d9ff; text-transform:uppercase; font-size:11px; font-weight:700; letter-spacing:2.4px; }
.hero h1 { font-size:clamp(27px,3.5vw,43px); margin:8px 0 6px; color:#fff; }
.hero p { color:#b5c8e7; margin:0; font-size:15px; }
.pill { display:inline-block; margin-top:17px; margin-right:8px; padding:6px 11px; border:1px solid rgba(134,190,255,.3); border-radius:99px; color:#d6e9ff; font-size:11px; background:rgba(9,20,40,.45); }
.metric { background:var(--panel); border:1px solid var(--line); border-radius:17px; padding:18px 19px; min-height:113px; }
.metric-label { color:#9caecb; font-size:12px; letter-spacing:.5px; }
.metric-value { color:#f3f7ff; font-family:'Space Grotesk',sans-serif; font-size:26px; font-weight:700; margin-top:6px; }
.metric-note { color:#80d9c5; font-size:11px; margin-top:3px; }
.section { color:#edf4ff; font-family:'Space Grotesk',sans-serif; font-size:20px; font-weight:700; margin:17px 0 10px; }
.info-card { border:1px solid var(--line); background:var(--panel); border-radius:18px; padding:19px; height:100%; }
.state-operational { color:#6ee7b7; }
.state-minor { color:#facc70; }
.state-major { color:#ffad66; }
.state-failure { color:#ff7e91; }
.small-muted { color:#9caecb; font-size:12px; }
div.stButton > button { border-radius:12px; border:1px solid #5eb8ed; background:linear-gradient(100deg,#2e8bc8,#5a6de0); color:white; font-weight:700; padding:.65rem 1rem; }
div.stButton > button:hover { border-color:#a5e8ff; color:white; }
.stSelectbox label,.stSlider label,.stNumberInput label { color:#c8d8f1 !important; }
hr { border-color:var(--line); }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

STATES_FALLBACK = ["Operational", "Minor Wear", "Major Fault", "Failure"]
P_FALLBACK = np.array([
    [0.7, 0.2, 0.1, 0.0],
    [0.2, 0.5, 0.2, 0.1],
    [0.0, 0.3, 0.4, 0.3],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

@st.cache_resource
def load_model():
    model_path = Path(__file__).with_name("machine_maintenance_markov_model.pkl")
    if model_path.exists():
        try:
            with model_path.open("rb") as f:
                data = pickle.load(f)
            states = list(data["states"])
            matrix = np.asarray(data["transition_matrix"], dtype=float)
            if matrix.shape == (len(states), len(states)) and np.allclose(matrix.sum(axis=1), 1):
                return states, matrix, True
        except Exception:
            pass
    return STATES_FALLBACK, P_FALLBACK.copy(), False

states, P, loaded_from_pickle = load_model()

# State metadata follows the states already present in the user's model.
COLORS = {"Operational":"#42d6a4", "Minor Wear":"#f6c85f", "Major Fault":"#ff9f5a", "Failure":"#ff647c"}
ICONS = {"Operational":"●", "Minor Wear":"◐", "Major Fault":"▲", "Failure":"✕"}
ACTIONS = {
    "Operational":"Continue normal operation and keep routine inspections scheduled.",
    "Minor Wear":"Schedule a preventive inspection and monitor wear indicators.",
    "Major Fault":"Arrange maintenance soon and assess the machine before heavy use.",
    "Failure":"Stop operation and arrange immediate technical inspection.",
}

st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Industrial intelligence · Markov model</div>
  <h1>Predictive Machine Maintenance</h1>
  <p>Understand machine-state transitions, simulate future conditions, and plan maintenance before downtime.</p>
  <span class="pill">⚙️ Markov Process</span><span class="pill">📊 Transition Analytics</span><span class="pill">🧠 Pickle Model {'Loaded' if loaded_from_pickle else 'Fallback Active'}</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Control panel")
    st.caption("Configure a machine scenario")
    current_state = st.selectbox("Current machine condition", states, index=0)
    horizon = st.slider("Forecast horizon (steps)", min_value=1, max_value=30, value=10)
    runs = st.slider("Simulation runs", min_value=100, max_value=5000, value=1000, step=100)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    run_sim = st.button("↻ Run forecast", use_container_width=True)
    st.divider()
    st.markdown("**Model details**")
    st.caption(f"States: {len(states)}")
    st.caption("Rows of the transition matrix represent current state; columns represent next state.")
    if loaded_from_pickle:
        st.success("Pickle file loaded successfully")
    else:
        st.warning("Pickle file not found or invalid. Built-in project matrix is being used.")

# Stable results unless user asks to rerun; controls also update the result naturally.
if "forecast_nonce" not in st.session_state:
    st.session_state.forecast_nonce = 0
if run_sim:
    st.session_state.forecast_nonce += 1

idx = states.index(current_state)
rng = np.random.default_rng(int(seed) + st.session_state.forecast_nonce)

# Probability distribution at the selected future step, calculated from the Markov chain.
distribution = np.zeros(len(states))
distribution[idx] = 1.0
prob_rows = []
for step in range(horizon + 1):
    row = {"Step": step}
    row.update({s: float(distribution[j]) for j, s in enumerate(states)})
    prob_rows.append(row)
    distribution = distribution @ P
forecast_df = pd.DataFrame(prob_rows)

# Simulate individual paths to show a plausible example path and terminal-state mix.
paths = []
terminal_counts = np.zeros(len(states), dtype=int)
example_path = [idx]
for sim in range(int(runs)):
    state_i = idx
    this_path = [state_i]
    for _ in range(horizon):
        state_i = int(rng.choice(len(states), p=P[state_i]))
        this_path.append(state_i)
    terminal_counts[state_i] += 1
    if sim == 0:
        example_path = this_path
terminal_pct = terminal_counts / max(1, int(runs)) * 100

failure_idx = states.index("Failure") if "Failure" in states else len(states)-1
failure_prob = float(forecast_df.iloc[-1][states[failure_idx]])
maintenance_prob = sum(float(forecast_df.iloc[-1][s]) for s in states if s in ("Major Fault", "Failure"))
most_likely_idx = int(np.argmax(distribution))
# distribution currently advanced one step beyond horizon; use exact row instead.
final_probs = forecast_df.iloc[-1][states].to_numpy(dtype=float)
most_likely_idx = int(np.argmax(final_probs))

c1,c2,c3,c4 = st.columns(4)
metrics = [
    ("CURRENT CONDITION", current_state, "Selected machine state"),
    (f"FAILURE PROBABILITY · STEP {horizon}", f"{failure_prob*100:.1f}%", "Markov forecast estimate"),
    ("FAULT RISK · STEP " + str(horizon), f"{maintenance_prob*100:.1f}%", "Major Fault or Failure"),
    ("MOST LIKELY STATE", states[most_likely_idx], f"{final_probs[most_likely_idx]*100:.1f}% probability"),
]
for col,(label,value,note) in zip([c1,c2,c3,c4],metrics):
    with col:
        st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section">📈 Forecast & state probabilities</div>', unsafe_allow_html=True)
left,right = st.columns([1.65,1])
with left:
    plot_df = forecast_df.melt(id_vars="Step", var_name="Machine state", value_name="Probability")
    fig = px.line(plot_df, x="Step", y="Probability", color="Machine state", markers=True,
                  color_discrete_map=COLORS, template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,20,40,.15)",
                      margin=dict(l=10,r=10,t=20,b=10), legend_title_text="",
                      yaxis=dict(tickformat=".0%", gridcolor="rgba(160,185,220,.12)", rangemode="tozero"),
                      xaxis=dict(title="Forecast step", gridcolor="rgba(160,185,220,.08)"),
                      font=dict(family="DM Sans",color="#dbe7ff"), height=350)
    st.plotly_chart(fig, use_container_width=True)
with right:
    final_chart = pd.DataFrame({"Machine state": states, "Probability": final_probs})
    fig2 = px.pie(final_chart, names="Machine state", values="Probability", hole=.64,
                  color="Machine state", color_discrete_map=COLORS, template="plotly_dark")
    fig2.update_traces(textposition="outside", textinfo="percent", marker=dict(line=dict(color="#0a1428",width=3)))
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       margin=dict(l=8,r=8,t=25,b=8), showlegend=True,
                       legend=dict(orientation="h",yanchor="bottom",y=-.15,xanchor="center",x=.5),
                       font=dict(family="DM Sans",color="#dbe7ff"), height=350)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="section">🧭 Maintenance recommendation</div>', unsafe_allow_html=True)
rec_col, detail_col = st.columns([1,1.5])
with rec_col:
    state_color = COLORS.get(states[most_likely_idx], "#8bbcff")
    st.markdown(f'''<div class="info-card"><div class="small-muted">MOST LIKELY CONDITION AT STEP {horizon}</div><h2 style="color:{state_color};margin:10px 0">{ICONS.get(states[most_likely_idx],'●')} {states[most_likely_idx]}</h2><div style="font-size:26px;font-weight:700;color:#fff">{final_probs[most_likely_idx]*100:.1f}%</div><div class="small-muted">model-estimated probability</div></div>''', unsafe_allow_html=True)
with detail_col:
    rec_state = states[most_likely_idx]
    st.markdown(f'''<div class="info-card"><h3 style="margin-top:0">Recommended next action</h3><p style="color:#c8d8f1;line-height:1.7">{ACTIONS.get(rec_state, 'Review the machine condition and schedule an inspection based on operational requirements.')}</p><hr><div class="small-muted">This is a probabilistic planning aid, not a sensor-based diagnosis. Validate recommendations against real maintenance logs and machine telemetry.</div></div>''', unsafe_allow_html=True)

st.markdown('<div class="section">🔁 Markov transition matrix</div>', unsafe_allow_html=True)
st.caption("Each cell is the probability of moving from the row state to the column state in one step.")
heat = go.Figure(data=go.Heatmap(z=P, x=states, y=states, zmin=0, zmax=1, colorscale=[[0,"#12213d"],[0.35,"#246b91"],[0.7,"#24b7b1"],[1,"#f6c85f"]],
                                 text=np.vectorize(lambda x:f"{x:.0%}")(P), texttemplate="%{text}",
                                 hovertemplate="From %{y}<br>To %{x}<br>Probability: %{z:.1%}<extra></extra>"))
heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   margin=dict(l=10,r=10,t=8,b=10), font=dict(family="DM Sans",color="#eaf1ff"), height=300,
                   xaxis_title="Next state", yaxis_title="Current state")
st.plotly_chart(heat, use_container_width=True)

with st.expander("View forecast probability table"):
    display_df = forecast_df.copy()
    for s in states:
        display_df[s] = display_df[s].map(lambda x: f"{x:.2%}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with st.expander("View simulation summary"):
    sim_df = pd.DataFrame({"Terminal state": states, "Simulated runs": terminal_counts, "Share": [f"{x:.1f}%" for x in terminal_pct]})
    st.dataframe(sim_df, use_container_width=True, hide_index=True)
    st.caption(f"Monte Carlo simulation: {int(runs):,} paths · {horizon} steps · seed {int(seed)}. The analytical forecast above is calculated from the transition matrix; the simulation is an approximate sample.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#8297b9;font-size:12px;padding:4px'>PREDICTIVE MACHINE MAINTENANCE · MARKOV PROCESS · Built with Streamlit</div>", unsafe_allow_html=True)
