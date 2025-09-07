import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import re
import google.generativeai as genai

# ==== GEMINI API CONFIG (SECURE METHOD) ====
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_AVAILABLE = True
except Exception:
    st.error("Gemini API key not found. Please add it to your Streamlit secrets.", icon="🚨")
    GEMINI_AVAILABLE = False

# ==== APP CONFIG & UI THEME ====
st.set_page_config(page_title="🩺 Comprehensive Neonatal & Infant Growth Analyzer", layout="wide")

st.markdown("""
<style>
/* --- Main Background & Font --- */
body {
    background: linear-gradient(to right, #89f7fe 0%, #66a6ff 100%);
    color: #212529;
    font-family: 'Poppins', sans-serif;
}

/* --- Main Input Card & Animation --- */
.card-section {
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 36px;
    margin-bottom: 32px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card-section:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.15);
}

/* --- Input Fields Styling & Animation --- */
input[type="number"] {
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
input[type="number"]:focus {
    border-color: #66a6ff !important;
    box-shadow: 0 0 0 2px rgba(102, 166, 255, 0.25) !important;
}

/* --- UPDATED: Eye-catching Section Headers --- */
.section-header {
    text-align: center;
    font-weight: 800; /* Made even bolder */
    color: #0a40a5;
    font-size: 2.2rem; /* Increased size significantly */
    margin-bottom: 25px; /* More space below heading */
    text-shadow: 2px 2px 5px rgba(0,0,0,0.15); /* Stronger shadow */
}

/* --- Centering Utility --- */
.center-content {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
}

/* --- UPDATED: Enhanced Button Styling & Animation --- */
.stButton > button {
    background: linear-gradient(90deg,#0072ff 0%, #00c6ff 100%);
    color: white;
    font-size: 1.1rem;
    border-radius: 10px;
    padding: 12px 40px;
    font-weight: 700;
    border: none;
    box-shadow: 0 4px 14px 0 rgba(0,118,255,0.39);
    transition: all 0.3s ease;
}
.stButton > button:hover {
  background: linear-gradient(90deg,#00c6ff 0%, #0072ff 100%);
  transform: translateY(-3px) scale(1.05); /* This line is updated */
  box-shadow: 0 6px 20px 0 rgba(0,118,255,0.23);
  filter: brightness(1.2);
}

/* --- UPDATED: Tab Styling for Visibility, Spacing, and Font Size --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px; /* Increased space between tabs significantly */
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    background-color: #f0f2f6;
    transition: all 0.3s ease;
    font-weight: 600;
    color: #0d47a1 !important; /* Set visible color for inactive tabs */
    font-size: 1.05rem; /* Increased font size */
    padding: 10px 20px; /* Adjusted padding for better fit */
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #6dd5ed 0%, #2193b0 100%);
    color: white !important;
    font-size: 1.1rem; /* Slightly larger for active tab */
    font-weight: 700; /* Bolder for active tab */
}
</style>
""", unsafe_allow_html=True)

# ==== LOAD MODELS ====
@st.cache_resource
def load_models():
    try:
        return {
            'wfa': joblib.load('models/wfa_interpolators.joblib'),
            'lhfa': joblib.load('models/lhfa_interpolators.joblib'),
            'hc': joblib.load('models/hc_interpolators.joblib'),
            'bmi': joblib.load('models/bmi_interpolators.joblib'),
            'predictor': joblib.load('models/growth_predictor.joblib')
        }
    except FileNotFoundError:
        st.error("Model files not found. Ensure the 'models' directory is present.", icon="📁")
        return None

models = load_models()
if not models:
    st.stop()

# ==== HELPER FUNCTIONS ====
def get_percentile(value_x, measurement_y, gender, model_key):
    model = models[model_key][gender]
    p_cols = list(model.keys())
    p_labels = [float(re.findall(r'(\d+\.?\d*)', col)[0]) for col in p_cols]
    p_values = [func(value_x) for func in model.values()]
    sorted_pairs = sorted(zip(p_values, p_labels))
    p_values_sorted, p_labels_sorted = zip(*sorted_pairs)
    percentile = np.interp(measurement_y, p_values_sorted, p_labels_sorted)
    return round(percentile, 2)

def classify_percentile(percentile):
    if percentile < 3: return "Very low – needs urgent medical review."
    elif 3 <= percentile < 15: return "Low – below average."
    elif 15 <= percentile <= 85: return "Healthy/Normal – within the average range."
    elif 85 < percentile <= 97: return "High – above average."
    else: return "Very high – professional review may be needed."

def plot_gauge(value, title, key):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={'suffix': "th %", 'font': {'size': 24, 'color': '#0d47a1'}},
        # UPDATED: Changed title color to white for maximum visibility
        title={'text': f"<b>{title}</b>", 'font': {'size': 18, 'color': 'white'}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#0072ff"},
               'steps': [{'range': [0, 15], 'color': "#e0f7fa"}, {'range': [15, 85], 'color': "#b2ebf2"}, {'range': [85, 100], 'color': "#e0f7fa"}],
               'threshold': {'line': {'color': "#d62728", 'width': 4}, 'thickness': 0.75, 'value': value}}))
    fig.update_layout(height=230, margin=dict(t=60, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, key=key)
# def plot_gauge(value, title, key):
#     fig = go.Figure(go.Indicator(
#         # CHANGED: Mode is now just 'gauge', the number is added separately
#         mode="gauge",
#         title={'text': f"<b>{title}</b>", 'font': {'size': 18, 'color': 'white'}},
#         gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#0072ff"},
#                'steps': [{'range': [0, 15], 'color': "#e0f7fa"}, {'range': [15, 85], 'color': "#b2ebf2"}, {'range': [85, 100], 'color': "#e0f7fa"}],
#                'threshold': {'line': {'color': "#d62728", 'width': 4}, 'thickness': 0.75, 'value': value}}))

#     # NEW: Add the number as a centered annotation
#     fig.add_annotation(
#         x=0.5, y=0.4,
#         text=f"{value:.0f}th %",
#         font=dict(
#             size=30,
#             color="white"
#         ),
#         showarrow=False,
#         xanchor="center",
#         yanchor="middle"
#     )

#     fig.update_layout(height=230, margin=dict(t=60, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
#     st.plotly_chart(fig, use_container_width=True, key=key)
def calculate_bmi(weight, height_cm):
    return round(weight / ((height_cm / 100) ** 2), 2) if height_cm else 0

def calculate_target_height(father_height, mother_height, gender):
    return (father_height + mother_height + (13 if gender == "Male" else -13)) / 2

def get_gemini_analysis(context):
    if not GEMINI_AVAILABLE: return ["Gemini analysis unavailable."]
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        prompt = "Explain in 2 or 3 simple bullet points for a parent: " + context
        response = model.generate_content(prompt)
        return [line.lstrip("-•* ").strip() for line in response.text.split("\n") if line.strip()]
    except Exception as e: return [f"Gemini API error: {e}"]

# ========== INPUT UI ==========
st.title("🩺 Comprehensive Neonatal & Infant Growth Analyzer")
st.markdown("Developed by **NIST University, Artificial Intelligence Global Innovation Center (GIC)**")

with st.form("input_form"):
    st.markdown('<h1 class="section-header">Child Details</h1>', unsafe_allow_html=True)
    st.markdown('<div class="center-content">', unsafe_allow_html=True)
    gender = st.radio("Gender", ("Male", "Female"), horizontal=True, key='gender', label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age_months = st.number_input("Age (months)", 0.0, 60.0, 12.0, 0.5)
        weight = st.number_input("Weight (kg)", 1.0, 40.0, 9.6, 0.1)
        height = st.number_input("Height / Length (cm)", 40.0, 130.0, 75.0, 0.5)
        head_circumference = st.number_input("Head Circumference (cm)", 25.0, 60.0, 45.0, 0.5)
    with c2:
        muac = st.number_input("Mid-Upper Arm Circumference (cm)", 8.0, 35.0, 15.0, 0.1)
        triceps_sf = st.number_input("Triceps Skinfold (mm)", 2.0, 30.0, 10.0, 0.5)
        subscapular_sf = st.number_input("Subscapular Skinfold (mm)", 2.0, 30.0, 8.0, 0.5)
        waist_circumference = st.number_input("Waist Circumference (cm)", 30.0, 100.0, 50.0, 0.5)

    st.markdown('<hr style="height:1px;border:none;color:#ddd;background-color:#ddd;" />', unsafe_allow_html=True)
    st.markdown('<h1 class="section-header">Parent Details</h1>', unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    with pc1: father_height = st.number_input("Father's Height (cm)", 140.0, 200.0, 175.0, 0.5)
    with pc2: mother_height = st.number_input("Mother's Height (cm)", 140.0, 200.0, 165.0, 0.5)

    # st.markdown("<br>", unsafe_allow_html=True)
    # st.markdown('<div class="center-content">', unsafe_allow_html=True)
    # submit_button = st.form_submit_button("Analyze Growth")
    st.markdown("<br>", unsafe_allow_html=True)

# Create 3 columns and place the button in the middle one
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        submit_button = st.form_submit_button(
            "Analyze Growth", 
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if not submit_button:
    st.stop()

# ========== CALCULATIONS & ANALYSIS ==========
bmi = calculate_bmi(weight, height)
wfa_perc = get_percentile(age_months, weight, gender, 'wfa')
lhfa_perc = get_percentile(age_months, height, gender, 'lhfa')
hc_perc = get_percentile(age_months, head_circumference, gender, 'hc')
bmi_perc = get_percentile(age_months, bmi, gender, 'bmi')
muac_perc = max(10, min(90, (muac - 12) * 10 + 50))
target_height = calculate_target_height(father_height, mother_height, gender)

risk_score = 0
risk_factors = []
if wfa_perc < 15:
    risk_factors.append("Low weight-for-age")
    risk_score += (3 if wfa_perc < 3 else 2)
elif wfa_perc > 85:
    risk_factors.append("High weight-for-age")
    risk_score += 2
if lhfa_perc < 15:
    risk_factors.append("Risk of stunting (low height)")
    risk_score += (3 if lhfa_perc < 3 else 2)
if muac < 12.5:
    risk_factors.append("Malnutrition risk (low MUAC)")
    risk_score += (4 if muac < 11.5 else 3)
if bmi_perc < 5:
    risk_factors.append("Very low BMI")
    risk_score += 2
elif bmi_perc > 90:
    risk_factors.append("High BMI (obesity risk)")
    risk_score += 3
if waist_circumference and height and (waist_circumference/height) >= 0.5:
    risk_factors.append("Central obesity risk")
    risk_score += 3

# ========== TABS OUTPUT (FULL VERSION) ==========
tab_labels = ["📈 Growth Charts", "💪 Body Composition", "🧬 Development", "🔮 Prediction", "⚠️ Risks", "📋 Summary"]
tabs = st.tabs(tab_labels)

with tabs[0]:
    st.header("Growth Chart Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_gauge(wfa_perc, "Weight-for-Age", "wfa_gauge")
        st.info(f"**Status:** {classify_percentile(wfa_perc)}")
    with col2:
        plot_gauge(lhfa_perc, "Height-for-Age", "lhfa_gauge")
        st.info(f"**Status:** {classify_percentile(lhfa_perc)}")
    with col3:
        plot_gauge(hc_perc, "Head Circ.-for-Age", "hc_gauge")
        st.info(f"**Status:** {classify_percentile(hc_perc)}")

    st.subheader("💡 AI-Powered Explanation")
    ai_context = f"A {age_months}-month-old {gender.lower()}'s weight is at the {wfa_perc}th percentile and height is at the {lhfa_perc}th percentile."
    for p in get_gemini_analysis(ai_context):
        st.markdown(f"🔹 {p}")

with tabs[1]:
    st.header("Body Composition Analysis")
    c1, c2 = st.columns([2, 1])
    with c1:
        plot_gauge(bmi_perc, "BMI-for-Age", "bmi_gauge")
        if muac: plot_gauge(muac_perc, "MUAC (Approx. %)", "muac_gauge")
    with c2:
        st.metric("Body Mass Index (BMI)", f"{bmi:.2f}")
        st.metric("Triceps Skinfold (mm)", f"{triceps_sf:.1f}")
        st.metric("Subscapular Skinfold (mm)", f"{subscapular_sf:.1f}")
        st.metric("Waist Circumference (cm)", f"{waist_circumference:.1f}")

    st.subheader("Explanation")
    ai_context = f"The child's BMI is at the {bmi_perc}th percentile. The MUAC is {muac:.1f} cm."
    for p in get_gemini_analysis(ai_context):
        st.markdown(f"🔹 {p}")

with tabs[2]:
    st.header("Development Insights")
    st.metric("Predicted Adult Height (Mid-Parental)", f"{target_height:.1f} cm")
    st.metric("Current Height Percentile", f"{lhfa_perc}th")
    st.subheader("Explanation")
    ai_context = f"A child's predicted adult height based on parental height is {target_height:.1f} cm. Their current height is at the {lhfa_perc}th percentile."
    for p in get_gemini_analysis(ai_context):
        st.markdown(f"🔹 {p}")

with tabs[3]:
    st.header("Growth Predictions")
    input_arr = np.array([[age_months, weight]])
    pred_weight_6m = models['predictor'].predict(input_arr)[0]
    pred_weight_12m = models['predictor'].predict([[age_months + 6, pred_weight_6m]])[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Weight (in 6 months)", f"{pred_weight_6m:.2f} kg")
    c2.metric("Predicted Weight (in 12 months)", f"{pred_weight_12m:.2f} kg")
    c3.metric("Predicted Adult Height", f"{target_height:.1f} cm")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[age_months, age_months + 6, age_months + 12], y=[weight, pred_weight_6m, pred_weight_12m],
        mode='lines+markers', name='Projected Weight', line=dict(color='#0072ff', dash='dash')))
    fig.update_layout(title="Predicted Weight Trajectory", height=250, xaxis_title="Age (months)", yaxis_title="Weight (kg)", margin={"t":40, "b":10})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Explanation")
    ai_context = f"A child aged {age_months} months weighing {weight}kg is predicted to weigh {pred_weight_6m:.2f}kg in 6 months."
    for p in get_gemini_analysis(ai_context):
        st.markdown(f"🔹 {p}")

with tabs[4]:
    st.header("Risk Assessment & Recommendations")
    st.metric("Overall Risk Score", f"{risk_score} / 20", help="A higher score indicates more potential growth concerns.")
    if risk_factors:
        st.warning("Potential risk factors identified:")
        for factor in risk_factors: st.markdown(f"- **{factor}**")
    else:
        st.success("No critical risk factors were identified based on the provided data.")
    st.subheader("Explanation")
    risk_context = ", ".join(risk_factors) if risk_factors else "No critical risk factors identified."
    for p in get_gemini_analysis("Key risks identified: " + risk_context):
        st.markdown(f"🔹 {p}")

with tabs[5]:
    st.header("Comprehensive Summary")
    summary_data = {
        "Metric": ["Gender", "Age (Months)", "Weight", "Height/Length", "Head Circumference", "BMI", "MUAC", "Target Adult Height", "Risk Score"],
        "Value": [
            gender, f"{age_months}", f"{weight:.1f} kg ({wfa_perc:.1f}th %)", f"{height:.1f} cm ({lhfa_perc:.1f}th %)",
            f"{head_circumference:.1f} cm ({hc_perc:.1f}th %)", f"{bmi:.2f} ({bmi_perc:.1f}th %)", f"{muac:.1f} cm",
            f"{target_height:.1f} cm", f"{risk_score} ({'Low' if not risk_factors else ', '.join(risk_factors)})"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    st.subheader("Summary Results Explanation")
    sum_context = f"Summary for a {age_months}-month-old {gender.lower()}: Weight is {weight}kg ({wfa_perc}th percentile). Height is {height}cm ({lhfa_perc}th percentile). BMI is {bmi}. Key risks: {', '.join(risk_factors) if risk_factors else 'None identified'}."
    for p in get_gemini_analysis(sum_context):
        st.markdown(f"🔹 {p}")

st.info("_Disclaimer: This tool is for informational purposes only. Always consult a qualified healthcare professional for medical advice._", icon="ℹ️")