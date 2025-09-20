import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import re

# ==== APP CONFIG & UI THEME ====
st.set_page_config(page_title="🩺 Neonatal & Infant Growth Analyzer", layout="wide")

# --- PREVIOUS UI STYLES ---
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

/* --- Eye-catching Section Headers --- */
.section-header {
    text-align: center;
    font-weight: 800;
    color: #0a40a5;
    font-size: 2.2rem;
    margin-bottom: 25px;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.15);
}

/* --- Centering Utility --- */
.center-content {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
}

/* --- Enhanced Button Styling & Animation --- */
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
  transform: translateY(-3px) scale(1.05);
  box-shadow: 0 6px 20px 0 rgba(0,118,255,0.23);
  filter: brightness(1.2);
}

/* --- Tab Styling for Visibility, Spacing, and Font Size --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    background-color: #f0f2f6;
    transition: all 0.3s ease;
    font-weight: 600;
    color: #0d47a1 !important;
    font-size: 1.05rem;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #6dd5ed 0%, #2193b0 100%);
    color: white !important;
    font-size: 1.1rem;
    font-weight: 700;
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
        title={'text': f"<b>{title}</b>", 'font': {'size': 18, 'color': 'white'}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#0072ff"},
               'steps': [{'range': [0, 15], 'color': "#e0f7fa"}, {'range': [15, 85], 'color': "#b2ebf2"}, {'range': [85, 100], 'color': "#e0f7fa"}],
               'threshold': {'line': {'color': "#d62728", 'width': 4}, 'thickness': 0.75, 'value': value}}))
    fig.update_layout(height=230, margin=dict(t=60, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, key=key)

def calculate_bmi(weight, height_cm):
    return round(weight / ((height_cm / 100) ** 2), 2) if height_cm else 0

def calculate_target_height(father_height, mother_height, gender):
    return (father_height + mother_height + (13 if gender == "Male" else -13)) / 2

# ==== NEW: DETAILED XAI FUNCTION WITH RISK HIGHLIGHTING ====
def get_local_explanation(topic, data, gender):
    explanations = []
    child = "your son" if gender == "Male" else "your daughter"
    he_she = "he" if gender == "Male" else "she"
    his_her = "his" if gender == "Male" else "her"

    # Helper function to create styled text for risks
    def risk_text(text):
        # Using an orange color for warnings
        return f"<p style='color:#eea41a;'>{text}</p>"
    
    # Helper function for normal text points
    def normal_text(text):
        return f"<p>{text}</p>"

    if topic == 'growth_charts':
        wfa = data.get('wfa_perc', 50)
        lhfa = data.get('lhfa_perc', 50)
        
        explanations.append(f"<b>Weight-for-Age ({wfa}th percentile):</b>")
        if wfa < 15:
            explanations.append(risk_text(f"  • This indicates {child} is in a lower weight range compared to {his_her} peers, which is a potential risk."))
            explanations.append(risk_text(f"  • Consistent low weight can impact energy levels and overall development."))
            explanations.append(risk_text(f"  • It is highly advisable to consult a pediatrician to discuss {his_her} diet and rule out any health issues."))
        elif wfa > 85:
            explanations.append(normal_text(f"  • {child.capitalize()} is heavier than the average for {his_her} age, which can be a positive sign of good nutrition."))
            explanations.append(normal_text(f"  • It's important to ensure this weight gain is balanced with steady height growth."))
            explanations.append(risk_text(f"  • Monitor this trend with your doctor to ensure {he_she} maintains a healthy growth curve and avoids excessive weight gain."))
        else:
            explanations.append(normal_text(f"  • This is a great sign! {his_her.capitalize()} weight is in the healthy, typical range for {his_her} age."))
            explanations.append(normal_text(f"  • This indicates {he_she} is receiving good nutrition and growing steadily."))
        
        explanations.append(f"<br><b>Height-for-Age ({lhfa}th percentile):</b>")
        if lhfa < 15:
            explanations.append(risk_text(f"  • {child.capitalize()} is shorter for {his_her} age. This is a condition known as **stunting** if it persists."))
            explanations.append(risk_text(f"  • Stunting is a serious issue as it relates to long-term nutrition and can affect overall development."))
            explanations.append(risk_text(f"  • Ensure {his_her} diet is rich in protein, calcium, and vitamins, and discuss this finding with a healthcare professional."))
        else:
            explanations.append(normal_text(f"  • Excellent! {his_her.capitalize()} height is on track or above average for {his_her} age."))
            explanations.append(normal_text(f"  • This is a strong indicator of healthy skeletal development and good long-term nutrition."))

    elif topic == 'body_composition':
        bmi = data.get('bmi_perc', 50)
        explanations.append(f"<b>BMI-for-Age ({bmi}th percentile):</b>")
        if bmi < 5:
            explanations.append(risk_text(f"  • BMI (Body Mass Index) shows {child} may be **underweight** for {his_her} height."))
            explanations.append(risk_text(f"  • This is a significant health indicator and suggests {he_she} may not be getting enough calories or nutrients."))
            explanations.append(risk_text(f"  • Please review this finding with a doctor to create a plan for healthy weight gain."))
        elif bmi > 85:
            explanations.append(risk_text(f"  • {his_her.capitalize()} BMI is in a higher range, suggesting a risk of being **overweight** or **obese**."))
            explanations.append(risk_text(f"  • This can be an early indicator of future health problems, so it's important to address."))
            explanations.append(risk_text(f"  • Focus on encouraging active play, limiting sugary drinks, and providing a diet of whole, unprocessed foods."))
        else:
            explanations.append(normal_text(f"  • Perfect. This shows a healthy balance between {his_her} weight and height."))
            explanations.append(normal_text(f"  • This is a key sign of good nutritional status and lowers the risk of many health issues."))

    elif topic == 'development':
        target_height = data.get('target_height', 170)
        explanations.append(f"<b>Predicted Adult Height:</b>")
        explanations.append(normal_text(f"  • Based on parental heights, the scientific estimate for {his_her} adult height is around **{target_height:.1f} cm**."))
        explanations.append(normal_text(f"  • This is often called 'mid-parental height' and provides a genetic target for {his_her} growth potential."))
        explanations.append(normal_text(f"  • Good nutrition, sleep, and a healthy lifestyle are crucial for {child} to reach this full potential."))

    elif topic == 'prediction':
        pred_6m = data.get('pred_weight_6m', 10)
        explanations.append(f"<b>Future Growth Projection:</b>")
        explanations.append(normal_text(f"  • Based on {his_her} current growth pattern, our model predicts {child} will weigh approximately **{pred_6m:.2f} kg** in six months."))
        explanations.append(normal_text(f"  • This is a statistical forecast, not a guarantee. Growth can have spurts and slowdowns."))
        explanations.append(normal_text(f"  • The best way to ensure healthy future growth is to continue with regular health check-ups and a balanced diet."))
    
    elif topic == 'risks':
        risks = data.get('risk_factors', [])
        if not risks:
            explanations.append(normal_text("<b>No Major Risks Identified:</b> Based on the data provided, your child appears to be on a healthy growth track. Continue with regular monitoring and healthy habits."))
        else:
            explanations.append("<b>Key Recommendations Based on Risks:</b>")
            if "Low weight-for-age" in risks or "Very low BMI" in risks:
                explanations.append(risk_text("  • <b>Address Underweight Status:</b> The low weight/BMI is a primary concern. Focus on providing calorie and nutrient-dense foods like avocado, full-fat dairy, and nut butters (if age appropriate). Consult a doctor to ensure there are no underlying medical issues affecting weight gain."))
            if "Risk of stunting (low height)" in risks:
                explanations.append(risk_text("  • <b>Focus on Linear Growth:</b> To combat stunting risk, ensure a diet rich in protein (for building blocks), calcium and Vitamin D (for bone health), and other essential micronutrients. This is crucial for long-term development."))
            if "Malnutrition risk (low MUAC)" in risks:
                explanations.append(risk_text("  • <b>Urgent Malnutrition Review:</b> A low MUAC (arm circumference) is a serious warning sign that requires immediate medical attention. Please see a healthcare professional as soon as possible to address acute malnutrition."))
            if "High BMI (obesity risk)" in risks or "Central obesity risk" in risks:
                explanations.append(risk_text("  • <b>Manage Obesity Risk:</b> To address high BMI or central obesity, reduce sugary snacks and drinks, encourage at least 60 minutes of active play per day, and focus family meals around vegetables, lean proteins, and whole grains."))
    return explanations

# ========== INPUT UI ==========
st.title("Neonatal & Infant Growth Analyzer")
st.markdown("Developed by **NIST University & SIBA, Artificial Intelligence Global Innovation Center (GIC)**")

with st.form("input_form"):
    st.markdown('<div class="card-section">', unsafe_allow_html=True)
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

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_button = st.form_submit_button("Analyze Growth", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if not submit_button:
    st.stop()

# ========== CALCULATIONS & ANALYSIS ==========
bmi = calculate_bmi(weight, height)
wfa_perc = get_percentile(age_months, weight, gender, 'wfa')
lhfa_perc = get_percentile(age_months, height, gender, 'lhfa')
hc_perc = get_percentile(age_months, head_circumference, gender, 'hc')
bmi_perc = get_percentile(age_months, bmi, gender, 'bmi')
target_height = calculate_target_height(father_height, mother_height, gender)

risk_score = 0
risk_factors = []
if wfa_perc < 15: risk_factors.append("Low weight-for-age"); risk_score += (3 if wfa_perc < 3 else 2)
elif wfa_perc > 85: risk_factors.append("High weight-for-age"); risk_score += 2
if lhfa_perc < 15: risk_factors.append("Risk of stunting (low height)"); risk_score += (3 if lhfa_perc < 3 else 2)
if muac < 12.5: risk_factors.append("Malnutrition risk (low MUAC)"); risk_score += (4 if muac < 11.5 else 3)
if bmi_perc < 5: risk_factors.append("Very low BMI"); risk_score += 2
elif bmi_perc > 90: risk_factors.append("High BMI (obesity risk)"); risk_score += 3
if waist_circumference and height and (waist_circumference/height) >= 0.5: risk_factors.append("Central obesity risk"); risk_score += 3

# ========== TABS OUTPUT (FULL VERSION) ==========
tab_labels = ["Growth Charts", "Body Composition", "Development", "Prediction", "Risks", "Summary"]
tabs = st.tabs(tab_labels)

def display_explanations(topic, data, gender):
    explanations = get_local_explanation(topic, data, gender)
    for p in explanations:
        st.markdown(p, unsafe_allow_html=True)

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

    st.subheader("Explanation")
    display_explanations('growth_charts', {'wfa_perc': wfa_perc, 'lhfa_perc': lhfa_perc}, gender)

with tabs[1]:
    st.header("Body Composition Analysis")
    c1, c2 = st.columns([2, 1])
    with c1:
        plot_gauge(bmi_perc, "BMI-for-Age", "bmi_gauge")
    with c2:
        st.metric("Body Mass Index (BMI)", f"{bmi:.2f}")
        st.metric("Triceps Skinfold (mm)", f"{triceps_sf:.1f}")
        st.metric("Subscapular Skinfold (mm)", f"{subscapular_sf:.1f}")
        st.metric("Waist Circumference (cm)", f"{waist_circumference:.1f}")

    st.subheader("Explanation")
    display_explanations('body_composition', {'bmi_perc': bmi_perc}, gender)

with tabs[2]:
    st.header("Development Insights")
    st.metric("Predicted Adult Height (Mid-Parental)", f"{target_height:.1f} cm")
    st.metric("Current Height Percentile", f"{lhfa_perc}th")
    st.subheader("Explanation")
    display_explanations('development', {'target_height': target_height}, gender)

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
    display_explanations('prediction', {'pred_weight_6m': pred_weight_6m}, gender)

with tabs[4]:
    st.header("Risk Assessment & Recommendations")
    st.metric("Overall Risk Score", f"{risk_score} / 20", help="A higher score indicates more potential growth concerns.")
    if risk_factors:
        st.warning("Potential risk factors identified:")
        for factor in risk_factors: st.markdown(f"- **{factor}**")
    else:
        st.success("No critical risk factors were identified based on the provided data.")
    st.subheader("Explanation")
    display_explanations('risks', {'risk_factors': risk_factors}, gender)

with tabs[5]:
    st.header("Comprehensive Summary")
    summary_data = {
        "Metric": ["Gender", "Age (Months)", "Weight", "Height/Length", "Head Circumference", "BMI", "MUAC", "Target Adult Height", "Risk Score"],
        "Value": [
            gender, f"{age_months}", f"{weight:.1f} kg ({wfa_perc:.1f}th %)", f"{height:.1f} cm ({lhfa_perc:.1f}th %)",
            f"{head_circumference:.1f} cm ({hc_perc:.1f}th %)", f"{bmi:.2f} ({bmi_perc:.1f}th %)", f"{muac:.1f} cm",
            f"{target_height:.1f} cm", f"{risk_score} ({'Low Risk' if not risk_factors else ', '.join(risk_factors)})"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    st.subheader("Summary Recommendations")
    display_explanations('risks', {'risk_factors': risk_factors}, gender)

st.info("_Disclaimer: This tool is for informational purposes only. Always consult a qualified healthcare professional for medical advice._", icon="ℹ️")