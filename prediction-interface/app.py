import streamlit as st
import pandas as pd
import joblib
import base64

import base64

with open("prediction-interface/assets/logo.png", "rb") as image_file:
    logo_base64 = base64.b64encode(image_file.read()).decode()

# ----------------------------------
# Load Model
# ----------------------------------

model = joblib.load(
    "Machine Learning/models/final_churn_pipeline.joblib"
)

# ----------------------------------
# Helper Function
# ----------------------------------

def yes_no_to_binary(value):
    return 1 if value == "Yes" else 0


# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="prediction-interface/assets/logo.png",
    layout="wide"
)

def get_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(
    BASE_DIR,
    "assets",
    "background.jpg"
)

img = get_base64(image_path)


st.markdown(
    f"""
    <style>

    .stApp {{
        background:
            linear-gradient(
                rgba(0,0,40,0.35),
                rgba(0,0,40,0.35)
            ),
            url("data:image/jpg;base64,{img}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .block-container {{
        padding-top: 2rem;
        max-width: 1350px;
    }}

    h1 {{
        color: white !important;
        text-align: center;
        font-size: 3.5rem !important;
        text-shadow: 0 0 20px #00E5FF;
    }}

    h2 {{
        color: #00E5FF !important;
        font-weight: bold;
    }}

    h3 {{
        color: #FF4FD8 !important;
    }}

    div[data-baseweb="select"] > div {{
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(0,229,255,0.3);
        border-radius: 12px;
    }}

    .stNumberInput input {{
        background: rgba(255,255,255,0.08) !important;
        color: white !important;
        border-radius: 12px;
    }}

    .stButton > button {{
        background: linear-gradient(
            90deg,
            #00E5FF,
            #7B61FF,
            #FF4FD8
        );
        color: white;
        border: none;
        border-radius: 15px;
        height: 65px;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 0 0 25px rgba(123,97,255,0.7);
    }}

    .stButton > button:hover {{
        transform: scale(1.02);
        transition: 0.3s;
        box-shadow: 0 0 35px #FF4FD8;
    }}

    div[data-testid="metric-container"] {{
        background: rgba(15,20,60,0.45);
        border: 1px solid rgba(0,229,255,0.2);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        padding: 20px;
        box-shadow: 0 0 20px rgba(0,229,255,0.15);
    }}

    .stProgress > div > div > div > div {{
        background: linear-gradient(
            90deg,
            #00E5FF,
            #7B61FF,
            #FF4FD8
        );
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------
# Header
# ----------------------------------

st.markdown(f"""
<div style="
background: rgba(15,20,60,0.45);
backdrop-filter: blur(15px);
padding:30px;
border-radius:20px;
text-align:center;
box-shadow:0 0 25px rgba(0,229,255,0.2);
">

<h1>
<img src="data:image/png;base64,{logo_base64}" width="50" style="vertical-align:middle;font-size:22px;
color:white;
">Customer Churn Prediction System</h1>
<p style="font-size:22px; color:white;">
Predict customer churn using AI and generate retention strategies.
</p>

</div>
""", unsafe_allow_html=True)

st.divider()

# ----------------------------------
# Customer Information
# ----------------------------------

st.markdown("""
<h2 style='
text-align:center;
color:#00E5FF;
margin-bottom:30px;
'>
👤 Customer Information
</h2>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["Yes", "No"]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

with col2:

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure_months = st.number_input(
        "Tenure Months",
        min_value=0,
        value=0
    )

# ----------------------------------
# Service Information
# ----------------------------------

st.divider()

st.markdown("""
<h2 style='
text-align:center;
color:#00E5FF
margin-bottom:30px;
'>
🛠 Service Information
</h2>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No"]
    )

with col2:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No"]
    )

# ----------------------------------
# Billing Information
# ----------------------------------

st.divider()

st.markdown("""
<h2 style='
text-align:center;
color:#00E5FF;
margin-bottom:30px;
'>
💳 Billing Information
</h2>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=0.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=0.0
    )

# ----------------------------------
# Predict Button
# ----------------------------------

st.divider()

predict = st.button(
    "🔍 Predict Churn",
    use_container_width=True
)

# ----------------------------------
# Prediction
# ----------------------------------

if predict:
    errors = []

    if total_charges < monthly_charges:
        errors.append(
            "Total Charges must be greater than or equal to Monthly Charges."
        )

    if tenure_months > 0 and total_charges == 0:
        errors.append(
            "Total Charges can not be zero when tenure is greater than zero."
        )

    if errors:
        for error in errors:
            st.error(error)

        st.stop()


    # Engineered Features

    fiber_monthly_risk = int(
        internet_service == "Fiber optic"
        and contract == "Month-to-month"
    )

    new_high_spend = int(
        tenure_months <= 12
        and monthly_charges >= 70
    )

    new_monthly_customer = int(
        tenure_months <= 12
        and contract == "Month-to-month"
    )

    security_tech_bundle = int(
        online_security == "Yes"
        and tech_support == "Yes"
    )

    # Input Data

    input_data = pd.DataFrame({

        "Senior Citizen": [yes_no_to_binary(senior_citizen)],
        "Partner": [yes_no_to_binary(partner)],
        "Dependents": [yes_no_to_binary(dependents)],
        "Tenure Months": [tenure_months],
        "Multiple Lines": [yes_no_to_binary(multiple_lines)],
        "Internet Service": [internet_service],
        "Online Security": [yes_no_to_binary(online_security)],
        "Online Backup": [yes_no_to_binary(online_backup)],
        "Device Protection": [yes_no_to_binary(device_protection)],
        "Tech Support": [yes_no_to_binary(tech_support)],
        "Streaming TV": [yes_no_to_binary(streaming_tv)],
        "Streaming Movies": [yes_no_to_binary(streaming_movies)],
        "Contract": [contract],
        "Paperless Billing": [yes_no_to_binary(paperless_billing)],
        "Payment Method": [1],
        "Monthly Charges": [monthly_charges],
        "Total Charges": [total_charges],
        "Fiber_Monthly_Risk": [fiber_monthly_risk],
        "New_High_Spend": [new_high_spend],
        "New_Monthly_Customer": [new_monthly_customer],
        "Security_Tech_Bundle": [security_tech_bundle]

    })

    # Predict

    probability = model.predict_proba(
        input_data
    )[0][1]

    probability_percent = probability * 100

    # Risk Level

    if probability <= 0.30:
        risk_level = "LOW"
    elif probability <= 0.60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # Confidence Level

    if probability >= 0.85:
        confidence = "Very Strong"
    elif probability >= 0.70:
        confidence = "Strong"
    else:
        confidence = "Moderate"

    # Risk Factors

    risk_factors = []

    if tenure_months <= 12:
        risk_factors.append("Short tenure")

    if contract == "Month-to-month":
        risk_factors.append("Month-to-month contract")

    if monthly_charges >= 70:
        risk_factors.append("High monthly charges")

    if internet_service == "Fiber optic":
        risk_factors.append("Fiber optic service")

    if tech_support == "No":
        risk_factors.append("No technical support")

    top_factors = risk_factors[:3]

    # Recommendations

    if risk_level == "HIGH":

        recommendation = (
            "Provide a personalized retention offer, assign a customer support representative, and contact the customer immediately."
        )

    elif risk_level == "MEDIUM":

        recommendation = (
            "Offer loyalty incentives, monitor customer engagement, and provide targeted promotions."
        )

    else:

        recommendation = (
            "Maintain regular engagement and continue providing quality service."
        )

    # ----------------------------------
    # Results
    # ----------------------------------

    st.divider()

    st.markdown("""
<h2 style='
text-align:center;
color:#00E5FF;
margin-bottom:30px;
'>
📈 Prediction Results
</h2>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )

    with col2:
        st.metric(
            "Risk Level",
            risk_level
        )

    with col3:
        st.metric(
            "Confidence",
            confidence
        )

    st.progress(probability)

    # Assessment

    st.subheader("📋 Customer Assessment")

    st.info(
        f"This customer is classified as {risk_level} risk with a churn probability of {probability_percent:.2f}%. The prediction is based on customer profile, subscription characteristics, billing behavior and service usage patterns."
    )

    # Risk Factors

    st.subheader("⚠ Top Churn Drivers")

    if top_factors:

        for factor in top_factors:
            st.write(f"• {factor}")

    else:

        st.write(
            "• No significant risk factors identified"
        )

    # Recommendation

    st.subheader("🎯 Recommended Action")

    st.success(recommendation)