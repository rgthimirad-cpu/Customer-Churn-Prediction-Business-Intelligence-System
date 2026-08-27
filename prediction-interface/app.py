import streamlit as st
import pandas as pd
import joblib

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
    page_icon="📊",
    layout="wide"
)

# ----------------------------------
# Header
# ----------------------------------

st.title("📊 Customer Churn Prediction System")

st.markdown(
    """
    Predict whether a customer is likely to churn and
    receive risk classification and retention recommendations.
    """
)

st.divider()

# ----------------------------------
# Customer Information
# ----------------------------------

st.header("👤 Customer Information")

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

st.header("🛠 Service Information")

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

st.header("💳 Billing Information")

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

    # Input DataFrame

    input_data = pd.DataFrame({

        "Senior Citizen": [
            yes_no_to_binary(senior_citizen)
        ],

        "Partner": [
            yes_no_to_binary(partner)
        ],

        "Dependents": [
            yes_no_to_binary(dependents)
        ],

        "Tenure Months": [
            tenure_months
        ],

        "Multiple Lines": [
            yes_no_to_binary(multiple_lines)
        ],

        "Internet Service": [
            internet_service
        ],

        "Online Security": [
            yes_no_to_binary(online_security)
        ],

        "Online Backup": [
            yes_no_to_binary(online_backup)
        ],

        "Device Protection": [
            yes_no_to_binary(device_protection)
        ],

        "Tech Support": [
            yes_no_to_binary(tech_support)
        ],

        "Streaming TV": [
            yes_no_to_binary(streaming_tv)
        ],

        "Streaming Movies": [
            yes_no_to_binary(streaming_movies)
        ],

        "Contract": [
            contract
        ],

        "Paperless Billing": [
            yes_no_to_binary(paperless_billing)
        ],

        "Payment Method": [
            1
        ],

        "Monthly Charges": [
            monthly_charges
        ],

        "Total Charges": [
            total_charges
        ],

        "Fiber_Monthly_Risk": [
            fiber_monthly_risk
        ],

        "New_High_Spend": [
            new_high_spend
        ],

        "New_Monthly_Customer": [
            new_monthly_customer
        ],

        "Security_Tech_Bundle": [
            security_tech_bundle
        ]
    })

    
    # Predict

    probability = model.predict_proba(
        input_data
    )[0][1]

    probability_percent = probability * 100

    # ----------------------------------
    # Risk Level
    # ----------------------------------

    if probability <= 0.30:
        risk_level = "LOW"

    elif probability <= 0.60:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    # ----------------------------------
    # Risk Factors
    # ----------------------------------

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

    # ----------------------------------
    # Recommendation Logic
    # ----------------------------------

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

    st.header("📈 Prediction Results")

    col1, col2 = st.columns(2)

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

    # ----------------------------------
    # Risk Factors
    # ----------------------------------

    st.subheader("⚠ Main Risk Factors")

    if risk_factors:

        for factor in risk_factors:
            st.write(f"• {factor}")

    else:

        st.write(
            "• No significant risk factors identified"
        )

    # ----------------------------------
    # Recommendation
    # ----------------------------------

    st.subheader("🎯 Recommended Action")

    st.success(recommendation)