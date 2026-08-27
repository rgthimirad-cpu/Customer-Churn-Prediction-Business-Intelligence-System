import streamlit as st

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
# Placeholder Results
# ----------------------------------

if predict:

    st.divider()

    st.header("📈 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            "-- %"
        )

    with col2:
        st.metric(
            "Risk Level",
            "--"
        )

    st.subheader("⚠ Main Risk Factors")

    st.write("• To be integrated with Group 5 outputs")
    st.write("• Placeholder")
    st.write("• Placeholder")

    st.subheader("🎯 Recommended Action")

    st.info(
        "Retention recommendation will be integrated in the next version."
    )