
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")

model = load_model()

st.title("Telecom Customer Churn Prediction")
st.write("Predict churn risk and support customer retention decisions.")

st.sidebar.header("Customer Details")

def yes_no(label):
    return st.sidebar.selectbox(label, ["Yes", "No"])

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = yes_no("Partner")
dependents = yes_no("Dependents")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

phone = yes_no("Phone Service")
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines", ["No", "Yes", "No phone service"]
)

internet = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

internet_options = ["No", "Yes", "No internet service"]

online_security = st.sidebar.selectbox(
    "Online Security", internet_options
)
online_backup = st.sidebar.selectbox(
    "Online Backup", internet_options
)
device_protection = st.sidebar.selectbox(
    "Device Protection", internet_options
)
tech_support = st.sidebar.selectbox(
    "Tech Support", internet_options
)
streaming_tv = st.sidebar.selectbox(
    "Streaming TV", internet_options
)
streaming_movies = st.sidebar.selectbox(
    "Streaming Movies", internet_options
)

contract = st.sidebar.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)
paperless = yes_no("Paperless Billing")

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.sidebar.number_input(
    "Monthly Charges", min_value=0.0, value=70.0
)
total = st.sidebar.number_input(
    "Total Charges", min_value=0.0,
    value=float(monthly * tenure)
)

customer = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}

input_df = pd.DataFrame([customer])

st.subheader("Customer Profile")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Churn Risk", type="primary"):
    probability = model.predict_proba(input_df)[0, 1]
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{probability:.1%}")
    col2.metric(
        "Model Prediction",
        "Likely to Churn" if prediction == 1 else "Likely to Stay"
    )

    if probability >= 0.70:
        st.error("High risk: prioritize a retention review.")
        st.write(
            "Suggested action: contact the customer, understand "
            "their concerns, and review suitable contract or service options."
        )
    elif probability >= 0.40:
        st.warning("Medium risk: consider a proactive check-in.")
        st.write(
            "Suggested action: review service satisfaction and "
            "offer relevant support."
        )
    else:
        st.success("Low risk: continue normal engagement.")
        st.write(
            "Suggested action: maintain service quality and "
            "monitor customer feedback."
        )

    st.caption(
        "Risk bands are illustrative business thresholds, not "
        "validated optimal cutoffs. Predictions are estimates, "
        "not guarantees."
    )
