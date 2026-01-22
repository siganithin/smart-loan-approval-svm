import streamlit as st
import pandas as pd
import joblib

# ------------------ Load saved objects ------------------
svm_linear = joblib.load("models/svm_linear.pkl")
svm_poly = joblib.load("models/svm_poly.pkl")
svm_rbf = joblib.load("models/svm_rbf.pkl")
scaler = joblib.load("models/scaler.pkl")
imputer = joblib.load("models/imputer.pkl")
features = joblib.load("models/features.pkl")

# ------------------ App UI ------------------
st.title("Smart Loan Approval System")
st.write("This system uses Support Vector Machines to predict loan approval.")

st.sidebar.header("Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0)
loan = st.sidebar.number_input("Loan Amount", min_value=0)
credit = st.sidebar.selectbox("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Yes", "No"])
area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

kernel = st.sidebar.radio(
    "Select SVM Kernel",
    ["RBF SVM", "Linear SVM", "Polynomial SVM"]
)


# ------------------ Prepare input ------------------
input_data = {
    "ApplicantIncome": income,
    "LoanAmount": loan,
    "Credit_History": 1 if credit == "Yes" else 0,
    "Self_Employed_Yes": 1 if employment == "Yes" else 0,
    "Property_Area_Semiurban": 1 if area == "Semiurban" else 0,
    "Property_Area_Urban": 1 if area == "Urban" else 0
}

input_df = pd.DataFrame([input_data])

# Align with training columns
input_df = input_df.reindex(columns=features, fill_value=0)

# Impute + scale
input_imputed = imputer.transform(input_df)
input_scaled = scaler.transform(input_imputed)

# ------------------ Prediction ------------------
if st.button("Check Loan Eligibility"):

    model = (
        svm_linear if kernel == "Linear SVM"
        else svm_poly if kernel == "Polynomial SVM"
        else svm_rbf
    )

    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.success("✅ Loan Approved")
        st.write("Applicant is likely to repay the loan.")
    else:
        st.error("❌ Loan Rejected")
        st.write("Applicant is unlikely to repay the loan.")

    st.info(f"Kernel Used: {kernel}")
