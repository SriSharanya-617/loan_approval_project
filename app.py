import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------------
# Load and prepare dataset
# -----------------------------
df = pd.read_csv("loan_data.csv")

features = [
    'ApplicantIncome',
    'LoanAmount',
    'Credit_History',
    'Self_Employed',
    'Property_Area'
]
target = 'Loan_Status'

X = df[features].copy()
y = df[target]

# Handle missing values
X['LoanAmount'] = X['LoanAmount'].fillna(X['LoanAmount'].mean())
X['Credit_History'] = X['Credit_History'].fillna(X['Credit_History'].mode()[0])
X['Self_Employed'] = X['Self_Employed'].fillna(X['Self_Employed'].mode()[0])

# Encode categorical columns
le = LabelEncoder()
X['Self_Employed'] = le.fit_transform(X['Self_Employed'])
X['Property_Area'] = le.fit_transform(X['Property_Area'])
y = le.fit_transform(y)

# Split & scale
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏦 Smart Loan Approval System")

st.write(
    "This system uses **Support Vector Machines (SVM)** to predict whether a loan "
    "should be **Approved or Rejected** based on applicant details."
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0)
loan_amt = st.sidebar.number_input("Loan Amount", min_value=0)
credit = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

credit_val = 1 if credit == "Yes" else 0
employment_val = 1 if employment == "Yes" else 0
property_val = le.transform([property_area])[0]

# -----------------------------
# Model Selection
# -----------------------------
st.sidebar.header("Select SVM Kernel")
kernel_choice = st.sidebar.radio(
    "Kernel Type",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

# Train selected model
if kernel_choice == "Linear SVM":
    model = SVC(kernel='linear', probability=True)
elif kernel_choice == "Polynomial SVM":
    model = SVC(kernel='poly', degree=3, probability=True)
else:
    model = SVC(kernel='rbf', probability=True)

model.fit(x_train, y_train)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("✅ Check Loan Eligibility"):
    user_data = np.array([[
        income,
        loan_amt,
        credit_val,
        employment_val,
        property_val
    ]])

    user_data = scaler.transform(user_data)

    prediction = model.predict(user_data)[0]
    confidence = model.predict_proba(user_data)[0][prediction]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Kernel Used:** {kernel_choice}")
    st.write(f"**Model Confidence:** {confidence:.2f}")

    # Business explanation
    if credit_val == 1 and income > loan_amt:
        st.info(
            "Based on strong credit history and income pattern, "
            "the applicant is likely to repay the loan."
        )
    else:
        st.warning(
            "Based on weak credit or income-to-loan ratio, "
            "the applicant is unlikely to repay the loan."
        )
