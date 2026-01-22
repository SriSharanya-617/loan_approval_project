import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="centered"
)

# -------------------------------------------------
# CUSTOM CSS (VISIBLE TEXT)
# -------------------------------------------------
st.markdown("""
<style>

/* ===== BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #f0f4ff, #e6f7f1, #fff7e6);
    background-attachment: fixed;
}

/* ===== MAIN CARD ===== */
.block-container {
    background: #ffffff !important;
    border-radius: 18px;
    padding: 2.5rem 3rem;
    box-shadow: 0 12px 35px rgba(0,0,0,0.12);
}

/* FORCE TEXT VISIBILITY */
.block-container * {
    color: #000000 !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1f3c88, #2a5298, #1e3c72);
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Sidebar inputs */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* ===== BUTTON ===== */
.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: #ffffff !important;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
    border: none;
}

/* ===== ALERTS ===== */
div.stAlert-success {
    background: #e6fffa !important;
    color: #065f46 !important;
    border-radius: 12px;
}

div.stAlert-error {
    background: #ffeaea !important;
    color: #7f1d1d !important;
    border-radius: 12px;
}

div.stAlert-info {
    background: #e8f3ff !important;
    color: #0b3c6f !important;
    border-radius: 12px;
}

div.stAlert-warning {
    background: #fff4cc !important;
    color: #7a4a00 !important;
    border-radius: 12px;
}

h1, h2, h3 {
    color: #1f3c88 !important;
    font-weight: 800;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🏦 Smart Loan Approval System")
st.markdown(
    "This system uses **Support Vector Machines (SVM)** to predict loan approval "
    "based on applicant financial details."
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("loan.csv")

df = load_data()

# -------------------------------------------------
# HANDLE MISSING VALUES
# -------------------------------------------------
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Education', 'Property_Area']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# -------------------------------------------------
# REMOVE OUTLIERS
# -------------------------------------------------
for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# -------------------------------------------------
# ENCODE CATEGORICAL
# -------------------------------------------------
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -------------------------------------------------
# FEATURES & TARGET
# -------------------------------------------------
X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Self_Employed', 'Property_Area']]
y = df['Loan_Status']

# -------------------------------------------------
# SPLIT & SCALE
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------
st.sidebar.header("📋 Applicant Details")

app_income = st.sidebar.number_input(
    "Applicant Income",
    min_value=500,
    value=5000
)

loan_amt = st.sidebar.number_input(
    "Loan Amount",
    min_value=10,
    value=150
)

credit_hist = st.sidebar.selectbox("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -------------------------------------------------
# MODEL SELECTION
# -------------------------------------------------
st.sidebar.header("⚙️ Model Selection")

kernel = st.sidebar.radio(
    "Choose SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

kernel_map = {
    "Linear SVM": ("linear", {}),
    "Polynomial SVM": ("poly", {"degree": 3}),
    "RBF SVM": ("rbf", {"gamma": "scale"})
}

kernel_name, params = kernel_map[kernel]

model = SVC(kernel=kernel_name, C=1, **params)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("✅ Check Loan Eligibility"):

    input_df = pd.DataFrame([{
        'ApplicantIncome': app_income,
        'LoanAmount': loan_amt,
        'Credit_History': 1.0 if credit_hist == "Yes" else 0.0,
        'Self_Employed': label_encoders['Self_Employed'].transform([employment])[0],
        'Property_Area': label_encoders['Property_Area'].transform([property_area])[0]
    }])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    confidence = abs(model.decision_function(input_scaled)[0])
    confidence = min(confidence / 3, 1.0) * 100

    st.markdown("---")
    st.subheader("📌 Loan Decision")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown(f"""
    **Kernel Used:** {kernel}  
    **Model Accuracy:** {accuracy:.2f}  
    **Confidence Score:** {confidence:.1f}%
    """)

    st.markdown("### 🧠 Business Explanation")

    if prediction == 1:
        st.info(
            "Based on the applicant’s income level and positive credit history, "
            "the model predicts a high likelihood of loan repayment."
        )
    else:
        st.warning(
            "Based on income and credit history patterns, "
            "the applicant is less likely to repay the loan."
        )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.caption("SVM-based FinTech System")
