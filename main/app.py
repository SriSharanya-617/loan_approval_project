import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ---------------- Page Config ----------------
st.set_page_config(page_title="Smart Loan Approval System", layout="wide")


# ---------------- CSS Styling ----------------
st.markdown("""
<style>
.main {background-color: #0b1220; color: white;}
h1, h2, h3 {color: white !important;}
.result-box {
    padding: 20px;
    border-radius: 15px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}
.approved {background-color: #0f5132; color: #d1e7dd;}
.rejected {background-color: #842029; color: #f8d7da;}
.info-box {
    padding: 15px;
    border-radius: 12px;
    background-color: #111827;
    color: white;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- Title & Description ----------------
st.title("🏦 Smart Loan Approval System")
st.write("This system uses **Support Vector Machines (SVM)** to predict loan approval based on applicant details.")


# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("main/loan.csv")

    return df


df = load_data()


# ---------------- Data Cleaning ----------------
def clean_data(df):
    df = df.copy()

    # Fill missing values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Drop ID column if exists
    if "Loan_ID" in df.columns:
        df.drop(columns=["Loan_ID"], inplace=True)

    return df


df = clean_data(df)

# ---------------- Select Important Columns ----------------
# We will use only the columns asked in instructions
required_cols = ["ApplicantIncome", "LoanAmount", "Credit_History", "Self_Employed", "Property_Area", "Loan_Status"]
df = df[required_cols]

# Encode target
le = LabelEncoder()
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])  # Y=1, N=0

# One-hot encode features
X = df.drop(columns=["Loan_Status"])
X = pd.get_dummies(X, drop_first=True)
y = df["Loan_Status"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling (SVM needs scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------- Sidebar Inputs ----------------
st.sidebar.header("📌 Enter Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0.0, value=5000.0, step=500.0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0, value=150.0, step=10.0)

credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])
credit_history_val = 1.0 if credit_history == "Yes" else 0.0

employment_status = st.sidebar.selectbox("Employment Status", ["Yes", "No"])  # Self_Employed
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.sidebar.header("⚙️ Model Selection")
kernel_choice = st.sidebar.radio("Choose SVM Kernel", ["Linear SVM", "Polynomial SVM", "RBF SVM"])

# Convert choice to sklearn kernel
kernel_map = {
    "Linear SVM": "linear",
    "Polynomial SVM": "poly",
    "RBF SVM": "rbf"
}
kernel = kernel_map[kernel_choice]


# ---------------- Train Selected Model ----------------
model = SVC(kernel=kernel, C=1, gamma="scale", probability=True)
model.fit(X_train_scaled, y_train)

# Accuracy (Optional display)
test_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, test_pred)


# ---------------- Prepare User Input Data ----------------
user_input = {
    "ApplicantIncome": app_income,
    "LoanAmount": loan_amount,
    "Credit_History": credit_history_val,
    "Self_Employed": employment_status,
    "Property_Area": property_area
}

user_df = pd.DataFrame([user_input])

# One-hot encode user input same as training
user_df = pd.get_dummies(user_df, drop_first=True)

# Match training columns
user_df = user_df.reindex(columns=X.columns, fill_value=0)

# Scale
user_scaled = scaler.transform(user_df)


# ---------------- Prediction Button ----------------
if st.sidebar.button("✅ Check Loan Eligibility"):
    pred = model.predict(user_scaled)[0]
    proba = model.predict_proba(user_scaled)[0]

    confidence = np.max(proba) * 100  # confidence %

    # Output Section
    st.subheader("📌 Loan Decision Result")

    if pred == 1:
        st.markdown('<div class="result-box approved">✅ Loan Approved</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box rejected">❌ Loan Rejected</div>', unsafe_allow_html=True)

    # Optional Details
    st.markdown(f"""
    <div class="info-box">
    <b>Kernel Used:</b> {kernel_choice} <br>
    <b>Model Test Accuracy:</b> {acc:.2f} <br>
    <b>Confidence Score:</b> {confidence:.2f}% 
    </div>
    """, unsafe_allow_html=True)

    # Business Explanation (Very Important)
    st.subheader("💡 Business Explanation")
    if credit_history_val == 1 and app_income > 3000:
        explanation = "Based on **good credit history** and **stable income pattern**, the applicant is likely to repay the loan."
    elif credit_history_val == 0:
        explanation = "Because the applicant has **no credit history**, the risk is higher and approval is less likely."
    else:
        explanation = "Based on the income and loan amount pattern, the applicant may have repayment risk."

    st.write(explanation)

else:
    st.info("👈 Enter applicant details from the sidebar and click **Check Loan Eligibility**.")
