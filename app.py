from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ================================
# Load dataset and preprocess
# ================================
def load_data():
    dataset_path = Path("cleaned_dataset.csv")

    if not dataset_path.exists():
        st.error(f"Dataset file not found at {dataset_path}. Upload it to the workspace folder.")
        st.stop()

    df = pd.read_csv(dataset_path)

    # Drop unnecessary columns
    if 'customerid' in df.columns:
        df.drop(columns=['customerid'], inplace=True)  

    # Encode categorical features
    le = LabelEncoder()
    categorical_cols = [
        'gender', 'partner', 'dependents', 'phoneservice', 'multiplelines',
        'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
        'streamingtv', 'streamingmovies', 'paperlessbilling'
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # Scale numerical features
    numeric_cols = ["tenure", "monthlycharges", "totalcharges"]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    X = df.drop('churn', axis=1)
    y = df['churn']

    return df, X, y, scaler

# ================================
# Load model
# ================================
def load_model():
    model_path = Path("logistic_regression_model.pkl")
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Upload it to the workspace folder.")
        st.stop()
    model = joblib.load(model_path)
    return model

# ================================
# Main Streamlit App
# ================================
def main():
    st.title("Telecom Customer Churn Prediction")

    df, X, y, scaler = load_data()
    model = load_model()

    # ================================
    # User Inputs
    # ================================
    gender = st.selectbox("Gender", ["Male", "Female"])
    seniorcitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", 0, 100, 10)
    phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
    multiplelines = st.selectbox("Multiple Lines", ["Yes", "No"])
    onlinesecurity = st.selectbox("Online Security", ["Yes", "No"])
    onlinebackup = st.selectbox("Online Backup", ["Yes", "No"])
    deviceprotection = st.selectbox("Device Protection", ["Yes", "No"])
    techsupport = st.selectbox("Tech Support", ["Yes", "No"])
    streamingtv = st.selectbox("Streaming TV", ["Yes", "No"])
    streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No"])
    paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    monthlycharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    totalcharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
    internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paymentmethod = st.selectbox("Payment Method", [
        "Electronic check",
        "Credit card (automatic)",
        "Bank transfer (automatic)",
        "Manual"
    ])

    input_dict = {
        "gender": gender,
        "seniorcitizen": 1 if seniorcitizen == "Yes" else 0,
        "partner": 1 if partner == "Yes" else 0,
        "dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "phoneservice": 1 if phoneservice == "Yes" else 0,
        "multiplelines": 1 if multiplelines == "Yes" else 0,
        "onlinesecurity": 1 if onlinesecurity == "Yes" else 0,
        "onlinebackup": 1 if onlinebackup == "Yes" else 0,
        "deviceprotection": 1 if deviceprotection == "Yes" else 0,
        "techsupport": 1 if techsupport == "Yes" else 0,
        "streamingtv": 1 if streamingtv == "Yes" else 0,
        "streamingmovies": 1 if streamingmovies == "Yes" else 0,
        "paperlessbilling": 1 if paperlessbilling == "Yes" else 0,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges,
        "internetservice": internetservice,
        "contract": contract,
        "paymentmethod": paymentmethod,
    }

    df_input = pd.DataFrame([input_dict])

    # One-hot encoding & align columns with training data
    df_input = pd.get_dummies(df_input)
    expected_cols = model.feature_names_in_
    df_input = df_input.reindex(columns=expected_cols, fill_value=0)

    # Predict
    if st.button("Predict Churn"):
        prediction = model.predict(df_input)[0]
        if prediction == 1:
            st.error("⚠ The customer is likely to CHURN!")
        else:
            st.success("✅ The customer is NOT likely to churn.")

# Run the app
if __name__ == "__main__":
    main()

