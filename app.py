import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")

# ==================== تحميل البيانات والموديل ====================
@st.cache_data
def load_data():
    dataset_path = Path("workspace") / "cleaned_dataset.csv"

    if not dataset_path.exists():
        st.error(f"Dataset file not found at {dataset_path}. Upload it to the workspace folder.")
        st.stop()

    df = pd.read_csv(dataset_path)
    return df

@st.cache_resource
def load_model():
    try:
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('ordinal_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except:
        return None, None

df = load_data()
model, encoder = load_model()

# ==================== Sidebar ====================
page = st.sidebar.radio("Navigate", ["Dashboard", "Prediction", "Insights"])

# ==================== Dashboard ====================
if page == "Dashboard":
    st.title("Customer Churn Dashboard")
    
    st.write("Total Customers:", len(df))
    st.write("Churned Customers:", len(df[df['Churn']=='Yes']))
    st.write("Churn Rate:", f"{(len(df[df['Churn']=='Yes'])/len(df)*100):.1f}%")
    st.write("Avg Monthly Charges:", f"${df['MonthlyCharges'].mean():.2f}")
    
    st.subheader("Churn Distribution")
    st.bar_chart(df['Churn'].value_counts())

# ==================== Prediction ====================
elif page == "Prediction":
    st.title("Predict Customer Churn")
    
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", value=50.0)
    contract = st.selectbox("Contract", ["Monthly", "One year", "Two year"])
    
    if st.button("Predict"):
        input_data = pd.DataFrame({
            'Gender':[gender],
            'SeniorCitizen':[1 if senior_citizen=="Yes" else 0],
            'Tenure':[tenure],
            'Contract':[contract],
            'MonthlyCharges':[monthly_charges]
        })
        
        if model and encoder:
            try:
                input_encoded = encoder.transform(input_data)
                pred = model.predict(input_encoded)[0]
                prob = model.predict_proba(input_encoded)[0][1]*100
                st.write("Prediction:", "Churn" if pred=="Yes" else "No Churn")
                st.write(f"Churn Probability: {prob:.1f}%")
            except:
                st.write("Error in prediction.")
        else:
            # Demo logic
            risk = 0
            if contract=="Monthly": risk+=30
            if monthly_charges>70: risk+=20
            if tenure<12: risk+=25
            st.write("Risk Score:", f"{risk}%")
            st.write("Prediction:", "Churn" if risk>50 else "No Churn")

# ==================== Insights ====================
elif page == "Insights":
    st.title("Insights & Recommendations")
    
    st.write("Top Churn Drivers:")
    st.write("- Month-to-Month contracts have higher churn")
    st.write("- Tenure less than 12 months increases churn risk")
    st.write("- Higher monthly charges correlate with higher churn")
