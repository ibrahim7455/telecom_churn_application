import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# تكوين الصفحة
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحميل البيانات والموديل
@st.cache_data
def load_data():
    df = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/TelecomCustomerChurn.csv')
    return df

@st.cache_resource
def load_model():
    try:
        with open('random_forest_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('ordinal_encoder.pkl', 'rb') as file:
            encoder = pickle.load(file)
        return model, encoder
    except:
        st.warning("⚠️ الموديل غير موجود. سيتم استخدام موديل افتراضي للتوضيح.")
        return None, None

df = load_data()
model, encoder = load_model()

# CSS مخصص
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .churn-yes {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .churn-no {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar للتنقل
st.sidebar.title("📱 Telecom Churn Analysis")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Prediction", "💡 Insights & Recommendations"])

# ==================== صفحة Dashboard ====================
if page == "📊 Dashboard":
    st.title("📊 Customer Churn Analytics Dashboard")
    st.markdown("---")
    
    # المقاييس الأساسية
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    churned_customers = len(df[df['Churn'] == 'Yes'])
    churn_rate = (churned_customers / total_customers) * 100
    avg_monthly = df['MonthlyCharges'].mean()
    
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("Churned Customers", f"{churned_customers:,}", f"{churn_rate:.1f}%")
    with col3:
        st.metric("Retention Rate", f"{100-churn_rate:.1f}%")
    with col4:
        st.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
    
    st.markdown("---")
    
    # الرسوم البيانية
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn Distribution Pie Chart
        churn_counts = df['Churn'].value_counts()
        fig_pie = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title="Churn Distribution",
            color_discrete_sequence=['#66bb6a', '#ef5350'],
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Contract Types Distribution
        contract_counts = df['Contract'].value_counts()
        fig_contract = px.pie(
            values=contract_counts.values,
            names=contract_counts.index,
            title="Contract Types Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        st.plotly_chart(fig_contract, use_container_width=True)
    
    # Monthly Charges vs Churn
    col1, col2 = st.columns(2)
    
    with col1:
        fig_box = px.box(
            df,
            x='Churn',
            y='MonthlyCharges',
            title="Monthly Charges vs Churn",
            color='Churn',
            color_discrete_map={'Yes': '#ef5350', 'No': '#66bb6a'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        fig_hist = px.histogram(
            df,
            x='MonthlyCharges',
            nbins=50,
            title="Distribution of Monthly Charges",
            color_discrete_sequence=['#42a5f5']
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Payment Method Analysis
    st.subheader("💳 Payment Method Analysis")
    payment_churn = df.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='Count')
    fig_payment = px.bar(
        payment_churn,
        x='PaymentMethod',
        y='Count',
        color='Churn',
        title="Payment Method vs Churn",
        barmode='group',
        color_discrete_map={'Yes': '#ef5350', 'No': '#66bb6a'}
    )
    st.plotly_chart(fig_payment, use_container_width=True)
    
    # Features Analysis
    st.subheader("📈 Features Analysis")
    features = ['TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling']
    
    cols = st.columns(3)
    for idx, feature in enumerate(features):
        with cols[idx % 3]:
            feature_churn = df.groupby([feature, 'Churn']).size().reset_index(name='Count')
            fig = px.bar(
                feature_churn,
                x=feature,
                y='Count',
                color='Churn',
                title=f"{feature} vs Churn",
                barmode='group',
                color_discrete_map={'Yes': '#ef5350', 'No': '#66bb6a'},
                height=400
            )
            fig.update_xaxis(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# ==================== صفحة التنبؤ ====================
elif page == "🔮 Prediction":
    st.title("🔮 Customer Churn Prediction")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Customer Information")
        
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    
    with col2:
        st.subheader("Additional Services")
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        st.subheader("Billing Information")
        contract = st.selectbox("Contract", ["Monthly", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
            ["Manual", "Bank transfer (automatic)", "Credit card (automatic)", "Mailed check"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0, step=10.0)
    
    st.markdown("---")
    
    if st.button("🔍 Predict Churn", type="primary", use_container_width=True):
        # إعداد البيانات للتنبؤ
        input_data = pd.DataFrame({
            'Gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'Tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        if model and encoder:
            # تطبيق الترميز والتنبؤ
            try:
                input_encoded = encoder.transform(input_data)
                prediction = model.predict(input_encoded)[0]
                prediction_proba = model.predict_proba(input_encoded)[0]
                
                churn_probability = prediction_proba[1] * 100
                
                st.markdown("### Prediction Result")
                
                if prediction == 'Yes':
                    st.markdown(f"""
                        <div class="prediction-box churn-yes">
                            ⚠️ High Risk of Churn<br>
                            Probability: {churn_probability:.1f}%
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="prediction-box churn-no">
                            ✅ Low Risk of Churn<br>
                            Retention Probability: {100-churn_probability:.1f}%
                        </div>
                    """, unsafe_allow_html=True)
                
                # عرض احتمالية التنبؤ
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Churn Probability", f"{churn_probability:.1f}%")
                with col2:
                    st.metric("Retention Probability", f"{100-churn_probability:.1f}%")
                    
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
        else:
            # محاكاة للتنبؤ إذا لم يكن الموديل موجودًا
            risk_score = 0
            if contract == "Monthly":
                risk_score += 30
            if monthly_charges > 70:
                risk_score += 20
            if tenure < 12:
                risk_score += 25
            if paperless_billing == "Yes":
                risk_score += 15
            if internet_service == "Fiber optic":
                risk_score += 10
                
            st.markdown("### Prediction Result (Demo Mode)")
            
            if risk_score > 50:
                st.markdown(f"""
                    <div class="prediction-box churn-yes">
                        ⚠️ High Risk of Churn<br>
                        Risk Score: {risk_score}%
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-box churn-no">
                        ✅ Low Risk of Churn<br>
                        Risk Score: {risk_score}%
                    </div>
                """, unsafe_allow_html=True)

# ==================== صفحة Insights ====================
elif page == "💡 Insights & Recommendations":
    st.title("💡 Insights & Recommendations")
    st.markdown("---")
    
    # Key Insights
    st.header("🔍 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Churn Drivers")
        
        insights = [
            {
                "title": "Contract Type Impact",
                "description": "Month-to-month contracts show significantly higher churn rates compared to yearly contracts.",
                "metric": f"{(df[df['Contract']=='Monthly']['Churn']=='Yes').mean()*100:.1f}% churn rate"
            },
            {
                "title": "Tenure Effect",
                "description": "Customers with tenure less than 12 months are at highest risk of churning.",
                "metric": f"{(df[df['Tenure']<12]['Churn']=='Yes').mean()*100:.1f}% churn rate"
            },
            {
                "title": "Monthly Charges",
                "description": "Customers paying higher monthly charges have increased churn probability.",
                "metric": f"${df[df['Churn']=='Yes']['MonthlyCharges'].mean():.2f} avg"
            }
        ]
        
        for insight in insights:
            with st.container():
                st.markdown(f"**{insight['title']}**")
                st.info(insight['description'])
                st.metric("", insight['metric'])
                st.markdown("---")
    
    with col2:
        st.subheader("🎯 Risk Factors")
        
        # حساب عوامل الخطورة
        risk_factors = []
        
        # تحليل العقود
        monthly_churn = (df[df['Contract']=='Monthly']['Churn']=='Yes').mean()
        risk_factors.append(("Month-to-Month Contract", monthly_churn * 100))
        
        # تحليل الدعم الفني
        no_tech_churn = (df[df['TechSupport']=='No']['Churn']=='Yes').mean()
        risk_factors.append(("No Tech Support", no_tech_churn * 100))
        
        # تحليل طريقة الدفع
        manual_churn = (df[df['PaymentMethod']=='Manual']['Churn']=='Yes').mean()
        risk_factors.append(("Manual Payment", manual_churn * 100))
        
        # تحليل الإنترنت
        fiber_churn = (df[df['InternetService']=='Fiber optic']['Churn']=='Yes').mean()
        risk_factors.append(("Fiber Optic Service", fiber_churn * 100))
        
        # عرض الرسم البياني
        risk_df = pd.DataFrame(risk_factors, columns=['Factor', 'Churn Rate'])
        fig = px.bar(
            risk_df,
            x='Churn Rate',
            y='Factor',
            orientation='h',
            title="Top Risk Factors for Churn",
            color='Churn Rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recommendations
    st.header("💼 Strategic Recommendations")
    
    recommendations = [
        {
            "icon": "🎁",
            "title": "Contract Incentives",
            "description": "Offer attractive discounts and benefits for customers switching to annual or biennial contracts.",
            "priority": "High",
            "impact": "Reduce churn by 15-20%"
        },
        {
            "icon": "👥",
            "title": "Early Engagement Program",
            "description": "Implement proactive retention program for customers in first 12 months with personalized support.",
            "priority": "High",
            "impact": "Reduce new customer churn by 25%"
        },
        {
            "icon": "🛡️",
            "title": "Tech Support Enhancement",
            "description": "Provide free tech support for first 6 months and promote its value to reduce technical issues.",
            "priority": "Medium",
            "impact": "Improve satisfaction by 30%"
        },
        {
            "icon": "💳",
            "title": "Payment Automation",
            "description": "Encourage automatic payment methods with small incentives or rewards programs.",
            "priority": "Medium",
            "impact": "Reduce payment-related churn by 10%"
        },
        {
            "icon": "📦",
            "title": "Service Bundle Optimization",
            "description": "Create competitive bundles combining internet, TV, and support services at attractive prices.",
            "priority": "High",
            "impact": "Increase customer lifetime value by 20%"
        },
        {
            "icon": "📊",
            "title": "Predictive Monitoring",
            "description": "Use this ML model to identify at-risk customers monthly and trigger retention campaigns.",
            "priority": "High",
            "impact": "Prevent 40% of predicted churns"
        }
    ]
    
    col1, col2 = st.columns(2)
    
    for idx, rec in enumerate(recommendations):
        with col1 if idx % 2 == 0 else col2:
            with st.container():
                st.markdown(f"### {rec['icon']} {rec['title']}")
                st.write(rec['description'])
                
                cols = st.columns([1, 1])
                with cols[0]:
                    priority_color = {
                        "High": "🔴",
                        "Medium": "🟡",
                        "Low": "🟢"
                    }
                    st.markdown(f"**Priority:** {priority_color[rec['priority']]} {rec['priority']}")
                with cols[1]:
                    st.markdown(f"**Impact:** {rec['impact']}")
                
                st.markdown("---")
    
    # Action Plan
    st.header("📋 Implementation Roadmap")
    
    timeline = pd.DataFrame({
        'Phase': ['Phase 1\n(Month 1-2)', 'Phase 2\n(Month 3-4)', 'Phase 3\n(Month 5-6)', 'Phase 4\n(Month 7-12)'],
        'Actions': [
            'Deploy predictive model\nLaunch contract incentives',
            'Implement early engagement\nEnhance tech support',
            'Optimize service bundles\nPromote auto-payment',
            'Monitor and optimize\nScale successful programs'
        ],
        'Expected Impact': [15, 25, 35, 45]
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.line(
            timeline,
            x='Phase',
            y='Expected Impact',
            title='Expected Cumulative Impact on Churn Reduction (%)',
            markers=True,
            text='Expected Impact'
        )
        fig.update_traces(textposition='top center', line_color='#42a5f5', marker_size=12)
        fig.update_layout(yaxis_range=[0, 50])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Target Goals")
        st.metric("Current Churn Rate", f"{churn_rate:.1f}%")
        st.metric("Target Churn Rate", f"{churn_rate*0.55:.1f}%", f"-{churn_rate*0.45:.1f}%")
        st.metric("Expected Revenue Impact", "$2.5M+", "annually")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📱 Telecom Customer Churn Prediction System | Built with Streamlit & Machine Learning</p>
    </div>
""", unsafe_allow_html=True)
