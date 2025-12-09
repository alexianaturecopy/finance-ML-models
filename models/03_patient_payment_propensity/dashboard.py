"""
Patient Payment Propensity Dashboard
Interactive prediction tool for healthcare payment likelihood
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from model import PaymentPropensityModel

# Page config
st.set_page_config(page_title="Patient Payment Propensity", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    h1 {color: #1e3a8a; font-weight: 700;}
    h2 {color: #2563eb; font-weight: 600;}
    h3 {color: #3b82f6;}
    .risk-high {color: #dc2626; font-weight: bold;}
    .risk-medium {color: #f59e0b; font-weight: bold;}
    .risk-low {color: #10b981; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = Path(__file__).parent / 'saved_models' / 'payment_propensity_model.pkl'
        model = PaymentPropensityModel.load_model(str(model_path))
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model()

# Title
st.title("🏥 Patient Payment Propensity Predictor")
st.markdown("AI-powered prediction system for healthcare patient payment likelihood")

# Sidebar
with st.sidebar:
    st.header("About This Model")
    if model and not error:
        st.success("✅ Model Loaded")
        st.metric("Accuracy", f"{model.training_metrics.get('accuracy', 0)*100:.1f}%")
        st.metric("ROC-AUC", f"{model.training_metrics.get('roc_auc', 0):.3f}")
    else:
        st.error(f"❌ Model Error: {error}")
    
    st.markdown("---")
    st.markdown("**Built by:** Alexia")
    st.markdown("**Model:** Random Forest Classifier")
    st.markdown("**Problem:** Predict patient payment likelihood")

# Main content
if not model or error:
    st.error("Please train the model first: `python train_model.py`")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Single Patient", "📁 Batch Analysis", "ℹ️ About"])

with tab1:
    st.header("Single Patient Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=18, max_value=95, value=45)
        income_level = st.selectbox("Income Level", ["High", "Medium", "Low"])
        employed = st.checkbox("Currently Employed", value=True)
        insurance_type = st.selectbox("Insurance Type", 
            ["Private", "Medicare", "Medicaid", "Self-Pay"])
    
    with col2:
        st.subheader("Financial Info")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=680)
        prior_collections = st.number_input("Prior Collections", min_value=0, max_value=10, value=0)
        prior_balance = st.number_input("Prior Balance ($)", min_value=0, max_value=50000, value=0)
        patient_responsibility = st.number_input("Patient Balance ($)", 
            min_value=0, max_value=50000, value=1500)
    
    with col3:
        st.subheader("Payment History")
        payment_history_score = st.slider("Payment History Score", 0, 100, 70)
        typical_days_to_pay = st.number_input("Typical Days to Pay", 
            min_value=0, max_value=180, value=45)
    
    if st.button("🔮 Predict Payment Likelihood", type="primary"):
        # Calculate derived features
        payment_burden_ratio = patient_responsibility / (patient_responsibility + 1000)
        affordability_score = 100 - (payment_burden_ratio * 50 + (850 - credit_score) / 8.5)
        
        insurance_map = {'Private': 80, 'Medicare': 65, 'Medicaid': 55, 'Self-Pay': 30}
        payment_capacity_score = (
            credit_score / 8.5 * 0.35 +
            payment_history_score * 0.35 +
            insurance_map[insurance_type] * 0.30
        )
        
        collection_risk_score = (
            prior_collections * 25 +
            (1 if patient_responsibility > 2000 else 0) * 15 +
            (1 if credit_score < 650 else 0) * 20 +
            (1 if insurance_type == 'Self-Pay' else 0) * 25 +
            (0 if employed else 1) * 15
        )
        
        # Prepare input
        patient_data = {
            'age': age,
            'credit_score': credit_score,
            'prior_collections': prior_collections,
            'patient_responsibility': patient_responsibility,
            'prior_balance': prior_balance,
            'payment_history_score': payment_history_score,
            'typical_days_to_pay': typical_days_to_pay,
            'payment_burden_ratio': payment_burden_ratio,
            'affordability_score': affordability_score,
            'payment_capacity_score': payment_capacity_score,
            'collection_risk_score': collection_risk_score,
            'high_balance_flag': 1 if patient_responsibility > 2000 else 0,
            'prior_collection_flag': 1 if prior_collections > 0 else 0,
            'low_credit_flag': 1 if credit_score < 650 else 0,
            'uninsured_flag': 1 if insurance_type == 'Self-Pay' else 0,
            'unemployed_flag': 0 if employed else 1
        }
        
        # Predict
        result = model.predict(patient_data)
        will_pay = result['will_pay'].iloc[0]
        probability = result['payment_probability'].iloc[0]
        risk_level = result['risk_level'].iloc[0]
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if will_pay:
                st.success("✅ **Will Pay Full**")
            else:
                st.error("❌ **Won't Pay Full**")
        
        with col2:
            st.metric("Payment Probability", f"{probability*100:.1f}%")
        
        with col3:
            risk_color = {"Low-Risk": "🟢", "Medium-Risk": "🟡", "High-Risk": "🔴"}
            st.info(f"{risk_color.get(risk_level, '⚪')} **{risk_level}**")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            title={'text': "Payment Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "#fee2e2"},
                    {'range': [30, 70], 'color': "#fef3c7"},
                    {'range': [70, 100], 'color': "#d1fae5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if probability >= 0.7:
            st.success("✅ **Standard collection process**")
            st.write("- Send standard payment reminder")
            st.write("- Offer payment plan options")
        elif probability >= 0.4:
            st.warning("⚠️ **Enhanced follow-up required**")
            st.write("- Make personal phone call")
            st.write("- Offer financial counseling")
            st.write("- Consider payment arrangements")
        else:
            st.error("🚨 **High-risk account**")
            st.write("- Immediate personal contact required")
            st.write("- Assess financial assistance eligibility")
            st.write("- Consider early collection agency referral")

with tab2:
    st.header("Batch Analysis")
    st.write("Upload a CSV file with patient data for batch predictions")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} patients")
            
            if st.button("🔮 Run Predictions", type="primary"):
                with st.spinner("Analyzing patients..."):
                    # Ensure all required columns exist
                    required_cols = model.feature_cols
                    missing_cols = set(required_cols) - set(df.columns)
                    
                    if missing_cols:
                        st.error(f"Missing columns: {missing_cols}")
                    else:
                        results = model.predict(df)
                        df['will_pay'] = results['will_pay']
                        df['payment_probability'] = results['payment_probability']
                        df['risk_level'] = results['risk_level']
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Patients", len(df))
                        with col2:
                            high_risk = (results['risk_level'] == 'High-Risk').sum()
                            st.metric("High Risk", high_risk)
                        with col3:
                            wont_pay = (~results['will_pay']).sum()
                            st.metric("Won't Pay", wont_pay)
                        with col4:
                            avg_prob = results['payment_probability'].mean()
                            st.metric("Avg Probability", f"{avg_prob*100:.1f}%")
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Risk distribution
                            risk_counts = results['risk_level'].value_counts()
                            fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                                       title="Risk Distribution",
                                       color_discrete_map={
                                           'Low-Risk': '#10b981',
                                           'Medium-Risk': '#f59e0b',
                                           'High-Risk': '#dc2626'
                                       })
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Payment likelihood
                            will_pay_counts = results['will_pay'].value_counts()
                            fig = px.bar(x=['Won\'t Pay', 'Will Pay'], 
                                       y=[will_pay_counts.get(False, 0), will_pay_counts.get(True, 0)],
                                       title="Payment Likelihood",
                                       color=['Won\'t Pay', 'Will Pay'],
                                       color_discrete_map={'Won\'t Pay': '#dc2626', 'Will Pay': '#10b981'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Top 10 high-risk
                        st.subheader("🚨 Top 10 High-Risk Patients")
                        high_risk_df = df.sort_values('payment_probability').head(10)
                        display_cols = ['patient_id', 'patient_responsibility', 'payment_probability', 
                                      'risk_level', 'insurance_type', 'credit_score']
                        available_cols = [col for col in display_cols if col in high_risk_df.columns]
                        st.dataframe(high_risk_df[available_cols], use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="payment_propensity_results.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab3:
    st.header("About This Model")
    
    st.markdown("""
    ### 🎯 Business Problem
    Healthcare providers face significant challenges collecting patient balances, with typical collection 
    rates of only 20-30%. This model predicts which patients are likely to pay their bills, enabling:
    
    - **Targeted collection strategies**
    - **Resource optimization**
    - **Early intervention for high-risk accounts**
    - **Financial assistance program targeting**
    
    ### 🤖 How It Works
    
    The model analyzes **16 factors** including:
    - Demographics (age, employment, income level)
    - Credit indicators (credit score, prior collections)
    - Insurance coverage (type, payment history)
    - Financial burden (balance amount, prior balance)
    - Payment behavior (typical payment timeline, history score)
    
    ### 📊 Model Performance
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{model.training_metrics.get('accuracy', 0)*100:.1f}%")
    with col2:
        st.metric("ROC-AUC", f"{model.training_metrics.get('roc_auc', 0):.3f}")
    with col3:
        st.metric("Training Samples", "1,000")
    
    st.markdown("""
    ### 💼 Business Impact
    
    **Example: $10M in Annual Patient AR**
    - Identify $3M in high-risk accounts
    - Target collection efforts effectively
    - Improve recovery by 30% = **$900K additional collections**
    - ROI: 10-15x implementation cost
    
    ### 🎯 Use Cases
    
    1. **Daily Collection Prioritization**
       - Score all outstanding balances
       - Focus efforts on medium-risk accounts (best ROI)
       - Early intervention for high-risk accounts
    
    2. **Financial Assistance Programs**
       - Identify patients needing assistance
       - Proactive outreach before accounts age
       - Reduce bad debt write-offs
    
    3. **Revenue Cycle Optimization**
       - Predict cash collections
       - Optimize staffing levels
       - Improve days in AR metrics
    
    ### 🔧 Technical Details
    
    - **Algorithm**: Random Forest Classifier
    - **Features**: 16 behavioral and financial indicators
    - **Training**: 1,000 patient records
    - **Validation**: 20% holdout set
    - **Deployment**: Streamlit Cloud
    
    ### 📈 Top Predictive Features
    
    1. Payment Capacity Score (23%)
    2. Affordability Score (14%)
    3. Collection Risk Score (11%)
    4. Payment History Score (11%)
    5. Credit Score (8%)
    
    ---
    
    **Built by**: Alexia | CFO Portfolio Project  
    **Technology**: Python, Scikit-learn, Streamlit  
    **Source Code**: [GitHub](https://github.com/alexianaturecopy/finance-ML-models)
    """)

# Footer
st.markdown("---")
st.markdown("*This is a demonstration model using synthetic data. For production use, train on actual patient data.*")
