"""
Claims Denial Prediction Dashboard
Interactive prediction tool for insurance claims approval/denial
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from model import ClaimsDenialModel

# Page config
st.set_page_config(page_title="Claims Denial Prediction", page_icon="📋", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    h1 {color: #1e3a8a; font-weight: 700;}
    h2 {color: #2563eb; font-weight: 600;}
    h3 {color: #3b82f6;}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = Path(__file__).parent / 'saved_models' / 'claims_denial_model.pkl'
        model = ClaimsDenialModel.load_model(str(model_path))
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model()

# Title
st.title("📋 Insurance Claims Denial Predictor")
st.markdown("AI-powered risk assessment for insurance claims approval")

# Sidebar
with st.sidebar:
    st.header("About This Model")
    if model and not error:
        st.success("✅ Model Loaded")
        st.metric("Training Samples", "1,000")
    else:
        st.error(f"❌ Model Error: {error}")
    
    st.markdown("---")
    st.markdown("**Built by:** Alexia")
    st.markdown("**Model:** Random Forest Classifier")
    st.markdown("**Problem:** Predict claims denial risk")

# Main content
if not model or error:
    st.error("Please train the model first: `python train_model.py`")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Single Claim", "📁 Batch Analysis", "ℹ️ About"])

with tab1:
    st.header("Single Claim Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Claim Information")
        claim_amount = st.number_input("Claim Amount ($)", 
            min_value=0, max_value=100000, value=5000, step=100)
        
        payer = st.selectbox("Insurance Payer", 
            ["BlueCross", "Aetna", "UnitedHealth", "Cigna", "Medicare", "Medicaid"])
        
        cpt_code = st.selectbox("CPT Code", [
            "99213 - Office Visit",
            "99214 - Complex Visit",
            "99223 - Hospital Admission",
            "71020 - Chest X-Ray",
            "80053 - Lab Panel",
            "93000 - EKG",
            "29881 - Arthroscopy",
            "27447 - Knee Replacement"
        ])
        
        days_to_submission = st.number_input("Days from Service to Submission", 
            min_value=0, max_value=180, value=15)
    
    with col2:
        st.subheader("Compliance Checklist")
        prior_auth = st.checkbox("Prior Authorization Obtained", value=True)
        documentation = st.checkbox("Documentation Complete", value=True)
        correct_coding = st.checkbox("Correct Coding Verified", value=True)
        in_network = st.checkbox("In-Network Provider", value=True)
        timely_filed = days_to_submission <= 30
        
        st.info(f"{'✅' if timely_filed else '❌'} Timely Filing: {days_to_submission} days")
        
        provider_denial_rate = st.slider("Provider Historical Denial Rate (%)", 
            0, 50, 10)
    
    if st.button("🔮 Predict Denial Risk", type="primary"):
        # Calculate derived features
        high_amount_flag = 1 if claim_amount > 10000 else 0
        risk_score = (
            (0 if prior_auth else 1) * 30 +
            (0 if documentation else 1) * 25 +
            (0 if correct_coding else 1) * 20 +
            (0 if in_network else 1) * 15 +
            (0 if timely_filed else 1) * 25
        )
        
        # Prepare input
        claim_data = {
            'claim_amount': claim_amount,
            'days_to_submission': days_to_submission,
            'prior_auth_obtained': 1 if prior_auth else 0,
            'documentation_complete': 1 if documentation else 0,
            'correct_coding': 1 if correct_coding else 0,
            'in_network': 1 if in_network else 0,
            'timely_filed': 1 if timely_filed else 0,
            'provider_denial_rate': provider_denial_rate,
            'high_amount_flag': high_amount_flag,
            'risk_score': risk_score
        }
        
        # Predict
        result = model.predict(claim_data)
        will_deny = result['will_deny'].iloc[0]
        denial_prob = result['denial_probability'].iloc[0]
        risk_level = result['risk_level'].iloc[0]
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if will_deny:
                st.error("❌ **Likely DENIAL**")
            else:
                st.success("✅ **Likely APPROVAL**")
        
        with col2:
            st.metric("Denial Probability", f"{denial_prob*100:.1f}%")
        
        with col3:
            risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            st.info(f"{risk_colors.get(risk_level, '⚪')} **{risk_level} Risk**")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=denial_prob*100,
            title={'text': "Denial Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "#d1fae5"},
                    {'range': [30, 70], 'color': "#fef3c7"},
                    {'range': [70, 100], 'color': "#fee2e2"}
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
        
        # Risk factors
        st.subheader("⚠️ Risk Factors Identified")
        risk_factors = []
        if not prior_auth:
            risk_factors.append("🔴 Missing prior authorization (30% risk)")
        if not documentation:
            risk_factors.append("🔴 Incomplete documentation (25% risk)")
        if not correct_coding:
            risk_factors.append("🟡 Coding not verified (20% risk)")
        if not in_network:
            risk_factors.append("🟡 Out-of-network provider (15% risk)")
        if not timely_filed:
            risk_factors.append("🔴 Late filing (25% risk)")
        if high_amount_flag:
            risk_factors.append("🟡 High claim amount (increases scrutiny)")
        if provider_denial_rate > 15:
            risk_factors.append(f"🟡 Provider has {provider_denial_rate}% denial rate")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(factor)
        else:
            st.success("✅ No major risk factors identified!")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if denial_prob >= 0.7:
            st.error("🚨 **High Risk - DO NOT SUBMIT YET**")
            st.write("**Required Actions Before Submission:**")
            if not prior_auth:
                st.write("- ❗ Obtain prior authorization immediately")
            if not documentation:
                st.write("- ❗ Complete all required documentation")
            if not correct_coding:
                st.write("- ❗ Verify CPT codes with coding specialist")
            if not in_network:
                st.write("- ❗ Check if single-case agreement possible")
            if not timely_filed:
                st.write("- ❗ Document reason for late filing")
        elif denial_prob >= 0.3:
            st.warning("⚠️ **Medium Risk - Review Recommended**")
            st.write("- Double-check all documentation")
            st.write("- Have coding specialist review")
            st.write("- Monitor closely after submission")
        else:
            st.success("✅ **Low Risk - Proceed with Submission**")
            st.write("- Submit claim as normal")
            st.write("- Standard follow-up procedures")

with tab2:
    st.header("Batch Analysis")
    st.write("Upload a CSV file with claims data for batch predictions")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} claims")
            
            if st.button("🔮 Run Predictions", type="primary"):
                with st.spinner("Analyzing claims..."):
                    # Ensure required columns
                    required_cols = model.feature_cols
                    missing_cols = set(required_cols) - set(df.columns)
                    
                    if missing_cols:
                        st.error(f"Missing columns: {missing_cols}")
                    else:
                        results = model.predict(df)
                        df['will_deny'] = results['will_deny']
                        df['denial_probability'] = results['denial_probability']
                        df['risk_level'] = results['risk_level']
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Claims", len(df))
                        with col2:
                            high_risk = (results['risk_level'] == 'High').sum()
                            st.metric("High Risk", high_risk)
                        with col3:
                            will_deny = results['will_deny'].sum()
                            st.metric("Likely Denials", will_deny)
                        with col4:
                            at_risk_amount = df[results['will_deny']]['claim_amount'].sum()
                            st.metric("Amount at Risk", f"${at_risk_amount:,.0f}")
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Risk distribution
                            risk_counts = results['risk_level'].value_counts()
                            fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                                       title="Risk Distribution",
                                       color_discrete_map={
                                           'Low': '#10b981',
                                           'Medium': '#f59e0b',
                                           'High': '#dc2626'
                                       })
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Denial prediction
                            denial_counts = results['will_deny'].value_counts()
                            fig = px.bar(x=['Will Approve', 'Will Deny'], 
                                       y=[denial_counts.get(False, 0), denial_counts.get(True, 0)],
                                       title="Denial Predictions",
                                       color=['Will Approve', 'Will Deny'],
                                       color_discrete_map={'Will Approve': '#10b981', 'Will Deny': '#dc2626'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Top 10 high-risk
                        st.subheader("🚨 Top 10 High-Risk Claims")
                        high_risk_df = df.sort_values('denial_probability', ascending=False).head(10)
                        display_cols = ['claim_id', 'claim_amount', 'denial_probability', 
                                      'risk_level', 'payer']
                        available_cols = [col for col in display_cols if col in high_risk_df.columns]
                        st.dataframe(high_risk_df[available_cols], use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="claims_denial_predictions.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab3:
    st.header("About This Model")
    
    st.markdown("""
    ### 🎯 Business Problem
    Insurance claims denial rates of 10-15% create significant cash flow problems for healthcare providers. 
    This model predicts denial risk BEFORE submission, enabling:
    
    - **Pre-submission correction** of likely denials
    - **Resource optimization** in claims management
    - **Cash flow improvement** through first-pass approvals
    - **Reduced administrative burden** from appeals
    
    ### 🤖 How It Works
    
    The model analyzes **10 key factors**:
    - Prior authorization status
    - Documentation completeness
    - Coding accuracy
    - Network status
    - Timeliness of filing
    - Claim amount
    - Provider denial history
    - Days to submission
    - Compliance risk score
    
    ### 📊 Model Performance
    
    - **Training Dataset**: 1,000 claims
    - **Binary Classification**: Approve / Deny
    - **Real-time predictions**: < 100ms per claim
    
    ### 💼 Business Impact
    
    **Example: $50M in Annual Claims**
    - Current denial rate: 12% = $6M denied
    - Pre-submission intervention on high-risk claims
    - Reduce denial rate to 6% = **$3M recovered annually**
    - Administrative savings: **$500K** (reduced appeals)
    - **Total Impact**: $3.5M per year
    
    ### 🎯 Use Cases
    
    1. **Pre-Submission Review**
       - Score all claims before submission
       - Flag high-risk claims for review
       - Correct issues proactively
    
    2. **Daily Claims Scrubbing**
       - Automated quality assurance
       - Coding verification
       - Documentation completeness check
    
    3. **Provider Education**
       - Identify providers with high denial risk
       - Targeted training on common issues
       - Improve submission quality
    
    4. **Cash Flow Forecasting**
       - Predict approval rates
       - Estimate collection timelines
       - Optimize working capital
    
    ### 🔧 Technical Details
    
    - **Algorithm**: Random Forest Classifier
    - **Features**: 10 compliance and risk indicators
    - **Deployment**: Streamlit Cloud
    - **Integration**: Can integrate with claims management systems
    
    ### 📈 Common Denial Reasons Predicted
    
    1. **Missing Prior Authorization** (30% of denials)
    2. **Incomplete Documentation** (25% of denials)
    3. **Coding Errors** (20% of denials)
    4. **Timely Filing Limits** (15% of denials)
    5. **Out of Network** (10% of denials)
    
    ### 🎓 Training & Validation
    
    - **Training Set**: 800 claims (80%)
    - **Test Set**: 200 claims (20%)
    - **Stratified sampling** to ensure balanced classes
    - **Cross-validation** for robust performance
    
    ---
    
    **Built by**: Alexia | CFO Portfolio Project  
    **Technology**: Python, Scikit-learn, Streamlit  
    **Source Code**: [GitHub](https://github.com/alexianaturecopy/finance-ML-models)
    """)

# Footer
st.markdown("---")
st.markdown("*This is a demonstration model using synthetic data. For production use, train on actual claims data.*")
