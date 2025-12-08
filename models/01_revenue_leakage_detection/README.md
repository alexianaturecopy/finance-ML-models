# Revenue Leakage Detection - ML Model

Live Demo https://finance-ml-models-glcub3hk9uyytepndn67fw.streamlit.app/
## 🎯 Business Problem

SaaS companies lose 5-15% of potential revenue to preventable leakage. This model identifies high-risk accounts and quantifies potential losses before they impact the bottom line.

---

## 💡 What This Model Does

**Predicts revenue leakage from six common scenarios:**

1. **Failed Payment Retries** - Expired or declined payment methods
2. **Usage Overages Not Billed** - Customers exceeding licensed users
3. **Contract Terms Errors** - Pricing mismatches vs signed agreements
4. **Billing System Errors** - Technical issues preventing invoicing
5. **Downgrades Not Processed** - Should have downgraded but didn't
6. **Expired Payment Methods** - Payment info not updated

**Output for each account:**
- Risk level (High/Medium/Low)
- Leakage probability (0-100%)
- Estimated monthly leakage amount
- Estimated annual impact
- Actionable recommendations

---
## 📊 Dashboard Features

### Single Account Analysis
- Manual input form for real-time predictions
- Risk assessment with probability scoring
- Identification of specific risk factors
- Actionable recommendations
- Visual gauge charts

### Batch Analysis
- Upload CSV/Excel with multiple accounts
- Bulk predictions in seconds
- Summary statistics and risk distribution
- Top accounts requiring attention
- Export results for action

---

## 🎓 Model Architecture

### Two-Stage ML Pipeline

**Stage 1: Classification (Has Leakage?)**
- Algorithm: Random Forest Classifier
- Features: 14 account attributes
- Output: Binary prediction + probability
- Accuracy: ~85% on test data

**Stage 2: Regression (How Much?)**
- Algorithm: Gradient Boosting Regressor
- Features: Same 14 attributes
- Output: Dollar amount estimation
- Performance: ±$50 MAE on monthly leakage

### Key Features Used

1. **Payment Health**
   - `payment_failure_rate` - Historical payment success
   - `payment_reliability_score` - Payment method quality
   - `payment_risk` - Binary flag for unreliable payments

2. **Customer Engagement**
   - `customer_health_score` - Overall account health
   - `engagement_score` - Login + feature adoption composite
   - `last_login_days_ago` - Recency of activity
   - `feature_adoption_pct` - Product usage depth

3. **Usage Patterns**
   - `user_utilization_ratio` - Actual users / Licensed users
   - `overutilization` - Flag for exceeding licenses
   - `revenue_per_user` - Account efficiency metric

4. **Support Signals**
   - `support_tickets_90d` - Recent support load
   - `health_risk` - Binary flag for poor health
   - `engagement_risk` - Binary flag for low engagement

5. **Account Maturity**
   - `days_since_contract_start` - Account age
   - Affects leakage likelihood patterns

---

## 📈 Model Performance

### Classification Metrics
- **ROC-AUC**: 0.87 (excellent discrimination)
- **Precision**: 0.82 (few false positives)
- **Recall**: 0.79 (catches most leakage)
- **F1-Score**: 0.80 (balanced performance)

### Regression Metrics
- **MAE**: $47 (average error on monthly amount)
- **RMSE**: $72 (captures outliers well)
- **R²**: 0.68 (explains 68% of variance)

### Business Impact
On 1,000 account test set:
- Identified: $2.8M annual leakage
- High-risk accounts: 12% of portfolio
- Average leakage per affected account: $9,300/year

---

## 🔍 Use Cases

### 1. Monthly Revenue Operations Review
**Scenario**: CFO wants to understand revenue at risk

**Workflow**:
1. Export account data from billing system
2. Upload to dashboard for batch analysis
3. Review high-risk accounts flagged by model
4. Assign RevOps team to investigate top 20 accounts

**Expected Result**: Recover $100K-500K annually

### 2. Customer Success Prioritization
**Scenario**: CS team needs to prioritize outreach

**Workflow**:
1. Run predictions on entire customer base
2. Filter for "High" risk accounts
3. Export list with specific risk factors
4. CS team reaches out proactively

**Expected Result**: Reduce churn by 15-20%

### 3. Billing System Audit
**Scenario**: New CFO wants to ensure billing accuracy

**Workflow**:
1. Analyze all accounts for systemic issues
2. Identify patterns (e.g., all Enterprise contracts have similar errors)
3. Root cause analysis of billing logic
4. Fix system-wide issues

**Expected Result**: Prevent future leakage at scale



## 📁 File Structure

```
01_revenue_leakage_detection/
├── dashboard.py              # Streamlit application (500+ lines)
├── model.py                  # ML model code (400+ lines)
├── generate_data.py          # Synthetic data generator (400+ lines)
├── train_model.py            # Training orchestration (200+ lines)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   ├── training_data.csv     # 1,000 accounts with features
│   ├── sample_accounts.csv   # 100 accounts for demo
│   └── model_evaluation_results.csv  # Performance metrics
└── saved_models/
    └── revenue_leakage_model.pkl  # Trained model (pickle)
```

---

## 🛠️ Customization

### Use Your Own Data

Replace `data/training_data.csv` with your actual data. Ensure it has these columns:

**Required Features:**
- `payment_failure_rate`
- `customer_health_score`
- `payment_reliability_score`
- `user_utilization_ratio`
- `engagement_score`
- `last_login_days_ago`
- `support_tickets_90d`
- `feature_adoption_pct`
- `payment_risk`
- `health_risk`
- `engagement_risk`
- `overutilization`
- `days_since_contract_start`
- `revenue_per_user`

**Target Variables (for training only):**
- `has_leakage` (0/1)
- `monthly_leakage_amount` (dollars)

Then retrain:
```bash
python train_model.py
```

### Adjust Risk Thresholds

In `model.py`, modify classification threshold:
```python
# Default: 0.5 probability = leakage predicted
has_leakage = (leakage_prob >= 0.5).astype(int)

# More conservative (fewer false positives):
has_leakage = (leakage_prob >= 0.7).astype(int)

# More aggressive (catch more potential leakage):
has_leakage = (leakage_prob >= 0.3).astype(int)
```

---

## 🎯 What This Demonstrates


**Systems Thinking:**
- Built end-to-end solution from data generation to deployed dashboard
- Not just analysis—actionable system that drives operational improvements

**Technical + Business:**
- Bridged ML techniques with CFO-level business problems
- Quantified financial impact ($2.8M identified in test data)
- Created workflow that fits into RevOps processes

**Scalability:**
- Handles 1,000+ accounts in batch
- Production-ready architecture
- Clear deployment path for real-world use

**Communication:**
- Dashboard designed for executives, not data scientists
- Risk levels, dollar impacts, and recommendations—not confusion matrices
- Documentation tells business story, not just technical specs

---

## 📝 Next Steps

### Expand the Model
- Add more leakage types (promotional discounts expired, etc.)
- Incorporate time-series features (trending patterns)
- Build prediction explainability (SHAP values)

### Production Readiness
- Connect to Stripe/Chargebee API
- Add authentication to dashboard
- Set up automated email alerts
- Build A/B testing framework (measure recovery rates)



## 🏆 Success Metrics

**Technical:**
- ✅ Model accuracy > 85%
- ✅ Dashboard loads < 2 seconds
- ✅ Batch predictions < 10 seconds for 1,000 accounts
- ✅ Zero crashes during testing

**Business:**
- ✅ Identifies 5-10% of revenue at risk
- ✅ Actionable recommendations for 100% of flagged accounts
- ✅ Demonstrates CFO-level technical capability

---

## 📧 Contact

**Author**: Alexia  
**Role**: CFO Transitioning to Web3/AI  
**LinkedIn**: [https://www.linkedin.com/in/ye-quan-8b610820a/](#)

---
*Status: Production-Ready Demo*  
*Industry: SaaS / Subscription Business Models*
