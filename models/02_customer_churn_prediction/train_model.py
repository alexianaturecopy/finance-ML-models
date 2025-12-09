"""
Customer Churn Prediction - Model Training Script
Orchestrates data generation, model training, and evaluation
"""

import os
import pandas as pd
from generate_data import ChurnDataGenerator
from model import ChurnPredictionModel

def ensure_directories():
    """Ensure required directories exist"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    print("✅ Directories verified")

def generate_training_data(num_customers=1000):
    """Generate synthetic training data"""
    print("\n" + "="*60)
    print("STEP 1: Generating Training Data")
    print("="*60)
    
    generator = ChurnDataGenerator(num_customers=num_customers)
    df = generator.generate_complete_dataset()
    
    # Save datasets
    df.to_csv('data/training_data.csv', index=False)
    df.head(100).to_csv('data/sample_customers.csv', index=False)
    
    return df

def train_models(df):
    """Train ML models"""
    print("\n" + "="*60)
    print("STEP 2: Training ML Model")
    print("="*60)
    
    model = ChurnPredictionModel()
    model.train(df)
    model.save_model('saved_models/churn_prediction_model.pkl')
    
    return model

def evaluate_model(model, df):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print("STEP 3: Model Evaluation")
    print("="*60)
    
    # Test on holdout sample
    test_sample = df.sample(n=100, random_state=42)
    predictions = model.predict(test_sample)
    
    # Combine with actual values
    results = pd.concat([
        test_sample[['customer_id', 'tier', 'status', 'overall_health_score']].reset_index(drop=True),
        predictions.reset_index(drop=True)
    ], axis=1)
    
    # Calculate accuracy metrics
    accuracy = (results['status'] == results['predicted_status']).mean()
    print(f"\nPrediction Accuracy: {accuracy:.1%}")
    
    # Status-specific accuracy
    print("\nAccuracy by Status:")
    for status in results['status'].unique():
        status_df = results[results['status'] == status]
        status_acc = (status_df['status'] == status_df['predicted_status']).mean()
        print(f"  {status}: {status_acc:.1%} ({len(status_df)} customers)")
    
    # Risk distribution
    print(f"\nRisk Distribution:")
    print(results['risk_level'].value_counts())
    
    # Save evaluation results
    results.to_csv('data/model_evaluation_results.csv', index=False)
    print("\n✅ Evaluation results saved to: data/model_evaluation_results.csv")
    
    return results

def generate_insights(model, df):
    """Generate business insights from model"""
    print("\n" + "="*60)
    print("STEP 4: Business Insights")
    print("="*60)
    
    # Feature importance
    print("\n📊 Top Predictive Features:")
    features = model.get_feature_importance()
    for i, feat in enumerate(features[:5], 1):
        print(f"{i}. {feat['feature']}: {feat['importance']:.4f}")
    
    # Churn analysis by tier
    print("\n💰 Churn Analysis by Tier:")
    churn_by_tier = df[df['status'] == 'Churned'].groupby('tier').size()
    total_by_tier = df.groupby('tier').size()
    churn_rate_by_tier = (churn_by_tier / total_by_tier * 100).round(1)
    
    for tier in churn_rate_by_tier.index:
        print(f"  {tier}: {churn_rate_by_tier[tier]:.1f}% ({churn_by_tier[tier]} churned)")
    
    # Primary churn reasons
    print("\n🔍 Top Churn Reasons:")
    churn_reasons = df[df['status'] == 'Churned']['primary_reason'].value_counts().head(5)
    for reason, count in churn_reasons.items():
        print(f"  {reason}: {count} customers")
    
    # At-risk customers
    at_risk_count = (df['status'] == 'At-Risk').sum()
    at_risk_value = df[df['status'] == 'At-Risk']['mrr'].sum()
    print(f"\n⚠️ At-Risk Customers:")
    print(f"  Count: {at_risk_count}")
    print(f"  Monthly Revenue at Risk: ${at_risk_value:,.0f}")
    print(f"  Annual Revenue at Risk: ${at_risk_value * 12:,.0f}")

def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("  CUSTOMER CHURN PREDICTION - MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Ensure directories exist
    ensure_directories()
    
    # Step 1: Generate data
    df = generate_training_data(num_customers=1000)
    
    # Step 2: Train models
    model = train_models(df)
    
    # Step 3: Evaluate
    results = evaluate_model(model, df)
    
    # Step 4: Insights
    generate_insights(model, df)
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  📁 data/training_data.csv - Full training dataset (1000 customers)")
    print("  📁 data/sample_customers.csv - Sample data for dashboard (100 customers)")
    print("  📁 data/model_evaluation_results.csv - Model performance evaluation")
    print("  📁 saved_models/churn_prediction_model.pkl - Trained ML model")
    print("\nNext Steps:")
    print("  🚀 Run: streamlit run dashboard.py")
    print("  📊 Upload data/sample_customers.csv in the dashboard")
    print("  🔍 Test predictions on your customers!")
    print("="*70)

if __name__ == "__main__":
    main()
