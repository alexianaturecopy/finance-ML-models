"""
Revenue Leakage Detection - Model Training Script
Orchestrates data generation, model training, and evaluation
"""

import os
import pandas as pd
from generate_data import SaaSDataGenerator
from model import RevenueLeakageModel

def ensure_directories():
    """Ensure required directories exist"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    print("✅ Directories verified")

def generate_training_data(num_accounts=1000):
    """Generate synthetic training data"""
    print("\n" + "="*60)
    print("STEP 1: Generating Training Data")
    print("="*60)
    
    generator = SaaSDataGenerator(num_accounts=num_accounts)
    df = generator.generate_complete_dataset()
    
    # Save datasets
    df.to_csv('data/training_data.csv', index=False)
    df.head(100).to_csv('data/sample_accounts.csv', index=False)
    
    return df

def train_models(df):
    """Train ML models"""
    print("\n" + "="*60)
    print("STEP 2: Training ML Models")
    print("="*60)
    
    model = RevenueLeakageModel()
    model.train(df)
    model.save_model('saved_models/revenue_leakage_model.pkl')
    
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
        test_sample[['account_id', 'tier', 'mrr', 'has_leakage', 'monthly_leakage_amount']].reset_index(drop=True),
        predictions.reset_index(drop=True)
    ], axis=1)
    
    # Calculate accuracy metrics
    accuracy = (results['has_leakage'] == results['predicted_leakage']).mean()
    print(f"\nClassification Accuracy: {accuracy:.1%}")
    
    # For accounts with leakage, check amount prediction
    leakage_actual = results[results['has_leakage'] == 1]
    if len(leakage_actual) > 0:
        amount_error = (leakage_actual['monthly_leakage_amount'] - leakage_actual['predicted_monthly_amount']).abs().mean()
        print(f"Average Amount Prediction Error: ${amount_error:.2f}")
    
    # Calculate total potential savings
    total_identified = results[results['predicted_leakage'] == 1]['predicted_annual_amount'].sum()
    print(f"\nTotal Annual Leakage Identified: ${total_identified:,.0f}")
    
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
    print("\n📊 Top Predictive Features (Classification):")
    class_features = model.get_feature_importance('classification')
    for i, feat in enumerate(class_features[:5], 1):
        print(f"{i}. {feat['feature']}: {feat['importance']:.4f}")
    
    # Leakage by tier
    tier_analysis = df[df['has_leakage']].groupby('tier').agg({
        'monthly_leakage_amount': ['count', 'sum', 'mean']
    }).round(2)
    
    print("\n💰 Leakage by Subscription Tier:")
    print(tier_analysis)
    
    # Leakage by type
    print("\n🔍 Leakage by Type:")
    type_summary = df[df['has_leakage']].groupby('leakage_type').agg({
        'monthly_leakage_amount': 'sum',
        'account_id': 'count'
    }).sort_values('monthly_leakage_amount', ascending=False)
    type_summary.columns = ['Total Monthly Leakage ($)', 'Account Count']
    print(type_summary)
    
    # Detection difficulty
    print("\n🎯 Detection Difficulty Distribution:")
    difficulty = df[df['has_leakage']]['detection_difficulty'].value_counts()
    print(difficulty)

def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("  REVENUE LEAKAGE DETECTION - MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Ensure directories exist
    ensure_directories()
    
    # Step 1: Generate data
    df = generate_training_data(num_accounts=1000)
    
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
    print("  📁 data/training_data.csv - Full training dataset (1000 accounts)")
    print("  📁 data/sample_accounts.csv - Sample data for dashboard (100 accounts)")
    print("  📁 data/model_evaluation_results.csv - Model performance evaluation")
    print("  📁 saved_models/revenue_leakage_model.pkl - Trained ML model")
    print("\nNext Steps:")
    print("  🚀 Run: streamlit run dashboard.py")
    print("  📊 Upload data/sample_accounts.csv in the dashboard")
    print("  🔍 Test predictions on your accounts!")
    print("="*70)

if __name__ == "__main__":
    main()
