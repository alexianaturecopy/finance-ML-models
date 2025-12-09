"""
Customer Churn Prediction - ML Model
Multi-class classification: Active / At-Risk / Churned
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, f1_score
)
import joblib
import json
from datetime import datetime

class ChurnPredictionModel:
    """Multi-class model for predicting customer churn risk"""
    
    def __init__(self):
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_cols = []
        self.feature_importance = {}
        self.training_metrics = {}
        
    def prepare_features(self, df):
        """Prepare features for modeling"""
        # Select feature columns
        self.feature_cols = [
            'tenure_days',
            'engagement_score',
            'support_health_score',
            'payment_health_score',
            'satisfaction_composite',
            'contract_strength',
            'usage_intensity',
            'renewal_risk',
            'low_engagement_flag',
            'support_issues_flag',
            'payment_issues_flag',
            'low_satisfaction_flag',
            'api_calls_per_day',
            'support_tickets_90d',
            'payment_failures'
        ]
        
        # Handle missing values
        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_cols,
            index=X.index
        )
        
        return X_scaled
    
    def train(self, df):
        """Train the churn prediction model"""
        print("=== Customer Churn Prediction - Model Training ===\n")
        print(f"Dataset size: {len(df)} customers")
        
        # Status distribution
        status_dist = df['status'].value_counts()
        print(f"\nStatus Distribution:")
        for status, count in status_dist.items():
            print(f"  {status}: {count} ({count/len(df):.1%})")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Encode target (Active=0, At-Risk=1, Churned=2)
        y = self.label_encoder.fit_transform(df['status'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        print("\n=== Training Classification Model ===")
        self.classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)
        
        # Metrics
        print("\nClassification Report:")
        target_names = self.label_encoder.classes_
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        
        # Feature importance
        feature_imp = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Features:")
        print(feature_imp.head(10))
        
        # Store metrics
        self.training_metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_imp.to_dict('records'),
            'class_distribution': status_dist.to_dict()
        }
        
        self.feature_importance = feature_imp.to_dict('records')
        
        print("\n✅ Model training complete!")
    
    def predict(self, customer_data):
        """Predict churn risk for new customer(s)"""
        if isinstance(customer_data, dict):
            # Single customer
            df = pd.DataFrame([customer_data])
        else:
            # DataFrame
            df = customer_data.copy()
        
        # Prepare features
        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_cols,
            index=X.index
        )
        
        # Predictions
        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)
        
        # Decode predictions
        status_pred = self.label_encoder.inverse_transform(predictions)
        
        # Churn probability (probability of being Churned)
        churned_idx = list(self.label_encoder.classes_).index('Churned')
        churn_prob = probabilities[:, churned_idx]
        
        # At-risk probability
        atrisk_idx = list(self.label_encoder.classes_).index('At-Risk')
        atrisk_prob = probabilities[:, atrisk_idx]
        
        # Risk score (0-100)
        risk_score = (churn_prob * 100 + atrisk_prob * 50)
        
        # Risk level
        risk_level = pd.cut(
            risk_score,
            bins=[0, 30, 60, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        # Compile results
        results = pd.DataFrame({
            'predicted_status': status_pred,
            'churn_probability': churn_prob,
            'atrisk_probability': atrisk_prob,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'active_probability': probabilities[:, 0]
        })
        
        return results
    
    def get_feature_importance(self):
        """Get feature importance"""
        return self.feature_importance
    
    def get_retention_recommendations(self, customer_features):
        """Generate retention recommendations based on customer profile"""
        recommendations = []
        
        if customer_features.get('low_engagement_flag', 0) == 1:
            recommendations.append("🎯 Launch re-engagement campaign with product training")
        
        if customer_features.get('support_issues_flag', 0) == 1:
            recommendations.append("🤝 Escalate to customer success for proactive support")
        
        if customer_features.get('payment_issues_flag', 0) == 1:
            recommendations.append("💳 Contact billing department to resolve payment issues")
        
        if customer_features.get('low_satisfaction_flag', 0) == 1:
            recommendations.append("📞 Schedule executive check-in call to address concerns")
        
        if customer_features.get('engagement_score', 100) < 40:
            recommendations.append("📚 Provide personalized onboarding session")
        
        if customer_features.get('contract_strength', 100) < 50:
            recommendations.append("📝 Offer annual contract with discount incentive")
        
        if not recommendations:
            recommendations.append("✅ Continue standard account management")
        
        return recommendations
    
    def save_model(self, path='saved_models/churn_prediction_model.pkl'):
        """Save trained model"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_cols': self.feature_cols,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(model_data, path)
        print(f"\n✅ Model saved to: {path}")
    
    @classmethod
    def load_model(cls, path='saved_models/churn_prediction_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(path)
        
        model = cls()
        model.classifier = model_data['classifier']
        model.scaler = model_data['scaler']
        model.label_encoder = model_data['label_encoder']
        model.feature_cols = model_data['feature_cols']
        model.feature_importance = model_data['feature_importance']
        model.training_metrics = model_data['training_metrics']
        
        print(f"✅ Model loaded from: {path}")
        print(f"Training date: {model_data['trained_date']}")
        
        return model


def main():
    """Train model on generated data"""
    # Load training data
    print("Loading training data...")
    df = pd.read_csv('data/training_data.csv')
    
    # Initialize and train model
    model = ChurnPredictionModel()
    model.train(df)
    
    # Save model
    model.save_model()
    
    # Test prediction on sample
    print("\n=== Testing Predictions on Sample ===")
    sample = df.head(5)
    predictions = model.predict(sample)
    
    print("\nSample Predictions:")
    results = pd.concat([
        sample[['customer_id', 'tier', 'status', 'overall_health_score']],
        predictions
    ], axis=1)
    print(results)
    
    print("\n✅ Model training and testing complete!")


if __name__ == "__main__":
    main()
