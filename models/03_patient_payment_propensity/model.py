"""
Patient Payment Propensity - ML Model
Binary classification: Will Pay / Won't Pay
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
from datetime import datetime

class PaymentPropensityModel:
    """Predicts patient payment likelihood"""
    
    def __init__(self):
        self.classifier = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.feature_importance = {}
        self.training_metrics = {}
        
    def prepare_features(self, df):
        """Prepare features for modeling"""
        self.feature_cols = [
            'age', 'credit_score', 'prior_collections', 'patient_responsibility',
            'prior_balance', 'payment_history_score', 'typical_days_to_pay',
            'payment_burden_ratio', 'affordability_score', 'payment_capacity_score',
            'collection_risk_score', 'high_balance_flag', 'prior_collection_flag',
            'low_credit_flag', 'uninsured_flag', 'unemployed_flag'
        ]
        
        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_cols,
            index=X.index
        )
        
        return X_scaled
    
    def train(self, df):
        """Train the payment propensity model"""
        print("=== Patient Payment Propensity - Model Training ===\n")
        print(f"Dataset size: {len(df)} patients")
        
        # Prepare features
        X = self.prepare_features(df)
        y = df['will_pay_full'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        print("\n=== Training Classification Model ===")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Won\'t Pay', 'Will Pay']))
        
        # Feature importance
        feature_imp = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Features:")
        print(feature_imp.head(10))
        
        self.training_metrics = {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc)
        }
        self.feature_importance = feature_imp.to_dict('records')
        
        print("\n✅ Model training complete!")
    
    def predict(self, patient_data):
        """Predict payment propensity for new patients"""
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = patient_data.copy()
        
        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_cols,
            index=X.index
        )
        
        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)[:, 1]
        
        results = pd.DataFrame({
            'will_pay': predictions,
            'payment_probability': probabilities,
            'risk_level': pd.cut(probabilities, bins=[0, 0.3, 0.7, 1.0], labels=['High-Risk', 'Medium-Risk', 'Low-Risk'])
        })
        
        return results
    
    def save_model(self, path='saved_models/payment_propensity_model.pkl'):
        """Save trained model"""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        joblib.dump(model_data, path)
        print(f"\n✅ Model saved to: {path}")
    
    @classmethod
    def load_model(cls, path='saved_models/payment_propensity_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(path)
        model = cls()
        model.classifier = model_data['classifier']
        model.scaler = model_data['scaler']
        model.feature_cols = model_data['feature_cols']
        model.feature_importance = model_data['feature_importance']
        model.training_metrics = model_data['training_metrics']
        return model
