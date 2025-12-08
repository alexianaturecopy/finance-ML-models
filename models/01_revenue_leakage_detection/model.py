"""
Revenue Leakage Detection - ML Model
Two-stage model: Classification (leakage yes/no) + Regression (amount estimation)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
import json
from datetime import datetime

class RevenueLeakageModel:
    """Two-stage ML model for detecting and quantifying revenue leakage"""
    
    def __init__(self):
        self.classifier = None  # Detects if leakage exists
        self.regressor = None   # Estimates leakage amount
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.feature_importance = {}
        self.training_metrics = {}
        
    def prepare_features(self, df):
        """Prepare features for modeling"""
        # Select feature columns
        self.feature_cols = [
            'payment_failure_rate',
            'customer_health_score',
            'payment_reliability_score',
            'user_utilization_ratio',
            'engagement_score',
            'last_login_days_ago',
            'support_tickets_90d',
            'feature_adoption_pct',
            'payment_risk',
            'health_risk',
            'engagement_risk',
            'overutilization',
            'days_since_contract_start',
            'revenue_per_user'
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
    
    def train_classification_model(self, X_train, y_train, X_test, y_test):
        """Train binary classification model (has_leakage yes/no)"""
        print("\n=== Training Classification Model ===")
        
        # Random Forest Classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        
        # Metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        # Feature importance
        feature_imp = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Features:")
        print(feature_imp.head(10))
        
        # Store metrics
        self.training_metrics['classification'] = {
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_imp.to_dict('records')
        }
        
        self.feature_importance['classification'] = feature_imp.to_dict('records')
        
        return y_pred, y_pred_proba
    
    def train_regression_model(self, X_train, y_train, X_test, y_test):
        """Train regression model for leakage amount (only on accounts with leakage)"""
        print("\n=== Training Regression Model ===")
        
        # Gradient Boosting Regressor
        self.regressor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        self.regressor.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.regressor.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nMean Absolute Error: ${mae:,.2f}")
        print(f"Root Mean Squared Error: ${rmse:,.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Feature importance
        feature_imp = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.regressor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Features:")
        print(feature_imp.head(10))
        
        # Store metrics
        self.training_metrics['regression'] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'feature_importance': feature_imp.to_dict('records')
        }
        
        self.feature_importance['regression'] = feature_imp.to_dict('records')
        
        return y_pred
    
    def train(self, df):
        """Train both models on provided data"""
        print("=== Revenue Leakage Detection - Model Training ===\n")
        print(f"Dataset size: {len(df)} accounts")
        print(f"Leakage rate: {df['has_leakage'].mean():.1%}")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Classification target (has_leakage)
        y_class = df['has_leakage'].astype(int)
        
        # Split data
        X_train, X_test, y_train_class, y_test_class = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        
        # Train classification model
        _, y_pred_proba = self.train_classification_model(
            X_train, y_train_class, X_test, y_test_class
        )
        
        # For regression, only use accounts with actual leakage
        leakage_mask = df['has_leakage'] == True
        X_leakage = X[leakage_mask]
        y_amount = df.loc[leakage_mask, 'monthly_leakage_amount']
        
        if len(X_leakage) > 50:  # Need enough samples
            X_train_reg, X_test_reg, y_train_amount, y_test_amount = train_test_split(
                X_leakage, y_amount, test_size=0.2, random_state=42
            )
            
            # Train regression model
            self.train_regression_model(
                X_train_reg, y_train_amount, X_test_reg, y_test_amount
            )
        else:
            print("\n⚠️ Insufficient leakage samples for regression training")
        
        print("\n✅ Model training complete!")
    
    def predict(self, account_data):
        """Predict leakage for new account(s)"""
        if isinstance(account_data, dict):
            # Single account
            df = pd.DataFrame([account_data])
        else:
            # DataFrame
            df = account_data.copy()
        
        # Prepare features
        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_cols,
            index=X.index
        )
        
        # Classification prediction
        leakage_prob = self.classifier.predict_proba(X_scaled)[:, 1]
        has_leakage = (leakage_prob >= 0.5).astype(int)
        
        # Amount prediction (only for predicted leakage cases)
        predicted_amounts = np.zeros(len(X_scaled))
        if self.regressor and has_leakage.sum() > 0:
            leakage_indices = np.where(has_leakage == 1)[0]
            predicted_amounts[leakage_indices] = self.regressor.predict(
                X_scaled.iloc[leakage_indices]
            )
        
        # Compile results
        results = pd.DataFrame({
            'leakage_probability': leakage_prob,
            'predicted_leakage': has_leakage,
            'predicted_monthly_amount': predicted_amounts,
            'predicted_annual_amount': predicted_amounts * 12,
            'risk_level': pd.cut(
                leakage_prob,
                bins=[0, 0.3, 0.6, 1.0],
                labels=['Low', 'Medium', 'High']
            )
        })
        
        return results
    
    def get_feature_importance(self, model_type='classification'):
        """Get feature importance for specified model"""
        return self.feature_importance.get(model_type, [])
    
    def save_model(self, path='saved_models/revenue_leakage_model.pkl'):
        """Save trained model"""
        model_data = {
            'classifier': self.classifier,
            'regressor': self.regressor,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(model_data, path)
        print(f"\n✅ Model saved to: {path}")
    
    @classmethod
    def load_model(cls, path='saved_models/revenue_leakage_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(path)
        
        model = cls()
        model.classifier = model_data['classifier']
        model.regressor = model_data['regressor']
        model.scaler = model_data['scaler']
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
    model = RevenueLeakageModel()
    model.train(df)
    
    # Save model
    model.save_model()
    
    # Test prediction on sample
    print("\n=== Testing Predictions on Sample ===")
    sample = df.head(5)
    predictions = model.predict(sample)
    
    print("\nSample Predictions:")
    results = pd.concat([
        sample[['account_id', 'tier', 'mrr', 'has_leakage', 'monthly_leakage_amount']],
        predictions
    ], axis=1)
    print(results)
    
    print("\n✅ Model training and testing complete!")


if __name__ == "__main__":
    main()
