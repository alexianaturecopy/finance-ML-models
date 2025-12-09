"""Claims Denial Prediction - ML Model"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
from datetime import datetime

class ClaimsDenialModel:
    def __init__(self):
        self.classifier = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        
    def prepare_features(self, df):
        self.feature_cols = [
            'claim_amount', 'days_to_submission', 'prior_auth_obtained',
            'documentation_complete', 'correct_coding', 'in_network',
            'timely_filed', 'provider_denial_rate', 'high_amount_flag', 'risk_score'
        ]
        X = df[self.feature_cols].fillna(0)
        X = X.astype({col: int for col in ['prior_auth_obtained', 'documentation_complete', 'correct_coding', 'in_network', 'timely_filed', 'high_amount_flag']})
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=self.feature_cols, index=X.index)
        return X_scaled
    
    def train(self, df):
        print("=== Claims Denial Prediction - Training ===\n")
        X = self.prepare_features(df)
        y = df['denied'].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        self.classifier.fit(X_train, y_train)
        
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("\n", classification_report(y_test, y_pred, target_names=['Approved', 'Denied']))
        print("\n✅ Training complete!")
    
    def predict(self, claim_data):
        if isinstance(claim_data, dict):
            df = pd.DataFrame([claim_data])
        else:
            df = claim_data.copy()
        
        X = df[self.feature_cols].fillna(0)
        X = X.astype({col: int for col in ['prior_auth_obtained', 'documentation_complete', 'correct_coding', 'in_network', 'timely_filed', 'high_amount_flag']})
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.feature_cols, index=X.index)
        
        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)[:, 1]
        
        results = pd.DataFrame({
            'will_deny': predictions,
            'denial_probability': probabilities,
            'risk_level': pd.cut(probabilities, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
        })
        return results
    
    def save_model(self, path='saved_models/claims_denial_model.pkl'):
        joblib.dump({'classifier': self.classifier, 'scaler': self.scaler, 'feature_cols': self.feature_cols, 'trained_date': datetime.now().strftime('%Y-%m-%d')}, path)
        print(f"✅ Model saved to: {path}")
    
    @classmethod
    def load_model(cls, path='saved_models/claims_denial_model.pkl'):
        data = joblib.load(path)
        model = cls()
        model.classifier = data['classifier']
        model.scaler = data['scaler']
        model.feature_cols = data['feature_cols']
        return model
