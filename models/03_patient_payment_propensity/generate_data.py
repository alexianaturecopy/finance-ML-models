"""
Patient Payment Propensity - Sample Data Generator
Generates realistic healthcare patient payment scenarios
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

class PatientPaymentDataGenerator:
    """Generate realistic patient payment data for healthcare providers"""
    
    def __init__(self, num_patients=1000):
        self.num_patients = num_patients
        
        # Insurance types and payment characteristics
        self.insurance_types = {
            'Private': {'coverage': 0.80, 'payment_rate': 0.85},
            'Medicare': {'coverage': 0.65, 'payment_rate': 0.75},
            'Medicaid': {'coverage': 0.70, 'payment_rate': 0.60},
            'Self-Pay': {'coverage': 0.00, 'payment_rate': 0.40}
        }
        
        # Procedure types and costs
        self.procedures = {
            'Emergency Visit': {'cost_range': (500, 3000), 'complexity': 'medium'},
            'Outpatient Surgery': {'cost_range': (3000, 15000), 'complexity': 'high'},
            'Diagnostic Test': {'cost_range': (200, 2000), 'complexity': 'low'},
            'Specialist Consultation': {'cost_range': (150, 800), 'complexity': 'low'},
            'Physical Therapy': {'cost_range': (100, 500), 'complexity': 'low'},
            'Inpatient Stay': {'cost_range': (5000, 50000), 'complexity': 'high'}
        }
    
    def generate_patients(self):
        """Generate patient demographics and characteristics"""
        patients = []
        
        for i in range(self.num_patients):
            patient_id = f"PT{str(i+1).zfill(6)}"
            
            # Demographics
            age = int(np.random.normal(50, 20))
            age = max(18, min(95, age))
            
            # Insurance (weighted by typical distribution)
            insurance_weights = [0.50, 0.25, 0.15, 0.10]
            insurance_type = np.random.choice(
                list(self.insurance_types.keys()), 
                p=insurance_weights
            )
            
            # Income level (affects payment ability)
            if insurance_type == 'Private':
                income_level = np.random.choice(['High', 'High', 'Medium', 'Medium', 'Low'])
            elif insurance_type == 'Medicare':
                income_level = np.random.choice(['Medium', 'Medium', 'Low', 'Low'])
            elif insurance_type == 'Medicaid':
                income_level = np.random.choice(['Low', 'Low', 'Low', 'Medium'])
            else:  # Self-Pay
                income_level = np.random.choice(['Low', 'Low', 'Medium'])
            
            # Credit indicators
            if income_level == 'High':
                credit_score = int(np.random.normal(750, 50))
                prior_collections = random.choice([0, 0, 0, 1])
            elif income_level == 'Medium':
                credit_score = int(np.random.normal(680, 60))
                prior_collections = random.choice([0, 0, 1, 1, 2])
            else:  # Low
                credit_score = int(np.random.normal(610, 70))
                prior_collections = random.choice([1, 1, 2, 2, 3])
            
            credit_score = max(300, min(850, credit_score))
            
            # Visit characteristics
            procedure_type = random.choice(list(self.procedures.keys()))
            procedure_info = self.procedures[procedure_type]
            
            total_charges = int(np.random.uniform(*procedure_info['cost_range']))
            
            # Insurance coverage
            insurance_info = self.insurance_types[insurance_type]
            if insurance_type == 'Self-Pay':
                insurance_paid = 0
                patient_responsibility = total_charges
            else:
                coverage_rate = insurance_info['coverage']
                insurance_paid = int(total_charges * coverage_rate * np.random.uniform(0.9, 1.0))
                patient_responsibility = total_charges - insurance_paid
            
            # Payment behavior factors
            days_to_bill = random.randint(5, 45)
            prior_balance = max(0, int(np.random.exponential(500) if prior_collections > 0 else 0))
            
            # Employment status
            if age >= 65:
                employed = random.choice([False, False, False, True])
            elif age >= 55:
                employed = random.choice([True, True, True, False])
            else:
                employed = random.choice([True, True, True, True, False])
            
            # Payment history
            if income_level == 'High' and credit_score > 700:
                payment_history_score = np.random.uniform(85, 100)
                typical_days_to_pay = int(np.random.normal(30, 10))
            elif income_level == 'Medium':
                payment_history_score = np.random.uniform(60, 85)
                typical_days_to_pay = int(np.random.normal(60, 20))
            else:
                payment_history_score = np.random.uniform(30, 65)
                typical_days_to_pay = int(np.random.normal(90, 30))
            
            typical_days_to_pay = max(15, typical_days_to_pay)
            
            # Determine payment outcome
            payment_propensity = (
                payment_history_score * 0.30 +
                (credit_score / 8.5) * 0.25 +
                (100 - prior_collections * 20) * 0.20 +
                (insurance_info['payment_rate'] * 100) * 0.15 +
                (100 if employed else 50) * 0.10
            )
            
            # Payment categories
            if payment_propensity >= 75:
                will_pay_full = True
                payment_timeline = 'On-Time'
                days_to_payment = int(np.random.normal(30, 10))
                payment_amount = patient_responsibility
            elif payment_propensity >= 55:
                will_pay_full = random.choice([True, True, False])
                payment_timeline = 'Late' if will_pay_full else 'Partial'
                days_to_payment = int(np.random.normal(75, 20))
                payment_amount = patient_responsibility if will_pay_full else int(patient_responsibility * np.random.uniform(0.4, 0.8))
            elif payment_propensity >= 35:
                will_pay_full = False
                payment_timeline = 'Partial'
                days_to_payment = int(np.random.normal(120, 30))
                payment_amount = int(patient_responsibility * np.random.uniform(0.2, 0.5))
            else:
                will_pay_full = False
                payment_timeline = 'Non-Payment'
                days_to_payment = None
                payment_amount = 0
            
            days_to_payment = max(0, days_to_payment) if days_to_payment else None
            
            # Financial assistance eligibility
            if income_level == 'Low' and patient_responsibility > 1000:
                financial_assistance_eligible = True
            else:
                financial_assistance_eligible = random.choice([True, False, False, False])
            
            patients.append({
                'patient_id': patient_id,
                'age': age,
                'insurance_type': insurance_type,
                'income_level': income_level,
                'credit_score': credit_score,
                'prior_collections': prior_collections,
                'employed': employed,
                'procedure_type': procedure_type,
                'total_charges': total_charges,
                'insurance_paid': insurance_paid,
                'patient_responsibility': patient_responsibility,
                'prior_balance': prior_balance,
                'payment_history_score': round(payment_history_score, 1),
                'typical_days_to_pay': typical_days_to_pay,
                'payment_propensity_score': round(payment_propensity, 1),
                'will_pay_full': will_pay_full,
                'payment_timeline': payment_timeline,
                'days_to_payment': days_to_payment,
                'payment_amount': payment_amount,
                'financial_assistance_eligible': financial_assistance_eligible
            })
        
        return pd.DataFrame(patients)
    
    def create_features(self, df):
        """Engineer features for ML model"""
        # Payment burden ratio
        df['payment_burden_ratio'] = df['patient_responsibility'] / (df['patient_responsibility'] + df['insurance_paid'] + 1)
        
        # Affordability score (inverse of burden + credit)
        df['affordability_score'] = 100 - (df['payment_burden_ratio'] * 50 + (850 - df['credit_score']) / 8.5)
        
        # Risk flags
        df['high_balance_flag'] = (df['patient_responsibility'] > 2000).astype(int)
        df['prior_collection_flag'] = (df['prior_collections'] > 0).astype(int)
        df['low_credit_flag'] = (df['credit_score'] < 650).astype(int)
        df['uninsured_flag'] = (df['insurance_type'] == 'Self-Pay').astype(int)
        df['unemployed_flag'] = (~df['employed']).astype(int)
        
        # Payment capacity score
        credit_component = df['credit_score'] / 8.5
        history_component = df['payment_history_score']
        insurance_component = df['insurance_type'].map({
            'Private': 80, 'Medicare': 65, 'Medicaid': 55, 'Self-Pay': 30
        })
        
        df['payment_capacity_score'] = (
            credit_component * 0.35 +
            history_component * 0.35 +
            insurance_component * 0.30
        )
        
        # Collection risk score
        df['collection_risk_score'] = (
            df['prior_collections'] * 25 +
            df['high_balance_flag'] * 15 +
            df['low_credit_flag'] * 20 +
            df['uninsured_flag'] * 25 +
            df['unemployed_flag'] * 15
        )
        
        return df
    
    def generate_complete_dataset(self):
        """Generate complete dataset with all features"""
        print("Generating patient data...")
        df = self.generate_patients()
        
        print("Creating features...")
        df = self.create_features(df)
        
        # Summary statistics
        print(f"\n=== Dataset Summary ===")
        print(f"Total Patients: {len(df)}")
        print(f"\nPayment Outcomes:")
        print(df['payment_timeline'].value_counts())
        print(f"\nFull Payment Rate: {df['will_pay_full'].mean():.1%}")
        print(f"\nInsurance Distribution:")
        print(df['insurance_type'].value_counts())
        print(f"\nAverage Patient Responsibility: ${df['patient_responsibility'].mean():,.0f}")
        print(f"Total Collectible: ${df['patient_responsibility'].sum():,.0f}")
        print(f"Expected Collections: ${df['payment_amount'].sum():,.0f}")
        print(f"Collection Rate: {(df['payment_amount'].sum() / df['patient_responsibility'].sum()):.1%}")
        
        return df


def main():
    """Generate and save sample data"""
    print("=== Patient Payment Propensity - Data Generator ===\n")
    
    generator = PatientPaymentDataGenerator(num_patients=1000)
    df = generator.generate_complete_dataset()
    
    # Save datasets
    df.to_csv('data/training_data.csv', index=False)
    print(f"\n✅ Training data saved to: data/training_data.csv")
    
    df.head(100).to_csv('data/sample_patients.csv', index=False)
    print(f"✅ Sample data saved to: data/sample_patients.csv")


if __name__ == "__main__":
    main()
