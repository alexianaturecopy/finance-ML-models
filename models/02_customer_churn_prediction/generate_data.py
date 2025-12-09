"""
Customer Churn Prediction - Sample Data Generator
Generates realistic SaaS customer data with churn indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class ChurnDataGenerator:
    """Generate realistic SaaS customer data with churn patterns"""
    
    def __init__(self, num_customers=1000):
        self.num_customers = num_customers
        self.subscription_tiers = {
            'Starter': {'mrr': 99, 'users': 5, 'churn_baseline': 0.15},
            'Professional': {'mrr': 299, 'users': 20, 'churn_baseline': 0.10},
            'Business': {'mrr': 799, 'users': 100, 'churn_baseline': 0.07},
            'Enterprise': {'mrr': 2499, 'users': 500, 'churn_baseline': 0.05}
        }
        
    def generate_customers(self):
        """Generate customer master data"""
        customers = []
        
        for i in range(self.num_customers):
            customer_id = f"CUST{str(i+1).zfill(6)}"
            
            # Assign subscription tier (weighted towards lower tiers)
            tier_weights = [0.40, 0.35, 0.20, 0.05]
            tier = np.random.choice(list(self.subscription_tiers.keys()), p=tier_weights)
            
            # Customer lifecycle
            signup_date = datetime.now() - timedelta(days=random.randint(30, 730))
            tenure_days = (datetime.now() - signup_date).days
            
            # Determine if churned and when
            churn_baseline = self.subscription_tiers[tier]['churn_baseline']
            will_churn = np.random.random() < churn_baseline
            
            if will_churn and tenure_days > 90:  # Only churn if at least 90 days tenure
                churn_date = signup_date + timedelta(days=random.randint(90, tenure_days))
                status = 'Churned'
                days_to_churn = 0
            elif will_churn and tenure_days <= 90:
                # At risk but haven't churned yet
                status = 'At-Risk'
                days_to_churn = random.randint(10, 60)
            else:
                status = 'Active'
                days_to_churn = None
                churn_date = None
            
            # Usage patterns (correlated with churn risk)
            if status == 'Active':
                login_frequency_score = np.random.beta(7, 2) * 100  # High engagement
                feature_usage_score = np.random.beta(6, 3) * 100
                api_calls_per_day = int(np.random.lognormal(5, 1.5))
            elif status == 'At-Risk':
                login_frequency_score = np.random.beta(3, 5) * 100  # Declining
                feature_usage_score = np.random.beta(3, 5) * 100
                api_calls_per_day = int(np.random.lognormal(3, 1.2))
            else:  # Churned
                login_frequency_score = np.random.beta(2, 7) * 100  # Very low
                feature_usage_score = np.random.beta(2, 7) * 100
                api_calls_per_day = int(np.random.lognormal(2, 1))
            
            # Support tickets (more tickets = more friction)
            if status == 'Active':
                support_tickets_90d = int(np.random.poisson(1.5))
                avg_resolution_days = np.random.uniform(1, 3)
            elif status == 'At-Risk':
                support_tickets_90d = int(np.random.poisson(4))
                avg_resolution_days = np.random.uniform(3, 7)
            else:
                support_tickets_90d = int(np.random.poisson(6))
                avg_resolution_days = np.random.uniform(5, 10)
            
            # Payment history
            if status == 'Active':
                payment_failures = 0
                days_overdue = 0
            elif status == 'At-Risk':
                payment_failures = random.choice([0, 1, 1, 2])
                days_overdue = random.choice([0, 0, 5, 15])
            else:
                payment_failures = random.choice([1, 2, 3, 4])
                days_overdue = random.choice([30, 45, 60, 90])
            
            # NPS and satisfaction
            if status == 'Active':
                nps_score = random.choice([8, 9, 9, 10])
                satisfaction_score = np.random.uniform(7, 10)
            elif status == 'At-Risk':
                nps_score = random.choice([5, 6, 6, 7])
                satisfaction_score = np.random.uniform(4, 7)
            else:
                nps_score = random.choice([0, 1, 2, 3, 4])
                satisfaction_score = np.random.uniform(1, 4)
            
            # Contract and pricing
            mrr = self.subscription_tiers[tier]['mrr']
            contract_term_months = random.choice([1, 1, 1, 12, 24, 36])  # Mostly monthly
            auto_renewal = random.choice([True, True, True, False])  # Mostly auto-renew
            
            # Engagement metrics
            team_size = int(self.subscription_tiers[tier]['users'] * np.random.uniform(0.3, 1.2))
            active_users_pct = login_frequency_score / 100
            
            customers.append({
                'customer_id': customer_id,
                'tier': tier,
                'mrr': mrr,
                'signup_date': signup_date.strftime('%Y-%m-%d'),
                'tenure_days': tenure_days,
                'status': status,
                'days_to_churn': days_to_churn,
                'churn_date': churn_date.strftime('%Y-%m-%d') if churn_date else None,
                'login_frequency_score': round(login_frequency_score, 1),
                'feature_usage_score': round(feature_usage_score, 1),
                'api_calls_per_day': api_calls_per_day,
                'support_tickets_90d': support_tickets_90d,
                'avg_resolution_days': round(avg_resolution_days, 1),
                'payment_failures': payment_failures,
                'days_overdue': days_overdue,
                'nps_score': nps_score,
                'satisfaction_score': round(satisfaction_score, 1),
                'contract_term_months': contract_term_months,
                'auto_renewal': auto_renewal,
                'team_size': team_size,
                'active_users_pct': round(active_users_pct, 2)
            })
        
        return pd.DataFrame(customers)
    
    def create_features(self, df):
        """Engineer features for ML model"""
        # Engagement composite score
        df['engagement_score'] = (
            df['login_frequency_score'] * 0.4 +
            df['feature_usage_score'] * 0.4 +
            (df['active_users_pct'] * 100) * 0.2
        )
        
        # Support health (inverse - more tickets = worse)
        df['support_health_score'] = np.maximum(0, 100 - (
            df['support_tickets_90d'] * 10 +
            df['avg_resolution_days'] * 5
        ))
        
        # Payment health
        df['payment_health_score'] = np.maximum(0, 100 - (
            df['payment_failures'] * 25 +
            df['days_overdue']
        ))
        
        # Contract strength (longer contracts = more committed)
        df['contract_strength'] = df['contract_term_months'].map({
            1: 10,
            12: 50,
            24: 75,
            36: 100
        })
        
        # Renewal risk
        df['renewal_risk'] = (~df['auto_renewal']).astype(int)
        
        # Usage intensity
        df['usage_intensity'] = np.log1p(df['api_calls_per_day'])
        
        # Tenure buckets (newer customers churn more)
        df['tenure_bucket'] = pd.cut(
            df['tenure_days'],
            bins=[0, 90, 180, 365, 730, np.inf],
            labels=['0-3m', '3-6m', '6-12m', '1-2y', '2y+']
        )
        
        # Satisfaction composite
        df['satisfaction_composite'] = (
            df['nps_score'] * 10 + df['satisfaction_score'] * 10
        ) / 2
        
        # Risk flags
        df['low_engagement_flag'] = (df['engagement_score'] < 40).astype(int)
        df['support_issues_flag'] = (df['support_tickets_90d'] > 5).astype(int)
        df['payment_issues_flag'] = (df['payment_failures'] > 0).astype(int)
        df['low_satisfaction_flag'] = (df['nps_score'] < 7).astype(int)
        
        # Overall health score
        df['overall_health_score'] = (
            df['engagement_score'] * 0.30 +
            df['support_health_score'] * 0.20 +
            df['payment_health_score'] * 0.20 +
            df['satisfaction_composite'] * 0.20 +
            df['contract_strength'] * 0.10
        )
        
        return df
    
    def generate_churn_reasons(self, df):
        """Generate reasons for churn/risk"""
        reasons = []
        
        for idx, row in df.iterrows():
            if row['status'] == 'Active':
                reason_list = ['None - Healthy Customer']
            elif row['status'] == 'At-Risk':
                reason_list = []
                if row['low_engagement_flag']:
                    reason_list.append('Declining Usage')
                if row['support_issues_flag']:
                    reason_list.append('Support Frustration')
                if row['payment_issues_flag']:
                    reason_list.append('Payment Issues')
                if row['low_satisfaction_flag']:
                    reason_list.append('Low Satisfaction')
                if not reason_list:
                    reason_list = ['General Disengagement']
            else:  # Churned
                # Primary reason weighted by severity
                possible_reasons = []
                if row['payment_failures'] > 2:
                    possible_reasons.extend(['Payment Failure'] * 3)
                if row['support_tickets_90d'] > 5:
                    possible_reasons.extend(['Poor Support Experience'] * 2)
                if row['engagement_score'] < 30:
                    possible_reasons.extend(['Lack of Product Value'] * 3)
                if row['nps_score'] <= 3:
                    possible_reasons.extend(['Competitor Offering'] * 2)
                if row['contract_term_months'] == 1 and not row['auto_renewal']:
                    possible_reasons.extend(['Budget Constraints'] * 1)
                
                if possible_reasons:
                    reason_list = [random.choice(possible_reasons)]
                else:
                    reason_list = ['General Dissatisfaction']
            
            reasons.append({
                'customer_id': row['customer_id'],
                'primary_reason': reason_list[0],
                'secondary_reasons': ', '.join(reason_list[1:]) if len(reason_list) > 1 else None
            })
        
        return pd.DataFrame(reasons)
    
    def generate_complete_dataset(self):
        """Generate complete dataset with all features"""
        print("Generating customer data...")
        df = self.generate_customers()
        
        print("Creating features...")
        df = self.create_features(df)
        
        print("Generating churn reasons...")
        reasons_df = self.generate_churn_reasons(df)
        df = df.merge(reasons_df, on='customer_id')
        
        # Summary statistics
        print(f"\n=== Dataset Summary ===")
        print(f"Total Customers: {len(df)}")
        print(f"\nStatus Distribution:")
        print(df['status'].value_counts())
        print(f"\nChurn Rate: {(df['status'] == 'Churned').mean():.1%}")
        print(f"At-Risk Rate: {(df['status'] == 'At-Risk').mean():.1%}")
        print(f"Active Rate: {(df['status'] == 'Active').mean():.1%}")
        
        print(f"\nTier Distribution:")
        print(df['tier'].value_counts())
        
        print(f"\nChurn by Tier:")
        print(df[df['status'] == 'Churned'].groupby('tier').size())
        
        print(f"\nTop Churn Reasons:")
        print(df[df['status'] != 'Active']['primary_reason'].value_counts().head())
        
        return df


def main():
    """Generate and save sample data"""
    print("=== Customer Churn Prediction - Data Generator ===\n")
    
    # Generate data
    generator = ChurnDataGenerator(num_customers=1000)
    df = generator.generate_complete_dataset()
    
    # Save to CSV
    output_path = 'data/training_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Training data saved to: {output_path}")
    
    # Create smaller sample for dashboard demo
    sample_df = df.head(100)
    sample_output = 'data/sample_customers.csv'
    sample_df.to_csv(sample_output, index=False)
    print(f"✅ Sample data saved to: {sample_output}")
    
    # Feature importance preview
    print(f"\n=== Key Features for ML Model ===")
    feature_cols = [
        'engagement_score', 'support_health_score', 'payment_health_score',
        'satisfaction_composite', 'contract_strength', 'tenure_days',
        'low_engagement_flag', 'support_issues_flag', 'payment_issues_flag'
    ]
    print(f"Features: {', '.join(feature_cols)}")
    print(f"Target: status (Active / At-Risk / Churned)")


if __name__ == "__main__":
    main()
