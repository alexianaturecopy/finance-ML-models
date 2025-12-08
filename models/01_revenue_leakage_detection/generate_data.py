"""
Revenue Leakage Detection - Sample Data Generator
Generates realistic SaaS billing data with various leakage scenarios
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class SaaSDataGenerator:
    """Generate realistic SaaS billing data with revenue leakage scenarios"""
    
    def __init__(self, num_accounts=1000):
        self.num_accounts = num_accounts
        self.subscription_tiers = {
            'Starter': {'mrr': 99, 'users': 5},
            'Professional': {'mrr': 299, 'users': 20},
            'Business': {'mrr': 799, 'users': 100},
            'Enterprise': {'mrr': 2499, 'users': 500}
        }
        
    def generate_accounts(self):
        """Generate account master data"""
        accounts = []
        
        for i in range(self.num_accounts):
            account_id = f"ACC{str(i+1).zfill(6)}"
            
            # Assign subscription tier (weighted towards lower tiers)
            tier_weights = [0.40, 0.35, 0.20, 0.05]
            tier = np.random.choice(list(self.subscription_tiers.keys()), p=tier_weights)
            
            # Account attributes
            contract_start = datetime.now() - timedelta(days=random.randint(90, 730))
            contract_term = random.choice([12, 24, 36])  # months
            
            # Payment method reliability (affects payment failures)
            payment_method_age = random.randint(1, 365)
            payment_reliability = min(1.0, payment_method_age / 365)  # Newer cards more likely to fail
            
            # Customer health score (affects churn and leakage)
            health_score = np.random.beta(8, 2) * 100  # Skewed towards healthy
            
            # Usage patterns
            licensed_users = self.subscription_tiers[tier]['users']
            actual_users = int(licensed_users * np.random.uniform(0.3, 1.2))  # Some over/under utilize
            
            # Invoice history
            num_invoices = max(1, int((datetime.now() - contract_start).days / 30))
            failed_payments = int(num_invoices * np.random.uniform(0, 0.3) * (1 - payment_reliability))
            
            accounts.append({
                'account_id': account_id,
                'tier': tier,
                'mrr': self.subscription_tiers[tier]['mrr'],
                'contract_start_date': contract_start.strftime('%Y-%m-%d'),
                'contract_term_months': contract_term,
                'payment_method_age_days': payment_method_age,
                'payment_reliability_score': round(payment_reliability, 3),
                'customer_health_score': round(health_score, 1),
                'licensed_users': licensed_users,
                'actual_users': actual_users,
                'total_invoices': num_invoices,
                'failed_payment_count': failed_payments,
                'last_login_days_ago': random.randint(0, 90),
                'support_tickets_90d': int(np.random.poisson(2)),
                'feature_adoption_pct': round(np.random.beta(6, 3) * 100, 1)
            })
        
        return pd.DataFrame(accounts)
    
    def generate_leakage_scenarios(self, accounts_df):
        """Generate specific revenue leakage scenarios"""
        leakage_data = []
        
        for idx, account in accounts_df.iterrows():
            # Determine if account has leakage (30% probability)
            has_leakage = np.random.random() < 0.30
            
            if not has_leakage:
                leakage_data.append({
                    'account_id': account['account_id'],
                    'has_leakage': False,
                    'leakage_type': 'None',
                    'monthly_leakage_amount': 0,
                    'estimated_annual_leakage': 0,
                    'detection_difficulty': 'N/A'
                })
                continue
            
            # Generate specific leakage type based on account characteristics
            leakage_scenarios = self._determine_leakage_type(account)
            
            leakage_data.append({
                'account_id': account['account_id'],
                'has_leakage': True,
                **leakage_scenarios
            })
        
        return pd.DataFrame(leakage_data)
    
    def _determine_leakage_type(self, account):
        """Determine specific type of revenue leakage based on account profile"""
        
        # Leakage type probabilities based on account characteristics
        leakage_types = []
        
        # 1. Failed payment retries (common with low payment reliability)
        if account['payment_reliability_score'] < 0.7:
            leakage_types.append({
                'type': 'Failed_Payment_Retry',
                'probability': 0.5,
                'amount': account['mrr'] * 0.15,  # 15% of MRR lost
                'difficulty': 'Easy'
            })
        
        # 2. Billing system errors (random, any account)
        leakage_types.append({
            'type': 'Billing_System_Error',
            'probability': 0.15,
            'amount': account['mrr'] * np.random.uniform(0.05, 0.25),
            'difficulty': 'Medium'
        })
        
        # 3. Contract terms not implemented (high-value accounts)
        if account['tier'] in ['Business', 'Enterprise']:
            leakage_types.append({
                'type': 'Contract_Terms_Error',
                'probability': 0.20,
                'amount': account['mrr'] * np.random.uniform(0.10, 0.30),
                'difficulty': 'Hard'
            })
        
        # 4. Usage overages not billed (high actual users vs licensed)
        if account['actual_users'] > account['licensed_users'] * 1.1:
            overage_users = account['actual_users'] - account['licensed_users']
            price_per_user = account['mrr'] / account['licensed_users']
            leakage_types.append({
                'type': 'Usage_Overage_Unbilled',
                'probability': 0.40,
                'amount': overage_users * price_per_user,
                'difficulty': 'Medium'
            })
        
        # 5. Downgrade not processed (unhealthy accounts)
        if account['customer_health_score'] < 50:
            leakage_types.append({
                'type': 'Downgrade_Not_Processed',
                'probability': 0.25,
                'amount': account['mrr'] * 0.40,  # Should have downgraded
                'difficulty': 'Easy'
            })
        
        # 6. Expired payment method (old payment methods)
        if account['payment_method_age_days'] > 300:
            leakage_types.append({
                'type': 'Expired_Payment_Method',
                'probability': 0.35,
                'amount': account['mrr'],
                'difficulty': 'Easy'
            })
        
        # Select leakage type based on probabilities
        if leakage_types:
            selected = random.choices(
                leakage_types,
                weights=[lt['probability'] for lt in leakage_types],
                k=1
            )[0]
            
            return {
                'leakage_type': selected['type'],
                'monthly_leakage_amount': round(selected['amount'], 2),
                'estimated_annual_leakage': round(selected['amount'] * 12, 2),
                'detection_difficulty': selected['difficulty']
            }
        else:
            # Fallback to generic billing error
            return {
                'leakage_type': 'Billing_System_Error',
                'monthly_leakage_amount': round(account['mrr'] * 0.10, 2),
                'estimated_annual_leakage': round(account['mrr'] * 0.10 * 12, 2),
                'detection_difficulty': 'Medium'
            }
    
    def create_features(self, accounts_df, leakage_df):
        """Create ML features from account and leakage data"""
        merged = accounts_df.merge(leakage_df, on='account_id')
        
        # Engineer additional features
        merged['payment_failure_rate'] = merged['failed_payment_count'] / merged['total_invoices']
        merged['user_utilization_ratio'] = merged['actual_users'] / merged['licensed_users']
        merged['days_since_contract_start'] = (
            (datetime.now() - pd.to_datetime(merged['contract_start_date'])).dt.days
        )
        merged['revenue_per_user'] = merged['mrr'] / merged['licensed_users']
        merged['engagement_score'] = (
            (100 - merged['last_login_days_ago']) * 0.4 +
            merged['feature_adoption_pct'] * 0.6
        )
        
        # Risk indicators
        merged['payment_risk'] = (merged['payment_reliability_score'] < 0.7).astype(int)
        merged['health_risk'] = (merged['customer_health_score'] < 60).astype(int)
        merged['engagement_risk'] = (merged['last_login_days_ago'] > 30).astype(int)
        merged['overutilization'] = (merged['user_utilization_ratio'] > 1.1).astype(int)
        
        return merged
    
    def generate_complete_dataset(self):
        """Generate complete dataset with all features"""
        print("Generating account data...")
        accounts_df = self.generate_accounts()
        
        print("Generating leakage scenarios...")
        leakage_df = self.generate_leakage_scenarios(accounts_df)
        
        print("Creating features...")
        complete_df = self.create_features(accounts_df, leakage_df)
        
        # Summary statistics
        total_leakage = complete_df[complete_df['has_leakage']]['estimated_annual_leakage'].sum()
        leakage_rate = complete_df['has_leakage'].mean()
        
        print(f"\n=== Dataset Summary ===")
        print(f"Total Accounts: {len(complete_df)}")
        print(f"Accounts with Leakage: {complete_df['has_leakage'].sum()} ({leakage_rate:.1%})")
        print(f"Total Estimated Annual Leakage: ${total_leakage:,.0f}")
        print(f"Average Leakage per Affected Account: ${total_leakage / complete_df['has_leakage'].sum():,.0f}")
        print(f"\nLeakage by Type:")
        print(complete_df[complete_df['has_leakage']]['leakage_type'].value_counts())
        print(f"\nLeakage by Difficulty:")
        print(complete_df[complete_df['has_leakage']]['detection_difficulty'].value_counts())
        
        return complete_df


def main():
    """Generate and save sample data"""
    print("=== Revenue Leakage Detection - Data Generator ===\n")
    
    # Generate data
    generator = SaaSDataGenerator(num_accounts=1000)
    df = generator.generate_complete_dataset()
    
    # Save to CSV
    output_path = 'data/training_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Training data saved to: {output_path}")
    
    # Create smaller sample for dashboard demo
    sample_df = df.head(50)
    sample_output = 'data/sample_accounts.csv'
    sample_df.to_csv(sample_output, index=False)
    print(f"✅ Sample data saved to: {sample_output}")
    
    # Feature importance preview
    print(f"\n=== Key Features for ML Model ===")
    feature_cols = [
        'payment_failure_rate', 'customer_health_score', 'payment_reliability_score',
        'user_utilization_ratio', 'engagement_score', 'last_login_days_ago',
        'payment_risk', 'health_risk', 'engagement_risk', 'overutilization'
    ]
    print(f"Features: {', '.join(feature_cols)}")
    print(f"Target: has_leakage")


if __name__ == "__main__":
    main()
