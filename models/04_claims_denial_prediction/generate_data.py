"""Claims Denial Prediction - Data Generator"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

class ClaimsDataGenerator:
    def __init__(self, num_claims=1000):
        self.num_claims = num_claims
        self.payers = ['BlueCross', 'Aetna', 'UnitedHealth', 'Cigna', 'Medicare', 'Medicaid']
        self.denial_reasons = ['Missing Info', 'Not Necessary', 'Prior Auth', 'Out of Network', 'Duplicate', 'Coding Error', 'Timely Filing', 'Coverage Lapsed']
        self.cpt_codes = {'99213': 'Office Visit', '99214': 'Complex Visit', '99223': 'Hospital', '71020': 'X-Ray', '80053': 'Lab Panel', '93000': 'EKG', '29881': 'Surgery', '27447': 'Knee Replace'}
    
    def generate_claims(self):
        claims = []
        for i in range(self.num_claims):
            claim_id = f"CLM{str(i+1).zfill(6)}"
            payer = random.choice(self.payers)
            cpt_code = random.choice(list(self.cpt_codes.keys()))
            
            if cpt_code in ['29881', '27447']:
                claim_amount = int(np.random.uniform(15000, 50000))
            elif cpt_code in ['99223']:
                claim_amount = int(np.random.uniform(5000, 15000))
            else:
                claim_amount = int(np.random.uniform(100, 2000))
            
            prior_auth = random.choice([True, True, True, False])
            docs_complete = random.choice([True, True, True, False])
            correct_code = random.choice([True, True, True, True, False])
            in_network = random.choice([True, True, True, False])
            days_to_sub = random.randint(1, 60)
            timely = days_to_sub <= 30
            provider_rate = np.random.beta(2, 8) * 100
            
            denial_prob = 0.08
            if not prior_auth: denial_prob += 0.30
            if not docs_complete: denial_prob += 0.25
            if not correct_code: denial_prob += 0.20
            if not in_network: denial_prob += 0.15
            if not timely: denial_prob += 0.25
            
            denied = np.random.random() < denial_prob
            denial_reason = random.choice(self.denial_reasons) if denied else None
            
            claims.append({
                'claim_id': claim_id, 'payer': payer, 'cpt_code': cpt_code,
                'service_description': self.cpt_codes[cpt_code], 'claim_amount': claim_amount,
                'days_to_submission': days_to_sub, 'prior_auth_obtained': prior_auth,
                'documentation_complete': docs_complete, 'correct_coding': correct_code,
                'in_network': in_network, 'timely_filed': timely,
                'provider_denial_rate': round(provider_rate, 1), 'denied': denied,
                'denial_reason': denial_reason
            })
        return pd.DataFrame(claims)
    
    def create_features(self, df):
        df['high_amount_flag'] = (df['claim_amount'] > 10000).astype(int)
        df['risk_score'] = (
            (~df['prior_auth_obtained']).astype(int) * 30 +
            (~df['documentation_complete']).astype(int) * 25 +
            (~df['correct_coding']).astype(int) * 20 +
            (~df['in_network']).astype(int) * 15 +
            (~df['timely_filed']).astype(int) * 25
        )
        return df
    
    def generate_complete_dataset(self):
        print("Generating claims...")
        df = self.generate_claims()
        df = self.create_features(df)
        print(f"Total: {len(df)}, Denied: {df['denied'].mean():.1%}")
        return df

def main():
    gen = ClaimsDataGenerator()
    df = gen.generate_complete_dataset()
    df.to_csv('data/training_data.csv', index=False)
    df.head(100).to_csv('data/sample_claims.csv', index=False)
    print("✅ Data saved!")

if __name__ == "__main__":
    main()
