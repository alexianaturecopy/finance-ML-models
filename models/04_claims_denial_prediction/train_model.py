"""Claims Denial Prediction - Training"""
import os
from generate_data import ClaimsDataGenerator
from model import ClaimsDenialModel

os.makedirs('data', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

print("\n" + "="*60)
print("CLAIMS DENIAL PREDICTION - TRAINING")
print("="*60)

gen = ClaimsDataGenerator()
df = gen.generate_complete_dataset()

model = ClaimsDenialModel()
model.train(df)
model.save_model()

df.to_csv('data/training_data.csv', index=False)
df.head(100).to_csv('data/sample_claims.csv', index=False)

print("\n" + "="*60)
print("✅ COMPLETE!")
print("="*60)
