"""Patient Payment Propensity - Training Script"""
import os
from generate_data import PatientPaymentDataGenerator
from model import PaymentPropensityModel

os.makedirs('data', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

print("\n" + "="*60)
print("PATIENT PAYMENT PROPENSITY - MODEL TRAINING")
print("="*60)

# Generate data
generator = PatientPaymentDataGenerator(num_patients=1000)
df = generator.generate_complete_dataset()

# Train model
model = PaymentPropensityModel()
model.train(df)
model.save_model()

# Save evaluation
df.to_csv('data/training_data.csv', index=False)
df.head(100).to_csv('data/sample_patients.csv', index=False)

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
