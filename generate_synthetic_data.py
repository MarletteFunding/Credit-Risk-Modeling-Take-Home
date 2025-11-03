"""
Generate synthetic credit risk dataset for the take-home assignment.
This script creates a realistic dataset with ~10,000 loan records.
"""

import random
import csv

# Set random seed for reproducibility
random.seed(42)

# Number of samples
n_samples = 10000

# Define column structure
columns = [
    'fico_score', 'income', 'debt_to_income', 'num_open_trades', 
    'utilization_rate', 'inquiries_last_6m', 'loan_amount', 'term', 
    'apr', 'channel', 'product_type', 'employment_length', 
    'state', 'age', 'default_12m'
]

print("Generating synthetic credit risk dataset...")

# Generate data
data = []
for i in range(n_samples):
    # Generate features
    fico_score = random.randint(550, 850)
    income = random.randint(20000, 200000)
    debt_to_income = round(random.uniform(0.0, 0.8), 2)
    num_open_trades = random.randint(0, 20)
    utilization_rate = round(random.uniform(0.0, 1.0), 2)
    inquiries_last_6m = random.randint(0, 10)
    loan_amount = random.randint(1000, 50000)
    term = random.choice([36, 36, 36, 36, 36, 36, 36, 60, 60, 60])  # 70% 36, 30% 60
    apr = round(random.uniform(5.0, 35.0), 2)
    
    # Categorical variables
    channel = random.choices(['Direct', 'Partner', 'Mail'], weights=[0.5, 0.35, 0.15])[0]
    product_type = random.choices(
        ['Personal', 'Debt_Consolidation', 'Home_Improvement', 'Auto', 'Business'],
        weights=[0.3, 0.25, 0.2, 0.15, 0.1]
    )[0]
    state = random.choices(
        ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI', 'Other'],
        weights=[0.15, 0.12, 0.1, 0.1, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.16]
    )[0]
    
    employment_length = random.randint(0, 40)
    age = random.randint(21, 75)
    
    # Generate default probability with realistic relationships
    default_prob = (
        0.05 +  # baseline
        (850 - fico_score) / 3000 +  # FICO effect
        debt_to_income * 0.15 +       # DTI effect
        utilization_rate * 0.08 +     # Utilization effect
        inquiries_last_6m * 0.01 +    # Inquiries effect
        (apr - 5) / 300 +              # APR effect
        random.gauss(0, 0.05)          # Random noise
    )
    
    # Clip probabilities to [0, 1]
    default_prob = max(0, min(1, default_prob))
    
    # Generate binary outcome
    default_12m = 1 if random.random() < default_prob else 0
    
    # Introduce some missing values (realistic scenario) - about 2%
    if random.random() < 0.02:
        employment_length = ''
    if random.random() < 0.02:
        debt_to_income = ''
    if random.random() < 0.02:
        inquiries_last_6m = ''
    
    # Add a few outliers (extremely high utilization)
    if random.random() < 0.005:
        utilization_rate = round(random.uniform(1.0, 1.5), 2)
    
    # Create row
    row = [
        fico_score, income, debt_to_income, num_open_trades,
        utilization_rate, inquiries_last_6m, loan_amount, term,
        apr, channel, product_type, employment_length,
        state, age, default_12m
    ]
    
    data.append(row)

# Write to CSV
output_path = 'data/synthetic_credit_risk_data.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(data)

# Calculate default rate
default_count = sum(1 for row in data if row[-1] == 1)
default_rate = default_count / n_samples

print(f"✅ Synthetic dataset created successfully!")
print(f"📁 Saved to: {output_path}")
print(f"📊 Shape: ({n_samples}, {len(columns)})")
print(f"📈 Default rate: {default_rate:.2%}")
print(f"\n🎉 Dataset is ready for candidates to use!")
