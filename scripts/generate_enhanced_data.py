"""
Generate enhanced synthetic credit risk data for take-home assignment.

This script creates a more realistic and challenging dataset with:
- Multiple vintages for out-of-time validation
- Imbalanced target (3-5% default rate)
- Special values (99999, -1, etc.) that need handling
- Sample weights for imbalanced learning
- Potential data leakage traps
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_enhanced_credit_data(n_samples=50000):
    """Generate enhanced credit risk dataset."""
    
    # Generate vintages (monthly from 2021-01 to 2024-09)
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2024, 9, 30)
    
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        # Move to next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    
    # Generate origination dates
    origination_dates = np.random.choice(dates, size=n_samples)
    vintages = [d.strftime('%Y%m') for d in origination_dates]
    origination_dates_str = [d.strftime('%Y-%m-%d') for d in origination_dates]
    
    # Calculate months on book (for data leakage trap - should NOT be used in modeling)
    mob = np.random.randint(0, 25, n_samples)  # 0-24 months on book
    
    # Generate FICO scores with special values (99999 for missing)
    fico_scores = np.random.randint(550, 851, n_samples)
    # Introduce 5% missing values as 99999
    missing_fico_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    fico_scores[missing_fico_idx] = 99999
    
    # Generate income with special values (-1 for missing)
    income = np.random.lognormal(10.8, 0.6, n_samples).astype(int)
    # Introduce 3% missing values as -1
    missing_income_idx = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    income[missing_income_idx] = -1
    
    # Generate other features
    debt_to_income = np.clip(np.random.beta(2, 3, n_samples), 0.01, 0.99)
    num_open_trades = np.random.poisson(8, n_samples)
    utilization_rate = np.clip(np.random.beta(2, 2, n_samples), 0, 1)
    
    # Inquiries with special value (99 for missing)
    inquiries_last_6m = np.random.poisson(2, n_samples)
    missing_inq_idx = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    inquiries_last_6m[missing_inq_idx] = 99
    
    loan_amount = np.random.randint(1000, 50000, n_samples)
    term = np.random.choice([36, 60], n_samples, p=[0.7, 0.3])
    apr = np.clip(np.random.normal(12, 7, n_samples), 3, 35.99)
    
    channel = np.random.choice(['Direct', 'Partner', 'Mail'], n_samples, p=[0.5, 0.4, 0.1])
    product_type = np.random.choice(
        ['Personal', 'Debt_Consolidation', 'Auto', 'Home_Improvement', 'Business'],
        n_samples,
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )
    
    employment_length = np.random.randint(0, 40, n_samples)
    state = np.random.choice(['CA', 'TX', 'FL', 'NY', 'IL', 'GA', 'Other'], n_samples,
                            p=[0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.37])
    age = np.random.randint(21, 75, n_samples)
    
    # Generate imbalanced default (3-5% default rate, varying by vintage)
    # Create propensity to default based on features
    default_propensity = (
        (fico_scores < 650) * 0.15 +  # Low FICO increases default
        (debt_to_income > 0.5) * 0.10 +  # High DTI increases default
        (utilization_rate > 0.8) * 0.08 +  # High utilization increases default
        (inquiries_last_6m > 5) * 0.07 +  # Many inquiries increases default
        (income < 40000) * 0.05 +  # Low income increases default
        np.random.normal(0, 0.02, n_samples)  # Random noise
    )
    
    # Handle special values in propensity calculation
    default_propensity[fico_scores == 99999] += 0.12  # Missing FICO is risky
    default_propensity[income == -1] += 0.08  # Missing income is risky
    default_propensity[inquiries_last_6m == 99] += 0.05  # Missing inquiries is risky
    
    # Scale to get ~4% overall default rate
    default_propensity = np.clip(default_propensity, 0, 1) * 0.20
    
    # Generate actual defaults
    default_12m = (np.random.random(n_samples) < default_propensity).astype(int)
    
    # Add vintage effect (older vintages have more mature data, so higher observed defaults)
    vintage_year = np.array([int(v[:4]) for v in vintages])
    vintage_multiplier = np.where(vintage_year <= 2022, 1.3, 
                                  np.where(vintage_year == 2023, 1.1, 0.85))
    default_12m = (np.random.random(n_samples) < default_propensity * vintage_multiplier).astype(int)
    
    # Generate sample weights (inversely proportional to class frequency + some randomness)
    # This simulates importance sampling or cost-sensitive learning
    base_weight = np.ones(n_samples)
    base_weight[default_12m == 1] = 3.0  # Upweight minority class
    # Add some random variation to weights
    sample_weight = base_weight * np.random.uniform(0.8, 1.2, n_samples)
    
    # Add performance data that SHOULD NOT be used for modeling (leakage trap)
    # This represents information only available after loan origination
    days_past_due_current = np.where(
        default_12m == 1,
        np.random.choice([30, 60, 90, 120], n_samples),
        np.random.choice([0, 0, 0, 0, 15], n_samples)  # Mostly 0, some slightly late
    )
    
    # Total payments made (also leakage - only known after origination)
    total_payments = np.where(
        default_12m == 1,
        np.random.uniform(0, 0.3, n_samples) * loan_amount,  # Defaulters paid less
        np.random.uniform(0.5, 1.0, n_samples) * loan_amount  # Non-defaulters paid more
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'loan_id': [f'LOAN_{i:06d}' for i in range(n_samples)],
        'origination_date': origination_dates_str,
        'vintage': vintages,
        'months_on_book': mob,  # LEAKAGE - don't use this
        'fico_score': fico_scores,
        'income': income,
        'debt_to_income': np.round(debt_to_income, 4),
        'num_open_trades': num_open_trades,
        'utilization_rate': np.round(utilization_rate, 4),
        'inquiries_last_6m': inquiries_last_6m,
        'loan_amount': loan_amount,
        'term': term,
        'apr': np.round(apr, 2),
        'channel': channel,
        'product_type': product_type,
        'employment_length': employment_length,
        'state': state,
        'age': age,
        'days_past_due_current': days_past_due_current,  # LEAKAGE - don't use this
        'total_payments_to_date': np.round(total_payments, 2),  # LEAKAGE - don't use this
        'sample_weight': np.round(sample_weight, 4),
        'default_12m': default_12m
    })
    
    return df


if __name__ == '__main__':
    print("Generating enhanced credit risk dataset...")
    df = generate_enhanced_credit_data(n_samples=50000)
    
    # Print summary statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"\nDefault rate: {df['default_12m'].mean():.2%}")
    print(f"\nDefault rate by vintage:")
    print(df.groupby('vintage')['default_12m'].agg(['mean', 'count']).sort_index())
    
    print(f"\nSpecial values:")
    print(f"  FICO = 99999: {(df['fico_score'] == 99999).sum()} ({(df['fico_score'] == 99999).mean():.1%})")
    print(f"  Income = -1: {(df['income'] == -1).sum()} ({(df['income'] == -1).mean():.1%})")
    print(f"  Inquiries = 99: {(df['inquiries_last_6m'] == 99).sum()} ({(df['inquiries_last_6m'] == 99).mean():.1%})")
    
    # Save to CSV
    output_path = '../data/credit_risk_data_enhanced.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Data saved to {output_path}")

