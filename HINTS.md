# Hints and Guidance

This document provides hints and clarifications for the take-home assignment. Use these if you get stuck, but try to solve problems independently first!

## 🎯 Core Assignment Hints

### 1. Handling Special Values

**Why use special values (99999, -1, 99) instead of NaN?**
- In production systems, missing data is often encoded with special values
- Different systems may use different conventions
- This tests your ability to handle real-world data quality issues

**Hint for handling:**
```python
# Option 1: Replace with NaN and impute
df['fico_score'] = df['fico_score'].replace(99999, np.nan)

# Option 2: Create a missing indicator
df['fico_missing'] = (df['fico_score'] == 99999).astype(int)
df['fico_score'] = df['fico_score'].replace(99999, df['fico_score'].median())

# Option 3: Treat as a separate category (for tree-based models)
# Some algorithms can handle this naturally
```

**Questions to consider:**
- Does missing data carry information? (e.g., missing FICO might indicate higher risk)
- How does your imputation strategy affect model performance?

### 2. Out-of-Time Validation

**Why not use random train/test split?**
- Credit models are used to predict **future** defaults
- Random splits can leak future information into training
- Vintage-based splits simulate real-world deployment

**Hint for implementation:**
```python
# Bad: Random split (temporal leakage)
X_train, X_test = train_test_split(X, y, test_size=0.2)

# Good: Vintage-based split
train_data = df[df['vintage'] <= '202212']  # Train on 2021-2022
val_data = df[(df['vintage'] >= '202301') & (df['vintage'] <= '202306')]  # Val on H1 2023
test_data = df[df['vintage'] >= '202307']  # Test on H2 2023-2024
```

**Questions to consider:**
- How does model performance vary across vintages?
- Are there seasonal patterns in defaults?
- How would you handle model degradation over time?

### 3. Identifying Data Leakage

**Features that should NOT be used:**

1. **`months_on_book`** 
   - Only known after loan origination
   - Perfect predictor if you know loan has been open for 18 months without default
   
2. **`days_past_due_current`**
   - Current delinquency status
   - Only known after origination
   - Directly related to target (defaulters are often DPD > 0)
   
3. **`total_payments_to_date`**
   - Performance data
   - Only available after origination
   - Defaulters pay less (obvious leakage)

**Red flags for leakage:**
- Features that are "too good to be true" (AUC > 0.95)
- Features that wouldn't be available at prediction time
- Features derived from the target variable

**Hint for detection:**
```python
# Check feature importance
# If months_on_book or days_past_due_current are top features, you have leakage!

# Validate by training with/without suspicious features
model_with_leakage = train_model(X_with_leakage, y)
model_without_leakage = train_model(X_without_leakage, y)

# If AUC drops dramatically, those features were leakage
```

### 4. Handling Imbalanced Data

**Why is this important?**
- Only ~3% of loans default
- Without proper handling, model may just predict "no default" for everyone
- Sample weights help the model focus on minority class

**Hints for implementation:**

```python
# Option 1: Use sample weights (provided in dataset)
model.fit(X_train, y_train, sample_weight=train_weights)

# Option 2: Class weights
LogisticRegression(class_weight='balanced')
XGBClassifier(scale_pos_weight=32)  # ratio of negative to positive

# Option 3: Resampling (be careful with time series!)
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
```

**Metrics for imbalanced data:**
- **AUC-ROC**: Good for ranking, but can be misleading
- **AUC-PR**: Better for imbalanced datasets
- **KS Statistic**: Common in credit risk (max separation between classes)
- **F1-Score at different thresholds**: Consider business costs

**Hint for threshold selection:**
```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Choose threshold based on business requirements
# e.g., 80% recall (catch 80% of defaults) at acceptable precision
```

---

## 🚀 Bonus Section Hints

### Metaflow Pipeline Structure

**Key concepts:**
1. **Steps**: Each step is a function decorated with `@step`
2. **Artifacts**: Data passed between steps via `self.variable`
3. **Parameters**: Run-time configuration (e.g., vintage split date)
4. **Cards**: Visual reports attached to steps

**Basic commands:**
```bash
# Run the pipeline
python credit_risk_flow.py run

# Run with custom parameters
python credit_risk_flow.py run --train_vintage_end 202212

# Show most recent run
python credit_risk_flow.py show

# List all runs
python credit_risk_flow.py list

# Access artifacts from a specific run
python credit_risk_flow.py output <run_id>
```

**Hints for implementation:**

1. **Start simple**: Get the basic pipeline working first
2. **Add logging**: Use print statements to track progress
3. **Save artifacts**: Store models, metrics, and plots
4. **Parameter sweeps**: Try multiple configurations
5. **Error handling**: Wrap steps in try/except

**Advanced features to consider:**
- `@retry` decorator for fault tolerance
- `@resources` for compute requirements
- `@batch` for parallel processing
- Custom decorators for timing/logging

---

## 💡 General Tips

### Time Management

**Suggested allocation (6-8 hours total):**
- EDA and data quality (1-2 hours)
- Feature engineering (1-2 hours)
- Model training and tuning (2-3 hours)
- Evaluation and documentation (1-2 hours)
- Code cleanup and testing (1 hour)

**If short on time, prioritize:**
1. ✅ Proper train/test split (vintage-based)
2. ✅ Handling special values correctly
3. ✅ Avoiding data leakage
4. ✅ One well-tuned model > multiple poor models
5. ✅ Clear documentation of your approach

### Code Quality

**We're looking for:**
- Modular functions (not one giant notebook cell)
- Clear variable names
- Comments explaining non-obvious decisions
- Reproducible results (set random seeds!)
- Proper error handling

**Example of good code structure:**
```python
def handle_special_values(df, config):
    """
    Replace special values with appropriate imputations.
    
    Args:
        df: Input dataframe
        config: Dict with column names and special values
    
    Returns:
        DataFrame with special values handled
    """
    df = df.copy()
    
    for col, special_val in config.items():
        if col in df.columns:
            # Create missing indicator
            df[f'{col}_missing'] = (df[col] == special_val).astype(int)
            
            # Impute with median
            median_val = df[df[col] != special_val][col].median()
            df[col] = df[col].replace(special_val, median_val)
    
    return df

# Usage
special_values_config = {
    'fico_score': 99999,
    'income': -1,
    'inquiries_last_6m': 99
}
df = handle_special_values(df, special_values_config)
```

### Common Mistakes to Avoid

1. ❌ Using random train/test split
2. ❌ Using leakage features
3. ❌ Ignoring class imbalance
4. ❌ Not handling special values
5. ❌ Overfitting to validation set
6. ❌ Not documenting assumptions
7. ❌ Presenting only final results (show your process!)

### What We're Really Testing

**Technical skills:**
- Understanding of credit risk modeling concepts
- Ability to handle messy, real-world data
- Proper ML methodology (train/val/test, cross-validation)
- Feature engineering creativity

**Soft skills:**
- Communication (can you explain your decisions?)
- Attention to detail (did you notice the leakage features?)
- Problem-solving (how did you handle challenges?)
- Production readiness (is your code maintainable?)

---

## ❓ FAQ

**Q: Can I use external data sources?**
A: No, please use only the provided dataset.

**Q: How much feature engineering is expected?**
A: We don't expect hours of feature engineering. A few well-thought-out features are better than dozens of random transformations.

**Q: Should I tune hyperparameters extensively?**
A: Basic tuning is good, but don't spend hours on it. Focus on methodology and code quality.

**Q: Can I use AutoML tools?**
A: For the core assignment, we prefer to see your modeling decisions. For the bonus section, AutoML is acceptable if you can explain what it's doing.

**Q: What if my model performance is poor?**
A: That's okay! Document what you tried and why you think it didn't work. Learning from failures is important.

**Q: How important is the bonus section?**
A: It's truly optional. A great core submission is better than a rushed attempt at both.

---

Good luck! Remember, we're more interested in your thought process and approach than perfect results. 🚀

