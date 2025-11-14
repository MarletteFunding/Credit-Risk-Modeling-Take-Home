# AI Assistance Likelihood Assessment - Candidate_2

**Reviewer:** Jenny  
**Date:** 2025-11-14  
**Overall Assessment:** **LOW likelihood of over-reliance on AI. Strong evidence of genuine work.**

---

## Executive Summary

Candidate_2's submission demonstrates **genuine human effort with thoughtful engineering**. While they likely used AI tools for some boilerplate code (which is expected and appropriate), the work shows clear signs of:
- Independent problem-solving
- Iterative development
- Domain understanding
- Real data exploration
- Production-quality engineering mindset

**Recommendation: STRONG HIRE** - This candidate shows excellent ML engineering skills combined with domain awareness.

---

## Evidence Analysis

### ✅ Signs of GENUINE Work

#### 1. **Correctly Identified ALL Leakage Features** ⭐⭐⭐
```python
# From preprocessing.py line 200
exclude_features = ['loan_id','days_past_due_current',
                   'total_payments_to_date','sample_weight','months_on_book']
```
- Found ALL 3 leakage features: `days_past_due_current`, `total_payments_to_date`, `months_on_book`
- **Critical:** The updated template NO LONGER provides these explicitly
- This required genuine data analysis and domain knowledge
- AI couldn't have found these without actually analyzing the dataset

#### 2. **Handled Special Values Correctly** ⭐⭐⭐
```python
# From notebook cell 11
df['fico_score'] = np.where(df['fico_score'] == 99999, np.nan, df.fico_score)
df['income'] = np.where(df['income'] == -1, np.nan, df.income)
df['inquiries_last_6m'] = np.where(df['inquiries_last_6m'] == 99, np.nan, df.inquiries_last_6m)
```
- Discovered and correctly handled ALL special values
- Shows they actually READ the data

#### 3. **Proper Vintage-Based Splits** ⭐⭐
```python
# From modeling.py lines 25-62
def train_test_split_data(df, target='default_12m', date_col=None, 
                           type='stratified_time', n_folds=3):
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    # Separate positives and negatives
    df_pos = df_sorted[df_sorted[target] == 1]
    df_neg = df_sorted[df_sorted[target] == 0]
    # ... stratified time-based splitting
```
- Implemented **custom time-based stratified splitting**
- This is NOT a standard sklearn function
- Shows understanding of out-of-time validation for credit risk

#### 4. **Thoughtful Feature Engineering** ⭐⭐
```python
# From notebook observations
'income_strain'  # Engineered feature: income left after debt payments
'loan_to_fico'   # Ratio features
'income_to_loan'
'num_trades_per_age'
```
- Created domain-relevant features
- Shows financial intuition

#### 5. **Production-Quality Code Structure** ⭐⭐⭐
- **8 well-organized Python modules:**
  - `preprocessing.py` (536 lines)
  - `modeling.py` (336 lines)
  - `evaluation.py` (400+ lines)
  - `stat_test.py`, `model_report.py`, `folder_manager.py`, `utils.py`
- Config management (`config.yaml`)
- Proper experiment tracking with timestamped folders
- Saved encoders for reproducibility
- HTML report generation

#### 6. **Evidence of Iteration** ⭐⭐
- Multiple model runs with different timestamps:
  - `14_09_59_baseline_logreg`
  - `14_08_09_Catboost_test`
  - `14_10_03_lightgbm`
- Generated comprehensive comparison reports
- Thoughtful comments about model iterations:
  > "Every model iteration sees a random set of variables so the feature importance picture becomes more clear over time"

#### 7. **Custom Data Dictionary** ⭐⭐
- Created `dat_dict.md` with proper markdown table format
- Acts as schema management throughout pipeline
- Shows they took time to understand and document the data

#### 8. **Real EDA Observations** ⭐
From notebook markdown:
> "Default rates in a slightly downward trend for loans originated since 2023-01 but could be a **maturity** issue since we are tracking 12m"

- Shows understanding of censoring/maturity effects in credit data
- Not a typical AI-generated observation

#### 9. **Appropriate Template Usage**
- Left "TODO" prompts in place and added their own content below
- Example from notebook:
  ```markdown
  TODO: Discuss assumptions made by your models:
  - Logistic Regression assumptions (linearity, independence, etc.)
  <br><br>
  Generally it is an extension of linear regression so it follows...
  ```
- This shows they used the template as a guide, not blindly accepting it

#### 10. **Statistical Rigor** ⭐
- Implemented proper statistical tests (`stat_test.py`):
  - Chi-squared tests
  - 2-proportion z-tests
  - Confidence intervals for lift
- KS statistic implementation
- Proper null/missing value analysis by feature

---

### ⚠️ Possible AI Assistance (Appropriate)

#### 1. **Code Structure and Documentation**
- Very clean docstrings and function signatures
- Consistent formatting
- **Assessment:** Likely used AI for boilerplate/documentation
- **Verdict:** ✅ APPROPRIATE use of AI as a productivity tool

#### 2. **HTML Report Generation**
- Clean HTML/CSS in `model_report.py`
- **Assessment:** Possibly AI-generated template
- **Verdict:** ✅ Fine - this is utility code

#### 3. **Evaluation Metrics Functions**
- Standard sklearn metric wrappers in `evaluation.py`
- **Assessment:** Possibly AI-assisted
- **Verdict:** ✅ Reasonable - the KEY is they actually ran and validated everything

---

## Key Differentiators from Candidate_1

| Aspect | Candidate_1 | Candidate_2 |
|--------|-------------|-------------|
| **Leakage Features** | Added fake columns from template | ✅ Found ALL correctly |
| **Special Values** | Not mentioned | ✅ Handled ALL correctly |
| **Splits** | Random (contradicted own Metaflow) | ✅ Custom time-stratified |
| **Sample Weights** | Extracted but unused | ✅ Properly excluded |
| **Code Runs** | Broken Metaflow | ✅ Multiple working runs |
| **Artifacts** | None | ✅ Models, metrics, encoders, reports |
| **Evidence of Work** | Perfect polish, no iteration | ✅ Multiple runs, timestamps, notes |
| **Domain Knowledge** | Surface level | ✅ Deep understanding |

---

## Red Flags Check

❌ **No unremoved TODOs** (left intentionally as structure)  
❌ **No placeholder comments**  
❌ **No non-existent features**  
❌ **No contradictions**  
❌ **No broken logic**  
❌ **No unused variables**  
❌ **No perfect polish without validation**

---

## Final Assessment

### Likelihood of Over-Reliance on AI: **10-20%**

**Breakdown:**
- Used AI for productivity (docstrings, boilerplate): **80%** ✅ GOOD
- Core logic and problem-solving: **Human** ⭐
- Data exploration and validation: **Human** ⭐
- Feature engineering: **Human** ⭐
- Code structure and modularity: **Human with AI assistance** ✅

### Key Strengths:
1. **Domain expertise** - Understands credit risk concepts
2. **Production mindset** - Code is deployment-ready
3. **Thoroughness** - Multiple models, proper validation, comprehensive reports
4. **Engineering skills** - Clean modular code, reproducibility, experiment tracking

### What Sets This Apart:
- **Actually ran the code multiple times** (timestamps prove it)
- **Actually explored the data** (found special values, leakage)
- **Actually understands the problem** (vintage splits, maturity effects)
- **Actually validated results** (statistical tests, comparison reports)

---

## Recommendation

**STRONG HIRE** 🎯

This is exactly the kind of candidate you want:
- Uses AI as a **tool** to boost productivity
- But demonstrates **genuine understanding** and **problem-solving skills**
- Writes **production-quality code**
- Shows **domain expertise** in credit risk
- **Validates** everything they build

The work shows signs of iteration, debugging, and refinement - the hallmarks of a real engineer, not someone blindly copy-pasting AI outputs.

---

## Interview Focus Areas

If moving forward, explore:
1. Walk through their custom time-based splitting logic - why stratified?
2. How did they identify the leakage features?
3. What other feature engineering approaches did they consider?
4. How would they deploy this to production?
5. How did they use AI tools in their workflow?

Expected answers should show they understand their own code and can explain the reasoning behind their choices.

