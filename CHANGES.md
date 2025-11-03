# Take-Home Assignment Improvements - Summary

## 📋 Overview

This document summarizes the enhancements made to the Credit Risk Modeling take-home assignment to make it more challenging and realistic.

---

## 🎯 Key Improvements

### 1. Enhanced Dataset Complexity

**Before:**
- Simple dataset with ~50% default rate
- No special values or missing data handling required
- No temporal structure

**After:**
- **50,000 records** with realistic **~3% default rate** (highly imbalanced)
- **Special values** requiring handling:
  - `fico_score = 99999` for missing (~5% of data)
  - `income = -1` for missing (~3% of data)
  - `inquiries_last_6m = 99` for missing (~2% of data)
- **Vintage structure**: Monthly vintages from 202101 to 202409
  - Tests understanding of out-of-time validation
  - Simulates real credit model development

### 2. Data Leakage Challenges

Added **three leakage features** that candidates must identify and avoid:

1. **`months_on_book`**: Time since origination (not available at decision time)
2. **`days_past_due_current`**: Current delinquency status (post-origination)
3. **`total_payments_to_date`**: Payment history (post-origination)

These are subtle traps that separate candidates who understand production ML from those who don't.

### 3. Sample Weights

Added **`sample_weight`** column to:
- Test understanding of weighted learning for imbalanced data
- Simulate cost-sensitive learning scenarios
- Evaluate proper use of imbalanced learning techniques

### 4. Out-of-Time Validation Requirement

**Critical for credit models:**
- Must use vintage-based train/test splits
- Cannot use random shuffling (causes temporal leakage)
- Tests understanding of time-series ML

### 5. Bonus Section: Production Pipeline

Added optional Metaflow challenge:
- Build reproducible ML pipeline
- Implement proper experiment tracking
- Create production-ready code structure
- Demonstrates engineering maturity

---

## 📁 New File Structure

```
Credit-Risk-Modeling-Take-Home/
├── README.md                               # ✨ Enhanced with new requirements
├── HINTS.md                                # 🆕 Detailed hints for candidates
├── CHANGES.md                              # 🆕 This file (for reviewers)
├── requirements.txt                        # ✨ Updated with Metaflow
├──  scripts/
│   └── generate_enhanced_data.py           # 🆕 Data generation script
├── data/
│   ├── credit_risk_data.csv                # Old dataset (keep for reference)
│   └── credit_risk_data_enhanced.csv       # 🆕 New enhanced dataset
├── src/
│   └── credit_risk_flow_template.py        # 🆕 Metaflow template (bonus)
└── notebooks/
    └── modeling.ipynb                      # To be completed by candidates
```

---

## 🎓 What We're Now Testing

### Core Skills (Required)

1. **Data Quality & Special Values** ⭐
   - Can they identify and handle non-standard missing values?
   - Do they understand why special values are used in production?

2. **Imbalanced Learning** ⭐⭐
   - Do they recognize the class imbalance problem?
   - Can they use sample weights or other techniques appropriately?
   - Do they choose appropriate metrics (AUC-PR, KS, not just accuracy)?

3. **Out-of-Time Validation** ⭐⭐⭐ (CRITICAL)
   - Do they understand temporal data cannot be randomly split?
   - Can they implement proper vintage-based validation?
   - Do they analyze performance across different time periods?

4. **Data Leakage Detection** ⭐⭐⭐ (CRITICAL)
   - Can they identify features not available at prediction time?
   - Do they understand why `months_on_book` is leakage?
   - Do they test model without suspicious features?

### Advanced Skills (Bonus)

5. **Production Pipeline Engineering** ⭐⭐
   - Can they structure code as a workflow?
   - Do they understand experiment tracking?
   - Can they create reproducible, testable pipelines?

---

## 📊 Evaluation Rubric

### Core Assignment (100 points)

| Category | Points | What to Look For |
|----------|--------|------------------|
| **Data Quality** | 20 | - Correctly handles all special values<br>- Documents imputation strategy<br>- Creates missing indicators if appropriate |
| **Modeling** | 25 | - Uses vintage-based train/test split<br>- Implements at least 2 models<br>- Proper hyperparameter tuning<br>- Uses sample weights |
| **Feature Engineering** | 15 | - Creative features<br>- Proper encoding of categoricals<br>- Feature interactions<br>- Scaling where appropriate |
| **Leakage Prevention** | 15 | - Identifies ALL three leakage features<br>- Explains why they're leakage<br>- Shows performance with/without |
| **Evaluation** | 15 | - Multiple metrics (AUC-ROC, AUC-PR, KS)<br>- Performance by vintage<br>- Calibration analysis<br>- Feature importance |
| **Communication** | 10 | - Clear documentation<br>- Visualizations<br>- Explains decisions<br>- Summary of findings |

### Bonus Section (Extra Credit)

| Category | Points | What to Look For |
|----------|--------|------------------|
| **Pipeline Design** | 30 | - Well-structured steps<br>- Proper data flow<br>- Reproducible |
| **Best Practices** | 30 | - Error handling<br>- Logging<br>- Parameter management<br>- Testing |
| **Artifacts** | 20 | - Model versioning<br>- Metric tracking<br>- Proper storage |
| **Documentation** | 20 | - Clear instructions<br>- Usage examples<br>- Explains design decisions |

---

## 🚀 How to Use This Assignment

### For Candidates

**Provide them with:**
1. Link to this repository
2. Expected completion time: 6-8 hours (core) + 2-4 hours (bonus)
3. Note that bonus section is truly optional

**Don't provide:**
- `CHANGES.md` (this file - it has the answers!)
- Solution notebooks or code

**Optional to provide:**
- `HINTS.md` - Can provide upfront or make available if they get stuck

### For Reviewers

**Red Flags:**
- ❌ Uses random train/test split → Doesn't understand time series ML
- ❌ Uses leakage features without comment → Missed critical issue
- ❌ Ignores special values → Poor data quality awareness
- ❌ Doesn't address imbalance → Weak ML fundamentals
- ❌ Only shows AUC-ROC on highly imbalanced data → Wrong metrics

**Green Flags:**
- ✅ Vintage-based validation with clear explanation
- ✅ Identifies and explains all leakage features
- ✅ Creates missing value indicators before imputation
- ✅ Uses appropriate metrics for imbalanced data
- ✅ Analyzes model performance across vintages
- ✅ Modular, well-documented code
- ✅ Discusses limitations and next steps

### Reviewing Submissions

**Suggested approach:**

1. **Quick scan (5 min)**:
   - Check train/test split methodology
   - Look for leakage features in model
   - Check if special values are handled

2. **Detailed review (30-45 min)**:
   - Read through notebook/code systematically
   - Evaluate each rubric category
   - Run code if time permits

3. **Decision making**:
   - **Strong Pass**: 80+ points, demonstrates all critical concepts
   - **Pass**: 70-79 points, good fundamentals with minor gaps
   - **Borderline**: 60-69 points, missing some critical concepts
   - **No Pass**: <60 points, significant gaps in understanding

---

## 🔄 Regenerating Data

If you need to regenerate the dataset (e.g., to change parameters):

```bash
cd scripts
python generate_enhanced_data.py
```

You can modify the script to:
- Change default rate
- Adjust vintage range
- Modify special value percentages
- Change sample size

---

## 📝 Notes for Future Updates

**Potential additions:**
1. Add multi-product complexity (different default rates by product)
2. Include macroeconomic features (unemployment, interest rates)
3. Add geographic risk variations
4. Include fraud indicators (red herring features)
5. Seasonal patterns in defaults

**Feedback to collect:**
- How long did candidates take?
- Which parts were most challenging?
- Did anyone miss the leakage features?
- Was the bonus section attempted?

---

## 🎉 Summary

This enhanced take-home now tests:
- ✅ Real-world data quality issues (special values)
- ✅ Critical credit risk concepts (vintages, out-of-time validation)
- ✅ Production ML awareness (data leakage prevention)
- ✅ Imbalanced learning techniques
- ✅ Engineering best practices (optional bonus section)

**Estimated difficulty:**
- Previous version: Junior/Mid-level
- Current version: **Mid/Senior-level**

**Time investment:**
- Core: 6-8 hours (realistic for quality submission)
- Bonus: +2-4 hours (truly optional)

The assignment now better reflects the actual challenges of building production credit models at Marlette Funding! 🚀

