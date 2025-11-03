# Take-Home Assignment Distribution Package

## 📦 Package Information

**File:** `credit-risk-takehome-candidate.zip`
**Size:** ~2.0 MB
**Location:** `/home/ob-workspace/jenny/credit-risk-takehome-candidate.zip`

---

## ✅ What's Included

### Core Files
- ✅ **README.md** - Full assignment instructions (vague, requires discovery)
- ✅ **INSTRUCTIONS.txt** - Quick start guide
- ✅ **HINTS.md** - Optional hints (guides thinking, doesn't give answers)
- ✅ **requirements.txt** - Python dependencies
- ✅ **.gitignore** - Standard git ignore file

### Data
- ✅ **data/credit_risk_data_enhanced.csv** (5.9 MB)
  - 50,000 records
  - Vintages: 202101-202409
  - ~3% default rate
  - Special values: 99999, -1, 99
  - **3 leakage features** (not labeled)
  - Sample weights included

### Code Templates
- ✅ **notebooks/modeling.ipynb** - Starter notebook
- ✅ **src/__init__.py**
- ✅ **src/preprocessing.py** - Empty template
- ✅ **src/modeling.py** - Empty template
- ✅ **src/evaluation.py** - Empty template
- ✅ **src/credit_risk_flow_template.py** - Metaflow bonus template

### Outputs
- ✅ **outputs/.gitkeep** - Placeholder for outputs

---

## ❌ What's NOT Included (Intentionally)

### Hidden from Candidates
- ❌ **CHANGES.md** - Contains all the answers and evaluation rubric
- ❌ **scripts/generate_enhanced_data.py** - Data generation script
- ❌ **data/credit_risk_data.csv** - Old baseline dataset
- ❌ **.git/** - Git history
- ❌ **GitHub repository link** - Prevents seeing commit history

---

## 🎯 What Candidates Must Discover

### Data Quality Issues (Not Explicitly Stated)
1. **Special Values**
   - FICO = 99999 (5% of data)
   - Income = -1 (3% of data)
   - Inquiries = 99 (2% of data)

2. **Data Leakage Features** (Not Labeled)
   - `months_on_book` - post-origination
   - `days_past_due_current` - performance data
   - `total_payments_to_date` - performance data

3. **Validation Strategy** (Not Prescribed)
   - Must realize random split is wrong
   - Must implement vintage-based out-of-time validation

4. **Class Imbalance** (Not Highlighted)
   - Only 3% default rate
   - Must address in methodology

---

## 📧 Distribution Instructions

### How to Send to Candidates

**Option 1: Email**
```
Subject: Data Scientist Take-Home Assignment

Hi [Candidate Name],

Thank you for your interest in the Data Scientist position at Marlette Funding!

Attached is a take-home assignment that we'd like you to complete. 
Please see the INSTRUCTIONS.txt file in the zip for quick start guide, 
and README.md for full details.

Estimated time: 6-8 hours for core assignment

Please submit your completed work by [DATE].

If you have any questions, feel free to reach out to jing.ren@bestegg.com

Best regards,
[Your Name]
```

**Option 2: File Sharing Service**
- Upload to Google Drive / Dropbox / etc.
- Share link with candidate
- Set expiration date if desired

---

## 🔍 What Strong Candidates Will Do

### Discovery
- ✅ Run EDA and discover special values
- ✅ Notice temporal structure and use vintage-based validation
- ✅ Identify leakage features through reasoning
- ✅ Recognize class imbalance

### Communication
- ✅ Ask clarifying questions when unsure
- ✅ Document their decision-making process
- ✅ Explain WHY they made certain choices
- ✅ Highlight concerns or assumptions

### Technical Execution
- ✅ Clean, modular code
- ✅ Proper validation methodology
- ✅ Appropriate metrics for imbalanced data
- ✅ Feature engineering with clear rationale

---

## 📊 Evaluation Checklist

### Critical Issues (Auto-Fail if Missed)
- [ ] Uses random train/test split (temporal leakage)
- [ ] Includes obvious leakage features without comment
- [ ] Doesn't address class imbalance at all
- [ ] Ignores special values (uses 99999 as real FICO)

### Red Flags
- [ ] No questions asked despite ambiguity
- [ ] Only shows final results (no process)
- [ ] Code is messy/uncommented
- [ ] Weak justification for choices

### Green Flags
- [ ] Vintage-based validation with explanation
- [ ] Identifies and excludes leakage features
- [ ] Handles special values thoughtfully
- [ ] Uses appropriate metrics (AUC-PR, KS)
- [ ] Analyzes performance across vintages
- [ ] Clean, documented code
- [ ] Asks good clarifying questions

---

## 🔄 Regenerating the Package

If you need to regenerate the package:

```bash
cd /home/ob-workspace/jenny/Credit-Risk-Modeling-Take-Home

# Regenerate data if needed
cd scripts
python generate_enhanced_data.py
cd ..

# Recreate package (run these commands manually or create script)
rm -rf /tmp/credit-risk-takehome-candidate
mkdir -p /tmp/credit-risk-takehome-candidate/{data,notebooks,src,outputs}

# Copy files
cp README.md HINTS.md requirements.txt .gitignore /tmp/credit-risk-takehome-candidate/
cp data/credit_risk_data_enhanced.csv /tmp/credit-risk-takehome-candidate/data/
cp notebooks/modeling.ipynb /tmp/credit-risk-takehome-candidate/notebooks/
cp src/*.py /tmp/credit-risk-takehome-candidate/src/

# Create INSTRUCTIONS.txt (copy from existing or recreate)

# Create zip
cd /tmp
python3 -c "
import zipfile
import os
def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, os.path.dirname(path))
            ziph.write(file_path, arcname)
with zipfile.ZipFile('credit-risk-takehome-candidate.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipdir('credit-risk-takehome-candidate', zipf)
"

# Move to workspace
cp credit-risk-takehome-candidate.zip /home/ob-workspace/jenny/
```

---

## 📝 Version History

**v1.0** (Nov 3, 2024)
- Initial enhanced version
- Removed prescriptive guidance
- Made challenges discoverable
- 50K records, 3% default, vintages 2021-2024
- 3 leakage traps, special values, sample weights

---

## ⚠️ Important Reminders

1. **Never share CHANGES.md** - It contains all the answers
2. **Keep GitHub repo private** - Prevents candidates from seeing history
3. **Candidates can email questions** - This is part of the test
4. **Expected completion time: 6-8 hours** - Not including bonus section
5. **Bonus is truly optional** - A strong core submission is better

---

## 🎯 Success Criteria Summary

The ideal candidate will:
- Independently discover data quality issues
- Implement proper vintage-based validation
- Identify and avoid leakage features
- Handle class imbalance appropriately
- Write clean, production-quality code
- Communicate effectively and ask questions
- Demonstrate both technical skills and business judgment

Good luck with your candidate evaluations! 🚀

