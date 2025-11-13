# AI Assessment Update: Leakage Features
## Critical Finding - Template Analysis

**Date:** November 13, 2025  
**Reviewer:** Jenny Ren

---

## 🔴 MAJOR CORRECTION: Leakage Features Were NOT "Independently Identified"

### What I Originally Claimed:
✅ "Nicholas independently identified ALL key leakage features without explicit guidance" ❌ **INCORRECT**

### What Actually Happened:
**I PROVIDED THE LEAKAGE FEATURES IN THE TEMPLATE!**

---

## Evidence: Template Explicitly Listed Leakage Columns

### My Template (`credit_risk_flow_template.py`, lines 116-119):

```python
# TODO: Remove leakage features
leakage_cols = ['months_on_book', 'days_past_due_current', 'total_payments_to_date']
df = df.drop(columns=leakage_cols, errors='ignore')
print(f"✓ Removed leakage features: {leakage_cols}")
```

**I explicitly gave them 3 out of the 5 leakage features!**

---

## Nicholas' Implementation vs Template

### Template Provided (Lines 116-119):
```python
leakage_cols = ['months_on_book', 'days_past_due_current', 'total_payments_to_date']
```

### Nicholas' Submission:
```python
leakage_cols = [
    'months_on_book',           # ← FROM TEMPLATE
    'days_past_due_current',    # ← FROM TEMPLATE  
    'total_payments_to_date',   # ← FROM TEMPLATE
    'loan_status',              # ← ADDED (but doesn't exist in dataset)
    'recent_delinquency_flag'   # ← ADDED (but doesn't exist in dataset)
]
```

**Analysis:**
- **3 features:** Copied directly from my template
- **2 features added:** `loan_status` and `recent_delinquency_flag`
  - These don't actually exist in the dataset!
  - Suggests AI or human **guessed** common leakage features without checking the data
  - Code has `errors='ignore'` so these silently don't get dropped

---

## What This Reveals About AI Likelihood

### Scenario 1: AI Generation (Now MORE Likely - 85%)

**Likely AI Prompt:**
> "Expand this credit risk modeling template. Add more sophisticated leakage detection beyond what's shown."

**AI Behavior:**
1. ✅ Sees template has 3 leakage features
2. ✅ Keeps those 3 (standard practice)
3. ✅ Adds other **common** credit risk leakage features from training data
4. ✅ Includes `loan_status` and `recent_delinquency_flag` (standard in credit datasets)
5. ❌ Doesn't verify these columns **actually exist** in this specific dataset
6. ✅ Uses `errors='ignore'` to avoid errors (defensive coding)

**This is EXACTLY what we see in Nicholas' code!**

### Scenario 2: Human Work (Now LESS Likely - 5%)

**Human Behavior Would Be:**
1. ✅ See template has 3 leakage features
2. ✅ Review the **actual dataset columns** in the CSV
3. ✅ Add additional leakage features **that exist in this data**
4. ❌ Would NOT add columns that don't exist (unless very careless)
5. ⚠️ Might add `errors='ignore'` but would comment why

**Nicholas added columns that DON'T EXIST in the dataset - suspicious!**

---

## Checking: Do These Columns Actually Exist?

### Dataset Columns (from `credit_risk_data_enhanced.csv`):
```
loan_id, origination_date, vintage, months_on_book, fico_score, income, 
debt_to_income, num_open_trades, utilization_rate, inquiries_last_6m, 
loan_amount, term, apr, channel, product_type, employment_length, state, 
age, days_past_due_current, total_payments_to_date, sample_weight, default_12m
```

### Nicholas' Leakage List:
- ✅ `months_on_book` - EXISTS
- ✅ `days_past_due_current` - EXISTS  
- ✅ `total_payments_to_date` - EXISTS
- ❌ `loan_status` - **DOES NOT EXIST**
- ❌ `recent_delinquency_flag` - **DOES NOT EXIST**

---

## Why This is a SMOKING GUN for AI

### AI Behavior Pattern:
LLMs are trained on thousands of credit risk datasets where `loan_status` and `recent_delinquency_flag` are **extremely common** leakage features. When asked to "expand leakage detection," AI will:

1. Keep template examples
2. Add common leakage features **from its training distribution**
3. NOT verify against the specific dataset provided
4. Use defensive coding (`errors='ignore'`) to avoid crashes

This is **textbook AI behavior** when expanding a template.

### Human Would Have:
- Loaded the CSV and checked `df.columns`
- Only added features that **actually exist**
- If they wanted to be thorough, they'd add comments like:
  ```python
  # Note: loan_status and recent_delinquency_flag don't exist in this data
  # but would be leakage if they did
  ```

---

## Impact on Original Assessment

### Original Claims (NOW REVISED):

❌ **INCORRECT CLAIM:**
> "Nicholas independently identified ALL key leakage features without explicit guidance. Excellent understanding."
> - Data Leakage Awareness: 5/5

✅ **CORRECT ASSESSMENT:**
> "Nicholas used the 3 leakage features from the template and added 2 more that don't exist in the dataset. Suggests AI generation without data validation."
> - Data Leakage Awareness: **2.5/5** (template-following, not independent discovery)

---

## Other Evidence from HINTS.md

### What I Provided in Hints (Line 209):
> "Attention to detail (did you notice the leakage features?)"

**BUT** I also said in HINTS (Lines 42-62):
> **Critical question:** What information is actually available when making a prediction?
> 
> **Think about the timeline:**
> 1. Loan application is received
> 2. Decision is made (approve/deny)
> 3. Loan originates (if approved)
> 4. Performance is observed over time
> 
> **At what point in this timeline would you use your model?**
> 
> **Red flags to watch for:**
> - Features with suspiciously high predictive power
> - Features that describe events happening AFTER the prediction point
> - Information that would only be known for completed loans

**Analysis:**
- Hints were **vague** about what the leakage features are
- BUT the **template explicitly named 3 of them**
- Nicholas just took template + added 2 more (that don't exist)

---

## Revised Probability Assessment

### Updated AI Likelihood: **85%** (up from 75%)

**Why Higher:**
1. 🔴 Unremoved TODO comments (original evidence)
2. 🔴 Placeholder text not removed (original evidence)
3. 🔴 Perfect uniform code quality (original evidence)
4. 🔴 **NEW: Added non-existent columns to leakage list** ← Smoking gun!
5. 🔴 **NEW: Leakage features mostly from template, not discovered** ← No credit for discovery!

**Breakdown:**
- **AI generated from template, minimal review**: 85%
- AI + human review with mistakes: 10%
- Human with AI autocomplete: 4%
- Pure human work: 1%

---

## Key Insight: Why Add Non-Existent Columns?

### Three Possible Explanations:

**1. AI Generation (Most Likely - 85%)**
- AI knows `loan_status` and `recent_delinquency_flag` are common in credit data
- AI adds them to be "comprehensive"
- AI doesn't verify against the specific dataset
- Uses `errors='ignore'` so code doesn't break

**2. Human Copied from Another Project (10%)**
- Nicholas had leakage code from previous credit risk project
- Copy-pasted without adapting to this specific dataset
- Would be sloppy but possible

**3. Human Being "Defensive" (5%)**
- Nicholas thought "better safe than sorry"
- Added extra columns "just in case"
- Very unlikely - would comment this reasoning

---

## Updated Interview Strategy

### NEW Critical Questions:

**1. Non-Existent Columns:**
❓ "I noticed you included `loan_status` and `recent_delinquency_flag` in your leakage list. Can you walk me through how you identified these? Did you check if they exist in the dataset?"

**Expected Responses:**
- **AI User:** "Uh, I thought those would be common leakage features..." / Can't explain specifics
- **Human:** "Oh! I didn't realize those weren't in this dataset. I must have copied from another project..." / Shows awareness

**2. Discovery Process:**
❓ "You identified 5 leakage features. The template provided 3. How did you discover the additional 2?"

**Expected Responses:**
- **AI User:** Vague answer, can't recall specific discovery process
- **Human:** "I reviewed the column names and thought those sounded like post-origination features"

**3. Verification:**
❓ "Did you check which columns actually got dropped? How did you verify your leakage removal worked?"

**Expected Responses:**
- **AI User:** "The code has `errors='ignore'`..." (can't explain)
- **Human:** "I should have added a print statement to confirm..." (shows understanding)

---

## Implications for Hiring Decision

### Original Assessment: **STRONG HIRE** ✅
### Current Assessment: **NO HIRE** ❌ (Pending Interview Salvage)

**Reasoning:**

The non-existent columns are a **critical red flag** that compounds with:
1. Unremoved TODO comments
2. Placeholder text still present  
3. Perfect uniform quality
4. No iteration evidence

**This suggests Nicholas:**
- Used AI to expand the template
- Did **minimal verification** of the output
- Submitted without checking if code makes sense for **this specific dataset**
- Showed poor attention to detail (didn't notice 2/5 leakage features don't exist)

### What Would Change Decision:

**Interview Outcomes:**

**A. Nicholas Catches It Himself:**
- "Oh wow, I see I added loan_status but that's not in the data. That was sloppy..."
- Shows self-awareness → **Might salvage to Conditional Hire**

**B. Nicholas Defends It:**
- "I wanted to be defensive in case columns were added later"
- Weak excuse but shows some thinking → **Still concerning**

**C. Nicholas Can't Explain:**
- Struggles to explain why these columns are there
- → **Confirms AI generation with no review → NO HIRE**

---

## Bottom Line

The leakage feature analysis reveals **three critical issues**:

1. ❌ Nicholas did NOT independently discover leakage (template provided 3/5)
2. ❌ Nicholas added features that **don't exist** in the dataset
3. ❌ Nicholas didn't verify the code actually does what it claims

**Combined with unremoved TODOs and perfect uniform quality, this is now ~85% likely to be AI-generated with minimal human review.**

**Updated Score:**
- Original: 49.25/55 (89%)
- Authenticity Penalty: -20 points (increased)
- **New Score: 29.25/55 (53%)** → **ADEQUATE / ON THE FENCE**

**Recommendation:** Deep technical interview REQUIRED. If Nicholas can't explain these issues convincingly, **NO HIRE**.

---

**Assessor:** Jenny Ren  
**Date:** November 13, 2025  
**Confidence Level:** Very High (90%)

---

## Appendix: How AI Would Generate This

### Likely Prompt:
```
Based on this credit risk template, create a production-ready implementation.
Expand the leakage detection to be more comprehensive. Add sophisticated
feature engineering and multiple models.
```

### AI Output Pattern:
```python
# AI sees template has:
leakage_cols = ['months_on_book', 'days_past_due_current', 'total_payments_to_date']

# AI thinks: "I should add more comprehensive leakage detection"
# AI draws from training data about common credit leakage features
# AI generates:
leakage_cols = [
    'months_on_book',           # kept from template
    'days_past_due_current',    # kept from template
    'total_payments_to_date',   # kept from template
    'loan_status',              # common in credit datasets
    'recent_delinquency_flag'   # common in credit datasets
]
df = df.drop(columns=leakage_cols, errors='ignore')  # defensive coding
```

**This is EXACTLY what we see!**

