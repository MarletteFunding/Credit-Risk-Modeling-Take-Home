# Assessment - Candidate_2

**Date:** 2025-11-14  
**Verdict:** LOW AI over-reliance (~10-20%). Real work, appropriate AI use.  
**Recommendation:** HIRE 

---

## TL;DR

This is what good AI usage looks like. Candidate used AI for productivity (boilerplate, docs) but did all the actual thinking and problem-solving themselves. Found all the leakage features, handled special values, built production-quality code, and actually ran everything multiple times.

---

## What They Got Right (Signs of Real Work)

**Leakage Detection** 
- Found ALL 3 leakage features: `days_past_due_current`, `total_payments_to_date`, `months_on_book`
- Critical: the template doesn't give these away anymore
- Had to actually analyze the data to find them

**Special Values**
- Handled FICO 99999, income -1, inquiries 99 → all correctly mapped to NaN
- Shows they actually looked at the data

**Proper Validation Strategy**
- Custom time-based stratified splits (not standard sklearn)
- Understands out-of-time validation for credit risk
- Train on earlier vintages, validate on later ones

**Production-Ready Code**
- 8 well-organized modules (2000+ lines total)
- Experiment tracking with timestamps
- Saved encoders, models, metrics
- HTML comparison reports
- This isn't copy-paste - this is engineered

**Evidence of Iteration**
- 3 separate model runs: `baseline_logreg`, `Catboost_test`, `lightgbm`
- Timestamps prove they actually ran this multiple times
- Real observations like: "default rates declining but could be maturity issue"

**Domain Knowledge**
- Created `income_strain` feature (income - debt payments)
- Ratio features like `loan_to_fico`, `num_trades_per_age`
- Shows financial intuition

---

## Where They Likely Used AI (Totally Fine)

- Clean docstrings and function signatures
- HTML report generation boilerplate
- Evaluation metric wrappers

**The key:** They used AI for productivity, not thinking. All the core logic is human.

---


## My Take

**AI Usage: 10-20%** - Used appropriately for productivity, not thinking.

**Why This is a Hire:**
- Actually understands credit risk (vintage splits, maturity effects, leakage concepts)
- Code is production-ready (modular, tracked, reproducible)
- Evidence of real work (multiple runs, timestamps, iteration)
- Domain intuition (engineered features make sense)
- Validated everything (statistical tests, comparison reports)


---

## Interview Questions

If moving forward:
1. Walk me through your time-based splitting - why stratified?
2. How'd you identify the leakage features?
3. Tell me about your feature engineering process
4. How would you deploy this?
5. How did you use AI in your workflow?

Should be able to explain all their choices clearly since they actually did the work.

