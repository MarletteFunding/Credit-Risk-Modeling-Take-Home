# AI Generation Likelihood Assessment
## Candidate: Nicholas

**Date:** November 13, 2025  
**Reviewer:** Jenny Ren

---

## Executive Summary

**AI Likelihood:** 
- **Metaflow/Utilities:** 95% AI-generated
- **Notebook:** 80% AI-generated
- **Overall:** 90% AI-generated with minimal human validation


---

## Red Flags

### 1. Non-Existent Columns Added 
- Added `loan_status` and `recent_delinquency_flag` to leakage removal
- **These columns don't exist in the dataset**
- Classic AI behavior: adds "common" features without verifying
- Code uses `errors='ignore'` so fails silently

### 2. Broken Sample Weights Logic 
- Creates `use_sample_weights` parameter
- Extracts `self.sample_weights` from data
- **Never passes weights to training functions**
- Template had this working; Nicholas' version is broken
- Also contradicts notebook which removes sample_weight as leakage

### 3. Unremoved Template TODOs
- 6+ instances of "TODO for candidates:" still in code
- Appears right above completed implementations
- Human would delete TODOs; AI preserves template structure

### 4. Placeholder Comments Still Present 
- 6+ instances of "This is a placeholder - candidates should implement"
- Followed by 50-100 lines of sophisticated code
- Human would remove placeholders after implementing

### 5. Leakage Features Provided in Template 
- I gave Nicholas 3/5 leakage features in template
- He didn't independently discover them
- Just copied template + added 2 that don't exist

---

## Metaflow Analysis (95% AI)

### Evidence It Was Never Tested:
-  Has critical bugs (unused sample_weights)
-  References non-existent columns
-  Contradictory logic (use weights vs remove as leakage)
-  Template TODOs not removed
-  Placeholder text not removed

### AI Indicators:
- ✓ Perfect uniform code quality across all files
- ✓ Identical docstring format (100% of functions)
- ✓ Numbered comment sections (Step 1, Step 2...)
- ✓ Zero debugging artifacts
- ✓ No iteration evidence
- ✓ Code expanded from 290 → 2,000+ lines

---

## Notebook Analysis (80% AI)

### Stats:
- 36 cells (19 code, 17 markdown)
- 18/19 cells have outputs (was executed)
- Execution: [1,2,3...10,11,12,**21**,22...27] ← deleted cells 13-20
- **No re-runs** (perfectly sequential)
- 630 lines of code

### Evidence of SOME Human Effort:
-  Actually ran the notebook (has outputs)
-  Deleted some cells (12→21 gap)
-  No critical bugs (unlike Metaflow)
-  Different validation approach (random vs vintage)


### The Smoking Gun - Inconsistency:
- **Notebook:** Uses random train/test split
- **Metaflow:** Uses vintage-based temporal split
- Generated separately, never integrated
- Shows lack of conceptual understanding

---


## What Actually Happened (My Theory)

### Metaflow Generation:
1. Pasted template into ChatGPT/Claude
2. Prompt: *"Expand into production-ready code with comprehensive features"*
3. Copy-pasted AI output
4. **Never tested it**
5. Submitted

### Notebook Generation:  
1. AI generated comprehensive EDA notebook
2. **Actually opened and ran it** (20% human effort)
3. Deleted a few cells (minor tweaks)
4. Submitted

### Key Issues:
- Code not tested (Metaflow has bugs)
- Code not reviewed (TODOs/placeholders remain)
- Code not understood (contradictions, non-existent columns)
- Components not integrated (random vs vintage splits)


---

## My Honest Take

**When I first reviewed this, I was impressed.** The code looked professional, comprehensive, well-organized. I was ready to give a strong hire.

**Then you asked about AI.** Once I started looking... it all fell apart.

**The Smoking Guns:**
- Added columns that **don't exist** in the data
- Code that **extracts but never uses** sample weights  
- TODOs **still in the code** after implementing
- Template **bugs fixed**, but Nicholas' version **broken**

**The Verdict:**
- Metaflow: 95% AI, 0% validation, has critical bugs
- Notebook: 80% AI, ran once, no critical bugs but no depth
- Overall: Submitted AI output without meaningful review


---

**Mood:** Initially impressed → now just impressed by AI's capabilities 
