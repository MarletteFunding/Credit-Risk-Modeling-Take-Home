# Making the Assignment More AI-Resistant

## Current Vulnerabilities

A sophisticated candidate could use AI assistants to:
- Generate standard ML code (data loading, modeling, evaluation)
- Handle special values if they point them out
- Implement validation strategies if they ask the right questions

## Recommended Enhancements

### 1. Add Written Explanation Requirements

**Current:** Tasks are mostly coding-focused

**Improved:** Add explicit sections requiring written explanations:

```markdown
### Required Written Sections (in your notebook):

1. **Data Quality Assessment** (1-2 paragraphs)
   - What data quality issues did you identify and HOW did you find them?
   - Why are these issues important for a credit model specifically?
   - What other issues might exist that you didn't have time to investigate?

2. **Feature Selection Justification** (1-2 paragraphs)
   - Walk through your decision process for which features to include/exclude
   - For any features you excluded, explain your reasoning
   - What additional features would you want in a production model?

3. **Validation Strategy Rationale** (1 paragraph)
   - Explain WHY you chose your specific train/test/validation approach
   - What are the risks of alternative approaches?
   - How would you validate this model in production?

4. **Business Impact Analysis** (1 paragraph)
   - If we deploy this model, what decisions will it drive?
   - What's the cost of a false positive vs. false negative?
   - How would you communicate model results to non-technical stakeholders?

5. **Model Limitations and Next Steps** (1-2 paragraphs)
   - What are the biggest weaknesses of your current approach?
   - What would you do differently with more time?
   - What data would you want that's not in this dataset?
```

**Why this helps:** AI can generate generic explanations, but domain-specific reasoning and personal decision-making process is harder to fake.

---

### 2. Add Contradictory or Red Herring Features

**Add to the dataset:**
```python
# Generate misleading features that seem predictive but aren't
df['credit_score_v2'] = df['fico_score'] * 0.95 + np.random.normal(0, 10, n)  # Highly correlated
df['risk_flag'] = (df['default_12m'] == 1).astype(int) + np.random.binomial(1, 0.1, n)  # 90% accurate but leakage
df['random_score'] = np.random.random(n)  # Should have no predictive power
```

**Question to add:**
"Explain why you included or excluded each feature. For any features with suspicious patterns, document your investigation."

**Why this helps:** Requires critical thinking to identify which features are legitimate vs. problematic. AI won't automatically flag subtle issues.

---

### 3. Include Follow-Up Interview Questions

**Add to DISTRIBUTION_NOTES.md:**

```markdown
## Follow-Up Interview Questions

Use these to probe understanding during technical interview:

### For Code/Methodology:
1. "I see you handled missing values by [X]. Why did you choose this over [Y]?"
2. "Walk me through how you decided which features to use."
3. "You used [validation strategy]. What would break if we did [alternative]?"
4. "Your model has AUC of X. Is that good? How do you know?"

### For Business Understanding:
5. "When in the loan lifecycle would we use this model?"
6. "If we reject a good applicant, what's the cost? What about approving a bad one?"
7. "You found default rate varies by vintage. What business factors might cause this?"
8. "How would you explain your model to a credit officer who will use it?"

### For Production Readiness:
9. "You're deploying this tomorrow. What's your biggest concern?"
10. "How would you monitor this model in production?"
11. "If performance drops 6 months from now, what would you check first?"

### AI Detection Questions:
12. "Tell me about something that didn't work. What did you try?"
13. "What was the hardest part of this assignment?"
14. "If you could redo one thing, what would it be?"
```

**Why this helps:** Follow-up questions reveal depth of understanding. Candidates using AI will struggle to explain their reasoning or discuss failures.

---

### 4. Add Time-Pressure Elements

**Current:** "6-8 hours estimated"

**Improved:** 
```markdown
### Submission Format

Please document your work chronologically:

1. **Initial Approach** (first 2 hours)
   - What did you do first?
   - What hypotheses did you form from initial EDA?

2. **Mid-Point Check** (after 4 hours)
   - What's working? What isn't?
   - What did you learn that changed your approach?

3. **Final Results**
   - Final model and performance
   - What would you do with more time?

Include timestamps in your notebook showing progression.
```

**Why this helps:** 
- Real work shows iteration and learning
- AI-generated solutions tend to be too perfect/linear
- Shows problem-solving process, not just final answer

---

### 5. Add Dataset Quirks That Require Investigation

**Enhancements to data generation:**

```python
# Add systematic missingness (not random)
# e.g., missing FICO only for certain states or channels
df.loc[(df['state'] == 'CA') & (np.random.random(n) < 0.15), 'fico_score'] = 99999

# Add time-dependent patterns
# e.g., default rate changes after certain date (economic shock)
df.loc[df['vintage'] >= '202303', 'default_12m'] *= 0.7  # Lower defaults in recent months

# Add interaction effects
# e.g., high utilization is only risky for low FICO
df['true_risk'] = np.where(
    (df['fico_score'] < 650) & (df['utilization_rate'] > 0.8),
    df['true_risk'] * 2,
    df['true_risk']
)
```

**Questions to add:**
- "Did you notice any patterns in missing data? What might cause them?"
- "Default rates vary significantly by time. What might explain this?"
- "Are there any interaction effects worth exploring?"

**Why this helps:** Requires genuine exploration and hypothesis testing, not just following standard workflows.

---

### 6. Require a "Mistake Log"

**Add to requirements:**

```markdown
### Required: Document Your Mistakes

Create a section titled "What Didn't Work" that includes:

1. At least 3 approaches you tried that didn't work as expected
2. Why you thought each approach would work
3. What you learned from each failure
4. How it influenced your final approach

Example format:
- **Mistake 1:** Used random train/test split initially
  - Why I tried it: Standard practice for most ML problems
  - What went wrong: Model performance collapsed on recent vintages
  - What I learned: Time-series data requires temporal validation
  - How I fixed it: Implemented vintage-based splits
```

**Why this helps:** 
- AI solutions rarely document failures
- Real candidates will have tried things that didn't work
- Shows authentic problem-solving process

---

### 7. Add Explainability Requirements

**Add to tasks:**

```markdown
### Model Interpretation (Required)

1. Generate feature importance and explain:
   - Top 3 most important features - do they make business sense?
   - Any surprising features? Why might they be important?
   - Any expected features that AREN'T important? Why not?

2. Pick 3 specific loans from test set:
   - One correctly predicted default
   - One correctly predicted non-default  
   - One wrong prediction
   
   For each, explain IN PLAIN ENGLISH why the model made its prediction.

3. Create 2-3 "what-if" scenarios:
   - "If we reduced APR by 5%, how would default risk change?"
   - "What's the minimum FICO score for our target default rate?"
```

**Why this helps:** Requires deep understanding of model behavior and business context. Hard for AI to generate authentic explanations.

---

## Detection Signals

### Green Flags (Likely Authentic):
- ✅ Shows iteration and learning process
- ✅ Documents dead ends and mistakes
- ✅ Inconsistent code style (natural variation)
- ✅ Domain-specific insights unique to their background
- ✅ References specific challenges they faced
- ✅ Custom visualizations that tell a story

### Red Flags (Possible AI Usage):
- 🚩 Perfect, linear progression (no mistakes)
- 🚩 Overly polished, blog-post style writing
- 🚩 Generic explanations without specific reasoning
- 🚩 All code follows identical patterns/style
- 🚩 Can't explain their choices in follow-up
- 🚩 Missing domain knowledge in interview
- 🚩 Code includes advanced techniques but weak fundamentals

---

## Recommended Changes (Priority Order)

### High Priority (Easy to Add):
1. ✅ Add written explanation requirements
2. ✅ Add "mistake log" section
3. ✅ Add follow-up interview questions to guide

### Medium Priority (Moderate Effort):
4. Add contradictory/red herring features
5. Add explainability requirements
6. Add time-pressure/chronological documentation

### Low Priority (More Work):
7. Add dataset quirks requiring investigation
8. Create alternative versions with different challenges

---

## Philosophy

**Old thinking:** "Make the problem so hard AI can't solve it"
**New thinking:** "Make the evaluation about reasoning, not just code"

**The goal isn't to prevent AI usage entirely** - in real work, engineers use tools including AI. 

**The goal is to test:**
- Can they identify problems independently?
- Do they understand WHY solutions work?
- Can they explain decisions to stakeholders?
- Do they show authentic problem-solving process?

A candidate who uses AI as a tool while demonstrating genuine understanding is potentially more valuable than one who codes everything from scratch but lacks business judgment.

---

## Implementation Plan

### Phase 1 (Immediate - No Data Changes):
- Add written explanation requirements to README
- Add "mistake log" requirement
- Create follow-up interview question bank
- Update evaluation criteria to emphasize reasoning

### Phase 2 (Next Iteration):
- Add contradictory features to dataset
- Add explainability requirements
- Create alternative versions for A/B testing

### Phase 3 (Future):
- Add time-dependent patterns
- Create more subtle data quality issues
- Develop automated detection tools

