# Credit Risk Modeling - Take-Home Assignment

**Role:** Data Scientist / Machine Learning Engineer  
**Estimated Time:** 4–6 hours

## 🎯 Objective

Build and document a small, production-ready modeling workflow that predicts the probability of 12-month loan default, using the provided dataset.

Your solution should demonstrate both **modeling skill** and **engineering maturity** — writing clean, modular, and reproducible Python code.

---

## 📁 Project Structure

```
Credit-Risk-Modeling-Take-Home/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Files to ignore in git
│
├── data/
│   └── synthetic_credit_risk_data.csv # Dataset for modeling
│
├── notebooks/
│   └── modeling.ipynb                 # Your main deliverable
│
├── src/                               # (Optional) Modular code
│   ├── __init__.py
│   ├── preprocessing.py               # Feature engineering utilities
│   ├── modeling.py                    # Model training utilities
│   └── evaluation.py                  # Evaluation metrics
│
└── outputs/                           # (Optional) Save models/plots here
```

---

## 🧮 Part 1: Modeling

### Dataset: `data/synthetic_credit_risk_data.csv`

Each record represents a loan at origination.

| Column | Type | Description |
|--------|------|-------------|
| `fico_score` | int | Credit score (550–850) |
| `income` | int | Annual income (USD) |
| `debt_to_income` | float | Ratio of debt to income |
| `num_open_trades` | int | Number of active credit trades |
| `utilization_rate` | float | Revolving credit utilization |
| `inquiries_last_6m` | int | Credit inquiries in past 6 months |
| `loan_amount` | int | Loan principal (USD) |
| `term` | int | Loan term in months (36 or 60) |
| `apr` | float | Annual percentage rate |
| `channel` | str | Origination channel (Direct, Partner, Mail) |
| `product_type` | str | Loan purpose |
| `employment_length` | int | Years of employment |
| `state` | str | Borrower's state |
| `age` | int | Borrower's age |
| `default_12m` | int | **Target variable** (1 = default within 12 months) |

### Tasks:

1. **EDA**: Perform basic exploratory data analysis and handle missing values or outliers.

2. **Feature Engineering**: Engineer useful features (transformations, encodings, scaling, etc.).

3. **Model Training**: Train and evaluate at least two models (e.g., Logistic Regression and Gradient Boosted Trees).

4. **Evaluation**: Compare metrics (AUC, KS, Precision-Recall) and interpret key drivers of default.

5. **Discussion**: Discuss model assumptions, potential leakage, and how you'd improve performance.

### Deliverable: 
`notebooks/modeling.ipynb` (or equivalent)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <repository-url>
cd Credit-Risk-Modeling-Take-Home
```

### 2. Set up Python environment
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start working
```bash
# Launch Jupyter
jupyter notebook notebooks/modeling.ipynb

# Or use JupyterLab
jupyter lab
```

---

## 📊 Evaluation Criteria

Your submission will be evaluated on:

- **Code Quality**: Clean, readable, and modular code
- **Modeling Approach**: Sound methodology and evaluation practices
- **Feature Engineering**: Creative and appropriate feature transformations
- **Communication**: Clear documentation and insights
- **Reproducibility**: Can we run your code and get the same results?

---

## 📤 Submission Instructions

1. Complete your analysis in `notebooks/modeling.ipynb`
2. (Optional) Add modular code to `src/` if you prefer a more structured approach
3. Push your code to a GitHub repository (or provide a zip file)
4. Ensure all outputs are saved in the notebook so we can review without running
5. Include a summary of your findings and next steps at the end of the notebook

---

## ❓ Questions?

If you have any questions about the assignment, please reach out to [contact email].

Good luck! 🎉
