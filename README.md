# Credit Risk Modeling - Take-Home Assignment

**Role:** Data Scientist / Machine Learning Engineer  
**Estimated Time:** 6–8 hours (Core) + 2-4 hours (Bonus)

## 🎯 Objective

Build and document a production-ready modeling workflow that predicts the probability of 12-month loan default, using a realistic, vintage-based credit dataset.

Your solution should demonstrate:
- **Modeling skill**: Sound methodology, proper validation, and evaluation
- **Engineering maturity**: Clean, modular, reproducible code
- **Domain knowledge**: Understanding of credit risk concepts (vintages, out-of-time validation, data leakage)

---

## 📁 Project Structure

```
Credit-Risk-Modeling-Take-Home/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Files to ignore in git
│
├── data/
│   └── credit_risk_data_enhanced.csv   # Dataset for modeling
│
├── notebooks/
│   └── modeling.ipynb                  # Your main deliverable
│
├── src/                                # (Optional) Modular code
│   ├── __init__.py
│   ├── preprocessing.py                # Feature engineering utilities
│   ├── modeling.py                     # Model training utilities
│   └── evaluation.py                   # Evaluation metrics
│
├── scripts/
│   └── generate_enhanced_data.py       # Data generation script
│
└── outputs/                            # (Optional) Save models/plots here
```

---

## 🧮 Part 1: Core Modeling Challenge

### Dataset: `data/credit_risk_data_enhanced.csv`

Each record represents a loan at origination. The dataset spans **multiple vintages** (2021-01 to 2024-09) with **~3% default rate** (imbalanced).

| Column | Type | Description | Special Values |
|--------|------|-------------|----------------|
| `loan_id` | str | Unique loan identifier | - |
| `origination_date` | date | Loan origination date | - |
| `vintage` | str | Origination month (YYYYMM) | - |
| `months_on_book` | int | **⚠️ LEAKAGE RISK** - Months since origination | - |
| `fico_score` | int | Credit score (550–850) | **99999 = Missing** |
| `income` | int | Annual income (USD) | **-1 = Missing** |
| `debt_to_income` | float | Ratio of debt to income | - |
| `num_open_trades` | int | Number of active credit trades | - |
| `utilization_rate` | float | Revolving credit utilization (0-1) | - |
| `inquiries_last_6m` | int | Credit inquiries in past 6 months | **99 = Missing** |
| `loan_amount` | int | Loan principal (USD) | - |
| `term` | int | Loan term in months (36 or 60) | - |
| `apr` | float | Annual percentage rate | - |
| `channel` | str | Origination channel | - |
| `product_type` | str | Loan purpose | - |
| `employment_length` | int | Years of employment | - |
| `state` | str | Borrower's state | - |
| `age` | int | Borrower's age | - |
| `days_past_due_current` | int | **⚠️ LEAKAGE RISK** - Current DPD status | - |
| `total_payments_to_date` | float | **⚠️ LEAKAGE RISK** - Payments made | - |
| `sample_weight` | float | Sample importance weight | - |
| `default_12m` | int | **Target variable** (1 = default within 12 months) | - |

### Key Challenges:

#### 1. **Data Quality & Special Values** ⭐
Handle special values that represent missing data:
- `fico_score = 99999` → Missing FICO
- `income = -1` → Missing income  
- `inquiries_last_6m = 99` → Missing inquiries

**Question to address**: How do you handle these missing values? Why not just use `NaN`?

#### 2. **Imbalanced Target** ⭐⭐
Default rate is ~3% (highly imbalanced). 
- Consider using `sample_weight` for weighted training
- Evaluate appropriate metrics (AUC-ROC, AUC-PR, KS statistic)
- Discuss precision/recall tradeoffs

#### 3. **Out-of-Time Validation** ⭐⭐⭐
Data spans multiple vintages (2021-2024).
- **DO NOT** use random train/test split
- Use **vintage-based time series split** (e.g., train on 2021-2022, validate on 2023, test on 2024)
- Explain why this is critical for credit models

#### 4. **Data Leakage Detection** ⭐⭐⭐
The dataset contains features that would **NOT** be available at origination:
- `months_on_book` - only known after origination
- `days_past_due_current` - performance data
- `total_payments_to_date` - performance data

**Question to address**: Identify ALL features with potential leakage and explain why they shouldn't be used.

### Tasks:

1. **EDA & Data Quality**
   - Analyze vintage distribution and default rates over time
   - Handle special values appropriately (99999, -1, 99)
   - Identify and document potential data leakage
   - Analyze class imbalance

2. **Feature Engineering**
   - Engineer features that would be available at origination
   - Handle missing values encoded as special values
   - Create appropriate encodings for categorical variables
   - Consider feature interactions

3. **Model Development**
   - Implement **out-of-time validation** using vintage splits
   - Train at least two models (e.g., Logistic Regression, Gradient Boosted Trees)
   - Use sample weights for imbalanced learning
   - Tune hyperparameters with proper validation

4. **Evaluation**
   - Compare models using multiple metrics (AUC-ROC, AUC-PR, KS)
   - Analyze model performance across different vintages
   - Generate feature importance and interpret results
   - Create calibration plots

5. **Discussion**
   - Discuss data leakage risks and how you avoided them
   - Explain your approach to handling imbalanced data
   - Discuss model assumptions and limitations
   - Propose improvements for production deployment

### Deliverable: 
`notebooks/modeling.ipynb` (or equivalent) with all outputs saved

---

## 🚀 Part 2: BONUS - Production Pipeline (Optional)

**Estimated Time**: 2-4 hours additional

Build a **Metaflow** or **similar workflow orchestration** pipeline that:

1. **Data Loading**: Load and validate the dataset
2. **Feature Engineering**: Apply transformations as a reproducible step
3. **Model Training**: Train model with proper experiment tracking
4. **Model Evaluation**: Generate metrics and artifacts
5. **Model Versioning**: Save model with metadata

### Requirements:

- Use [Metaflow](https://metaflow.org/) or similar (Prefect, Airflow, etc.)
- Each step should be a separate, testable function
- Include proper logging and error handling
- Save model artifacts and evaluation metrics
- **Bonus points**: Add parameter sweeps or A/B testing capability

### Example Metaflow Structure:

```python
from metaflow import FlowSpec, step, Parameter

class CreditRiskFlow(FlowSpec):
    
    vintage_split = Parameter('vintage_split', default='202301')
    
    @step
    def start(self):
        """Load and validate data"""
        self.next(self.preprocess)
    
    @step
    def preprocess(self):
        """Feature engineering and handling special values"""
        self.next(self.train)
    
    @step
    def train(self):
        """Train model with out-of-time validation"""
        self.next(self.evaluate)
    
    @step
    def evaluate(self):
        """Generate metrics and save artifacts"""
        self.next(self.end)
    
    @step
    def end(self):
        """Pipeline complete"""
        pass

if __name__ == '__main__':
    CreditRiskFlow()
```

**Deliverable**: 
- `src/credit_risk_flow.py` or similar
- Documentation on how to run the pipeline
- Example of viewing results from different runs

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

### 3. Generate the dataset (already done)
```bash
# The dataset is already generated, but you can regenerate it:
cd scripts
python generate_enhanced_data.py
```

### 4. Start working
```bash
# Launch Jupyter
jupyter notebook notebooks/modeling.ipynb

# Or use JupyterLab
jupyter lab
```

---

## 📊 Evaluation Criteria

### Core Assignment (Required):

- **Data Quality (20%)**: Proper handling of special values and missing data
- **Modeling Approach (25%)**: Sound methodology, out-of-time validation, handling imbalance
- **Feature Engineering (15%)**: Creative and appropriate transformations
- **Leakage Prevention (15%)**: Correctly identify and avoid data leakage
- **Code Quality (15%)**: Clean, readable, modular code
- **Communication (10%)**: Clear documentation, insights, and visualizations

### Bonus Section (Optional):

- **Pipeline Design (30%)**: Well-structured, reproducible workflow
- **Best Practices (30%)**: Proper logging, error handling, testing
- **Artifacts Management (20%)**: Model versioning and experiment tracking
- **Documentation (20%)**: Clear instructions and examples

---

## 📤 Submission Instructions

1. Complete your analysis in `notebooks/modeling.ipynb`
2. (Optional) Add modular code to `src/` 
3. (Optional) Add Metaflow pipeline for bonus section
4. Push your code to a GitHub repository (or provide a zip file)
5. Ensure all notebook outputs are saved so we can review without running
6. Include a summary section at the end with:
   - Key findings
   - Model performance summary
   - Identified data quality issues
   - Recommendations for improvement
   - (If bonus) Instructions to run your pipeline

---

## ⚡ Quick Tips

- **Start simple**: Get a baseline model working first
- **Document decisions**: Explain your choices (especially for handling special values and leakage)
- **Show your work**: Include visualizations and intermediate results
- **Time management**: Focus on core requirements first, bonus is truly optional

---

## ❓ Questions?

If you have any questions about the assignment, please reach out to jing.ren@bestegg.com

Good luck! 🎉
