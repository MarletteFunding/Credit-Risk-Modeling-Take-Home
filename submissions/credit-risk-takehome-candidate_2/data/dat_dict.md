| Column | Type | Description | Special Values |
|--------|------|-------------|----------------|
| `loan_id` | str | Unique loan identifier | - |
| `origination_date` | date | Loan origination date | - |
| `vintage` | str | Origination month (YYYYMM) | - |
| `months_on_book` | int | Months since origination | - |
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
| `days_past_due_current` | int | Current DPD status | - |
| `total_payments_to_date` | float | Payments made | - |
| `sample_weight` | float | Sample importance weight | - |
| `default_12m` | int | **Target variable** (1 = default within 12 months) | - |