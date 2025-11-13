"""
Metaflow pipeline template for Credit Risk Modeling

This is a TEMPLATE to help you get started with the bonus section.
Feel free to modify, extend, or completely rewrite this structure.

To run:
    python credit_risk_flow_template.py run
    
To view results:
    python credit_risk_flow_template.py show
"""

from metaflow import FlowSpec, step, Parameter, card
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# === Import your modular project utilities ===
from src.preprocessing import handle_missing_values, handle_outliers, create_features, encode_categorical
from src.modeling import train_logistic_regression, train_gradient_boosting
from src.evaluation import calculate_model_metrics, evaluate_model


class CreditRiskFlow(FlowSpec):
    """
    Credit Risk Modeling Pipeline
    
    A production-ready ML pipeline that:
    1. Loads and validates data
    2. Handles special values and missing data
    3. Engineers features
    4. Trains model with out-of-time validation
    5. Evaluates and saves artifacts
    """
    
    # Parameters
    data_path = Parameter(
        'data_path',
        default='../data/credit_risk_data_enhanced.csv',
        help='Path to input data'
    )
    
    train_vintage_end = Parameter(
        'train_vintage_end',
        default='202212',
        help='Last vintage to include in training (YYYYMM)'
    )
    
    test_vintage_start = Parameter(
        'test_vintage_start',
        default='202401',
        help='First vintage to include in test set (YYYYMM)'
    )
    
    use_sample_weights = Parameter(
        'use_sample_weights',
        default=True,
        help='Whether to use sample weights in training'
    )
    
    @card
    @step
    def start(self):
        """
        Start: Load and validate dataset
        
        TODO for candidates:
        - Load data from data_path
        - Validate required columns exist
        - Check for data quality issues
        - Log basic statistics
        """
        print("=" * 60)
        print("STEP 1: Loading Data")
        print("=" * 60)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
        print(f"✓ Vintages: {self.df['vintage'].min()} to {self.df['vintage'].max()}")
        print(f"✓ Default rate (class imbalance): {self.df['default_12m'].mean():.2%}")

        # TODO: Add more validation checks
        # - Check for unexpected values
        # - Validate data types
        # - Check for duplicates
        required_cols = {'loan_id', 'vintage', 'default_12m'}
        missing = required_cols - set(self.df.columns)
        assert not missing, f"Missing required columns: {missing}"
        
        self.next(self.preprocess)
    
    @step
    def preprocess(self):
        """
        Preprocessing: Handle special values and split data by vintage
        
        TODO for candidates:
        - Handle special values (99999, -1, 99)
        - Identify and remove leakage features
        - Create train/validation/test splits by vintage
        - Basic feature engineering
        """
        print("\n" + "=" * 60)
        print("STEP 2: Preprocessing")
        print("=" * 60)
        
        df = self.df.copy()
        
        # Handle known placeholder encodings and outliers
        # (99999 for FICO, -1 for income, 99 for inquiries)
        df = handle_missing_values(df)
        df = handle_outliers(df)
        print("✓ Special placeholder values (99999, -1, 99) handled and missing flags created.")
        
        # TODO: Remove leakage features that occur after origination
        leakage_cols = [
            'months_on_book', 'days_past_due_current', 'total_payments_to_date',
            'loan_status', 'recent_delinquency_flag'
        ]
        df.drop(columns=leakage_cols, errors='ignore', inplace=True)
        print(f"✓ Removed potential leakage features: {leakage_cols}")
        
        # Temporal (vintage-based) split for fair validation
        self.df_train = df[df['vintage'] <= self.train_vintage_end].copy()
        self.df_test = df[df['vintage'] >= self.test_vintage_start].copy()
        
        print(f"✓ Train: {len(self.df_train):,} rows ({self.df_train['vintage'].min()}–{self.df_train['vintage'].max()})")
        print(f"✓ Test:  {len(self.df_test):,} rows ({self.df_test['vintage'].min()}–{self.df_test['vintage'].max()})")
        print(f"✓ Train default rate: {self.df_train['default_12m'].mean():.2%}")
        print(f"✓ Test default rate:  {self.df_test['default_12m'].mean():.2%}")
        
        self.next(self.feature_engineering)
    
    @step
    def feature_engineering(self):
        """
        Feature Engineering: Create modeling features
        
        TODO for candidates:
        - Impute missing values
        - Encode categorical variables
        - Create feature interactions
        - Scale numerical features
        - Create feature matrix (X) and target (y)
        """
        print("\n" + "=" * 60)
        print("STEP 3: Feature Engineering")
        print("=" * 60)
        
        # Apply feature creation logic
        self.df_train = create_features(self.df_train)
        self.df_test = create_features(self.df_test)
        
        # Encode categorical variables (One-Hot Encoding)
        cat_cols = self.df_train.select_dtypes(include=['object', 'category']).columns.tolist()
        self.df_train = encode_categorical(self.df_train, cat_cols, method='onehot')
        self.df_test = encode_categorical(self.df_test, cat_cols, method='onehot')
        self.df_test = self.df_test.reindex(columns=self.df_train.columns, fill_value=0)
        
        # Select feature columns (exclude target and identifiers)
        drop_cols = ['loan_id', 'origination_date', 'vintage', 'default_12m']
        feature_cols = [c for c in self.df_train.columns if c not in drop_cols]
        
        self.X_train = self.df_train[feature_cols]
        self.y_train = self.df_train['default_12m']
        self.X_test = self.df_test[feature_cols]
        self.y_test = self.df_test['default_12m']
        
        if self.use_sample_weights and 'sample_weight' in self.df_train.columns:
            self.sample_weights = self.df_train['sample_weight'].values
        else:
            self.sample_weights = None
        
        print(f"✓ Final feature set: {len(feature_cols)} features")
        print(f"✓ X_train: {self.X_train.shape}, X_test: {self.X_test.shape}")
        
        self.next(self.train)
    
    @step
    def train(self):
        """
        Training: Train model with proper handling of imbalanced data
        
        TODO for candidates:
        - Try multiple algorithms
        - Use sample weights
        - Hyperparameter tuning
        - Cross-validation within training window
        """
        print("\n" + "=" * 60)
        print("STEP 4: Model Training")
        print("=" * 60)
        
        # Train Logistic Regression and Gradient Boosting models
        self.log_reg_model = train_logistic_regression(self.X_train, self.y_train)
        self.gb_model, self.model_type = train_gradient_boosting(self.X_train, self.y_train)
        
        print(f"✓ Models trained successfully: Logistic Regression and {self.model_type.upper()}")
        self.next(self.evaluate)
    
    @card
    @step
    def evaluate(self):
        """
        Evaluation: Generate metrics and analyze model performance
        
        TODO for candidates:
        - Calculate AUC-ROC, AUC-PR, KS
        - Create calibration plots
        - Analyze feature importance
        - Performance by vintage
        - Generate confusion matrix
        """
        print("\n" + "=" * 60)
        print("STEP 5: Model Evaluation")
        print("=" * 60)
        
        # Evaluate both models
        y_prob_lr = self.log_reg_model.predict_proba(self.X_test)[:, 1]
        y_prob_gb = self.gb_model.predict_proba(self.X_test)[:, 1]
        
        metrics_lr = calculate_model_metrics(self.y_test, y_prob_lr)
        metrics_gb = calculate_model_metrics(self.y_test, y_prob_gb)
        
        print("\n📊 Validation Metrics Summary:")
        print(pd.DataFrame([
            ["Logistic Regression", metrics_lr["AUC"], metrics_lr["KS"], metrics_lr["PR-AUC"]],
            [f"Gradient Boosting ({self.model_type})", metrics_gb["AUC"], metrics_gb["KS"], metrics_gb["PR-AUC"]],
        ], columns=["Model", "AUC", "KS", "PR-AUC"]))
        
        # Detailed model-level analysis
        print("\n🔹 Logistic Regression Evaluation:")
        evaluate_model(self.y_test, y_prob_lr, threshold=0.5)
        
        print("\n🔹 Gradient Boosting Evaluation:")
        evaluate_model(self.y_test, y_prob_gb, threshold=0.54)
        
        self.metrics = {"log_reg": metrics_lr, "gradient_boosting": metrics_gb}
        
        # TODO: Add more sophisticated evaluation
        # - KS statistic
        # - AUC-PR for imbalanced data
        # - Calibration plots
        # - Performance by vintage
        # - Feature importance
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        End: Save model and artifacts
        
        TODO for candidates:
        - Save model to disk
        - Save feature engineering pipeline
        - Export metrics and plots
        - Create model card
        """
        print("\n" + "=" * 60)
        print("STEP 6: Pipeline Complete!")
        print("=" * 60)
        
        print("\n✅ Final Results:")
        for model, metric in self.metrics.items():
            print(f"   {model}: {metric}")
        
        # Save models and metrics
        import joblib, json, os
        os.makedirs("../outputs", exist_ok=True)
        joblib.dump(self.log_reg_model, "../outputs/log_reg_model.pkl")
        joblib.dump(self.gb_model, "../outputs/gb_model.pkl")
        with open("../outputs/metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)
        
        print("\n" + "=" * 60)
        print("Pipeline execution completed successfully!")
        print("=" * 60)


if __name__ == '__main__':
    CreditRiskFlow()
