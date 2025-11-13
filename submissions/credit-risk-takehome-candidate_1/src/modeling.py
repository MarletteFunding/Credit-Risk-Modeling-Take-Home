"""
Model training and prediction utilities.

This module provides helper functions for training and evaluating models.
Candidates can use, modify, or replace these as needed.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


def train_test_split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets.

    Parameters
    ----------
    X : array-like or pd.DataFrame
        Feature matrix.
    y : array-like or pd.Series
        Target variable.
    test_size : float, default=0.2
        Proportion of data to use for testing.
    val_size : float, default=0.2
        Proportion of training data to use for validation.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_valid, X_test, y_train, y_valid, y_test : tuple
        Training, validation, and test splits.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    # --- Step 1: Validation ---
    if X is None or y is None:
        raise ValueError("❌ Both X and y must be provided.")
    if len(X) != len(y):
        raise ValueError(f"❌ X and y must have the same number of samples. Got {len(X)} and {len(y)}.")

    # --- Step 2: Stratify if target is binary ---
    stratify = y if len(np.unique(y)) > 1 else None

    # --- Step 3: Initial train/test split ---
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # --- Step 4: Create validation split from training data ---
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full if stratify is not None else None
    )

    # --- Step 5: Summary ---
    print("\n✅ Random Train/Validation/Test Split Complete")
    print(f"Train size: {len(X_train):,}")
    print(f"Valid size: {len(X_valid):,}")
    print(f"Test  size: {len(X_test):,}")
    if hasattr(y, "mean"):
        print(
            f"Default rate - "
            f"Train: {np.mean(y_train):.2%} | "
            f"Valid: {np.mean(y_valid):.2%} | "
            f"Test: {np.mean(y_test):.2%}"
        )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def train_logistic_regression(X_train, y_train, drop_leakage=True, **kwargs):
    """
    Train a logistic regression model with optional leakage removal
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    drop_leakage : bool, default=True
        Whether to drop known post-event / leakage columns
    **kwargs : dict
        Additional parameters for LogisticRegression
    
    Returns:
    --------
    model : trained model
    """
    # This is a placeholder - candidates should implement their own logic
    # --- Step 1: Remove potential leakage columns ---
    if drop_leakage and isinstance(X_train, pd.DataFrame):
        leakage_cols = [
            "days_past_due_current",
            "months_on_book",
            "total_payments_to_date",
            "loan_status",
            "sample_weight",
            "recent_delinquency_flag",
        ]
        found = [c for c in leakage_cols if c in X_train.columns]
        if found:
            print(f"⚠️  Removing potential leakage columns: {found}")
            X_train = X_train.drop(columns=found)

    # --- Step 2: Default hyperparameters ---
    default_params = dict(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model_params = {**default_params, **kwargs}

    # --- Step 3: Create pipeline ---
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("log_reg", LogisticRegression(**model_params)),
        ]
    )

    # --- Step 4: Fit model ---
    model.fit(X_train, y_train)

    print("\n✅ Logistic Regression model successfully trained (leakage-safe).")
    print(f"Parameters: {model_params}")

    return model


def train_gradient_boosting(X_train, y_train, **kwargs):
    """
    Train a gradient boosting model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    **kwargs : dict
        Additional parameters for GradientBoostingClassifier
    
    Returns:
    --------
    model : trained model
    """
    # --- Step 1: Detect imbalance ratio ---
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    imbalance_ratio = n_neg / max(n_pos, 1)
    print(f"\n⚖️  Class imbalance ratio (neg/pos): {imbalance_ratio:.1f}")

    # --- Step 2: Choose library ---
    try:
        from xgboost import XGBClassifier
        model_type = "xgboost"
    except ImportError:
        try:
            from lightgbm import LGBMClassifier
            model_type = "lightgbm"
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model_type = "sklearn"

    # --- Step 3: Default parameters ---
    if model_type == "xgboost":
        default_params = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=imbalance_ratio,  # key addition
        )
        default_params.update(kwargs)
        model = XGBClassifier(**default_params)

    elif model_type == "lightgbm":
        default_params = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        default_params.update(kwargs)
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**default_params)

    else:
        from sklearn.ensemble import GradientBoostingClassifier
        default_params = dict(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42,
        )
        default_params.update(kwargs)
        model = GradientBoostingClassifier(**default_params)

    # --- Step 4: Train model ---
    model.fit(X_train, y_train)

    print(f"\n✅ Gradient Boosting model ({model_type}) successfully trained.")
    print(f"Parameters: {default_params}")

    return model, model_type
