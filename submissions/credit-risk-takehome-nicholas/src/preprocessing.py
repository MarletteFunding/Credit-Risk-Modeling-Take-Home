"""
Feature engineering and preprocessing utilities.

This module provides helper functions for data preprocessing and feature engineering.
Candidates can use, modify, or replace these as needed.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def handle_missing_values(df, strategy='median'):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Strategy for imputation ('median', 'mean', 'mode', or 'drop')
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    # This is a placeholder - candidates should implement their own logic
    df = df.copy()

    # --- Step 1: Replace special placeholders with NaN ---
    special_values = {
        'fico_score': 99999,
        'income': -1,
        'inquiries_last_6m': 99
    }

    for col, val in special_values.items():
        if col in df.columns:
            df[col] = df[col].replace(val, np.nan)
            df[f"{col}_missing_flag"] = df[col].isna().astype(int)

    # --- Step 2: Impute numerical columns ---
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        if df[col].isna().sum() > 0:
            if strategy == 'median':
                fill_val = df[col].median()
            elif strategy == 'mean':
                fill_val = df[col].mean()
            elif strategy == 'mode':
                fill_val = df[col].mode()[0]
            elif strategy == 'drop':
                df = df.dropna(subset=[col])
                continue
            else:
                raise ValueError("Invalid imputation strategy")

            df[col] = df[col].fillna(fill_val)

    # --- Step 3: Handle categorical columns ---
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

def handle_outliers(df, method='iqr', z_thresh=3.5, cap_percentile=0.01):
    """
    Detect and handle outliers in numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    method : str
        Outlier handling method: 
        - 'iqr': Remove or cap using interquartile range.
        - 'zscore': Cap based on z-score threshold.
        - 'percentile': Cap extreme percentiles.
    z_thresh : float
        Z-score threshold (used if method='zscore').
    cap_percentile : float
        Percentile cutoff for capping (used if method='percentile').

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers capped/handled.
    """

    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        # Skip target variable if exists
        if col.lower().startswith('default'):
            continue

        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)

        elif method == 'zscore':
            mean, std = df[col].mean(), df[col].std()
            z_scores = (df[col] - mean) / std
            df[col] = np.where(z_scores > z_thresh, mean + z_thresh * std,
                               np.where(z_scores < -z_thresh, mean - z_thresh * std, df[col]))

        elif method == 'percentile':
            lower = df[col].quantile(cap_percentile)
            upper = df[col].quantile(1 - cap_percentile)
            df[col] = np.clip(df[col], lower, upper)

    return df

def create_features(df):
    """
    Create additional features from existing columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with raw features
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional engineered features
    """
    # This is a placeholder - candidates should implement their own logic
    df = df.copy()

    # --- 1. Credit utilization and payment ratios ---
    if {'income', 'loan_amount'}.issubset(df.columns):
        df['income_to_loan_ratio'] = df['income'] / (df['loan_amount'] + 1e-6)

    if {'debt_to_income', 'utilization_rate'}.issubset(df.columns):
        df['debt_utilization_interaction'] = df['debt_to_income'] * df['utilization_rate']

    # --- 2. Risk-based financial metrics ---
    if 'apr' in df.columns and 'fico_score' in df.columns:
        df['apr_to_fico_ratio'] = df['apr'] / (df['fico_score'] + 1e-6)

    # --- 3. Employment and term interactions ---
    if {'employment_length', 'term'}.issubset(df.columns):
        df['employment_to_term_ratio'] = df['employment_length'] / (df['term'] + 1e-6)

    # --- 4. Derived log-scaled features ---
    for col in ['income', 'loan_amount']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])

    # --- 5. Age-related risk metrics ---
    if 'age' in df.columns:
        df['is_senior'] = (df['age'] >= 60).astype(int)
        df['age_bucket'] = pd.cut(
            df['age'],
            bins=[18, 30, 40, 50, 60, 70, 100],
            labels=['<30', '30-40', '40-50', '50-60', '60-70', '70+']
        )

    # --- 6. Channel / product encoding ---
    if 'channel' in df.columns:
        df['is_online_channel'] = df['channel'].str.contains('online', case=False, na=False).astype(int)

    # --- 7. Clean invalid or infinite values ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Only fill NaN in numeric columns, not categoricals
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df

def encode_categorical(df, columns, method='onehot'):
    """
    Encode categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of column names to encode
    method : str
        Encoding method ('onehot', 'label', or 'target')
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded categorical variables
    """
    # This is a placeholder - candidates should implement their own logic
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not columns:
        print("⚠️ No categorical columns found for encoding.")
        return df

    print(f"Encoding {len(columns)} categorical columns using '{method}' method...")

    if method == 'onehot':
        df = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)

    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in columns:
            df[col] = le.fit_transform(df[col].astype(str))

    elif method == 'target':
        if 'default_12m' not in df.columns:
            raise ValueError("Target encoding requires 'default_12m' in the dataset.")
        for col in columns:
            mapping = df.groupby(col)['default_12m'].mean()
            df[col + '_target_enc'] = df[col].map(mapping)
        df.drop(columns=columns, inplace=True)

    else:
        raise ValueError("Invalid encoding method. Choose from ['onehot', 'label', 'target'].")

    return df

