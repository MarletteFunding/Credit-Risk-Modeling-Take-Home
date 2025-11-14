"""
Feature engineering and preprocessing utilities.

This module provides helper functions for data preprocessing and feature engineering.
Candidates can use, modify, or replace these as needed.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import folder_manager
import os
import yaml
import joblib
import random

# Config:
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
home_dir = config['HOME_DIRECTORY']


def handle_missing_values(df, strategy={}):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : dict
        Strategy for imputation (col:imputation_strategy)
        # Needs further methods for imputation implemented but you get the idea,
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    df = df.copy()

    for col, strat in strategy.items():
        if col not in df.columns:
            continue
        
        if strat.startswith("quantile_"):    
            q = float(strat.split("_")[1]) / 100
            fill_value = df[col].quantile(q)

        elif strat == "median":
            fill_value = df[col].median()
        
        elif strat == "mean":
            fill_value = df[col].median()

        elif strat.startswith("value_"):
            fill_value = float(strat.split("_")[1])

        else:
            raise ValueError(f"Unknown strategy: {strat}")

        df[col] = df[col].fillna(fill_value)

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
    # Implemented in notebook, can add tried and tested features after repeated experiment success
    pass


def encode_categorical(df, dat_dict, encoder_path=None):
    """
    Encode categorical variables and update the data dictionary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    dat_dict : pd.DataFrame
        Data dictionary.
    encoder_path : str, optional
        Path to save encoders. Defaults to None.
    
    Returns:
    --------
    (pd.DataFrame, pd.DataFrame)
        A tuple containing:
        - The dataframe with encoded categorical variables.
        - The updated data dictionary.
    """
    
    cat_cols = dat_dict[(dat_dict.model_features == 1) & (dat_dict.type == 'str')]['column']
    new_rows = []

    for col in [c for c in cat_cols if c != 'vintage']:
        print(f"For feature: {col}")
        path = encoder_path if encoder_path else folder_manager.encoding_path

        # Mark original column as not a model feature
        dat_dict.loc[dat_dict['column'] == col, 'model_features'] = 0

        if df[col].nunique() < 10:
            # One-hot encoding
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df[[col]])
            encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index, dtype=int)
            df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
            
            for new_col in encoded_cols:
                new_rows.append({
                    'column': new_col,
                    'type': 'int',
                    'model_features': 1,
                    'description': f'One-hot encoded from {col}',
                    'missingvalues': np.nan,
                })

            if encoder_path:
                joblib.dump(encoder, os.path.join(path, f'ohe_{col}.pkl'))
            else:
                joblib.dump(encoder, os.path.join(folder_manager.encoding_path, f'ohe_{col}.pkl'))

        else:
            # Ordinal encoding for generic high dimension encoding
            from sklearn.preprocessing import OrdinalEncoder
            encoder = OrdinalEncoder()
            new_col_name = col + '_oe'
            df[new_col_name] = encoder.fit_transform(df[[col]]).ravel().astype(int)
            
            new_rows.append({
                'column': new_col_name,
                'type': 'int',
                'model_features': 1,
                'description': f'Ordinal encoded from {col}',
                'missingvalues': np.nan,
            })
            if encoder_path:
                joblib.dump(encoder, os.path.join(path, f'oe_{col}.pkl'))
            else:
                joblib.dump(encoder, os.path.join(folder_manager.encoding_path, f'oe_{col}.pkl'))

    if new_rows:
        new_rows_df = pd.DataFrame(new_rows)
        dat_dict = pd.concat([dat_dict, new_rows_df], ignore_index=True)

    return df, dat_dict


def load_dat_dict(loc = "../data/dat_dict.md",exclude_features = None,ppi_features = None,target = 'default_12m'):

    '''
    This is the main view to the data schema while running analysis and updates need to be implemented everytime the raw data is touched
    Read data dictionary from disc and separate independent features for processing and modeling
    loc (str): location of data dict:
    exclude_features (list): features to exclude
    ppi_features (list): ppi features to exclude
    target: target metric from original data
    '''
    dat_dict = None
    if loc:
        print("reading data dictionary from disc")
        dat_dict = pd.read_csv("../data/dat_dict.md", sep="|", skipinitialspace=True, engine="python")
    else:
        # read columns and data type from original data
        pass

    dat_dict = dat_dict.dropna(how="all", axis=1).iloc[1:-1].reset_index(drop=True)
    dat_dict.columns = [c.strip() for c in dat_dict.columns]
    dat_dict.columns = [c.lower().replace(" ","") for c in dat_dict.columns]
    
    dat_dict['column'] = dat_dict['column'].str.replace("`","").str.strip()
    dat_dict['type'] = dat_dict['type'].str.strip()
    
    dat_dict['specialvalues'] = dat_dict['specialvalues'].str.split('=').str[0].str.replace("*","").str.strip()
    dat_dict.rename(columns = {'specialvalues':'missingvalues'},inplace = True)

    # Additional Filters
    if not exclude_features:
        exclude_features = ['loan_id','days_past_due_current','total_payments_to_date','sample_weight','months_on_book']
    
    if not ppi_features:
        dat_dict['model_features'] = np.where(~dat_dict['column'].isin(exclude_features + [target]),1,0)
    else:
        dat_dict['model_features'] = np.where(~dat_dict['column'].isin(exclude_features + ppi_features + [target]),1,0)
        
    return dat_dict
        

def plot_null_rate_by_feature(null_rate_series, null_feature, feature, path="../outputs/"):
    """
    null_rate_series : pd.Series
        A Series where the index is unique values of feature_b and values are
        the null rates of feature_a (output of check_null_rate_by_feature).
    feature_a_name : str
        The name of the feature whose null rate is plotted.
    feature_b_name : str
        The name of the feature by which the null rate is grouped.
    path : str, optional
        The folder to save the plot in. Defaults to "../outputs/".
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    null_rate_series.sort_values(ascending=False).plot(kind='bar', ax=ax)
    ax.set_title(f"Null Rate of '{null_feature}' by '{feature}'")
    ax.set_xlabel(feature)
    ax.set_ylabel(f"Null Rate of {null_feature}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

        
def check_null_rate_by_feature(df, null_feature, feature, dat_dict):
    
    temp_df = df[[null_feature, feature]].copy()
    missing_indicator = dat_dict.loc[dat_dict['column'] == null_feature, 'missingvalues'].iloc[0]
    is_null_a = pd.isnull(temp_df[null_feature])
    
    if pd.notna(missing_indicator) and missing_indicator.strip() != '':
    
        is_null_a = is_null_a | (temp_df[null_feature].astype(str) == str(missing_indicator))
    
    temp_df['is_null_a'] = is_null_a
    
    
    null_rate = temp_df.groupby(feature)['is_null_a'].mean()

    plot_null_rate_by_feature(null_rate,null_feature,feature)
    return null_rate
        
def analyze_feature(feature_series, feature_name):
    """
    feature_series : The feature column to analyze.
    feature_name : The name of the feature.
    """
    
    inferred_type = 'unknown'
    num_unique = feature_series.nunique()
    
    if pd.api.types.is_numeric_dtype(feature_series):
        # Treat as categorical if it has few unique values, Can decide on threshold, here i have 20
        if num_unique <= 20:
            inferred_type = 'categorical_numeric'
        else:
            inferred_type = 'numeric'
    elif pd.api.types.is_string_dtype(feature_series) or pd.api.types.is_categorical_dtype(feature_series):
        inferred_type = 'categorical'
    elif pd.api.types.is_datetime64_any_dtype(feature_series):
        inferred_type = 'datetime'

    print(f"Feature '{feature_name}' inferred as: {inferred_type}")

    # Distribution Plots
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Distribution Analysis for '{feature_name}'", fontsize=16)

    if inferred_type == 'numeric':
        ax1 = plt.subplot(1, 2, 1)
        sns.histplot(feature_series, kde=True, ax=ax1)
        ax1.set_title('Histogram & KDE')
        
        ax2 = plt.subplot(1, 2, 2)
        sns.boxplot(x=feature_series, ax=ax2)
        ax2.set_title('Box Plot')


    # For Two types categorical_numercic and just categorical
    elif inferred_type.startswith('categorical'):
        feature_series.value_counts().sort_index().plot(kind='bar', ax=plt.gca())
        plt.title('Count Plot')
        plt.xlabel(feature_name)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')

    elif inferred_type == 'datetime':
        feature_series.value_counts().sort_index().plot(kind='line', ax=plt.gca())
        plt.title('Counts over Time')
        plt.xlabel('Date')
        plt.ylabel('Frequency')

    else:
        print("Could not determine dat/plot type.")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    print("saving report to file")
    plt.savefig(folder_manager.feature_report_path + f'/{feature_name}.pdf', dpi = 300,bbox_inches='tight')
    plt.show()

    summary_data = {
        'Metric': [],
        'Value': []
    }
    
    # Helper to add to summary
    def add_summary(metric, value):
        summary_data['Metric'].append(metric)
        summary_data['Value'].append(value)

    # Common stats
    add_summary('Data Type (Pandas)', feature_series.dtype)
    add_summary('Inferred Type', inferred_type)
    add_summary('Number of Records', len(feature_series))
    add_summary('Missing Values', feature_series.isnull().sum())
    add_summary('Missing Values (%)', f"{feature_series.isnull().mean() * 100:.2f}% ")
    add_summary('Unique Values', num_unique)
    
    if num_unique > 0:
        most_frequent = feature_series.mode()
        if not most_frequent.empty:
            add_summary('Most Frequent Value', most_frequent.iloc[0])

    if inferred_type == 'numeric':
        # Basic stats
        desc = feature_series.describe()
        for idx, val in desc.items():
            add_summary(idx.capitalize(), f"{val:.2f}")
            
        # Outlier detection: values outside 25-75 quantiles
        Q1 = feature_series.quantile(0.25)
        Q3 = feature_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = feature_series[(feature_series < lower_bound) | (feature_series > upper_bound)]
        add_summary('Number of Outliers (IQR)', len(outliers))
        if len(outliers) > 0:
            add_summary('Outlier Percentage', f"{len(outliers) / len(feature_series) * 100:.2f}%")

        # Other checks
        add_summary('Zeros', (feature_series == 0).sum())
        if (feature_series < 0).any():
            add_summary('Negative Values', (feature_series < 0).sum())

    elif inferred_type.startswith('categorical'):
        # Value counts for top 5
        value_counts = feature_series.value_counts()
        add_summary('Top 5 Value Counts', value_counts.head(5).to_dict())
        
        # Weird values (low frequency)
        low_freq_threshold = max(5, len(feature_series) * 0.001) # e.g., less than 5 or 0.1%
        low_freq_values = value_counts[value_counts < low_freq_threshold]
        if not low_freq_values.empty:
            add_summary(f'Values with < {int(low_freq_threshold)} occurrences', low_freq_values.to_dict())

    summary_df = pd.DataFrame(summary_data)
    
    print(f"Analysis Summary for '{feature_name}'")
    print(summary_df.to_string(index=False))
    
    return summary_df


def analyze_feature_with_target(feature_series, feature_name, target):
    """
    Analyzes a feature and its relationship with a target variable.

    feature_series : pd.Series
        The feature column to analyze.
    feature_name : str
        The name of the feature.
    target : pd.Series
        The target variable series.
    """
    
    inferred_type = 'unknown'
    num_unique = feature_series.nunique()
    
    # use the pandas.api to infer types
    if pd.api.types.is_numeric_dtype(feature_series):
        # Treat as categorical if it has few unique values, Can decide on threshold, here i have 20
        if num_unique <= 20:
            inferred_type = 'categorical_numeric'
        else:
            inferred_type = 'numeric'
    elif pd.api.types.is_string_dtype(feature_series) or pd.api.types.is_categorical_dtype(feature_series):
        inferred_type = 'categorical'
    elif pd.api.types.is_datetime64_any_dtype(feature_series):
        inferred_type = 'datetime'

    print(f"Feature '{feature_name}' inferred as: {inferred_type}")

    ## Distribution plots
    plt.figure(figsize=(12, 6))
    target_name = target.name if target.name else 'target'
    plt.suptitle(f"Distribution of '{feature_name}' by '{target_name}'", fontsize=16)

    df_plot = pd.concat([feature_series, target], axis=1)
    df_plot.rename(columns={target.name: target_name}, inplace=True)


    if inferred_type == 'numeric':
        ax1 = plt.subplot(1, 2, 1)
        sns.kdeplot(data=df_plot, x=feature_name, hue=target_name, fill=True, common_norm=False, ax=ax1)
        ax1.set_title('KDE by Target')
        
        ax2 = plt.subplot(1, 2, 2)
        sns.boxplot(data=df_plot, x=target_name, y=feature_name, ax=ax2)
        ax2.set_title('Box Plot by Target')

    elif inferred_type.startswith('categorical'):
        sns.countplot(data=df_plot, x=feature_name, hue=target_name, ax=plt.gca())
        plt.title('Count Plot by Target')
        plt.xlabel(feature_name)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')

    elif inferred_type == 'datetime':
        # Ensure the feature is datetime
        df_plot[feature_name] = pd.to_datetime(df_plot[feature_name])
        
        # Group by month and target to count occurrences
        counts_by_target = df_plot.groupby([pd.Grouper(key=feature_name, freq='M'), target_name]).size().unstack(fill_value=0)
        
        counts_by_target.plot(kind='line', ax=plt.gca())
        plt.title('Counts over Time by Target')
        plt.xlabel('Date')
        plt.ylabel('Frequency')

    else:
        print("Could not determine data/plot type.")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    print("saving report to file")
    plt.savefig(folder_manager.feature_report_path + f'/{feature_name}_by_{target_name}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    summary_data = {
        'Metric': [],
        'Value': []
    }
    
    # Helper to add to summary
    def add_summary(metric, value):
        summary_data['Metric'].append(metric)
        summary_data['Value'].append(value)

    # Common stats
    add_summary('Data Type (Pandas)', feature_series.dtype)
    add_summary('Inferred Type', inferred_type)
    add_summary('Number of Records', len(feature_series))
    add_summary('Missing Values', feature_series.isnull().sum())
    add_summary('Missing Values (%)', f"{feature_series.isnull().mean() * 100:.2f}% ")
    add_summary('Unique Values', num_unique)
    
    if num_unique > 0:
        most_frequent = feature_series.mode()
        if not most_frequent.empty:
            add_summary('Most Frequent Value', most_frequent.iloc[0])

    if inferred_type == 'numeric':
        # Basic stats
        desc = feature_series.describe()
        for idx, val in desc.items():
            add_summary(idx.capitalize(), f"{val:.2f}")
            
        # Outlier detection: values outside 25-75 quantiles
        Q1 = feature_series.quantile(0.25)
        Q3 = feature_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = feature_series[(feature_series < lower_bound) | (feature_series > upper_bound)]
        add_summary('Number of Outliers (IQR)', len(outliers))
        if len(outliers) > 0:
            add_summary('Outlier Percentage', f"{len(outliers) / len(feature_series) * 100:.2f}%")

        # Other checks
        add_summary('Zeros', (feature_series == 0).sum())
        if (feature_series < 0).any():
            add_summary('Negative Values', (feature_series < 0).sum())

    elif inferred_type.startswith('categorical'):
        # Value counts for top 5
        value_counts = feature_series.value_counts()
        add_summary('Top 5 Value Counts', value_counts.head(5).to_dict())
        
        # Weird values (low frequency)
        low_freq_threshold = max(5, len(feature_series) * 0.001) # e.g., less than 5 or 0.1%
        low_freq_values = value_counts[value_counts < low_freq_threshold]
        if not low_freq_values.empty:
            add_summary(f'Values with < {int(low_freq_threshold)} occurrences', low_freq_values.to_dict())

    summary_df = pd.DataFrame(summary_data)
    
    
    print(f"Analysis Summary for '{feature_name}'")
    print(summary_df.to_string(index=False))
    
    return summary_df



def flag_highly_correlated_features(df,dat_dict, threshold=0.8):
    """
    df : pandas.DataFram
    dat_dict: data dictionary
    threshold : float Absolute correlation cutoff.
    """
    corr = df[dat_dict[(dat_dict.model_features == 1) & (dat_dict.type.isin(['int','float'])) ]['column'].values].corr().abs()

    # Take upper triangle so each pair considered once
    upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_pairs = corr.where(upper)

    to_drop = set()

    # Go through all pairs > threshold
    for col_i in corr_pairs.columns:
        for col_j in corr_pairs.index:
            if col_i == col_j:
                continue
            if corr_pairs.loc[col_j, col_i] > threshold:
                # randomly choose which to drop
                drop_candidate = random.choice([col_i, col_j])
                to_drop.add(drop_candidate)

    return to_drop