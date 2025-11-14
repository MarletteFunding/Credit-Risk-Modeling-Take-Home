"""
Model training and prediction utilities.

This module provides helper functions for training and evaluating models.
Candidates can use, modify, or replace these as needed.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from evaluation import *
from model_report import generate_html_report
import folder_manager
import joblib
import os
import optuna

def train_test_split_data(df,target ='default_12m',date_col = None, type = 'stratified_time', n_folds = 3,test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Parameters:
    -----------
    df: Full data
    target: target column
    type: type of splits, values:[stratified_time, stratified_kfold, kfold, time] 
    n_folds: number of train, test pairs
    test_size : float
    random_state : int, Random seed for reproducibility
    
    
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)

    # Separate positives and negatives
    df_pos = df_sorted[df_sorted[target] == 1]
    df_neg = df_sorted[df_sorted[target] == 0]

    # Compute per-fold sizes for positives and negatives
    pos_splits = np.array_split(df_pos.index.values, n_folds)
    neg_splits = np.array_split(df_neg.index.values, n_folds)

    folds = []

    for i in range(n_folds):
        # Validation indices for this fold
        val_idx = np.concatenate([pos_splits[i], neg_splits[i]])

        # Training indices: all data **before the earliest val date**
        val_start_date = df_sorted.loc[val_idx, date_col].min()
        train_idx = df_sorted[df_sorted[date_col] < val_start_date].index.values

        folds.append((train_idx, val_idx))

    return folds[1:]


# Probability threshold tuning:
def tune_prob_threshold(actual,pred_proba):
    
    thresholds = np.linspace(0.0,0.1,101)

    threshold = {}

    # For F1
    from sklearn.metrics import f1_score
    f1_scores = [f1_score(actual,pred_proba >= t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    threshold['f1'] = best_threshold
    print(f"Threshold maximizing F1 score: {best_threshold}")
    

    # Youden's J Statistic
    from sklearn.metrics import roc_curve
    fpr, tpr, thres = roc_curve(actual,pred_proba)
    youden_index = tpr - fpr
    threshold['tpr_fpr'] = thres[np.argmax(youden_index)]
    print(f"Threshold maximizing tpr-fpr: {threshold['tpr_fpr']}")

    return threshold


def train_logistic_regression(df_clean, dat_dict, exclude_features,folds, baseline_threshold=0.028, max_iter=700):
    """
    Train a logistic regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    **kwargs : dict
        Additional parameters for LogisticRegression
    
    Returns:
    --------
    model : trained model
    """
    all_fold_data = []
    all_metrics_df = pd.DataFrame()

    for i, (train_idx, val_idx) in enumerate(folds):
        if len(train_idx) == 0:
            print(f"Skipping fold {i} (no training data)")
            continue

        # Select numeric features
        feature_cols = dat_dict[(dat_dict.model_features == 1) &
                                (dat_dict.type.isin(['int','float']))]['column'].values
        #exclude features            
        feature_cols = [col for col in feature_cols if col not in exclude_features]

        # Split data
        X_train = df_clean.loc[train_idx, feature_cols]
        y_train = df_clean.loc[train_idx, 'default_12m']

        X_val = df_clean.loc[val_idx, feature_cols]
        y_val = df_clean.loc[val_idx, 'default_12m']

        # Fit logistic regression
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)

        # Predict probabilities
        y_pred_train_prob = model.predict_proba(X_train)[:, 1]
        y_pred_prob = model.predict_proba(X_val)[:, 1]

        # Threshold tuning
        thres_new = tune_prob_threshold(y_train, y_pred_train_prob)
        
        # Predictions
        y_pred_train_tuned = (y_pred_train_prob > thres_new['tpr_fpr']).astype(int)
        y_pred_tuned = (y_pred_prob > thres_new['tpr_fpr']).astype(int)

        # Evaluate and collect data
        print(f"Fold: {i}")
        train_figs, train_metrics = discrete_evaluations(y_train, y_pred_train_tuned, y_pred_train_prob, type="train", model_path=folder_manager.output_path, fold=i)
        test_figs, test_metrics = discrete_evaluations(y_val, y_pred_tuned, y_pred_prob, type="test", model_path=folder_manager.output_path, fold=i)
        
        all_fold_data.append({'fold': i, 'type': 'train', 'figures': train_figs, 'metrics': train_metrics})
        all_fold_data.append({'fold': i, 'type': 'test', 'figures': test_figs, 'metrics': test_metrics})
        all_metrics_df = pd.concat([all_metrics_df, train_metrics, test_metrics], ignore_index=True)

    # Save all metrics to CSV
    metrics_filepath = os.path.join(folder_manager.output_path, 'metrics', 'logistic_regression_metrics.csv')
    os.makedirs(os.path.dirname(metrics_filepath), exist_ok=True)
    all_metrics_df.to_csv(metrics_filepath, index=False)
    print(f"All metrics saved to: {metrics_filepath}")

    # Generate HTML report
    generate_html_report(all_fold_data, os.path.join(folder_manager.output_path, 'logistic_regression_report.html'))

def get_params(algorithm_name, problem_type='classification'):
    '''algorithm_name: select from [lgbm,xgb,catboost,adaboost]'''
    if 'catboost' in algorithm_name:
        # CatBoost is not used in a pipeline in this script, so no pipeline_key prefix.
        if problem_type == 'classification':
            return {
                "depth": [4, 6, 8],
                "learning_rate": [0.02, 0.05],
                "iterations": [200, 400, 700],
                "l2_leaf_reg": [3, 7, 17],
                "border_count": [16, 32],
                "scale_pos_weight": [1,7,10, 20, 30]
            }
        else:  # regression
            return {
                'iterations': [100, 300, 500],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'l2_leaf_reg': [1, 3, 5, 7],
            }

    if problem_type == 'classification':
        if 'lightgbm' in algorithm_name:
            return {
               "num_leaves": [15, 31, 63],
                "max_depth": [3, 4, 5],
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [200, 500],
                "scale_pos_weight": [5, 10, 20],  # Imbalance parameter
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.7, 0.9],
                "min_child_samples": [20, 50]
            }
        elif 'xgboost' in algorithm_name:
            return {
               "max_depth": [3, 4, 6],          
                "learning_rate": [0.01, 0.03, 0.05], 
                "n_estimators": [200, 500, 1000],   
                "subsample":  [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_weight": [1, 5, 10],
                "gamma": [0, 0.5, 1.0],      
                "reg_alpha": [0, 0.1, 1.0],     
                "reg_lambda": [1.0, 5.0, 20.0],
                "scale_pos_weight": [1,7,10, 20, 30],  # Imbalance parameter
                
            }
        elif 'adaboost' in algorithm_name:
            return {
                'n_estimators': [50, 100, 200, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                'algorithm': ['SAMME', 'SAMME.R']
            }

def train_and_evaluate_gradient_boosting(df_clean, dat_dict, target, exclude_features, folds, algorithm,n_selected_features, model_path, baseline_threshold=0.028):
    
    all_fold_data = []
    all_metrics_df = pd.DataFrame()

    print(f"Performing Hyperparameter Tuning and Feature Selection for {algorithm} on Fold 0")
    train_idx, val_idx = folds[0]
    feature_cols = dat_dict[(dat_dict.model_features == 1) & (dat_dict.type.isin(['int','float']))]['column'].values
    feature_cols = [col for col in feature_cols if col not in exclude_features]
    print(f"Features used: {feature_cols}")
    X_train = df_clean.loc[train_idx, feature_cols]
    y_train = df_clean.loc[train_idx, target]

    # Define the parameter grid for GridSearchCV
    param_grid = get_params(algorithm)

    # Initialize the model
    if algorithm == 'xgboost':
        model = XGBClassifier()
        fit_params = {}
    elif algorithm == 'lightgbm':
        model = LGBMClassifier()
        fit_params = {
            'eval_set':[(df_clean.loc[val_idx, feature_cols], df_clean.loc[val_idx, target])],
            'eval_metric': 'auc',
            }
    elif algorithm == 'catboost':
        model = CatBoostClassifier( loss_function="Logloss", eval_metric="AUC",verbose=0) 
        fit_params = {
            "eval_set": [(df_clean.loc[val_idx, feature_cols], df_clean.loc[val_idx, target])], 
            "early_stopping_rounds": 75
            }
    elif algorithm == 'adaboost':
        model = AdaBoostClassifier()
        fit_params = {}
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Set up GridSearchCV for hyper parameter tuning
    # Could potentially leak data due to GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train, **fit_params)

    print(f"Best parameters found: {grid_search.best_params_}")
    
    # Use the best estimator for feature importance
    best_model = grid_search.best_estimator_
    
    if algorithm == 'catboost':
        feature_importances = best_model.get_feature_importance()
    else:
        feature_importances = best_model.feature_importances_

    feature_importance_df = pd.DataFrame({'feature_names': X_train.columns, 'feature_importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='feature_importance', ascending=False)
    
    # Clip N number of features if specifid in the input
    top_25_features = feature_importance_df.head(n_selected_features)
    selected_features = top_25_features['feature_names'].tolist()
    
    feature_importance_fig = plot_feature_importance(top_25_features.sort_values(by='feature_importance', ascending=True))
    all_fold_data.append({'fold': 'feature_importance', 'type': 'general', 'figures': {'feature_importance': feature_importance_fig}, 'metrics': pd.DataFrame()})

    # Get the best parameters for retraining on folds
    model_params = grid_search.best_params_

    for i, (train_idx, val_idx) in enumerate(folds):
        print(f"\n Training {algorithm} on Fold {i}")
        if len(train_idx) == 0:
            print(f"Skipping fold {i} (no training data)")
            continue

        X_train_fold = df_clean.loc[train_idx, selected_features]
        y_train_fold = df_clean.loc[train_idx, target]
        X_val_fold = df_clean.loc[val_idx, selected_features]
        y_val_fold = df_clean.loc[val_idx, target]
        
        # Initialize the model with the best parameters learned from gridsearchCV
        if algorithm == 'xgboost':
            model = XGBClassifier(**model_params)
        elif algorithm == 'lightgbm':
            model = LGBMClassifier(**model_params)
        elif algorithm == 'catboost':
            model = CatBoostClassifier(**model_params, verbose=0)
        elif algorithm == 'adaboost':
            model = AdaBoostClassifier(**model_params)

        model.fit(X_train_fold, y_train_fold)
        joblib.dump(model, os.path.join(model_path, f'{algorithm}_model_fold_{i}.pkl'))
        
        y_pred_train_prob = model.predict_proba(X_train_fold)[:, 1]
        y_pred_prob = model.predict_proba(X_val_fold)[:, 1]
        
        # tune probability threshold to maximize tpr-fpr or other stat
        thres_new = tune_prob_threshold(y_train_fold, y_pred_train_prob)
        
        y_pred_train_tuned = (y_pred_train_prob > thres_new['tpr_fpr']).astype(int)
        y_pred_tuned = (y_pred_prob > thres_new['tpr_fpr']).astype(int)


        print("Train Metrics: \n")
        train_figs, train_metrics = discrete_evaluations(y_train_fold, y_pred_train_tuned, y_pred_train_prob, type=f"train_{algorithm}", model_path=model_path, fold=i)
        print("Test Metrics: \n")
        test_figs, test_metrics = discrete_evaluations(y_val_fold, y_pred_tuned, y_pred_prob, type=f"test_{algorithm}", model_path=model_path, fold=i)

        # Evaluate business metrics
        eval_df = df_clean.loc[val_idx, ['loan_amount', 'apr', 'term']].copy()
        eval_df['p_default'] = y_pred_prob
        business_metrics = evaluate_model(eval_df)
        print(eval_df.sort_index().head(5),business_metrics.sort_index().head(5))
        
        all_fold_data.append({'fold': i, 'type': 'train', 'figures': train_figs, 'metrics': train_metrics})
        all_fold_data.append({'fold': i, 'type': 'test', 'figures': test_figs, 'metrics': test_metrics, 'summary_business_metrics': business_metrics.describe().T})
        all_metrics_df = pd.concat([all_metrics_df, train_metrics, test_metrics], ignore_index=True)
    
    # Save all metrics to CSV for experiment tracking
    metrics_filepath = os.path.join(model_path, 'metrics', f'{algorithm}_metrics.csv')
    os.makedirs(os.path.dirname(metrics_filepath), exist_ok=True)
    all_metrics_df.to_csv(metrics_filepath, index=False)
    print(f"All metrics saved to: {metrics_filepath}")

    generate_html_report(all_fold_data, os.path.join(model_path, f'{algorithm}_model_report.html'))