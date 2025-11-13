"""
Model evaluation and metrics utilities.

This module provides helper functions for evaluating model performance.
Candidates can use, modify, or replace these as needed.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_ks_statistic(y_true, y_pred_proba):
    """
    Calculate Kolmogorov-Smirnov (KS) statistic.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    
    Returns:
    --------
    float : KS statistic
    """
    # This is a placeholder - candidates should implement their own logic
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return np.max(tpr - fpr)

def calculate_model_metrics(y_true, y_pred_proba):
    """
    Compute key model metrics: AUC, KS, and PR-AUC.
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    ks = calculate_ks_statistic(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    return {"AUC": auc, "KS": ks, "PR-AUC": pr_auc}

def plot_roc_curve(y_true, y_pred_proba, model_name='Model'):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model for the plot title
    """
    # This is a placeholder - candidates should implement their own logic
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, model_name='Model'):
    """
    Plot Precision-Recall curve.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model for the plot title
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"{model_name} (PR-AUC={pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve - {model_name}")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.show()


def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    """
    Comprehensive model evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    threshold : float
        Classification threshold
    
    Returns:
    --------
    dict : Dictionary containing various evaluation metrics
    """
    # Binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    ks = calculate_ks_statistic(y_true, y_pred_proba)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # Print summary
    print(f"\n📊 Model Evaluation Summary")
    print(f"AUC: {auc:.4f} | KS: {ks:.4f} | Threshold: {threshold:.2f}")
    print("\nClassification Report:")
    print(pd.DataFrame(report).transpose())

    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return {
        "AUC": auc,
        "KS": ks,
        "ConfusionMatrix": cm,
        "ClassificationReport": report
    }
