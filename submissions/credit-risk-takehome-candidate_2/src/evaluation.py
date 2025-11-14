"""
Model evaluation and metrics utilities.

This module provides helper functions for evaluating model performance.
Candidates can use, modify, or replace these as needed.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import glob
from sklearn.preprocessing import MinMaxScaler
import base64
from io import BytesIO


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
    # Combine and sort by predicted probability (descending)
    data = pd.DataFrame({'y': y_true, 'score': y_pred_proba})
    data = data.sort_values(by='score', ascending=False)
    
    # Cumulative % of bads and goods
    data['cum_bad'] = np.cumsum(data['y'] == 1) / (data['y'] == 1).sum()
    data['cum_good'] = np.cumsum(data['y'] == 0) / (data['y'] == 0).sum()
    
    # KS distance at each threshold
    data['ks'] = np.abs(data['cum_bad'] - data['cum_good'])
    
    # Best KS and corresponding threshold
    idx = data['ks'].idxmax()
    ks = data.loc[idx, 'ks']
    threshold = data.loc[idx, 'score']
    
    return ks, threshold, data


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
    pass

def plot_confusion_matrix(cm,type_dat):
    fig = ff.create_annotated_heatmap(cm, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'])
    fig.update_layout(title_text=f'Confusion Matrix {type_dat}')
    return fig
    

def plot_roc(fpr,tpr,threshold_roc,roc_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_score:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dash')))
    fig.update_layout(title_text='Receiver Operating Characteristic (ROC) Curve')
    return fig
    

def plot_precision_recall(recall_curve,precision_curve,pr_auc_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode='lines', name=f'Precision-Recall curve (area = {pr_auc_score:.2f})'))
    fig.update_layout(title_text='Precision-Recall Curve')
    return fig
    

def plot_feature_importance(top_25_features):
    num_features = len(top_25_features)
    # Calculate height dynamically, e.g., 25 pixels per feature, with a minimum height
    plot_height = max(600, num_features * 25) # Minimum 400px, then 25px per feature
    fig = px.bar(top_25_features, x='feature_importance', y='feature_names', orientation='h', title='Top 25 Feature Importances', height=plot_height)
    
    # Add more margin to the left to make space for feature names
    fig.update_layout(
        bargap = 0.5
    )
    return fig
    


def discrete_evaluations(actual,pred,pred_proba=None,type = "test",model_path = "",fold = -1):
    """
    helper for general classification metrics
    actual: true target
    pred: predicted target
    pred_proba: predicted probabilities from the model
    path: path to the submodel folder
    Check usual discrete target evaluations
    """
    
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    roc_score = roc_auc_score(actual, pred_proba)
    ks_statistic, _, _ = calculate_ks_statistic(actual, pred_proba)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"ROC AUC Score: {roc_score}")
    print(f"KS Statistic: {ks_statistic}")

    metrics = {
        'phase' : type,
        'fold': fold,
        'datetime': datetime.now(),
        'n_samples': len(actual),
        'mean_target': actual.mean(),
        'precision' : precision,
        'recall' : recall,
        'f1' : f1,
        'auc_score' : roc_score,
        'ks_statistic': ks_statistic
    }
    
    # Plots
    cm = confusion_matrix(actual,pred)
    cm_fig = plot_confusion_matrix(cm,type)

    fpr, tpr, thresholds_roc = roc_curve(actual, pred_proba)
    roc_fig = plot_roc(fpr,tpr,thresholds_roc,roc_score)

    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(actual, pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    pr_fig = plot_precision_recall(recall_curve,precision_curve,pr_auc)

    figures = {
        'confusion_matrix': cm_fig,
        'roc_curve': roc_fig,
        'pr_curve': pr_fig
    }
    
    return figures, pd.DataFrame([metrics])


def evaluate_model(df,  p_default_col="p_default",  principal_col="loan_amount", apr_col="apr", term_col="term",recovery_rate = .2):
    """
    Comprehensive model evaluation using expected value
    
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
    
    df = df.copy()

    expected_interest = df[principal_col] * (df[apr_col]/100) * (df[term_col] / 12)

    payoff_if_repaid = df[principal_col] + expected_interest

    payoff_if_default = df[principal_col] * recovery_rate

    # Binomial EV
    df["expected_value"] = (1 - df[p_default_col]) * payoff_if_repaid + df[p_default_col] * payoff_if_default

    # Translate to relative gain
    df["expected_roi"] = (df["expected_value"] - df[principal_col]) / df[principal_col]

    return df[["expected_value", "expected_roi"]]
    

def compare_model_metrics(folder_names, outputs_dir='../outputs'):
    """
    Generates head-to-head comparison plots and tables for model metrics.

    Args:
        folder_names (list): A list of folder names within the outputs directory.
        outputs_dir (str): The path to the outputs directory.
    """
    if isinstance(folder_names, str):
        try:
            import ast
            folder_names = ast.literal_eval(folder_names)
        except (ValueError, SyntaxError):
            # Not a list literal, treat as a comma-separated string
            folder_names = [name.strip() for name in folder_names.split(',')]

    if not isinstance(folder_names, list):
        print(f"Error: folder_names must be a list of strings, but got {type(folder_names)}")
        return
        
    all_metrics = []
    for folder_name in folder_names:
        metrics_path = os.path.join(outputs_dir, folder_name, 'metrics')
        csv_files = glob.glob(os.path.join(metrics_path, '*_metrics.csv'))
        
        if not csv_files:
            print(f"No metrics CSV found in {metrics_path}. Skipping.")
            continue
        
        # Take the first metrics file found
        metrics_file = csv_files[0]
        try:
            df = pd.read_csv(metrics_file)
            df['model'] = folder_name
            all_metrics.append(df)
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
            continue

    if not all_metrics:
        print("No metrics found to compare.")
        return

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    
    # Filter for test metrics and average over folds
    test_metrics = combined_metrics[combined_metrics['phase'].str.contains('test', case=False, na=False)].copy()
    
    if test_metrics.empty:
        print("No 'test' phase metrics found to compare.")
        return

    metric_cols = ['precision', 'recall', 'f1', 'auc_score', 'ks_statistic']
    
    # Ensure metric columns exist
    metric_cols = [col for col in metric_cols if col in test_metrics.columns]
    if not metric_cols:
        print("None of the specified metric columns were found.")
        return

    comparison_df = test_metrics.groupby('model')[metric_cols].mean().reset_index()

    # Create a directory for the comparison report
    report_dir_name = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_dir = os.path.join(outputs_dir, report_dir_name)
    os.makedirs(report_dir, exist_ok=True)
    print(f"Comparison report directory created at: {report_dir}")

    html_report_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .plot-container { margin-bottom: 30px; border: 1px solid #eee; padding: 15px; border-radius: 5px; background-color: #f9f9f9; }
            img { max-width: 100%; height: auto; display: block; margin: 10px auto; }
            .plotly-graph-div { margin: 10px auto; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Model Comparison Report</h1>
        <p>Report Generated On: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        <p>Compared Models: """ + ", ".join(folder_names) + """</p>
    """

    # 1. Comparison Table
    try:
        html_report_content += "<h2>Model Metrics Comparison Table:</h2>\n"
        styled_table = comparison_df.set_index('model').style.background_gradient(cmap='viridis', axis=1).set_caption("Model Metrics Comparison")
        html_report_content += styled_table.to_html()
    except Exception as e:
        html_report_content += f"<p>Error generating comparison table: {e}</p>\n"
        print(f"Could not generate styled table: {e}")

    # 2. Bar Plots for each metric
    html_report_content += "<h2>Individual Metric Bar Plots:</h2>\n"
    for metric in metric_cols:
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=comparison_df, x='model', y=metric)
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            html_report_content += f"<div class='plot-container'><h3>{metric.replace('_', ' ').title()} Comparison:</h3>\n"
            html_report_content += f"<img src='data:image/png;base64,{img_base64}' alt='{metric} Comparison'></div>\n"
            plt.close()
        except Exception as e:
            html_report_content += f"<p>Error generating bar plot for {metric}: {e}</p>\n"
            print(f"Could not generate bar plot for {metric}: {e}")

    # 3. Heatmap
    try:
        heatmap_df = comparison_df.set_index('model')
        if not heatmap_df.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_df, annot=True, fmt=".4f", cmap="viridis")
            plt.title('Metrics Heatmap')
            plt.yticks(rotation=0)
            plt.tight_layout()

            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            html_report_content += f"<div class='plot-container'><h2>Metrics Heatmap:</h2>\n"
            html_report_content += f"<img src='data:image/png;base64,{img_base64}' alt='Metrics Heatmap'></div>\n"
            plt.close()
    except Exception as e:
        html_report_content += f"<p>Error generating heatmap: {e}</p>\n"
        print(f"Could not generate heatmap: {e}")

    # 4. Radar Plot
    try:
        heatmap_df = comparison_df.set_index('model')
        if not heatmap_df.empty:
            scaler = MinMaxScaler()
            normalized_metrics = scaler.fit_transform(heatmap_df)
            normalized_df = pd.DataFrame(normalized_metrics, index=heatmap_df.index, columns=heatmap_df.columns)
            
            fig = go.Figure()

            for index, row in normalized_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=row.values,
                    theta=normalized_df.columns,
                    fill='toself',
                    name=row.name
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Model Comparison Radar Plot"
            )
            
            radar_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            html_report_content += f"<div class='plot-container'><h2>Model Comparison Radar Plot:</h2>\n"
            html_report_content += radar_html + "</div>\n"
    except Exception as e:
        html_report_content += f"<p>Error generating radar plot: {e}</p>\n"
        print(f"Could not generate radar plot: {e}")

    html_report_content += """
    </body>
    </html>
    """

    # Final write to a single file
    report_file_path = os.path.join(report_dir, 'model_comparison_report.html')
    with open(report_file_path, 'w') as f:
        f.write(html_report_content)
    print(f"Full model comparison report saved to: {report_file_path}")