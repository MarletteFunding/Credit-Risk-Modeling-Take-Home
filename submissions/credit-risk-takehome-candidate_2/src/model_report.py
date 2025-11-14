import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

def generate_html_report(all_fold_data, report_filepath):
    """
    Generates a self-contained HTML report with evaluation metrics and plots for each fold. 
    
    Args:
        all_fold_data (list): A list of dictionaries, where each dictionary contains
                              'fold', 'type', 'figures', and 'metrics' for a specific evaluation.
        report_filepath (str): The full path including filename for the output HTML report.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }
            .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; }
            .tab button:hover { background-color: #ddd; }
            .tab button.active { background-color: #ccc; }
            .tabcontent { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }
            .plot-container { margin-bottom: 20px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px;}
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>

    <h1>Model Evaluation Report</h1>

    <div class="tab">
    """

    # Extract feature importance plot
    feature_importance_plot = None
    for data in all_fold_data:
        if data['type'] == 'general' and 'feature_importance' in data.get('figures', {}):
            feature_importance_plot = data['figures']['feature_importance']
            break

    if feature_importance_plot:
        feature_importance_plot.update_yaxes(categoryorder="total ascending")

    # Add tab buttons, skipping the general 'feature_importance' tab
    for i, data in enumerate(all_fold_data):
        if data['type'] == 'general':
            continue
        tab_id = f"tab-{i}"
        
        # Safely determine fold_display_id
        if isinstance(data['fold'], int):
            fold_display_id = data['fold'] + 1
        else:
            fold_display_id = data['fold']

        label = f"{data['type'].capitalize()} - Fold {fold_display_id}"
        html_content += f"<button class='tablinks' onclick=\"openTab(event, '{tab_id}')\">{label}</button>\n"
    html_content += "</div>\n"

    # Add tab content
    for i, data in enumerate(all_fold_data):
        if data['type'] == 'general':
            continue
        tab_id = f"tab-{i}"
        
        # Safely determine fold_display_id for content
        if isinstance(data['fold'], int):
            fold_display_id = data['fold'] + 1
        else:
            fold_display_id = data['fold']

        eval_type = data['type']
        figures = data['figures']
        metrics_df = data['metrics']

        html_content += f"<div id='{tab_id}' class='tabcontent'>\n"
        html_content += f"<h2>{eval_type.capitalize()} - Fold {fold_display_id}</h2>\n"
        
        # Add metrics table
        if not metrics_df.empty:
            html_content += "<h3>Model Metrics:</h3>\n"
            formatters = {col: '{:.4f}'.format for col in metrics_df.select_dtypes(include=['float']).columns}
            html_content += metrics_df.to_html(index=False, formatters=formatters)

        # Add business metrics table
        if 'summary_business_metrics' in data and not data['summary_business_metrics'].empty:
            html_content += "<h3>Business Metrics Summary:</h3>\n"
            business_metrics_df = data['summary_business_metrics'].reset_index()
            business_metrics_df.columns = ['Metric', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
            
            # Convert expected_value to millions and rename
            ev_mask = business_metrics_df['Metric'] == 'expected_value'
            numeric_cols = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
            if ev_mask.any():
                business_metrics_df.loc[ev_mask, numeric_cols] /= 1_000

            business_metrics_df['Metric'] = business_metrics_df['Metric'].replace({
                'expected_value': 'Expected_Revenue(In Thousands)',
                'expected_roi': 'Expected ROI(Gain)'
            })
            
            formatters = {col: '{:.4f}'.format for col in numeric_cols}
            formatters['count'] = '{:.0f}'.format
            html_content += business_metrics_df.to_html(index=False, formatters=formatters)
        
        # Add figures
        for fig_name, fig_obj in figures.items():
            if fig_obj: # Ensure figure object exists
                html_content += f"<h3>{fig_name.replace('_', ' ').title()}:</h3>\n"
                html_content += pio.to_html(fig_obj, full_html=False, include_plotlyjs='cdn', div_id=f"{tab_id}-{fig_name}")
            else:
                html_content += f"<p>No plot available for {fig_name}.</p>\n"
        
        # Add feature importance plot to each test tab
        if eval_type == 'test' and feature_importance_plot:
            html_content += "<h3>Feature Importance:</h3>\n"
            html_content += pio.to_html(feature_importance_plot, full_html=False, include_plotlyjs='cdn', div_id=f"{tab_id}-feature_importance")

        html_content += "</div>\n"

    html_content += """
    <script>
    function openTab(evt, tabName) {
        var i, tabcontent, tablinks;
        tabcontent = document.getElementsByClassName("tabcontent");
        for (i = 0; i < tabcontent.length; i++) {
            tabcontent[i].style.display = "none";
        }
        tablinks = document.getElementsByClassName("tablinks");
        for (i = 0; i < tablinks.length; i++) {
            tablinks[i].className = tablinks[i].className.replace(" active", "");
        }
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";
    }

    // Open the first tab by default
    document.addEventListener("DOMContentLoaded", function() {
        if (document.getElementsByClassName("tablinks").length > 0) {
            document.getElementsByClassName("tablinks")[0].click();
        }
    });
    </script>

    </body>
    </html>
    """

    with open(report_filepath, 'w') as f:
        f.write(html_content)

    print(f"HTML report generated at: {report_filepath}")