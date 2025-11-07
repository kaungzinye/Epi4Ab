"""
Interactive HTML Visualizer for Epi4Ab Inference Results

Generates comprehensive visualization including:
- Per-residue predictions
- Performance metrics (ROC curves, confusion matrices)
- 3D structure visualization
- Summary statistics
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False
    print("Warning: py3Dmol not available. 3D visualization will be limited. Install with: pip install py3dmol")


# Label color mapping
LABEL_COLORS = {
    0: '#808080',  # Grey - Non-epitope
    1: '#90EE90',  # Light green - CIPS (direct contact)
    2: '#87CEEB'   # Sky blue - BepiPred/Ellipro (predicted epitope)
}

LABEL_NAMES = {
    0: 'Non-epitope',
    1: 'CIPS (Direct Contact)',
    2: 'BepiPred/Ellipro (Predicted Epitope)'
}


def load_prediction_results(test_record_dir: str, pdb_id: str) -> Optional[pd.DataFrame]:
    """Load prediction results for a single PDB."""
    result_file = Path(test_record_dir) / f'{pdb_id}_final_result.txt'
    if not result_file.exists():
        return None
    
    df = pd.read_csv(result_file, sep='\t')
    return df


def load_detailed_results(test_record_dir: str, pdb_id: str) -> Optional[pd.DataFrame]:
    """Load detailed results with all probabilities."""
    result_file = Path(test_record_dir) / f'{pdb_id}.txt'
    if not result_file.exists():
        return None
    
    df = pd.read_csv(result_file, sep='\t')
    return df


def create_probability_plot(df: pd.DataFrame, df_detailed: Optional[pd.DataFrame], pdb_id: str) -> str:
    """Create probability plot showing epitope probability vs residue position."""
    if not PLOTLY_AVAILABLE:
        return "<p>Plotly not available for probability plot.</p>"
    
    # Get residue IDs and probabilities
    res_ids = df['res_id'].values
    
    # Get epitope probability (prob_2 for label 2 - BepiPred/Ellipro predicted epitope)
    if df_detailed is not None and 'prob_2' in df_detailed.columns:
        epitope_probs = df_detailed['prob_2'].values
        true_labels = df_detailed['true_y'].values if 'true_y' in df_detailed.columns else None
    else:
        # Fallback: use prob. column if available, or estimate from pred_label
        if 'prob.' in df.columns:
            # If prob. is the probability of predicted label, we need prob_2
            # For now, use 1 - prob. as approximation if pred_label is 0
            epitope_probs = np.where(df['pred_label'] == 2, df['prob.'].values, 
                                    np.where(df['pred_label'] == 1, df['prob.'].values * 0.5, 
                                            df['prob.'].values * 0.1))
        else:
            epitope_probs = np.zeros(len(df))
        true_labels = None
    
    # Create figure
    fig = go.Figure()
    
    # Add background regions for true labels if available
    if true_labels is not None:
        # Find regions with different true labels
        prev_label = None
        start_idx = 0
        for i, label in enumerate(true_labels):
            if prev_label is not None and label != prev_label:
                # Add background rectangle for previous label
                color = LABEL_COLORS.get(int(prev_label), '#E0E0E0')
                fig.add_shape(
                    type="rect",
                    x0=res_ids[start_idx] - 0.5,
                    y0=0,
                    x1=res_ids[i-1] + 0.5,
                    y1=1,
                    fillcolor=color,
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
                start_idx = i
            prev_label = label
        
        # Add final rectangle
        if prev_label is not None:
            color = LABEL_COLORS.get(int(prev_label), '#E0E0E0')
            fig.add_shape(
                type="rect",
                x0=res_ids[start_idx] - 0.5,
                y0=0,
                x1=res_ids[-1] + 0.5,
                y1=1,
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
            )
    
    # Add probability line
    fig.add_trace(go.Scatter(
        x=res_ids,
        y=epitope_probs,
        mode='lines+markers',
        name='Epitope Probability',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=3, color='#2E86AB'),
        hovertemplate='Residue: %{x}<br>Probability: %{y:.4f}<extra></extra>'
    ))
    
    # Add threshold line at 0.5
    max_res = res_ids.max() if len(res_ids) > 0 else 607
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="Threshold (0.5)",
        annotation_position="right"
    )
    
    # Update layout
    fig.update_layout(
        title=f'Epitope Probability vs Residue Position - {pdb_id}',
        xaxis_title='Residue Position',
        yaxis_title='Probability of Being Epitope',
        yaxis=dict(range=[0, 1]),
        xaxis=dict(range=[0, max_res]),
        height=500,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig.to_html(include_plotlyjs='cdn', div_id=f'prob_plot_{pdb_id}')


def create_per_residue_table(df: pd.DataFrame, pdb_id: str) -> str:
    """Create interactive per-residue predictions table (kept for backward compatibility)."""
    if not PLOTLY_AVAILABLE:
        # Fallback to simple HTML table
        html = f"""
        <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
            <tr style="background-color: #f2f2f2;">
                <th style="border: 1px solid #ddd; padding: 8px;">Residue ID</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Residue Name</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Predicted Label</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Probability</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Score</th>
            </tr>
        """
        for _, row in df.iterrows():
            label_color = LABEL_COLORS.get(int(row['pred_label']), '#808080')
            label_name = LABEL_NAMES.get(int(row['pred_label']), f"Label {row['pred_label']}")
            html += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">{row['res_id']}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{row['res_name']}</td>
                <td style="border: 1px solid #ddd; padding: 8px; background-color: {label_color};">{row['pred_label']} ({label_name})</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{row['prob.']:.4f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{row['score']:.4f}</td>
            </tr>
            """
        html += "</table>"
        return html
    
    # Add color column based on predicted label
    df['color'] = df['pred_label'].map(LABEL_COLORS)
    df['label_name'] = df['pred_label'].map(LABEL_NAMES)
    
    # Get true labels if available
    true_labels = df.get('true_y', ['N/A'] * len(df))
    if isinstance(true_labels, pd.Series):
        true_labels = true_labels.tolist()
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Residue ID', 'Residue Name', 'True Label', 'Predicted Label', 
                   'Probability', 'Score', 'Label Type'],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[
                df['res_id'].tolist(),
                df['res_name'].tolist(),
                true_labels,
                df['pred_label'].tolist(),
                [f'{p:.4f}' for p in df['prob.']],
                [f'{s:.4f}' for s in df['score']],
                df['label_name'].tolist()
            ],
            fill_color=[['white' if i % 2 == 0 else 'lightgray' for i in range(len(df))]],
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title=f'Per-Residue Predictions - {pdb_id}',
        height=600
    )
    
    return fig.to_html(include_plotlyjs='cdn', div_id=f'table_{pdb_id}')


def create_confusion_matrix(true_labels: np.ndarray, pred_labels: np.ndarray, 
                           pdb_id: str, class_names: List[str] = None) -> str:
    """Create confusion matrix visualization."""
    if not PLOTLY_AVAILABLE:
        return "<p>Plotly not available for confusion matrix.</p>"
    
    from sklearn.metrics import confusion_matrix
    
    if class_names is None:
        all_classes = sorted(set(true_labels) | set(pred_labels))
        class_names = [LABEL_NAMES.get(c, f'Class {c}') for c in all_classes]
    
    cm = confusion_matrix(true_labels, pred_labels, labels=sorted(set(true_labels) | set(pred_labels)))
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {pdb_id}',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=500
    )
    
    return fig.to_html(include_plotlyjs='cdn', div_id=f'cm_{pdb_id}')


def create_roc_curves(true_labels: np.ndarray, prob_matrix: np.ndarray, 
                      pdb_id: str) -> str:
    """Create ROC curves for each class."""
    if not PLOTLY_AVAILABLE:
        return "<p>Plotly not available for ROC curves.</p>"
    
    from sklearn.metrics import roc_curve, auc
    
    fig = go.Figure()
    
    # Plot ROC curve for each class
    for class_idx in range(prob_matrix.shape[1]):
        if class_idx not in true_labels:
            continue
        
        # Create binary labels for this class
        y_binary = (true_labels == class_idx).astype(int)
        y_scores = prob_matrix[:, class_idx]
        
        if len(np.unique(y_binary)) < 2:
            continue
        
        fpr, tpr, _ = roc_curve(y_binary, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{LABEL_NAMES.get(class_idx, f"Class {class_idx}")} (AUC = {roc_auc:.3f})',
            line=dict(color=LABEL_COLORS.get(class_idx, '#000000'), width=2)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title=f'ROC Curves - {pdb_id}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500
    )
    
    return fig.to_html(include_plotlyjs='cdn', div_id=f'roc_{pdb_id}')


def create_label_distribution_chart(true_labels: np.ndarray, pred_labels: np.ndarray,
                                   pdb_id: str) -> str:
    """Create label distribution comparison chart."""
    if not PLOTLY_AVAILABLE:
        return "<p>Plotly not available for distribution chart.</p>"
    
    true_counts = pd.Series(true_labels).value_counts().sort_index()
    pred_counts = pd.Series(pred_labels).value_counts().sort_index()
    
    labels = [LABEL_NAMES.get(i, f'Class {i}') for i in sorted(set(true_labels) | set(pred_labels))]
    true_values = [true_counts.get(i, 0) for i in sorted(set(true_labels) | set(pred_labels))]
    pred_values = [pred_counts.get(i, 0) for i in sorted(set(true_labels) | set(pred_labels))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='True Labels',
        x=labels,
        y=true_values,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Predicted Labels',
        x=labels,
        y=pred_values,
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title=f'Label Distribution - {pdb_id}',
        xaxis_title='Label Type',
        yaxis_title='Count',
        barmode='group',
        width=700,
        height=400
    )
    
    return fig.to_html(include_plotlyjs='cdn', div_id=f'dist_{pdb_id}')


def create_3d_structure_html(pdb_id: str, pred_labels: np.ndarray, 
                            res_ids: np.ndarray, pdb_file_path: Optional[str] = None) -> str:
    """Create 3D structure visualization with py3Dmol."""
    # Try to find PDB file
    if pdb_file_path is None:
        # Try common locations
        possible_paths = [
            f'/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files/{pdb_id}.pdb',
            f'./pdb_files/{pdb_id}.pdb',
            f'../pdb_files/{pdb_id}.pdb'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pdb_file_path = path
                break
    
    if pdb_file_path is None or not os.path.exists(pdb_file_path):
        return f"""
        <div style="padding: 20px; background-color: #f0f0f0; border-radius: 5px;">
            <p><b>3D Structure Visualization</b></p>
            <p>PDB file not found for {pdb_id}.</p>
            <p>Expected locations: {', '.join(possible_paths)}</p>
            <p><i>Note: Install py3dmol and provide PDB file path for 3D visualization.</i></p>
        </div>
        """
    
    # Read PDB file
    try:
        with open(pdb_file_path, 'r') as f:
            pdb_content = f.read()
    except Exception as e:
        return f"<p>Error reading PDB file: {e}</p>"
    
    # Create color mapping for residues
    label_to_color = {
        0: 'gray',
        1: 'green',
        2: 'blue'
    }
    
    # Create residue color mapping
    residue_colors = {int(rid): label_to_color.get(int(pl), 'gray') 
                      for rid, pl in zip(res_ids, pred_labels)}
    
    # Escape PDB content for JavaScript
    pdb_content_escaped = pdb_content.replace('`', '\\`').replace('$', '\\$')
    
    # Create visualization HTML using py3Dmol CDN
    html = f"""
    <div style="margin: 20px 0;">
        <h4>3D Structure with Predicted Epitopes</h4>
        <div id="structure_{pdb_id}" style="width: 800px; height: 600px; position: relative;"></div>
        <script src="https://cdn.jsdelivr.net/npm/3dmol@2.1.0/build/3Dmol-min.js"></script>
        <script>
            (function() {{
                var viewer = $3Dmol.createViewer(document.getElementById('structure_{pdb_id}'), {{
                    defaultcolors: $3Dmol.rasmolElementColors
                }});
                
                var pdbData = `{pdb_content_escaped}`;
                viewer.addModel(pdbData, "pdb");
                
                // Set default style
                viewer.setStyle({{}}, {{cartoon: {{color: 'lightgray'}}}});
                
                // Color residues by predicted labels
                var residueColors = {json.dumps(residue_colors)};
                
                for (var resId in residueColors) {{
                    var color = residueColors[resId];
                    viewer.setStyle({{resi: parseInt(resId)}}, {{cartoon: {{color: color}}}});
                }}
                
                viewer.zoomTo();
                viewer.render();
            }})();
        </script>
        <div style="margin-top: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
            <p><b>Color Legend:</b></p>
            <ul style="margin: 5px 0;">
                <li><span style="color: gray;">Gray</span> = Non-epitope (Label 0)</li>
                <li><span style="color: green;">Green</span> = CIPS - Direct Contact (Label 1)</li>
                <li><span style="color: blue;">Blue</span> = Predicted Epitope (Label 2)</li>
            </ul>
        </div>
    </div>
    """
    
    return html


def create_summary_metrics_html(metrics: Dict[str, float], pdb_id: str) -> str:
    """Create summary metrics display."""
    html = f"""
    <div class="metrics-summary">
        <h3>Performance Metrics - {pdb_id}</h3>
        <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
            <tr style="background-color: #f2f2f2;">
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Metric</th>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
            </tr>
    """
    
    metric_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score',
        'roc_auc': 'ROC AUC',
        'avg_precision': 'Average Precision'
    }
    
    for key, value in metrics.items():
        if key in metric_names:
            display_name = metric_names[key]
            if isinstance(value, float) and np.isnan(value):
                display_value = 'N/A'
            else:
                display_value = f'{value:.4f}' if isinstance(value, (int, float)) else str(value)
            
            html += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;"><b>{display_name}</b></td>
                <td style="border: 1px solid #ddd; padding: 8px;">{display_value}</td>
            </tr>
            """
    
    html += """
        </table>
    </div>
    """
    
    return html


def generate_pdb_visualization(test_record_dir: str, pdb_id: str, 
                               evaluation_metrics: Optional[Dict] = None,
                               pdb_file_path: Optional[str] = None) -> str:
    """Generate complete HTML visualization for a single PDB."""
    
    # Load results
    df_final = load_prediction_results(test_record_dir, pdb_id)
    df_detailed = load_detailed_results(test_record_dir, pdb_id)
    
    if df_final is None:
        return f"<p>No results found for {pdb_id}</p>"
    
    # Extract data
    pred_labels = df_final['pred_label'].values
    res_ids = df_final['res_id'].values
    
    # Get true labels if available
    if df_detailed is not None and 'true_y' in df_detailed.columns:
        true_labels = df_detailed['true_y'].values
        prob_matrix = df_detailed[['prob_0', 'prob_1', 'prob_2']].values
    else:
        true_labels = None
        prob_matrix = None
    
    # Start building HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Epi4Ab Results - {pdb_id}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
                border-bottom: 2px solid #ddd;
                padding-bottom: 5px;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background-color: #fafafa;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Epi4Ab Inference Results - {pdb_id}</h1>
    """
    
    # Add summary metrics if available
    if evaluation_metrics:
        html += '<div class="section">'
        html += create_summary_metrics_html(evaluation_metrics, pdb_id)
        html += '</div>'
    
    # Add probability plot (replacing per-residue table)
    html += '<div class="section">'
    html += '<h2>Epitope Probability vs Residue Position</h2>'
    html += create_probability_plot(df_final, df_detailed, pdb_id)
    html += '</div>'
    
    # Add label distribution
    html += '<div class="section">'
    html += '<h2>Label Distribution</h2>'
    if true_labels is not None:
        html += create_label_distribution_chart(true_labels, pred_labels, pdb_id)
    else:
        html += f"<p>True labels not available. Predicted distribution: {pd.Series(pred_labels).value_counts().to_dict()}</p>"
    html += '</div>'
    
    # Add confusion matrix if true labels available
    if true_labels is not None:
        html += '<div class="section">'
        html += '<h2>Confusion Matrix</h2>'
        html += create_confusion_matrix(true_labels, pred_labels, pdb_id)
        html += '</div>'
        
        # Add ROC curves if probabilities available
        if prob_matrix is not None:
            html += '<div class="section">'
            html += '<h2>ROC Curves</h2>'
            html += create_roc_curves(true_labels, prob_matrix, pdb_id)
            html += '</div>'
    
    # Add 3D structure visualization
    html += '<div class="section">'
    html += '<h2>3D Structure Visualization</h2>'
    html += create_3d_structure_html(pdb_id, pred_labels, res_ids, pdb_file_path)
    html += '</div>'
    
    # Close HTML
    html += """
        </div>
    </body>
    </html>
    """
    
    return html


def generate_aggregate_visualization(test_record_dir: str, all_pdb_ids: List[str],
                                     evaluation_df: Optional[pd.DataFrame] = None) -> str:
    """Generate aggregate HTML visualization for all PDBs."""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Epi4Ab Results - Aggregate Summary</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
                border-bottom: 2px solid #ddd;
                padding-bottom: 5px;
            }}
            .pdb-link {{
                display: inline-block;
                margin: 10px;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }}
            .pdb-link:hover {{
                background-color: #45a049;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Epi4Ab Inference Results - Aggregate Summary</h1>
            
            <div class="section">
                <h2>Individual PDB Results</h2>
                <p>Click on a PDB ID to view detailed results:</p>
    """
    
    # Add links to individual PDB visualizations
    for pdb_id in sorted(all_pdb_ids):
        html += f'<a href="{pdb_id}_results.html" class="pdb-link">{pdb_id}</a>\n'
    
    html += '</div>'
    
    # Add summary metrics table if available
    if evaluation_df is not None:
        html += '<div class="section">'
        html += '<h2>Performance Metrics Summary</h2>'
        
        # Try to find PDB column (could be first column or named 'PDB')
        pdb_col = None
        if 'PDB' in evaluation_df.columns:
            pdb_col = 'PDB'
        elif len(evaluation_df.columns) > 0:
            pdb_col = evaluation_df.columns[0]  # Assume first column is PDB ID
        
        if pdb_col and PLOTLY_AVAILABLE:
            # Try to find metric columns (case-insensitive)
            metric_cols = {}
            for col in evaluation_df.columns:
                col_lower = col.lower()
                if 'accuracy' in col_lower:
                    metric_cols['accuracy'] = col
                elif 'roc' in col_lower or 'auc' in col_lower:
                    metric_cols['roc_auc'] = col
                elif 'f1' in col_lower:
                    metric_cols['f1'] = col
            
            if metric_cols:
                # Group by PDB and calculate mean
                agg_dict = {col: 'mean' for col in metric_cols.values()}
                summary_table = evaluation_df.groupby(pdb_col).agg(agg_dict).reset_index()
                
                # Build table values
                header_values = ['PDB ID']
                cell_values = [summary_table[pdb_col].tolist()]
                
                for metric_name, col_name in metric_cols.items():
                    header_values.append(metric_name.replace('_', ' ').title())
                    cell_values.append([
                        f'{v:.4f}' if not np.isnan(v) else 'N/A' 
                        for v in summary_table[col_name]
                    ])
                
                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=header_values,
                        fill_color='paleturquoise',
                        align='left'
                    ),
                    cells=dict(
                        values=cell_values,
                        fill_color='white',
                        align='left'
                    )
                )])
                
                html += fig.to_html(include_plotlyjs='cdn', div_id='summary_table')
            else:
                html += "<p>Could not find metric columns in evaluation data.</p>"
        else:
            html += "<p>Evaluation data format not recognized.</p>"
        
        html += '</div>'
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

