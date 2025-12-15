#!/usr/bin/env python3
"""
Epi4Ab Results Dashboard - Comprehensive visualization for inference results.

Generates an interactive HTML dashboard with:
- Classification metrics (confusion matrix, F1, precision, recall, accuracy)
- Per-position probability plots
- Amino acid analysis
- Consolidated summary across all PDBs

Usage:
    python scripts/visualize_results.py --test_record_dir output_inference/2025-12-12_GNNResNet_7/test_record
    python scripts/visualize_results.py --test_record_dir output_inference/2025-12-12_GNNResNet_7/test_record --pdb_id 1n8z_BAC
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT))

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Error: Plotly is required. Install with: pip install plotly")
    sys.exit(1)

from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, precision_recall_fscore_support
)

# Label definitions
LABEL_NAMES = {
    0: 'Non-epitope',
    1: 'CIPS (Direct Contact)',
    2: 'BepiPred/Ellipro'
}

LABEL_COLORS = {
    0: '#808080',  # Grey
    1: '#2ecc71',  # Green
    2: '#3498db'   # Blue
}

# Amino acid properties
AA_PROPERTIES = {
    'hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P', 'G'],
    'polar': ['S', 'T', 'C', 'Y', 'N', 'Q'],
    'positive': ['K', 'R', 'H'],
    'negative': ['D', 'E']
}

AA_PROPERTY_COLORS = {
    'hydrophobic': '#e74c3c',
    'polar': '#9b59b6',
    'positive': '#3498db',
    'negative': '#e67e22'
}


def get_aa_property(aa: str) -> str:
    """Get the property category of an amino acid."""
    for prop, aas in AA_PROPERTIES.items():
        if aa in aas:
            return prop
    return 'other'


def load_detailed_results(test_record_dir: str, pdb_id: str) -> pd.DataFrame:
    """Load detailed results with true_y, pred_y, and all probabilities."""
    result_file = Path(test_record_dir) / f'{pdb_id}.txt'
    if not result_file.exists():
        raise FileNotFoundError(f"Detailed result file not found: {result_file}")
    
    df = pd.read_csv(result_file, sep='\t')
    return df


def load_final_results(test_record_dir: str, pdb_id: str) -> pd.DataFrame:
    """Load final results with residue names."""
    result_file = Path(test_record_dir) / f'{pdb_id}_final_result.txt'
    if not result_file.exists():
        raise FileNotFoundError(f"Final result file not found: {result_file}")
    
    df = pd.read_csv(result_file, sep='\t')
    return df


def load_all_pdb_results(test_record_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load results from all PDBs and combine into single DataFrame."""
    test_record_path = Path(test_record_dir)
    
    # Find all PDB IDs
    result_files = list(test_record_path.glob('*_final_result.txt'))
    pdb_ids = [f.stem.replace('_final_result', '') for f in result_files]
    
    if not pdb_ids:
        raise FileNotFoundError(f"No result files found in {test_record_dir}")
    
    all_data = []
    for pdb_id in pdb_ids:
        try:
            # Load detailed results
            detailed = load_detailed_results(test_record_dir, pdb_id)
            final = load_final_results(test_record_dir, pdb_id)
            
            # Merge to get residue names
            merged = detailed.copy()
            merged['res_name'] = final['res_name']
            merged['score'] = final['score']
            merged['pdb_id'] = pdb_id
            
            all_data.append(merged)
        except Exception as e:
            print(f"Warning: Could not load {pdb_id}: {e}")
    
    if not all_data:
        raise ValueError("No valid PDB results could be loaded")
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined, pdb_ids


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute classification metrics."""
    # Get unique labels present in data
    labels = sorted(set(y_true) | set(y_pred))
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels),
        'labels': labels
    }
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    metrics['per_class'] = {}
    for i, label in enumerate(labels):
        metrics['per_class'][label] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': int(support[i])
        }
    
    # Macro and weighted averages
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    return metrics


def compute_per_pdb_metrics(df: pd.DataFrame, pdb_ids: List[str]) -> pd.DataFrame:
    """Compute metrics for each PDB separately."""
    records = []
    for pdb_id in pdb_ids:
        pdb_data = df[df['pdb_id'] == pdb_id]
        if len(pdb_data) == 0:
            continue
        
        y_true = pdb_data['true_y'].values
        y_pred = pdb_data['pred_y'].values
        
        metrics = compute_metrics(y_true, y_pred)
        
        records.append({
            'pdb_id': pdb_id,
            'n_residues': len(pdb_data),
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall']
        })
    
    return pd.DataFrame(records)


def create_confusion_matrix_figure(cm: np.ndarray, labels: List[int]) -> go.Figure:
    """Create an interactive confusion matrix heatmap."""
    label_names = [LABEL_NAMES.get(l, str(l)) for l in labels]
    
    # Create text annotations
    text = [[str(val) for val in row] for row in cm]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=label_names,
        y=label_names,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14},
        colorscale='Blues',
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400,
        width=500
    )
    
    return fig


def create_metrics_table(metrics: Dict) -> go.Figure:
    """Create a metrics table figure."""
    labels = metrics['labels']
    
    # Build table data
    headers = ['Label', 'Precision', 'Recall', 'F1-Score', 'Support']
    rows = []
    
    for label in labels:
        pc = metrics['per_class'][label]
        rows.append([
            LABEL_NAMES.get(label, str(label)),
            f"{pc['precision']:.4f}",
            f"{pc['recall']:.4f}",
            f"{pc['f1']:.4f}",
            str(pc['support'])
        ])
    
    # Add summary rows
    rows.append(['---', '---', '---', '---', '---'])
    rows.append(['Accuracy', '', '', f"{metrics['accuracy']:.4f}", ''])
    rows.append(['Macro Avg', f"{metrics['macro_precision']:.4f}", 
                 f"{metrics['macro_recall']:.4f}", f"{metrics['macro_f1']:.4f}", ''])
    rows.append(['Weighted Avg', '', '', f"{metrics['weighted_f1']:.4f}", ''])
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color='#3498db',
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color=[['#f8f9fa', '#ffffff'] * (len(rows) // 2 + 1)][:len(rows)],
            font=dict(size=11),
            align='center',
            height=25
        )
    )])
    
    fig.update_layout(
        title='Classification Metrics',
        height=350,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    
    return fig


def create_per_position_plot(df: pd.DataFrame, pdb_id: str) -> go.Figure:
    """Create per-position probability plot for a single PDB."""
    pdb_data = df[df['pdb_id'] == pdb_id].sort_values('res_id')
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f'Prediction Probabilities - {pdb_id}', 'True vs Predicted Labels']
    )
    
    res_ids = pdb_data['res_id'].values
    
    # Add probability traces
    for label, color in LABEL_COLORS.items():
        col_name = f'prob_{label}'
        if col_name in pdb_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=res_ids,
                    y=pdb_data[col_name].values,
                    mode='lines',
                    name=LABEL_NAMES[label],
                    line=dict(color=color, width=1.5),
                    hovertemplate=f'{LABEL_NAMES[label]}<br>Position: %{{x}}<br>Prob: %{{y:.4f}}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Add true vs predicted labels
    fig.add_trace(
        go.Scatter(
            x=res_ids,
            y=pdb_data['true_y'].values,
            mode='markers',
            name='True Label',
            marker=dict(color='black', size=6, symbol='circle'),
            hovertemplate='True Label: %{y}<br>Position: %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=res_ids,
            y=pdb_data['pred_y'].values + 0.1,  # Slight offset for visibility
            mode='markers',
            name='Predicted Label',
            marker=dict(color='red', size=6, symbol='x'),
            hovertemplate='Pred Label: %{y:.0f}<br>Position: %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text='Residue Position', row=2, col=1)
    fig.update_yaxes(title_text='Probability', range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text='Label', tickvals=[0, 1, 2], row=2, col=1)
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    
    return fig


def create_per_position_plot_full(df: pd.DataFrame, pdb_id: str) -> go.Figure:
    """Create full-width scrollable per-position probability plot for a single PDB."""
    pdb_data = df[df['pdb_id'] == pdb_id].sort_values('res_id')
    n_residues = len(pdb_data)
    
    # Make width proportional to residue count (min 3 pixels per residue)
    width = max(1200, n_residues * 3)
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f'Prediction Probabilities - {pdb_id} (Full View)', 'True vs Predicted Labels']
    )
    
    res_ids = pdb_data['res_id'].values
    
    # Add probability traces
    for label, color in LABEL_COLORS.items():
        col_name = f'prob_{label}'
        if col_name in pdb_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=res_ids,
                    y=pdb_data[col_name].values,
                    mode='lines+markers',
                    name=LABEL_NAMES[label],
                    line=dict(color=color, width=2),
                    marker=dict(size=3),
                    hovertemplate=f'{LABEL_NAMES[label]}<br>Position: %{{x}}<br>Prob: %{{y:.4f}}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Add true vs predicted labels
    fig.add_trace(
        go.Scatter(
            x=res_ids,
            y=pdb_data['true_y'].values,
            mode='markers',
            name='True Label',
            marker=dict(color='black', size=8, symbol='circle'),
            hovertemplate='True Label: %{y}<br>Position: %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=res_ids,
            y=pdb_data['pred_y'].values + 0.1,  # Slight offset for visibility
            mode='markers',
            name='Predicted Label',
            marker=dict(color='red', size=8, symbol='x'),
            hovertemplate='Pred Label: %{y:.0f}<br>Position: %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text='Residue Position', row=2, col=1, rangeslider_visible=True)
    fig.update_yaxes(title_text='Probability', range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text='Label', tickvals=[0, 1, 2], row=2, col=1)
    
    fig.update_layout(
        height=600,
        width=width,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    
    return fig


def create_per_position_plot_binned(df: pd.DataFrame, pdb_id: str, bin_size: int = 50) -> go.Figure:
    """Create binned/averaged per-position probability plot for a single PDB."""
    pdb_data = df[df['pdb_id'] == pdb_id].sort_values('res_id').copy()
    
    if len(pdb_data) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data available for {pdb_id}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Bin the data
    pdb_data['bin'] = (pdb_data['res_id'] // bin_size) * bin_size
    
    # Aggregate data by bins
    binned_data = pdb_data.groupby('bin').agg({
        'prob_0': 'mean' if 'prob_0' in pdb_data.columns else lambda x: 0,
        'prob_1': 'mean' if 'prob_1' in pdb_data.columns else lambda x: 0,
        'prob_2': 'mean' if 'prob_2' in pdb_data.columns else lambda x: 0,
        'true_y': lambda x: x.mode()[0] if len(x) > 0 else 0,
        'pred_y': lambda x: x.mode()[0] if len(x) > 0 else 0
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f'Prediction Probabilities - {pdb_id} (Binned by {bin_size})', 'True vs Predicted Labels']
    )
    
    bin_centers = binned_data['bin'].values
    
    # Add probability traces
    for label, color in LABEL_COLORS.items():
        col_name = f'prob_{label}'
        if col_name in binned_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=binned_data[col_name].values,
                    mode='lines+markers',
                    name=LABEL_NAMES[label],
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    hovertemplate=f'{LABEL_NAMES[label]}<br>Bin: %{{x}}<br>Avg Prob: %{{y:.4f}}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Add true vs predicted labels
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=binned_data['true_y'].values,
            mode='markers',
            name='True Label (Mode)',
            marker=dict(color='black', size=10, symbol='circle'),
            hovertemplate='True Label: %{y}<br>Bin: %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=binned_data['pred_y'].values + 0.1,  # Slight offset for visibility
            mode='markers',
            name='Predicted Label (Mode)',
            marker=dict(color='red', size=10, symbol='x'),
            hovertemplate='Pred Label: %{y:.0f}<br>Bin: %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text=f'Residue Position (Binned by {bin_size})', row=2, col=1)
    fig.update_yaxes(title_text='Average Probability', range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text='Label', tickvals=[0, 1, 2], row=2, col=1)
    
    fig.update_layout(
        height=500,
        width=1200,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    
    return fig


def create_aa_analysis(df: pd.DataFrame) -> go.Figure:
    """Create amino acid analysis charts."""
    # Compute accuracy per amino acid
    aa_stats = []
    for aa in df['res_name'].unique():
        aa_data = df[df['res_name'] == aa]
        correct = (aa_data['true_y'] == aa_data['pred_y']).sum()
        total = len(aa_data)
        # Clamp accuracy to max 1.0 to prevent display issues
        accuracy = min(correct / total, 1.0) if total > 0 else 0
        aa_stats.append({
            'aa': aa,
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'property': get_aa_property(aa)
        })
    
    aa_df = pd.DataFrame(aa_stats).sort_values('accuracy', ascending=True)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Accuracy by Amino Acid', 'Sample Count by Amino Acid'],
        horizontal_spacing=0.1
    )
    
    # Accuracy bar chart
    colors = [AA_PROPERTY_COLORS.get(p, '#95a5a6') for p in aa_df['property']]
    
    fig.add_trace(
        go.Bar(
            x=aa_df['accuracy'],
            y=aa_df['aa'],
            orientation='h',
            marker_color=colors,
            text=[f"{acc:.2%}" for acc in aa_df['accuracy']],
            textposition='outside',
            hovertemplate='AA: %{y}<br>Accuracy: %{x:.2%}<br>Samples: %{customdata}<extra></extra>',
            customdata=aa_df['total']
        ),
        row=1, col=1
    )
    
    # Sample count bar chart
    fig.add_trace(
        go.Bar(
            x=aa_df['total'],
            y=aa_df['aa'],
            orientation='h',
            marker_color=colors,
            text=aa_df['total'],
            textposition='outside',
            hovertemplate='AA: %{y}<br>Count: %{x}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text='Accuracy', range=[0, 1.05], row=1, col=1)
    fig.update_xaxes(title_text='Sample Count', row=1, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text='Amino Acid Analysis'
    )
    
    return fig


def create_pdb_performance_heatmap(per_pdb_df: pd.DataFrame) -> go.Figure:
    """Create a heatmap showing performance metrics per PDB."""
    # Prepare data
    metrics_cols = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']
    metric_names = ['Accuracy', 'Macro F1', 'Weighted F1', 'Macro Prec', 'Macro Recall']
    
    # Replace NaN values with 0.0 to prevent empty cells
    z_data = per_pdb_df[metrics_cols].fillna(0.0).values
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=metric_names,
        y=per_pdb_df['pdb_id'].values,
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        text=[[f"{val:.3f}" for val in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='PDB: %{y}<br>Metric: %{x}<br>Value: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Performance Metrics by PDB',
        height=max(400, len(per_pdb_df) * 25 + 100),
        xaxis_title='Metric',
        yaxis_title='PDB ID'
    )
    
    return fig


def create_label_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create donut charts showing true vs predicted label distributions."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'pie'}]],
        subplot_titles=['True Label Distribution', 'Predicted Label Distribution']
    )
    
    # True labels - filter to only ground truth labels (0 and 1)
    # Label 2 (BepiPred) should not appear in ground truth
    true_counts = df[df['true_y'].isin([0, 1])]['true_y'].value_counts().sort_index()
    fig.add_trace(
        go.Pie(
            labels=[LABEL_NAMES.get(l, str(l)) for l in true_counts.index],
            values=true_counts.values,
            hole=0.4,
            marker_colors=[LABEL_COLORS.get(l, '#95a5a6') for l in true_counts.index],
            textinfo='label+percent',
            hovertemplate='%{label}<br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Predicted labels
    pred_counts = df['pred_y'].value_counts().sort_index()
    fig.add_trace(
        go.Pie(
            labels=[LABEL_NAMES.get(l, str(l)) for l in pred_counts.index],
            values=pred_counts.values,
            hole=0.4,
            marker_colors=[LABEL_COLORS.get(l, '#95a5a6') for l in pred_counts.index],
            textinfo='label+percent',
            hovertemplate='%{label}<br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Label Distribution (All PDBs Combined)',
        height=400
    )
    
    return fig


def create_score_distribution(df: pd.DataFrame) -> go.Figure:
    """Create score distribution histogram."""
    fig = go.Figure()
    
    # Check if score column exists and has valid data
    if 'score' not in df.columns or df['score'].isna().all():
        fig.add_annotation(
            text="Score data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#7f8c8d')
        )
        fig.update_layout(
            title='Score Distribution by Predicted Label',
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    # Plot histogram for each label
    for label in sorted(df['pred_y'].unique()):
        label_data = df[df['pred_y'] == label]['score'].dropna()
        if len(label_data) > 0:
            fig.add_trace(
                go.Histogram(
                    x=label_data,
                    name=LABEL_NAMES.get(label, str(label)),
                    marker_color=LABEL_COLORS.get(label, '#95a5a6'),
                    opacity=0.7,
                    nbinsx=30  # Explicit bin count for better visualization
                )
            )
    
    fig.update_layout(
        title='Score Distribution by Predicted Label',
        xaxis_title='Score',
        yaxis_title='Count',
        barmode='overlay',
        height=400,
        showlegend=True
    )
    
    return fig


def create_summary_cards(df: pd.DataFrame, per_pdb_df: pd.DataFrame, metrics: Dict) -> str:
    """Create HTML summary cards."""
    n_pdbs = len(per_pdb_df)
    n_residues = len(df)
    overall_acc = metrics['accuracy']
    mean_f1 = per_pdb_df['macro_f1'].mean()
    
    # Count correct/incorrect
    correct = (df['true_y'] == df['pred_y']).sum()
    incorrect = n_residues - correct
    
    cards_html = f"""
    <div style="display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0;">
        <div style="background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 20px; border-radius: 10px; min-width: 150px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 32px; font-weight: bold;">{n_pdbs}</div>
            <div style="font-size: 14px; opacity: 0.9;">Total PDBs</div>
        </div>
        <div style="background: linear-gradient(135deg, #9b59b6, #8e44ad); color: white; padding: 20px; border-radius: 10px; min-width: 150px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 32px; font-weight: bold;">{n_residues:,}</div>
            <div style="font-size: 14px; opacity: 0.9;">Total Residues</div>
        </div>
        <div style="background: linear-gradient(135deg, #2ecc71, #27ae60); color: white; padding: 20px; border-radius: 10px; min-width: 150px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 32px; font-weight: bold;">{overall_acc:.1%}</div>
            <div style="font-size: 14px; opacity: 0.9;">Overall Accuracy</div>
        </div>
        <div style="background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; padding: 20px; border-radius: 10px; min-width: 150px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 32px; font-weight: bold;">{mean_f1:.3f}</div>
            <div style="font-size: 14px; opacity: 0.9;">Mean F1 Score</div>
        </div>
        <div style="background: linear-gradient(135deg, #1abc9c, #16a085); color: white; padding: 20px; border-radius: 10px; min-width: 150px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 32px; font-weight: bold;">{correct:,}</div>
            <div style="font-size: 14px; opacity: 0.9;">Correct Predictions</div>
        </div>
        <div style="background: linear-gradient(135deg, #e67e22, #d35400); color: white; padding: 20px; border-radius: 10px; min-width: 150px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 32px; font-weight: bold;">{incorrect:,}</div>
            <div style="font-size: 14px; opacity: 0.9;">Incorrect Predictions</div>
        </div>
    </div>
    """
    return cards_html


def create_pdb_ranking_table(per_pdb_df: pd.DataFrame) -> go.Figure:
    """Create a sortable PDB ranking table."""
    sorted_df = per_pdb_df.sort_values('accuracy', ascending=False)
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Rank', 'PDB ID', 'Residues', 'Accuracy', 'Macro F1', 'Weighted F1'],
            fill_color='#2c3e50',
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=[
                list(range(1, len(sorted_df) + 1)),
                sorted_df['pdb_id'].values,
                sorted_df['n_residues'].values,
                [f"{v:.4f}" for v in sorted_df['accuracy'].values],
                [f"{v:.4f}" for v in sorted_df['macro_f1'].values],
                [f"{v:.4f}" for v in sorted_df['weighted_f1'].values]
            ],
            fill_color=[['#ecf0f1', '#ffffff'] * (len(sorted_df) // 2 + 1)][:len(sorted_df)],
            font=dict(size=11),
            align='center',
            height=25
        )
    )])
    
    fig.update_layout(
        title='PDB Performance Ranking (by Accuracy)',
        height=max(300, len(sorted_df) * 30 + 100)
    )
    
    return fig


def _generate_tab_buttons_html(position_html: Dict) -> str:
    """Generate HTML for tab buttons."""
    buttons = []
    for i, pdb_id in enumerate(position_html.keys()):
        active = " active" if i == 0 else ""
        buttons.append(f'<button class="tab{active}" onclick="showTab(\'tab-{pdb_id}\')">{pdb_id}</button>')
    return "".join(buttons)


def _generate_tab_contents_html(position_html: Dict) -> str:
    """Generate HTML for tab contents with embedded plots."""
    contents = []
    for i, (pdb_id, html) in enumerate(position_html.items()):
        active = " active" if i == 0 else ""
        contents.append(f'<div id="tab-{pdb_id}" class="tab-content{active}">{html}</div>')
    return "".join(contents)


def _generate_pdb_links(pdb_ids: List[str], output_path: Path) -> str:
    """Generate HTML links to individual PDB visualization files."""
    # Determine relative path to visualizations folder
    vis_dir = output_path.parent / 'visualizations'
    links = []
    
    for pdb_id in sorted(pdb_ids):
        # Check if individual visualization file exists
        vis_file = vis_dir / f'{pdb_id}_probability_plot.html'
        if vis_file.exists():
            # Use relative path
            rel_path = f'visualizations/{pdb_id}_probability_plot.html'
            links.append(
                f'<a href="{rel_path}" target="_blank" '
                f'style="display: block; padding: 12px; background: #3498db; color: white; '
                f'text-decoration: none; border-radius: 5px; text-align: center; '
                f'transition: background 0.3s; font-weight: 500;" '
                f'onmouseover="this.style.background=\'#2980b9\'" '
                f'onmouseout="this.style.background=\'#3498db\'">{pdb_id}</a>'
            )
        else:
            # Show PDB ID but without link if file doesn't exist
            links.append(
                f'<div style="display: block; padding: 12px; background: #95a5a6; color: white; '
                f'border-radius: 5px; text-align: center; opacity: 0.7;">{pdb_id} (no viz)</div>'
            )
    
    return "".join(links)


def generate_dashboard_html(
    df: pd.DataFrame,
    pdb_ids: List[str],
    per_pdb_df: pd.DataFrame,
    overall_metrics: Dict,
    output_path: Path
):
    """Generate the complete HTML dashboard."""
    
    # Create all figures with explicit dimensions
    cm_fig = create_confusion_matrix_figure(
        overall_metrics['confusion_matrix'], 
        overall_metrics['labels']
    )
    cm_fig.update_layout(height=450, width=600)
    
    metrics_table = create_metrics_table(overall_metrics)
    metrics_table.update_layout(height=400)
    
    label_dist = create_label_distribution_chart(df)
    label_dist.update_layout(height=450)
    
    pdb_heatmap = create_pdb_performance_heatmap(per_pdb_df)
    # Already has dynamic height
    
    pdb_ranking = create_pdb_ranking_table(per_pdb_df)
    # Already has dynamic height
    
    aa_analysis = create_aa_analysis(df)
    aa_analysis.update_layout(height=600)
    
    score_dist = create_score_distribution(df)
    score_dist.update_layout(height=450)
    
    # Per-PDB position plots (create tabs) - with explicit dimensions
    position_plots = {}
    for pdb_id in pdb_ids[:10]:  # Limit to first 10 for performance
        try:
            plot = create_per_position_plot(df, pdb_id)
            plot.update_layout(height=600, width=1000)  # Explicit size for tabs
            position_plots[pdb_id] = plot
        except Exception as e:
            print(f"Warning: Could not create position plot for {pdb_id}: {e}")
    
    # Generate summary cards HTML
    summary_cards = create_summary_cards(df, per_pdb_df, overall_metrics)
    
    # Convert figures to div-only HTML (not full HTML documents)
    cm_html = cm_fig.to_html(include_plotlyjs='cdn', full_html=False, div_id='confusion-matrix')
    metrics_html = metrics_table.to_html(include_plotlyjs=False, full_html=False, div_id='metrics-table')
    label_html = label_dist.to_html(include_plotlyjs=False, full_html=False, div_id='label-distribution')
    heatmap_html = pdb_heatmap.to_html(include_plotlyjs=False, full_html=False, div_id='pdb-heatmap')
    ranking_html = pdb_ranking.to_html(include_plotlyjs=False, full_html=False, div_id='pdb-ranking')
    aa_html = aa_analysis.to_html(include_plotlyjs=False, full_html=False, div_id='aa-analysis')
    score_html = score_dist.to_html(include_plotlyjs=False, full_html=False, div_id='score-distribution')
    
    # Convert position plots to div-only HTML
    position_html = {}
    for pdb_id, fig in position_plots.items():
        position_html[pdb_id] = fig.to_html(include_plotlyjs=False, full_html=False, div_id=f'position-{pdb_id}')
    
    # Build HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Epi4Ab Results Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f6fa;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 10px;
        }}
        .tab {{
            padding: 8px 16px;
            background: #ecf0f1;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
        }}
        .tab:hover {{
            background: #bdc3c7;
        }}
        .tab.active {{
            background: #3498db;
            color: white;
        }}
        .tab-content {{
            display: none;
            padding: 20px;
            min-height: 600px;
        }}
        .tab-content.active {{
            display: block;
        }}
        .tab-content > div {{
            width: 100%;
            min-height: 500px;
        }}
        @media (max-width: 1200px) {{
            .grid-2 {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Epi4Ab Epitope Prediction Dashboard</h1>
        <p>Comprehensive analysis of {len(pdb_ids)} PDB structures | {len(df):,} total residues</p>
    </div>
    
    <div class="section">
        <h2>Summary Overview</h2>
        {summary_cards}
    </div>
    
    <div class="section">
        <h2>Classification Performance</h2>
        <div class="grid-2">
            {cm_html}
            {metrics_html}
        </div>
    </div>
    
    <div class="section">
        <h2>Label Distribution</h2>
        {label_html}
    </div>
    
    <div class="section">
        <h2>Per-PDB Performance</h2>
        {heatmap_html}
        <div style="margin-top: 20px;">{ranking_html}</div>
    </div>
    
    <div class="section">
        <h2>Amino Acid Analysis</h2>
        {aa_html}
    </div>
    
    <div class="section">
        <h2>Score Distribution</h2>
        {score_html}
    </div>
    
    <div class="section">
        <h2>Per-Position Probability Plots</h2>
        <p style="margin-bottom: 15px; color: #666;">
            Interactive plots for individual PDBs. Use tabs below to switch between PDBs (first 10 shown inline).
            For all PDBs, see individual visualization files linked below.
        </p>
        <div class="tabs">
            {_generate_tab_buttons_html(position_html)}
        </div>
        {_generate_tab_contents_html(position_html)}
    </div>
    
    <div class="section">
        <h2>Individual PDB Visualizations</h2>
        <p style="margin-bottom: 15px; color: #666;">
            Click on any PDB ID below to open its detailed visualization in a new tab:
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">
            {_generate_pdb_links(pdb_ids, output_path)}
        </div>
    </div>
    
    <script>
        // Tab switching function (plots are embedded as HTML with their own scripts)
        function showTab(tabId) {{
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(el => {{
                el.classList.remove('active');
                el.style.display = 'none';
            }});
            
            // Remove active class from all buttons
            document.querySelectorAll('.tab').forEach(el => {{
                el.classList.remove('active');
            }});
            
            // Show selected tab
            const selectedTab = document.getElementById(tabId);
            if (selectedTab) {{
                selectedTab.classList.add('active');
                selectedTab.style.display = 'block';
            }}
            
            // Activate clicked button
            const clickedButton = document.querySelector(`button[onclick*="${{tabId}}"]`);
            if (clickedButton) {{
                clickedButton.classList.add('active');
            }}
            
            // Trigger resize to fix plotly rendering
            window.dispatchEvent(new Event('resize'));
        }}
    </script>
</body>
</html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard saved to: {output_path}")


def generate_individual_pdb_html(df: pd.DataFrame, pdb_id: str, vis_dir: Path):
    """Generate comprehensive individual PDB visualization HTML file."""
    pdb_data = df[df['pdb_id'] == pdb_id]
    
    if len(pdb_data) == 0:
        print(f"  ⚠ No data found for {pdb_id}")
        return
    
    # Compute metrics for this PDB
    pdb_metrics = compute_metrics(pdb_data['true_y'].values, pdb_data['pred_y'].values)
    
    # Create all visualizations
    cm_fig = create_confusion_matrix_figure(pdb_metrics['confusion_matrix'], pdb_metrics['labels'])
    label_dist = create_label_distribution_chart(pdb_data)
    aa_analysis = create_aa_analysis(pdb_data)
    score_dist = create_score_distribution(pdb_data)
    
    # Two versions of position plot
    pos_plot_full = create_per_position_plot_full(df, pdb_id)
    pos_plot_binned = create_per_position_plot_binned(df, pdb_id, bin_size=50)
    
    # Convert figures to div-only HTML (not full HTML documents)
    cm_html = cm_fig.to_html(include_plotlyjs='cdn', full_html=False, div_id='confusion-matrix')
    label_html = label_dist.to_html(include_plotlyjs=False, full_html=False, div_id='label-dist')
    aa_html = aa_analysis.to_html(include_plotlyjs=False, full_html=False, div_id='aa-analysis')
    score_html = score_dist.to_html(include_plotlyjs=False, full_html=False, div_id='score-dist')
    full_html = pos_plot_full.to_html(include_plotlyjs=False, full_html=False, div_id='position-full')
    binned_html = pos_plot_binned.to_html(include_plotlyjs=False, full_html=False, div_id='position-binned')
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{pdb_id} - Detailed Epitope Prediction Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f6fa;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .scrollable {{
            overflow-x: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background: #f9f9f9;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 12px;
            opacity: 0.9;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{pdb_id}</h1>
        <p>Detailed Epitope Prediction Analysis</p>
    </div>
    
    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{pdb_metrics['accuracy']:.3f}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{pdb_metrics['macro_f1']:.3f}</div>
                <div class="metric-label">Macro F1</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{pdb_metrics['weighted_f1']:.3f}</div>
                <div class="metric-label">Weighted F1</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(pdb_data)}</div>
                <div class="metric-label">Total Residues</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Confusion Matrix</h2>
        {cm_html}
    </div>
    
    <div class="section">
        <h2>Label Distribution</h2>
        {label_html}
    </div>
    
    <div class="section">
        <h2>Full Position Plot (Scrollable)</h2>
        <p style="color: #666; margin-bottom: 15px;">
            This plot shows all residues in detail. Scroll horizontally to see the entire sequence.
            Use the range slider at the bottom to navigate.
        </p>
        <div class="scrollable">
            {full_html}
        </div>
    </div>
    
    <div class="section">
        <h2>Position Plot (Binned by 50 Residues)</h2>
        <p style="color: #666; margin-bottom: 15px;">
            This plot shows averaged predictions for easier interpretation. Each point represents ~50 residues.
        </p>
        {binned_html}
    </div>
    
    <div class="section">
        <h2>Amino Acid Analysis</h2>
        {aa_html}
    </div>
    
    <div class="section">
        <h2>Score Distribution</h2>
        {score_html}
    </div>
</body>
</html>"""
    
    # Write HTML file
    vis_file = vis_dir / f'{pdb_id}_probability_plot.html'
    vis_file.write_text(html)


def main():
    parser = argparse.ArgumentParser(description='Generate Epi4Ab Results Dashboard')
    parser.add_argument('--test_record_dir', type=str, required=True,
                       help='Path to test_record directory')
    parser.add_argument('--pdb_id', type=str, default=None,
                       help='Specific PDB ID to visualize (if not provided, visualizes all)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for dashboard (default: test_record_dir/../)')
    
    args = parser.parse_args()
    
    test_record_dir = Path(args.test_record_dir)
    if not test_record_dir.exists():
        print(f"Error: Test record directory not found: {test_record_dir}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = test_record_dir.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {test_record_dir}")
    
    # Load all data
    if args.pdb_id:
        # Single PDB mode
        detailed = load_detailed_results(str(test_record_dir), args.pdb_id)
        final = load_final_results(str(test_record_dir), args.pdb_id)
        detailed['res_name'] = final['res_name']
        detailed['score'] = final['score']
        detailed['pdb_id'] = args.pdb_id
        df = detailed
        pdb_ids = [args.pdb_id]
    else:
        # All PDBs mode
        df, pdb_ids = load_all_pdb_results(str(test_record_dir))
    
    print(f"Loaded {len(pdb_ids)} PDB(s) with {len(df):,} total residues")
    
    # Compute metrics
    print("Computing metrics...")
    overall_metrics = compute_metrics(df['true_y'].values, df['pred_y'].values)
    per_pdb_df = compute_per_pdb_metrics(df, pdb_ids)
    
    print(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Macro F1: {overall_metrics['macro_f1']:.4f}")
    
    # Generate dashboard
    print("Generating dashboard...")
    output_path = output_dir / 'dashboard.html'
    generate_dashboard_html(df, pdb_ids, per_pdb_df, overall_metrics, output_path)
    
    # Generate individual PDB visualization files
    print("Generating individual PDB visualizations...")
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    for pdb_id in pdb_ids:
        try:
            generate_individual_pdb_html(df, pdb_id, vis_dir)
            print(f"  ✓ Generated {pdb_id}_probability_plot.html")
        except Exception as e:
            print(f"  ✗ Failed to generate visualization for {pdb_id}: {e}")
    
    print(f"\nDashboard generation complete!")
    print(f"Open {output_path} in a web browser to view.")
    print(f"Individual PDB visualizations saved in: {vis_dir}")


if __name__ == '__main__':
    main()
