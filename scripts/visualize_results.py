#!/usr/bin/env python3
"""
Visualize Epi4Ab inference results - similar to fork visualization.

Usage:
    python scripts/visualize_results.py --test_record_dir output_inference/2025-12-04_GNNResNet_1/test_record
    python scripts/visualize_results.py --test_record_dir output_inference/2025-12-04_GNNResNet_1/test_record --pdb_id 1n8z_BAC
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

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
    print("Note: Plotly not available. Using matplotlib only. Install with: pip install plotly")

# Label color mapping (matching fork)
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


def load_prediction_results(test_record_dir: str, pdb_id: str) -> pd.DataFrame:
    """Load prediction results for a single PDB."""
    result_file = Path(test_record_dir) / f'{pdb_id}_final_result.txt'
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    
    df = pd.read_csv(result_file, sep='\t')
    return df


def create_probability_plot_matplotlib(df: pd.DataFrame, pdb_id: str, output_dir: Path):
    """Create probability plot using matplotlib."""
    fig, ax = plt.subplots(figsize=(15, 6))
    
    res_ids = df['res_id'].values
    probs = df['prob.'].values
    pred_labels = df['pred_label'].values
    
    # Color points by predicted label
    colors = [LABEL_COLORS[label] for label in pred_labels]
    
    # Plot probability line
    ax.plot(res_ids, probs, 'b-', alpha=0.3, linewidth=1, label='Probability')
    ax.scatter(res_ids, probs, c=colors, s=20, alpha=0.6, edgecolors='black', linewidths=0.5)
    
    # Add threshold line
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    
    # Add labels for different prediction types
    for label in [0, 1, 2]:
        mask = pred_labels == label
        if mask.any():
            ax.scatter(res_ids[mask], probs[mask], 
                      c=LABEL_COLORS[label], s=30, alpha=0.8,
                      label=LABEL_NAMES[label], edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('Prediction Probability', fontsize=12)
    ax.set_title(f'Epitope Prediction Probabilities - {pdb_id}', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    output_file = output_dir / f'{pdb_id}_probability_plot.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved probability plot: {output_file}")
    
    return output_file


def create_probability_plot_plotly(df: pd.DataFrame, pdb_id: str, output_dir: Path):
    """Create interactive probability plot using plotly."""
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = go.Figure()
    
    res_ids = df['res_id'].values
    probs = df['prob.'].values
    pred_labels = df['pred_label'].values
    
    # Add traces for each label type
    for label in [0, 1, 2]:
        mask = pred_labels == label
        if mask.any():
            fig.add_trace(go.Scatter(
                x=res_ids[mask],
                y=probs[mask],
                mode='markers+lines',
                name=LABEL_NAMES[label],
                line=dict(color=LABEL_COLORS[label], width=2),
                marker=dict(size=5, color=LABEL_COLORS[label]),
                hovertemplate=f'Residue: %{{x}}<br>Probability: %{{y:.4f}}<br>Label: {LABEL_NAMES[label]}<extra></extra>'
            ))
    
    # Add threshold line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="Threshold (0.5)",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=f'Epitope Prediction Probabilities - {pdb_id}',
        xaxis_title='Residue Position',
        yaxis_title='Prediction Probability',
        yaxis=dict(range=[0, 1]),
        height=500,
        hovermode='x unified',
        showlegend=True,
        template='plotly_white'
    )
    
    output_file = output_dir / f'{pdb_id}_probability_plot.html'
    fig.write_html(str(output_file))
    print(f"  ✓ Saved interactive plot: {output_file}")
    
    return output_file


def create_summary_statistics(df: pd.DataFrame, pdb_id: str, output_dir: Path):
    """Create summary statistics table."""
    stats = {
        'Total Residues': len(df),
        'Predicted Non-epitope (0)': len(df[df['pred_label'] == 0]),
        'Predicted CIPS (1)': len(df[df['pred_label'] == 1]),
        'Predicted BepiPred/Ellipro (2)': len(df[df['pred_label'] == 2]),
        'Mean Probability': df['prob.'].mean(),
        'Max Probability': df['prob.'].max(),
        'Min Probability': df['prob.'].min(),
        'Mean Score': df['score'].mean(),
    }
    
    # Create summary text
    summary_text = f"""
# Prediction Summary for {pdb_id}

## Statistics
"""
    for key, value in stats.items():
        if isinstance(value, float):
            summary_text += f"- **{key}**: {value:.4f}\n"
        else:
            summary_text += f"- **{key}**: {value}\n"
    
    summary_text += f"""
## Label Distribution
- Non-epitope (0): {stats['Predicted Non-epitope (0)']} residues ({stats['Predicted Non-epitope (0)']/stats['Total Residues']*100:.1f}%)
- CIPS (1): {stats['Predicted CIPS (1)']} residues ({stats['Predicted CIPS (1)']/stats['Total Residues']*100:.1f}%)
- BepiPred/Ellipro (2): {stats['Predicted BepiPred/Ellipro (2)']} residues ({stats['Predicted BepiPred/Ellipro (2)']/stats['Total Residues']*100:.1f}%)

## High Confidence Predictions
"""
    high_conf = df[df['prob.'] > 0.8]
    if len(high_conf) > 0:
        summary_text += f"- {len(high_conf)} residues with probability > 0.8\n"
        summary_text += f"- Top 5 highest probabilities:\n"
        for _, row in high_conf.nlargest(5, 'prob.').iterrows():
            summary_text += f"  - Residue {row['res_id']} ({row['res_name']}): {row['prob.']:.4f} (Label {row['pred_label']})\n"
    else:
        summary_text += "- No residues with probability > 0.8\n"
    
    output_file = output_dir / f'{pdb_id}_summary.md'
    with open(output_file, 'w') as f:
        f.write(summary_text)
    print(f"  ✓ Saved summary: {output_file}")
    
    return output_file


def visualize_pdb(test_record_dir: str, pdb_id: str, output_dir: Path):
    """Generate all visualizations for a single PDB."""
    print(f"\nVisualizing {pdb_id}...")
    
    # Load results
    df = load_prediction_results(test_record_dir, pdb_id)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    create_probability_plot_matplotlib(df, pdb_id, output_dir)
    if PLOTLY_AVAILABLE:
        create_probability_plot_plotly(df, pdb_id, output_dir)
    create_summary_statistics(df, pdb_id, output_dir)
    
    print(f"  ✓ Completed visualization for {pdb_id}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Epi4Ab inference results')
    parser.add_argument('--test_record_dir', type=str, required=True,
                       help='Path to test_record directory')
    parser.add_argument('--pdb_id', type=str, default=None,
                       help='Specific PDB ID to visualize (if not provided, visualizes all)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for visualizations (default: test_record_dir/../visualizations)')
    
    args = parser.parse_args()
    
    test_record_dir = Path(args.test_record_dir)
    if not test_record_dir.exists():
        print(f"Error: Test record directory not found: {test_record_dir}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = test_record_dir.parent / 'visualizations'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDB IDs
    if args.pdb_id:
        pdb_ids = [args.pdb_id]
    else:
        # Find all result files
        result_files = list(test_record_dir.glob('*_final_result.txt'))
        pdb_ids = [f.stem.replace('_final_result', '') for f in result_files]
    
    if not pdb_ids:
        print(f"Error: No result files found in {test_record_dir}")
        sys.exit(1)
    
    print(f"Found {len(pdb_ids)} PDB(s) to visualize")
    print(f"Output directory: {output_dir}")
    
    # Visualize each PDB
    for pdb_id in pdb_ids:
        try:
            visualize_pdb(str(test_record_dir), pdb_id, output_dir)
        except Exception as e:
            print(f"  ✗ Error visualizing {pdb_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Visualization complete! Results in: {output_dir}")


if __name__ == '__main__':
    main()

