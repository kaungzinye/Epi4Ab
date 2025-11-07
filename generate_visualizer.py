#!/usr/bin/env python3
"""
Standalone script to generate interactive HTML visualizations from inference results.

Usage:
    python generate_visualizer.py --test_record_dir /path/to/test_record --output_dir /path/to/output
    python generate_visualizer.py --test_record_dir /path/to/test_record --pdb_id 1N8Z
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from interface_prediction.evaluation_and_plot.visualizer import (
    generate_pdb_visualization,
    generate_aggregate_visualization,
    load_prediction_results,
    load_detailed_results
)


def parse_evaluation_file(eval_file: str) -> pd.DataFrame:
    """Parse evaluation_each_pdb.txt file if it exists."""
    if not os.path.exists(eval_file):
        return None
    
    try:
        # Read the file and skip the separator line (line with dashes)
        with open(eval_file, 'r') as f:
            lines = f.readlines()
        
        # Find the header line and data lines (skip separator lines with dashes)
        header_line = None
        data_lines = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            # Skip empty lines and separator lines (lines with mostly dashes)
            if not line_stripped or line_stripped.replace('-', '').replace(' ', '').replace('|', '') == '':
                continue
            
            # First non-empty, non-separator line is likely the header
            if header_line is None and line_stripped:
                header_line = line_stripped
            elif header_line is not None:
                # Check if this is a separator line (mostly dashes)
                if not (len(line_stripped.replace('-', '').replace(' ', '').replace('|', '')) < 5):
                    data_lines.append(line_stripped)
        
        if header_line is None or not data_lines:
            # Fallback to pandas read_csv
            try:
                df = pd.read_csv(eval_file, sep='\s+', skiprows=1)
                return df
            except:
                return None
        
        # Parse header (split by whitespace)
        headers = [h.strip() for h in header_line.split() if h.strip()]
        
        # Parse data lines
        rows = []
        for line in data_lines:
            # Split by whitespace and clean up
            values = [v.strip() for v in line.split() if v.strip()]
            if len(values) >= len(headers):
                rows.append(values[:len(headers)])
        
        if rows:
            df = pd.DataFrame(rows, columns=headers)
            # Convert numeric columns
            numeric_cols = ['Recall', 'Precision', 'f1', 'Accuracy', 'ROC AUC', 'Average Precision']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        else:
            return None
            
    except Exception as e:
        print(f"Warning: Error parsing evaluation file: {e}")
        # Fallback to pandas read_csv
        try:
            df = pd.read_csv(eval_file, sep='\s+', skiprows=1)
            return df
        except:
            return None


def extract_metrics_for_pdb(evaluation_df: pd.DataFrame, pdb_id: str) -> dict:
    """Extract metrics for a specific PDB from evaluation dataframe."""
    if evaluation_df is None:
        return None
    
    # Find rows for this PDB
    pdb_rows = evaluation_df[evaluation_df.iloc[:, 0] == pdb_id]
    if len(pdb_rows) == 0:
        return None
    
    # Try to find the 'all' label row (not 'cips')
    all_row = pdb_rows[pdb_rows.iloc[:, -1] == 'all']
    if len(all_row) == 0:
        all_row = pdb_rows.iloc[0:1]  # Use first row if no 'all' label
    
    row = all_row.iloc[0]
    
    # Map common column names to metrics
    metrics = {}
    col_mapping = {
        'Accuracy': 'accuracy',
        'ROC_AUC': 'roc_auc',
        'F1': 'f1',
        'Precision': 'precision',
        'Recall': 'recall',
        'Average_Precision': 'avg_precision'
    }
    
    for col_name, metric_key in col_mapping.items():
        if col_name in row.index:
            metrics[metric_key] = row[col_name]
        # Try alternative names
        elif col_name.lower() in [c.lower() for c in row.index]:
            matching_col = [c for c in row.index if c.lower() == col_name.lower()][0]
            metrics[metric_key] = row[matching_col]
    
    return metrics if metrics else None


def generate_visualizations(test_record_dir: str, output_dir: str = None, 
                           pdb_id: str = None, evaluation_file: str = None,
                           pdb_files_dir: str = '/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files'):
    """
    Generate visualizations programmatically (without argparse).
    
    Args:
        test_record_dir: Directory containing test_record files
        output_dir: Output directory for HTML files (default: same as test_record_dir)
        pdb_id: Generate visualization for single PDB (default: all PDBs)
        evaluation_file: Path to evaluation_each_pdb.txt file (optional)
        pdb_files_dir: Directory containing PDB files for 3D visualization
    """
    test_record_path = Path(test_record_dir)
    if not test_record_path.exists():
        print(f"Error: Test record directory not found: {test_record_dir}")
        return
    
    output_path = Path(output_dir) if output_dir else test_record_path.parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation metrics if available
    evaluation_df = None
    if evaluation_file:
        evaluation_df = parse_evaluation_file(evaluation_file)
    else:
        # Try to find evaluation file in parent directory
        eval_file = test_record_path.parent / 'evaluation_each_pdb.txt'
        if eval_file.exists():
            evaluation_df = parse_evaluation_file(str(eval_file))
    
    # Find all PDB IDs
    if pdb_id:
        pdb_ids = [pdb_id]
    else:
        result_files = list(test_record_path.glob('*_final_result.txt'))
        pdb_ids = sorted([f.stem.replace('_final_result', '') for f in result_files])
    
    if not pdb_ids:
        print(f"Warning: No result files found in {test_record_dir}")
        return
    
    print(f"Generating visualizations for {len(pdb_ids)} PDB(s)...")
    
    # Generate individual PDB visualizations
    for pdb_id_item in pdb_ids:
        print(f"  Processing {pdb_id_item}...")
        
        # Get metrics for this PDB
        metrics = None
        if evaluation_df is not None:
            metrics = extract_metrics_for_pdb(evaluation_df, pdb_id_item)
        
        # Find PDB file
        pdb_file_path = Path(pdb_files_dir) / f'{pdb_id_item}.pdb'
        if not pdb_file_path.exists():
            pdb_file_path = None
        
        # Generate HTML
        html = generate_pdb_visualization(
            str(test_record_path),
            pdb_id_item,
            evaluation_metrics=metrics,
            pdb_file_path=str(pdb_file_path) if pdb_file_path else None
        )
        
        # Save HTML file
        output_file = output_path / f'{pdb_id_item}_results.html'
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"    Saved: {output_file}")
    
    # Generate aggregate visualization
    print("Generating aggregate visualization...")
    aggregate_html = generate_aggregate_visualization(
        str(test_record_path),
        pdb_ids,
        evaluation_df=evaluation_df
    )
    
    aggregate_file = output_path / 'aggregate_results.html'
    with open(aggregate_file, 'w') as f:
        f.write(aggregate_html)
    
    print(f"  Saved: {aggregate_file}")
    print(f"\nVisualization complete! Open {aggregate_file} in a web browser to view results.")


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML visualizations from inference results'
    )
    parser.add_argument(
        '--test_record_dir',
        type=str,
        required=True,
        help='Directory containing test_record files (*_final_result.txt)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for HTML files (default: same as test_record_dir)'
    )
    parser.add_argument(
        '--pdb_id',
        type=str,
        default=None,
        help='Generate visualization for single PDB (default: all PDBs)'
    )
    parser.add_argument(
        '--evaluation_file',
        type=str,
        default=None,
        help='Path to evaluation_each_pdb.txt file (optional)'
    )
    parser.add_argument(
        '--pdb_files_dir',
        type=str,
        default='/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files',
        help='Directory containing PDB files for 3D visualization'
    )
    
    args = parser.parse_args()
    
    generate_visualizations(
        test_record_dir=args.test_record_dir,
        output_dir=args.output_dir,
        pdb_id=args.pdb_id,
        evaluation_file=args.evaluation_file,
        pdb_files_dir=args.pdb_files_dir
    )


if __name__ == '__main__':
    main()

