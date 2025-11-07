#!/usr/bin/env python3
"""
Diagnostic script to inspect label distributions and understand the CIPS evaluation issue.

This script shows:
1. What labels are in the data (0, 1, 2)
2. What happens during CIPS evaluation conversion
3. Why ROC AUC fails
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def inspect_labels(file_path: str, verbose: bool = False):
    """
    Inspect label file and show distribution.
    
    Expected structure:
    - node_label_pi.parquet contains column 'isInterface' with values:
        - 0: Non-epitope (background residues)
        - 1: CIPS (direct antibody-interacting, 5Å cut-off)
        - 2: Ellipro + BepiPred consensus (potential epitopes)
    """
    label_file = Path(file_path)
    
    if not label_file.exists():
        print(f"ERROR: File not found: {file_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"Inspecting label file: {file_path}")
    print(f"{'='*70}\n")
    
    # Load labels
    df = pd.read_parquet(file_path)
    
    if 'isInterface' not in df.columns:
        print(f"ERROR: Column 'isInterface' not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    labels = df['isInterface'].values
    
    print(f"Total residues: {len(labels)}")
    print(f"\nLabel Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        percentage = (count / len(labels)) * 100
        label_name = {
            0: "Non-epitope (background)",
            1: "CIPS (direct antibody-interacting)",
            2: "Ellipro+BepiPred (potential epitope)"
        }.get(int(label), f"Unknown label {label}")
        print(f"  Label {int(label)} ({label_name}): {count:6d} residues ({percentage:5.2f}%)")
    
    print(f"\n{'='*70}")
    print("CIPS Evaluation Conversion Analysis:")
    print(f"{'='*70}\n")
    
    # Simulate CIPS evaluation conversion
    # CIPS evaluation: true_interface = np.where(true_interface == 1, True, False)
    cips_binary = np.where(labels == 1, True, False)
    
    print(f"After CIPS conversion (label==1 -> True, else -> False):")
    unique_cips, counts_cips = np.unique(cips_binary, return_counts=True)
    
    has_both_classes = len(unique_cips) == 2
    for val, count in zip(unique_cips, counts_cips):
        percentage = (count / len(cips_binary)) * 100
        print(f"  {str(val):5} : {count:6d} residues ({percentage:5.2f}%)")
    
    print(f"\n{'='*70}")
    if has_both_classes:
        print("✓ PASS: Both classes present - ROC AUC can be computed")
    else:
        print("✗ FAIL: Only one class present - ROC AUC will fail!")
        if unique_cips[0] == True:
            print("   All residues are labeled as CIPS (label=1)")
            print("   This means your test data only contains CIPS residues")
        else:
            print("   No residues are labeled as CIPS (label=1)")
            print("   All residues are either non-epitopes (0) or Ellipro (2)")
            print("   CIPS evaluation requires at least SOME residues with label=1")
    print(f"{'='*70}\n")
    
    # Show what regular evaluation would see
    print("Regular Evaluation (non-CIPS) Conversion:")
    print("  Conversion: label != 0 -> True, label == 0 -> False")
    regular_binary = labels != 0
    unique_reg, counts_reg = np.unique(regular_binary, return_counts=True)
    for val, count in zip(unique_reg, counts_reg):
        percentage = (count / len(regular_binary)) * 100
        print(f"  {str(val):5} : {count:6d} residues ({percentage:5.2f}%)")
    
    print(f"\n{'='*70}")
    print("Interpretation:")
    print(f"{'='*70}")
    print("""
CIPS Evaluation Purpose:
  - Evaluates model's ability to detect ONLY direct antibody-interacting residues (label=1)
  - Ignores Ellipro predictions (label=2) and background (label=0)
  - Binary classification: CIPS (1) vs. Everything else (0,2)

Regular Evaluation Purpose:
  - Evaluates model's ability to detect ANY epitope (label=1 or 2)
  - Binary classification: Epitope (1,2) vs. Background (0)

The Problem:
  If your test data has:
    - ALL label=1 (all CIPS): After conversion, all True → ROC AUC fails
    - NO label=1 (all 0 or 2): After conversion, all False → ROC AUC fails
  
  ROC AUC requires BOTH True and False cases to compute a meaningful score.

Why This Happens:
  1. Test data might only contain structures with CIPS residues (unlikely but possible)
  2. Test data might have incorrect labels (missing CIPS annotations)
  3. Label generation pipeline might not have marked any residues as CIPS
  4. Filtering might have removed all non-CIPS or all CIPS residues
    """)
    
    if verbose:
        print(f"\n{'='*70}")
        print("Full Label Array (first 50 residues):")
        print(f"{'='*70}")
        print(labels[:50])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect label distribution in node_label_pi.parquet files")
    parser.add_argument("label_file", help="Path to node_label_pi.parquet file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed label array")
    
    args = parser.parse_args()
    inspect_labels(args.label_file, args.verbose)

