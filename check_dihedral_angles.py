#!/usr/bin/env python3
"""
Script to check if processed PDBs have valid dihedral angles
"""

import pandas as pd
import numpy as np
import os
import sys

def check_dihedral_angles(pdb_id, processed_dir):
    """
    Check if a PDB has valid dihedral angles in its node features
    """
    node_file = f"{processed_dir}/{pdb_id}/node_feature.parquet"
    
    if not os.path.exists(node_file):
        return False, "Node features file not found"
    
    try:
        df = pd.read_parquet(node_file)
        
        # Check if dihedral columns exist
        dihedral_cols = ['phi', 'psi', 'omega', 'chi']
        missing_cols = [col for col in dihedral_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing dihedral columns: {missing_cols}"
        
        # Check for valid dihedral angles (should be in radians, typically -π to π)
        results = {}
        
        for col in dihedral_cols:
            values = df[col].values
            
            # Check for NaN values
            nan_count = np.isnan(values).sum()
            
            # Check for extreme values (outside reasonable range)
            extreme_count = np.sum((values < -10) | (values > 10))
            
            # Check for zero values (might indicate calculation failed)
            zero_count = np.sum(values == 0.0)
            
            # Calculate statistics
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values)
            min_val = np.nanmin(values)
            max_val = np.nanmax(values)
            
            results[col] = {
                'total': len(values),
                'nan_count': nan_count,
                'extreme_count': extreme_count,
                'zero_count': zero_count,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'valid': nan_count == 0 and extreme_count == 0
            }
        
        # Overall assessment
        all_valid = all(results[col]['valid'] for col in dihedral_cols)
        
        return all_valid, results
        
    except Exception as e:
        return False, f"Error reading file: {e}"

def main():
    processed_dir = "/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed"
    
    # Get list of processed PDBs
    pdbs = []
    if os.path.exists(processed_dir):
        pdbs = [d for d in os.listdir(processed_dir) 
                if os.path.isdir(os.path.join(processed_dir, d))]
    
    if not pdbs:
        print("No processed PDBs found!")
        return
    
    print("=== Dihedral Angle Validation Report ===")
    print(f"Checking {len(pdbs)} PDBs...")
    print()
    
    valid_count = 0
    invalid_count = 0
    
    for pdb in sorted(pdbs):
        print(f"=== {pdb} ===")
        
        is_valid, result = check_dihedral_angles(pdb, processed_dir)
        
        if isinstance(result, str):
            # Error message
            print(f"❌ {result}")
            invalid_count += 1
        else:
            # Detailed results
            if is_valid:
                print("✅ All dihedral angles valid")
                valid_count += 1
            else:
                print("❌ Invalid dihedral angles found")
                invalid_count += 1
            
            # Show details for each angle type
            for angle_type, stats in result.items():
                status = "✅" if stats['valid'] else "❌"
                print(f"  {angle_type}: {status}")
                print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                print(f"    Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
                if stats['nan_count'] > 0:
                    print(f"    NaN values: {stats['nan_count']}")
                if stats['extreme_count'] > 0:
                    print(f"    Extreme values: {stats['extreme_count']}")
                if stats['zero_count'] > 0:
                    print(f"    Zero values: {stats['zero_count']}")
        
        print()
    
    print("=== Summary ===")
    print(f"Valid PDBs: {valid_count}")
    print(f"Invalid PDBs: {invalid_count}")
    print(f"Total: {len(pdbs)}")
    
    if invalid_count > 0:
        print("\n⚠️  Some PDBs have invalid dihedral angles!")
        print("This might indicate issues with the dihedral calculation.")
    else:
        print("\n✅ All PDBs have valid dihedral angles!")

if __name__ == "__main__":
    main()
