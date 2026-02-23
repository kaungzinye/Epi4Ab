#!/usr/bin/env python3
"""
Extract residue number mapping between sequential numbering and PDB residue numbers.

This script reads CIF/PDB structure files and creates a mapping between:
- Sequential position (1, 2, 3, ...) - used in node features
- PDB residue numbers (e.g., 330, 331, 332, ...) - used in raw_ground_truth.csv
"""

import os
import json
import argparse
from Bio.PDB import MMCIFParser, PDBParser
from pathlib import Path


def extract_mapping_from_cif(cif_file, antigen_chain_id=None):
    """
    Extract residue mapping from CIF file.
    
    Args:
        cif_file: Path to CIF file
        antigen_chain_id: Optional chain ID for antigen (if None, uses largest chain)
        
    Returns:
        dict: {
            'sequential_to_pdb': {1: 330, 2: 331, ...},
            'pdb_to_sequential': {330: 1, 331: 2, ...}
        }
    """
    if not os.path.exists(cif_file):
        raise FileNotFoundError(f"CIF file not found: {cif_file}")
    
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('structure', cif_file)
    model = structure[0]
    
    # Find antigen chain
    if antigen_chain_id:
        chain = model[antigen_chain_id]
        if chain is None:
            raise ValueError(f"Chain {antigen_chain_id} not found in structure")
    else:
        # Use largest chain (usually antigen)
        chains = list(model.get_chains())
        if not chains:
            raise ValueError("No chains found in structure")
        
        # Find chain with most residues (likely antigen)
        chain_sizes = [(len(list(chain.get_residues())), chain.id) for chain in chains]
        largest_chain_size, largest_chain_id = max(chain_sizes)
        chain = model[largest_chain_id]
        print(f"  Using chain {largest_chain_id} ({largest_chain_size} residues)")
    
    # Extract residue mapping
    residues = list(chain.get_residues())
    sequential_to_pdb = {}
    pdb_to_sequential = {}
    
    for seq_idx, residue in enumerate(residues, start=1):
        # Residue ID format: (' ', resnum, ' ')
        # resnum is the PDB residue number
        res_id = residue.id
        if isinstance(res_id, tuple) and len(res_id) >= 2:
            pdb_resnum = res_id[1]  # PDB residue number
        else:
            # Try to get from full_id
            full_id = residue.full_id
            if len(full_id) >= 4:
                pdb_resnum = full_id[3][1]  # (' ', resnum, ' ')
            else:
                print(f"  Warning: Could not extract residue number for residue {seq_idx}")
                continue
        
        sequential_to_pdb[seq_idx] = pdb_resnum
        pdb_to_sequential[pdb_resnum] = seq_idx
    
    return {
        'sequential_to_pdb': sequential_to_pdb,
        'pdb_to_sequential': pdb_to_sequential
    }


def extract_mapping_from_pdb(pdb_file, antigen_chain_id=None):
    """
    Extract residue mapping from PDB file.
    
    Args:
        pdb_file: Path to PDB file
        antigen_chain_id: Optional chain ID for antigen (if None, uses largest chain)
        
    Returns:
        dict: {
            'sequential_to_pdb': {1: 330, 2: 331, ...},
            'pdb_to_sequential': {330: 1, 331: 2, ...}
        }
    """
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    model = structure[0]
    
    # Find antigen chain
    if antigen_chain_id:
        chain = model[antigen_chain_id]
        if chain is None:
            raise ValueError(f"Chain {antigen_chain_id} not found in structure")
    else:
        # Use largest chain (usually antigen)
        chains = list(model.get_chains())
        if not chains:
            raise ValueError("No chains found in structure")
        
        chain_sizes = [(len(list(chain.get_residues())), chain.id) for chain in chains]
        largest_chain_size, largest_chain_id = max(chain_sizes)
        chain = model[largest_chain_id]
        print(f"  Using chain {largest_chain_id} ({largest_chain_size} residues)")
    
    # Extract residue mapping
    residues = list(chain.get_residues())
    sequential_to_pdb = {}
    pdb_to_sequential = {}
    
    for seq_idx, residue in enumerate(residues, start=1):
        res_id = residue.id
        if isinstance(res_id, tuple) and len(res_id) >= 2:
            pdb_resnum = res_id[1]
        else:
            full_id = residue.full_id
            if len(full_id) >= 4:
                pdb_resnum = full_id[3][1]
            else:
                print(f"  Warning: Could not extract residue number for residue {seq_idx}")
                continue
        
        sequential_to_pdb[seq_idx] = pdb_resnum
        pdb_to_sequential[pdb_resnum] = seq_idx
    
    return {
        'sequential_to_pdb': sequential_to_pdb,
        'pdb_to_sequential': pdb_to_sequential
    }


def extract_residue_mapping(structure_file, antigen_chain_id=None):
    """
    Extract residue mapping from structure file (CIF or PDB).
    
    Args:
        structure_file: Path to CIF or PDB file
        antigen_chain_id: Optional chain ID for antigen
        
    Returns:
        dict: Mapping dictionary
    """
    file_ext = os.path.splitext(structure_file)[1].lower()
    
    if file_ext == '.cif':
        return extract_mapping_from_cif(structure_file, antigen_chain_id)
    elif file_ext in ['.pdb', '.ent']:
        return extract_mapping_from_pdb(structure_file, antigen_chain_id)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .cif or .pdb")


def save_mapping(mapping, output_file):
    """Save mapping to JSON file."""
    # Convert int keys to strings for JSON
    mapping_json = {
        'sequential_to_pdb': {str(k): v for k, v in mapping['sequential_to_pdb'].items()},
        'pdb_to_sequential': {str(k): v for k, v in mapping['pdb_to_sequential'].items()}
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(mapping_json, f, indent=2)
    
    print(f"  Saved mapping to: {output_file}")


def load_mapping(mapping_file):
    """Load mapping from JSON file."""
    if not os.path.exists(mapping_file):
        return None
    
    with open(mapping_file, 'r') as f:
        mapping_json = json.load(f)
    
    # Convert string keys back to int
    mapping = {
        'sequential_to_pdb': {int(k): v for k, v in mapping_json['sequential_to_pdb'].items()},
        'pdb_to_sequential': {int(k): v for k, v in mapping_json['pdb_to_sequential'].items()}
    }
    
    return mapping


def main():
    parser = argparse.ArgumentParser(description='Extract residue number mapping from structure files')
    parser.add_argument('structure_file', help='Path to CIF or PDB structure file')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file path')
    parser.add_argument('--chain', '-c', help='Antigen chain ID (optional, uses largest chain if not specified)')
    
    args = parser.parse_args()
    
    print(f"Extracting residue mapping from: {args.structure_file}")
    
    try:
        mapping = extract_residue_mapping(args.structure_file, args.chain)
        
        print(f"  Mapping extracted: {len(mapping['sequential_to_pdb'])} residues")
        print(f"  Sequential range: 1 - {max(mapping['sequential_to_pdb'].keys())}")
        print(f"  PDB range: {min(mapping['sequential_to_pdb'].values())} - {max(mapping['sequential_to_pdb'].values())}")
        
        # Show sample mappings
        sample_keys = sorted(list(mapping['sequential_to_pdb'].keys()))[:5]
        print(f"  Sample mappings (sequential -> PDB):")
        for seq in sample_keys:
            print(f"    {seq} -> {mapping['sequential_to_pdb'][seq]}")
        
        save_mapping(mapping, args.output)
        print("  ✓ Mapping extraction complete")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

