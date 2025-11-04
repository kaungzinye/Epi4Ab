#!/usr/bin/env python3
"""
Sequence fetching utilities for Epi4Ab pipeline.
Supports multiple sources: FASTA files, RCSB PDB API, or PDB structure extraction.
"""

import requests
from pathlib import Path
from typing import Optional, Union
import MDAnalysis as mda


def fetch_from_rcsb_api(pdb_id: str, chain_id: str = 'A') -> Optional[str]:
    """
    Fetch sequence from RCSB PDB REST API.
    This gets the exact sequence that was crystallized.
    
    Args:
        pdb_id: PDB ID (e.g., '1N8Z')
        chain_id: Chain ID to fetch (default: 'A')
    
    Returns:
        Sequence string or None if fetch fails
    """
    pdb_id = pdb_id.upper()
    
    try:
        # RCSB PDB Data API endpoint for polymer entity sequences
        url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Try to get the sequence from the response
        # The entity_poly field contains the sequence
        if 'entity_poly' in data and 'pdbx_seq_one_letter_code_can' in data['entity_poly']:
            sequence = data['entity_poly']['pdbx_seq_one_letter_code_can']
            # Remove any whitespace or newlines
            sequence = ''.join(sequence.split())
            return sequence
        
        # Alternative: try the sequence from rcsb_polymer_entity
        if 'rcsb_polymer_entity' in data:
            for entity in data.get('rcsb_polymer_entity', []):
                if 'pdbx_seq_one_letter_code' in entity:
                    sequence = entity['pdbx_seq_one_letter_code']
                    sequence = ''.join(sequence.split())
                    return sequence
        
        print(f"Warning: Could not find sequence in RCSB API response for {pdb_id}")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Warning: RCSB API request failed for {pdb_id}: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Warning: Could not parse RCSB API response for {pdb_id}: {e}")
        return None


def load_from_fasta(fasta_path: Path) -> Optional[str]:
    """
    Load sequence from FASTA file.
    
    Args:
        fasta_path: Path to FASTA file
    
    Returns:
        Sequence string or None if load fails
    """
    try:
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines (starting with '>'), concatenate sequence lines
        sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
        
        if sequence:
            return sequence
        else:
            print(f"Warning: Empty sequence in FASTA file {fasta_path}")
            return None
            
    except Exception as e:
        print(f"Warning: Could not load FASTA file {fasta_path}: {e}")
        return None


def extract_from_pdb(universe: mda.Universe, chain_id: str = 'A') -> str:
    """
    Extract sequence from PDB structure using MDAnalysis.
    This is the fallback method - may have gaps or mutations.
    
    Args:
        universe: MDAnalysis Universe object
        chain_id: Chain ID to extract (default: 'A')
    
    Returns:
        Sequence string
    """
    # Three-letter to one-letter amino acid code mapping
    aa_map = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    try:
        # Select CA atoms from specified chain
        selection = f'chainid {chain_id} and name CA'
        ca_atoms = universe.select_atoms(selection)
        
        if len(ca_atoms) == 0:
            # Fallback: try without chain specification
            ca_atoms = universe.select_atoms('name CA')
        
        sequence = []
        for residue in ca_atoms.residues:
            res_name = residue.resname
            aa_code = aa_map.get(res_name, 'X')  # X for unknown
            sequence.append(aa_code)
        
        return ''.join(sequence)
        
    except Exception as e:
        print(f"Warning: Could not extract sequence from PDB: {e}")
        return ""


def get_sequence_auto(
    pdb_id: str,
    universe: mda.Universe,
    fasta_dir: Optional[Path] = None,
    chain_id: str = 'A',
    sequence_source: str = 'auto'
) -> tuple[str, str]:
    """
    Main dispatcher for sequence fetching with intelligent fallback.
    
    Priority order (for 'auto' mode):
    1. Check provided FASTA directory (if specified)
    2. Try RCSB PDB API (gets exact crystallized sequence)
    3. Fall back to PDB extraction (if API fails)
    
    Args:
        pdb_id: PDB ID (e.g., '1N8Z')
        universe: MDAnalysis Universe object for PDB extraction fallback
        fasta_dir: Optional directory containing FASTA files
        chain_id: Chain ID to extract (default: 'A')
        sequence_source: Source mode ('auto', 'pdb', 'fasta', 'rcsb')
    
    Returns:
        Tuple of (sequence string, source used: 'fasta'|'rcsb'|'pdb')
    """
    sequence = None
    source = None
    
    # Mode: FASTA only
    if sequence_source == 'fasta':
        if fasta_dir:
            # Try both .fasta and .fa extensions
            for ext in ['.fasta', '.fa', '.faa']:
                fasta_path = fasta_dir / f"{pdb_id}{ext}"
                if fasta_path.exists():
                    sequence = load_from_fasta(fasta_path)
                    if sequence:
                        source = 'fasta'
                        print(f"  ✓ Loaded sequence from FASTA: {len(sequence)} residues")
                        return sequence, source
        
        print(f"  ✗ FASTA mode specified but no FASTA file found for {pdb_id}")
        raise FileNotFoundError(f"No FASTA file found for {pdb_id}")
    
    # Mode: RCSB only
    elif sequence_source == 'rcsb':
        sequence = fetch_from_rcsb_api(pdb_id, chain_id)
        if sequence:
            source = 'rcsb'
            print(f"  ✓ Fetched sequence from RCSB API: {len(sequence)} residues")
            return sequence, source
        else:
            print(f"  ✗ RCSB mode specified but API fetch failed for {pdb_id}")
            raise RuntimeError(f"RCSB API fetch failed for {pdb_id}")
    
    # Mode: PDB only
    elif sequence_source == 'pdb':
        sequence = extract_from_pdb(universe, chain_id)
        source = 'pdb'
        print(f"  ✓ Extracted sequence from PDB: {len(sequence)} residues")
        return sequence, source
    
    # Mode: AUTO (intelligent fallback)
    else:  # sequence_source == 'auto'
        # 1. Try FASTA first (if directory provided)
        if fasta_dir:
            for ext in ['.fasta', '.fa', '.faa']:
                fasta_path = fasta_dir / f"{pdb_id}{ext}"
                if fasta_path.exists():
                    sequence = load_from_fasta(fasta_path)
                    if sequence:
                        source = 'fasta'
                        print(f"  ✓ Loaded sequence from FASTA: {len(sequence)} residues")
                        return sequence, source
        
        # 2. Try RCSB PDB API
        print(f"  → Attempting to fetch sequence from RCSB API...")
        sequence = fetch_from_rcsb_api(pdb_id, chain_id)
        if sequence:
            source = 'rcsb'
            print(f"  ✓ Fetched sequence from RCSB API: {len(sequence)} residues")
            return sequence, source
        
        # 3. Fall back to PDB extraction
        print(f"  → Falling back to PDB extraction...")
        sequence = extract_from_pdb(universe, chain_id)
        source = 'pdb'
        print(f"  ✓ Extracted sequence from PDB: {len(sequence)} residues")
        return sequence, source

