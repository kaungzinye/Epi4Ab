#!/usr/bin/env python3
"""
Epi4Ab Data Processing Pipeline
Based on the detailed methodology from the Epi4Ab paper.

This pipeline implements the exact data processing steps described in the paper:
1. Node Features: Sequence (protBERT), Structural (FreeSASA RSA, PDB2PQR charges, MDAnalysis geometry), Biophysical (IMGT, Kyte-Doolittle)
2. Epitope Labels: CIPS (5Å cut-off), Ellipro + BepiPred 3.0 consensus
3. Graph Connectivity: Cα-Cα distance cut-off of 10Å
4. Edge Attributes: Bond (1/d), Lennard-Jones, Charge (q1*q2/d) potentials
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
import subprocess
import tempfile
import json
import string
warnings.filterwarnings('ignore')

# Add the Epi4Ab directory to the path
sys.path.append('/leonardo_work/EUHPC_D29_035/Epi4Ab')

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.dihedrals import Dihedral
    from MDAnalysis.analysis.distances import distance_array
    from transformers import AutoTokenizer, EsmModel
    import torch
except ImportError as e:
    print(f"Required packages not available: {e}")
    print("Install with: pip install MDAnalysis transformers torch")
    sys.exit(1)

# Import sequence utilities
from preprocess.scripts.label_generation.sequence_utils import get_sequence_auto

# Amino acid three-letter to one-letter mapping (for sequence reconstruction)
AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# Maximum solvent-accessible surface area per amino acid (Å²)
# Values from Tien et al. (2013) "Maximum allowed solvent accessibilities of residues in proteins"
TIEN_RSA_MAX = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
}

# Map of modified/alternate residue names to canonical residues for RSA normalization
ALT_RESNAME_MAP = {
    'MSE': 'MET', 'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS',
    'CYX': 'CYS', 'CYM': 'CYS', 'SEC': 'CYS',
    'PYL': 'LYS', 'GLX': 'GLU', 'ASX': 'ASP',
    'SEP': 'SER', 'TPO': 'THR', 'PTR': 'TYR',
    'CSO': 'CYS', 'CSD': 'CYS', 'CSX': 'CYS',
    'MEN': 'ASN', 'MHO': 'MET', 'KCX': 'LYS',
    'UNK': 'GLY'
}

class Epi4AbDataProcessor:
    """
    Epi4Ab Data Processing Pipeline
    Implements the exact methodology from the paper
    """
    
    def __init__(self, pdb_file: str, output_dir: str, fasta_dir: Optional[str] = None, 
                 sequence_source: str = 'auto', antigen_chain: str = 'A'):
        self.pdb_file = pdb_file
        self.pdb_id = Path(pdb_file).stem
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fasta_dir = Path(fasta_dir) if fasta_dir else None
        self.sequence_source = sequence_source
        self.antigen_chain = antigen_chain
        
        # Load PDB structure
        self.universe = mda.Universe(pdb_file)

        # Antigen/antibody chain auto-detection
        # Default assumptions: antigen=A, antibody=B/C. If A is empty, pick the largest non-B/C chain.
        def _select_antigen_ca() -> mda.core.groups.AtomGroup:
            try:
                ag = self.universe.select_atoms('chainid A and name CA')
                if len(ag) > 0:
                    return ag
                # Fallback: pick largest chain by CA count excluding B and C
                candidate_ids = []
                # Attempt to gather unique chain IDs from CA atoms
                ca_atoms = self.universe.select_atoms('name CA')
                # MDAnalysis stores chain IDs in .segments or atom.tempfactor? Use residues.chainIDs when available
                # We'll try all uppercase letters heuristically and select the largest
                import string
                best_chain = None
                best_size = -1
                for cid in string.ascii_uppercase:
                    if cid in ['B', 'C']:
                        continue
                    grp = self.universe.select_atoms(f'chainid {cid} and name CA')
                    if len(grp) > best_size:
                        best_size = len(grp)
                        best_chain = grp
                return best_chain if best_chain is not None else ag
            except Exception:
                return self.universe.select_atoms('chainid A and name CA')

        def _select_antibody_ca() -> mda.core.groups.AtomGroup:
            """
            Robust antibody chain detection using multiple strategies:
            1. Try standard chain IDs (B, C, H, L)
            2. Identify by size (heavy ~110-130, light ~100-110 residues)
            3. Identify by sequence patterns (IMGT V-gene characteristics)
            4. Fallback to non-antigen chains
            """
            # Strategy 1: Try standard antibody chain IDs
            standard_ab_chains = ['B', 'C', 'H', 'L', 'D', 'E']  # Common antibody chain labels
            for chain_id in standard_ab_chains:
                try:
                    ab = self.universe.select_atoms(f'chainid {chain_id} and name CA')
                    if len(ab) > 0:
                        print(f"Found antibody chain(s) using standard ID: {chain_id} ({len(ab)} CA atoms)")
                        return ab
                except Exception:
                    continue
            
            # Strategy 2: Identify by chain size and characteristics
            # Antibody chains typically: heavy ~110-130 residues, light ~100-110 residues
            # Antigen chains are usually larger (often >200 residues)
            try:
                all_chains = {}
                for chain_id in string.ascii_uppercase:
                    chain_atoms = self.universe.select_atoms(f'chainid {chain_id} and name CA')
                    if len(chain_atoms) > 0:
                        all_chains[chain_id] = len(chain_atoms)
                
                # Identify antigen chain (usually largest, or chain A)
                antigen_size = len(self.antigen_ca) if self.antigen_ca else 0
                
                # Find chains that are likely antibodies (smaller than antigen, typical antibody sizes)
                ab_chains = []
                for chain_id, size in all_chains.items():
                    # Skip if this is the antigen chain
                    if chain_id == self.antigen_chain:
                        continue
                    # Antibody chains are typically 90-150 residues
                    if 90 <= size <= 150:
                        ab_chains.append((chain_id, size))
                
                if ab_chains:
                    # Combine all potential antibody chains
                    chain_ids = [cid for cid, _ in ab_chains]
                    selection_expr = ' or '.join([f'chainid {cid}' for cid in chain_ids])
                    ab = self.universe.select_atoms(f'({selection_expr}) and name CA')
                    if len(ab) > 0:
                        print(f"Found antibody chain(s) by size analysis: {chain_ids} (sizes: {[s for _, s in ab_chains]})")
                        return ab
            except Exception as e:
                print(f"Warning: Size-based antibody detection failed: {e}")
            
            # Strategy 3: Try non-antigen chains (fallback)
            try:
                if self.antigen_chain:
                    ab = self.universe.select_atoms(f'name CA and (not chainid {self.antigen_chain})')
                    if len(ab) > 0:
                        print(f"Warning: Using fallback antibody detection (all non-{self.antigen_chain} chains, {len(ab)} CA atoms)")
                        return ab
            except Exception:
                pass
            
            # Final fallback: return empty selection
            print(f"WARNING: Could not detect antibody chains for {self.pdb_id}")
            return self.universe.select_atoms('name CA and resnum -1')  # Empty selection

        # Select antigen and antibody chains
        # Note: _select_antigen_ca() may auto-detect the chain if A doesn't exist
        antigen_selection_result = _select_antigen_ca()
        self.antigen_ca = antigen_selection_result
        self.antibody_ca = _select_antibody_ca()
        
        # Validate chain detection and log results
        self._validate_chain_detection()
        
        # Auto-detect the actual antigen chain ID from the selection
        # This ensures sequence extraction matches the selected antigen chain
        # When _select_antigen_ca() falls back to largest chain (chain A doesn't exist),
        # we need to update self.antigen_chain to match so sequence extraction uses the same chain
        if self.antigen_ca is not None and len(self.antigen_ca) > 0:
            # Try to determine which chain was actually selected by testing chain selections
            original_chain = self.antigen_chain
            for chain_id in string.ascii_uppercase:
                test_selection = self.universe.select_atoms(f'chainid {chain_id} and name CA')
                if len(test_selection) == len(self.antigen_ca) and len(test_selection) > 0:
                    # Compare atom indices to confirm it's the same chain
                    matches = sum(1 for i in range(min(10, len(self.antigen_ca))) 
                                if test_selection[i].index == self.antigen_ca[i].index)
                    if matches == min(10, len(self.antigen_ca)):
                        self.antigen_chain = chain_id
                        if chain_id != original_chain:
                            print(f"Auto-detected antigen chain: {chain_id} (was specified as {original_chain})")
                        break
        
        # Initialize ESM2 model lazily (only load when needed)
        # This avoids hanging on compute nodes when model isn't cached
        self.esm2_tokenizer = None
        self.esm2_model = None
        self._esm2_loaded = False
        self._esm2_load_attempted = False
        
        # Define residue properties (Kyte-Doolittle hydrophobicity scale)
        self.hydrophobicity = {
            'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
            'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
            'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
            'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
        }
        
        # Residue weights (molecular weight in Da) - from IMGT
        self.residue_weights = {
            'ALA': 89.09, 'ARG': 174.20, 'ASN': 132.12, 'ASP': 133.10, 'CYS': 121.16,
            'GLN': 146.15, 'GLU': 147.13, 'GLY': 75.07, 'HIS': 155.16, 'ILE': 131.17,
            'LEU': 131.17, 'LYS': 146.19, 'MET': 149.21, 'PHE': 165.19, 'PRO': 115.13,
            'SER': 105.09, 'THR': 119.12, 'TRP': 204.23, 'TYR': 181.19, 'VAL': 117.15
        }
        
        # Residue volumes (in Å³) - from IMGT
        self.residue_volumes = {
            'ALA': 88.6, 'ARG': 173.4, 'ASN': 114.1, 'ASP': 111.1, 'CYS': 108.5,
            'GLN': 143.8, 'GLU': 138.4, 'GLY': 60.1, 'HIS': 153.2, 'ILE': 166.7,
            'LEU': 166.7, 'LYS': 168.6, 'MET': 162.9, 'PHE': 189.9, 'PRO': 112.7,
            'SER': 89.0, 'THR': 116.1, 'TRP': 227.8, 'TYR': 193.6, 'VAL': 140.0
        }
        
        # Isoelectric points
        self.isoelectric_points = {
            'ALA': 6.0, 'ARG': 10.8, 'ASN': 5.4, 'ASP': 2.8, 'CYS': 5.1,
            'GLN': 5.7, 'GLU': 3.2, 'GLY': 6.0, 'HIS': 7.6, 'ILE': 6.0,
            'LEU': 6.0, 'LYS': 9.7, 'MET': 5.7, 'PHE': 5.5, 'PRO': 6.3,
            'SER': 5.7, 'THR': 5.6, 'TRP': 5.9, 'TYR': 5.7, 'VAL': 6.0
        }
        
        # Number of atoms per residue
        self.atom_counts = {
            'ALA': 5, 'ARG': 11, 'ASN': 8, 'ASP': 8, 'CYS': 6,
            'GLN': 9, 'GLU': 9, 'GLY': 4, 'HIS': 10, 'ILE': 8,
            'LEU': 8, 'LYS': 9, 'MET': 8, 'PHE': 11, 'PRO': 7,
            'SER': 6, 'THR': 7, 'TRP': 14, 'TYR': 12, 'VAL': 7
        }
        
        # Deduplicate residues to handle PDB files with duplicate residue entries
        self._deduplicate_residues()
    
    def _validate_chain_detection(self):
        """
        Validate and log chain detection results for debugging.
        """
        antigen_count = len(self.antigen_ca) if self.antigen_ca else 0
        antibody_count = len(self.antibody_ca) if self.antibody_ca else 0
        
        print(f"\n{'='*60}")
        print(f"Chain Detection Results for {self.pdb_id}:")
        print(f"{'='*60}")
        print(f"Antigen chain ({self.antigen_chain}): {antigen_count} CA atoms")
        print(f"Antibody chains: {antibody_count} CA atoms")
        
        if antibody_count == 0:
            print(f"WARNING: No antibody chains detected! Label 1 (CIPS) will be all zeros.")
            print(f"         This may indicate:")
            print(f"         - PDB structure doesn't contain antibody")
            print(f"         - Chain IDs are non-standard")
            print(f"         - Antibody chains are labeled differently")
        elif antibody_count < 100:
            print(f"WARNING: Low antibody atom count ({antibody_count}). Expected ~200-260 for full antibody.")
        else:
            print(f"✓ Antibody detection successful")
        print(f"{'='*60}\n")
        
    def _deduplicate_residues(self):
        """
        Remove duplicate residues from antigen_ca to handle PDB files with duplicate entries
        """
        if self.antigen_ca is None or len(self.antigen_ca) == 0:
            return
            
        # Group by residue number and keep only the first occurrence
        seen_resnums = set()
        unique_atoms = []
        
        for atom in self.antigen_ca:
            resnum = atom.resnum
            if resnum not in seen_resnums:
                seen_resnums.add(resnum)
                unique_atoms.append(atom)
            else:
                print(f"Warning: Duplicate residue {resnum} found in {self.pdb_id}, keeping first occurrence")
        
        # Create new AtomGroup with unique residues
        if len(unique_atoms) < len(self.antigen_ca):
            print(f"Removed {len(self.antigen_ca) - len(unique_atoms)} duplicate residues from {self.pdb_id}")
            self.antigen_ca = mda.core.groups.AtomGroup(unique_atoms)
    
    def _calc_dihedral(self, p1, p2, p3, p4):
        """
        Calculate dihedral angle between 4 points using the standard formula
        """
        import numpy as np
        
        # Convert to numpy arrays
        p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
        
        # Calculate vectors
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        # Calculate normal vectors
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # Calculate dihedral angle
        cos_angle = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
        
        angle = np.arccos(cos_angle)
        
        # Determine sign using the sign of (n1 × n2) · b2
        sign = np.sign(np.dot(np.cross(n1, n2), b2))
        
        return sign * angle
    
    def calculate_dihedral_angles(self, residue):
        """
        Calculate phi, psi, omega, and chi angles using MDAnalysis
        """
        try:
            # Get atoms for dihedral calculations
            atoms = residue.atoms
            
            # Phi angle (C-N-CA-C)
            phi_atoms = []
            for atom in atoms:
                if atom.name == 'C' and atom.resnum == residue.resnum - 1:
                    phi_atoms.append(atom)
                elif atom.name == 'N' and atom.resnum == residue.resnum:
                    phi_atoms.append(atom)
                elif atom.name == 'CA' and atom.resnum == residue.resnum:
                    phi_atoms.append(atom)
                elif atom.name == 'C' and atom.resnum == residue.resnum:
                    phi_atoms.append(atom)
            
            phi = 0.0
            if len(phi_atoms) == 4:
                phi = self._calc_dihedral(
                    phi_atoms[0].position, phi_atoms[1].position,
                    phi_atoms[2].position, phi_atoms[3].position
                )
            
            # Psi angle (N-CA-C-N)
            psi_atoms = []
            for atom in atoms:
                if atom.name == 'N' and atom.resnum == residue.resnum:
                    psi_atoms.append(atom)
                elif atom.name == 'CA' and atom.resnum == residue.resnum:
                    psi_atoms.append(atom)
                elif atom.name == 'C' and atom.resnum == residue.resnum:
                    psi_atoms.append(atom)
                elif atom.name == 'N' and atom.resnum == residue.resnum + 1:
                    psi_atoms.append(atom)
            
            psi = 0.0
            if len(psi_atoms) == 4:
                psi = self._calc_dihedral(
                    psi_atoms[0].position, psi_atoms[1].position,
                    psi_atoms[2].position, psi_atoms[3].position
                )
            
            # Omega angle (CA-C-N-CA)
            omega_atoms = []
            for atom in atoms:
                if atom.name == 'CA' and atom.resnum == residue.resnum:
                    omega_atoms.append(atom)
                elif atom.name == 'C' and atom.resnum == residue.resnum:
                    omega_atoms.append(atom)
                elif atom.name == 'N' and atom.resnum == residue.resnum + 1:
                    omega_atoms.append(atom)
                elif atom.name == 'CA' and atom.resnum == residue.resnum + 1:
                    omega_atoms.append(atom)
            
            omega = 0.0
            if len(omega_atoms) == 4:
                omega = self._calc_dihedral(
                    omega_atoms[0].position, omega_atoms[1].position,
                    omega_atoms[2].position, omega_atoms[3].position
                )
            
            # Chi angle (simplified - would need proper side chain atoms)
            chi = 0.0
            
            return phi, psi, omega, chi
            
        except Exception as e:
            print(f"Warning: Could not calculate dihedral angles for residue {residue.resnum}: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def calculate_rsa(self, residue, cache_dir: Optional[str] = None) -> float:
        """
        Calculate Relative Solvent Accessibility (RSA) using FreeSASA.

        Strategy:
            1. Attempt to load cached FreeSASA results (works on compute nodes without FreeSASA installed)
            2. Otherwise run FreeSASA locally (pure Python dependency)
            3. Cache results to disk for subsequent runs

        Args:
            residue: MDAnalysis residue object
            cache_dir: Optional directory to cache FreeSASA results (persistent across runs)

        Returns:
            float: RSA value (0.0-1.0) for the residue, or 0.0 if calculation fails
        """
        if not hasattr(self, '_rsa_cache'):
            self._rsa_cache = {}
            self._rsa_cache_initialized = False
            self._rsa_cache_hits = 0
            self._rsa_calculations = 0
            self._rsa_cache_source = 'uninitialized'

        if not self._rsa_cache_initialized:
            self._initialize_rsa_cache(cache_dir)

        try:
            # Determine chain ID
            chain_id = 'A'
            if hasattr(residue, 'segid') and residue.segid:
                chain_id = residue.segid
            elif hasattr(residue, 'segment') and hasattr(residue.segment, 'segid') and residue.segment.segid:
                chain_id = residue.segment.segid
            elif hasattr(residue, 'chainID') and residue.chainID:
                chain_id = residue.chainID
            elif len(residue.atoms) > 0:
                atom = residue.atoms[0]
                if hasattr(atom, 'segid') and atom.segid:
                    chain_id = atom.segid
                elif hasattr(atom, 'chainID') and atom.chainID:
                    chain_id = atom.chainID

            # Residue numbers / identifiers
            resnum = residue.resnum if hasattr(residue, 'resnum') else getattr(residue, 'resid', None)
            resnum_str = str(resnum) if resnum is not None else None
            insertion_code = getattr(residue, 'icode', '') or getattr(residue, 'insertion_code', '')
            insertion_code = insertion_code.strip() if isinstance(insertion_code, str) else ''
            if insertion_code:
                resnum_with_icode = f"{resnum_str}{insertion_code}"
            else:
                resnum_with_icode = None

            lookup_keys = []
            if resnum is not None:
                lookup_keys.extend([
                    (chain_id, resnum),
                    (chain_id.upper(), resnum),
                    (chain_id.lower(), resnum),
                    (resnum,),
                ])
            if resnum_str is not None:
                lookup_keys.extend([
                    (chain_id, resnum_str),
                    (chain_id.upper(), resnum_str),
                    (chain_id.lower(), resnum_str),
                    (resnum_str,),
                ])
            if resnum_with_icode:
                lookup_keys.extend([
                    (chain_id, resnum_with_icode),
                    (chain_id.upper(), resnum_with_icode),
                    (chain_id.lower(), resnum_with_icode),
                    (resnum_with_icode,),
                ])

            for key in lookup_keys:
                if key in self._rsa_cache:
                    return float(self._rsa_cache[key])

            return 0.0

        except Exception as e:
            print(f"Warning: Could not retrieve RSA for residue {getattr(residue, 'resnum', 'unknown')}: {e}")
            return 0.0

    def _initialize_rsa_cache(self, cache_dir: Optional[str]) -> None:
        """Load RSA cache from disk or compute with FreeSASA."""
        cache_file = None
        rsa_entries: List[Dict[str, float]] = []

        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_file = cache_path / f"{self.pdb_id}_rsa_freesasa.json"

        if cache_file and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    rsa_entries = json.load(f)
                self._rsa_cache_hits = len(rsa_entries)
                self._rsa_cache_source = 'cache'
                print(f"✓ Using cached FreeSASA RSA results for {self.pdb_id}")
            except Exception as e:
                print(f"Warning: Could not read FreeSASA cache ({cache_file}): {e}")
                rsa_entries = []

        if not rsa_entries:
            rsa_entries = self._compute_freesasa_rsa()
            self._rsa_calculations = len(rsa_entries)
            self._rsa_cache_source = 'computed'
            if cache_file and rsa_entries:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(rsa_entries, f, indent=2)
                    print(f"✓ Cached FreeSASA RSA results for {self.pdb_id} to {cache_file}")
                except Exception as e:
                    print(f"Warning: Could not cache FreeSASA results: {e}")

        self._rsa_cache = self._expand_rsa_entries(rsa_entries)
        self._rsa_cache_initialized = True

    def _compute_freesasa_rsa(self) -> List[Dict[str, float]]:
        """Run FreeSASA and return per-residue RSA entries."""
        try:
            import freesasa
        except ImportError:
            print("Warning: FreeSASA not installed. Install with: pip install freesasa")
            return []

        try:
            structure = freesasa.Structure(self.pdb_file)
        except Exception as e:
            print(f"Warning: Could not load PDB into FreeSASA for {self.pdb_id}: {e}")
            return []

        try:
            try:
                parameters = freesasa.Parameters({'algorithm': 'lee-richards'})
            except Exception:
                parameters = freesasa.Parameters()
            result = freesasa.calc(structure, parameters)
        except Exception as e:
            print(f"Warning: FreeSASA calculation failed for {self.pdb_id}: {e}")
            return []

        residue_areas = result.residueAreas()
        rsa_entries: List[Dict[str, float]] = []

        for chain_id, chain_residues in residue_areas.items():
            try:
                chain = (chain_id or '').strip() or 'A'
            except Exception:
                chain = 'A'

            for _, area in chain_residues.items():
                try:
                    resname_raw = (getattr(area, 'residueType', '') or '').strip().upper()
                    resname_norm = resname_raw
                    if resname_norm not in TIEN_RSA_MAX and resname_norm in ALT_RESNAME_MAP:
                        resname_norm = ALT_RESNAME_MAP[resname_norm]

                    asa_total = float(area.total)
                    asa_max = TIEN_RSA_MAX.get(resname_norm)
                    if asa_max is None or asa_max <= 0:
                        asa_max = float(np.mean(list(TIEN_RSA_MAX.values())))

                    rsa = float(np.clip(asa_total / asa_max, 0.0, 1.0))

                    resnum_label = str(getattr(area, 'residueNumber', '')).strip()
                    insertion_code = ''
                    # residueNumber may already include insertion code, but ensure uppercase
                    if resnum_label:
                        resnum_label = resnum_label.upper()

                    resnum_digits = ''.join(ch for ch in resnum_label if (ch.isdigit() or ch == '-' or ch == '+'))
                    resnum_int = None
                    if resnum_digits and any(ch.isdigit() for ch in resnum_digits):
                        try:
                            resnum_int = int(resnum_digits)
                        except ValueError:
                            resnum_int = None

                    rsa_entries.append({
                        'chain_id': chain,
                        'resname': resname_norm,
                        'resname_original': resname_raw,
                        'resnum': resnum_label,
                        'resnum_int': resnum_int,
                        'rsa': rsa,
                        'sasa': asa_total,
                        'insertion_code': insertion_code
                    })
                except Exception:
                    continue

        if rsa_entries:
            print(f"✓ Computed FreeSASA for {len(rsa_entries)} residues ({self.pdb_id})")

        return rsa_entries

    def _expand_rsa_entries(self, entries: List[Dict[str, float]]) -> Dict:
        """Expand cached FreeSASA entries into lookup dictionary."""
        rsa_dict: Dict = {}
        for entry in entries:
            try:
                chain = entry.get('chain_id', 'A')
                rsa_val = float(entry.get('rsa', 0.0))
                resnum_label = entry.get('resnum')
                resnum_int = entry.get('resnum_int')

                keys = set()
                if resnum_label:
                    keys.add((chain, resnum_label))
                    keys.add((chain.upper(), resnum_label))
                    keys.add((chain.lower(), resnum_label))
                    keys.add((resnum_label,))
                if resnum_int is not None:
                    keys.add((chain, resnum_int))
                    keys.add((chain.upper(), resnum_int))
                    keys.add((chain.lower(), resnum_int))
                    keys.add((resnum_int,))

                for key in keys:
                    rsa_dict[key] = rsa_val
            except Exception:
                continue

        return rsa_dict
    
    def _is_compute_node(self) -> bool:
        """
        Detect if running on compute node (no internet) vs login node (has internet).
        On Leonardo HPC, compute nodes typically have SLURM environment variables.
        """
        import os
        import socket
        
        # Check hostname first (most reliable)
        hostname = os.environ.get('HOSTNAME', '')
        if hostname:
            # Leonardo login nodes are typically named like "loginXX.leonardo.local"
            if 'login' in hostname.lower():
                return False
            # Compute nodes often have different naming patterns
            if any(pattern in hostname.lower() for pattern in ['node', 'compute', 'r', 'gpu']):
                return True
        
        # Check for SLURM environment (indicates compute node typically)
        if os.environ.get('SLURM_JOB_ID'):
            # Additional check: try to detect if we can reach internet quickly
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=2)
                return False  # Can reach internet, likely login node
            except (socket.error, OSError):
                return True  # Cannot reach internet, likely compute node
        
        # Default: assume login node if uncertain (safer for API calls)
        return False
    
    def get_bepipred_predictions(self, sequence: str, cache_dir: Optional[str] = None) -> Optional[Dict[int, float]]:
        """
        Get BepiPred 3.0 epitope predictions for a protein sequence.
        
        HPC-aware implementation:
        1. Try local tool execution first (works on compute nodes, best performance)
        2. Check cache (works on compute nodes, from pre-fetched predictions)
        3. Try web API last (only works on login node, warns if on compute node)
        
        Args:
            sequence: Protein sequence (one-letter code)
            cache_dir: Directory to cache results (avoid repeated API calls)
            
        Returns:
            Dictionary mapping residue index (1-based) to BepiPred score, or None if failed
        """
        # Create cache directory if provided
        cache_file = None
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_file = cache_path / f"{self.pdb_id}_bepipred.json"
        
        # PRIORITY 1: Check cache FIRST (works on compute nodes, avoids unnecessary regeneration)
        # This is Priority 1 because regenerating ESM-2 encodings and BepiPred scores is expensive
        if cache_file and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    # Validate sequence matches (or allow if no sequence stored)
                    if cached_data.get('sequence') == sequence or 'sequence' not in cached_data:
                        print(f"Using cached BepiPred predictions for {self.pdb_id}")
                        return {int(k): float(v) for k, v in cached_data['scores'].items()}
                    else:
                        print(f"Warning: Cached BepiPred sequence mismatch for {self.pdb_id}, regenerating...")
            except Exception as e:
                print(f"Warning: Could not read BepiPred cache: {e}")
        
        # PRIORITY 2: Try Python API (bp3 package) if cache not available (works on compute nodes)
        try:
            from bp3 import bepipred3
            import tempfile
            
            # Create temporary FASTA file and ESM encoding directory
            temp_dir = Path(tempfile.mkdtemp(prefix=f'bp3_{self.pdb_id}_'))
            fasta_file = temp_dir / 'sequence.fasta'
            esm_dir = temp_dir / 'esm_encodings'
            
            # Write sequence to FASTA
            with open(fasta_file, 'w') as f:
                f.write(f">{self.pdb_id}\n{sequence}\n")
            
            # Run bp3 prediction
            antigens = bepipred3.Antigens(fasta_file, esm_dir)
            predictor = bepipred3.BP3EnsemblePredict(antigens)
            predictor.run_bp3_ensemble()
            
            # Extract probabilities (handle tensor/array conversion)
            if hasattr(antigens, 'ensemble_probs') and antigens.ensemble_probs:
                probs = antigens.ensemble_probs[0]
                
                # bp3 returns ensemble of multiple models (typically 5)
                # Average across ensemble models to get single prediction per residue
                import torch
                if len(probs) > 0 and isinstance(probs[0], torch.Tensor):
                    # Stack tensors and average across ensemble dimension
                    stacked = torch.stack(probs)
                    averaged = torch.mean(stacked, dim=0)
                    averaged_list = averaged.tolist()
                else:
                    # Fallback: average manually if not tensors
                    import numpy as np
                    stacked = np.array([item.tolist() if hasattr(item, 'tolist') else item for item in probs])
                    averaged_list = np.mean(stacked, axis=0).tolist()
                
                # Convert to dict: {residue_index: score}
                predictions = {i+1: float(prob) for i, prob in enumerate(averaged_list)}
                
                # Cache results
                if cache_dir and cache_file:
                    cache_data = {
                        'sequence': sequence,
                        'scores': {str(k): float(v) for k, v in predictions.items()}
                    }
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f)
                
                # Clean up temp directory
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                print(f"✓ Generated BepiPred-3.0 predictions using bp3 Python API for {self.pdb_id}")
                return predictions
        except ImportError:
            print("  → bp3 package not available, trying command-line tools...")
        except Exception as e:
            print(f"  ⚠️  bp3 Python API failed: {e}")
        
        # PRIORITY 3: Try command-line tools (works on compute nodes)
        try:
            # Check multiple possible command names
            for cmd_name in ['bepipred', 'bepipred-3.0', 'bepipred3.0']:
                result = subprocess.run(['which', cmd_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    # Run local BepiPred
                    predictions = self._run_local_bepipred(sequence, cmd_name)
                    if predictions:
                        # Cache results for future use
                        if cache_dir and cache_file:
                            cache_data = {
                                'sequence': sequence,
                                'scores': {str(k): float(v) for k, v in predictions.items()}
                            }
                            with open(cache_file, 'w') as f:
                                json.dump(cache_data, f)
                        print(f"✓ Generated BepiPred predictions using local tool for {self.pdb_id}")
                        return predictions
                    break
        except Exception as e:
            print(f"Warning: Local BepiPred execution failed: {e}")
        
        # PRIORITY 4: Try web API (only works on login node, not recommended - use cache instead)
        # Note: This is rarely reached since cache should exist from preprocessing
        is_compute = self._is_compute_node()
        if is_compute:
            print(f"Warning: Running on compute node (no internet). Cannot fetch BepiPred via API.")
            print(f"         Pre-fetch predictions on login node or install local tools.")
        else:
            try:
                import requests
                url = "https://tools.iedb.org/bcell/webservice"
                
                # Prepare request data
                data = {
                    'sequence': sequence,
                    'method': 'bepipred-3.0'
                }
                
                response = requests.post(url, data=data, timeout=30)
                if response.status_code == 200:
                    # Parse response (format depends on API)
                    predictions = self._parse_bepipred_response(response.text)
                    
                    # Cache results for future use (especially for compute nodes)
                    if cache_dir and cache_file and predictions:
                        cache_data = {
                            'sequence': sequence,
                            'scores': {str(k): float(v) for k, v in predictions.items()}
                        }
                        with open(cache_file, 'w') as f:
                            json.dump(cache_data, f)
                        print(f"✓ Fetched and cached BepiPred predictions for {self.pdb_id}")
                    
                    return predictions
            except ImportError:
                print("Warning: requests library not available. Install with: pip install requests")
            except Exception as e:
                print(f"Warning: BepiPred web API failed: {e}")
        
        print(f"Warning: Could not obtain BepiPred predictions for {self.pdb_id}.")
        return None
    
    def _parse_bepipred_response(self, response_text: str) -> Dict[int, float]:
        """
        Parse BepiPred API response.
        Note: Actual format depends on API - this is a placeholder.
        """
        # Placeholder parsing - adjust based on actual API response format
        predictions = {}
        lines = response_text.strip().split('\n')
        for i, line in enumerate(lines, start=1):
            try:
                # Assume format: residue_index score
                parts = line.strip().split()
                if len(parts) >= 2:
                    score = float(parts[1])
                    predictions[i] = score
            except (ValueError, IndexError):
                continue
        return predictions
    
    def _run_local_bepipred(self, sequence: str, cmd_name: str = 'bepipred') -> Optional[Dict[int, float]]:
        """
        Run BepiPred locally if installed.
        
        Args:
            sequence: Protein sequence
            cmd_name: Command name to use (bepipred, bepipred-3.0, etc.)
        """
        try:
            # Write sequence to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
                f.write(f">{self.pdb_id}\n{sequence}\n")
                temp_fasta = f.name
            
            # Run BepiPred (try with different argument formats)
            # BepiPred might use: bepipred input.fasta or bepipred -i input.fasta
            for cmd_format in [
                [cmd_name, temp_fasta],
                [cmd_name, '-i', temp_fasta],
                [cmd_name, '--input', temp_fasta]
            ]:
                try:
                    result = subprocess.run(
                        cmd_format,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        predictions = self._parse_bepipred_response(result.stdout)
                        # Clean up temp file
                        os.unlink(temp_fasta)
                        return predictions
                except Exception:
                    continue
            
            # Clean up temp file
            os.unlink(temp_fasta)
        except Exception as e:
            print(f"Warning: Local BepiPred execution error: {e}")
        
        return None
    
    def get_ellipro_predictions(self, pdb_file: str, cache_dir: Optional[str] = None) -> Optional[Dict[int, float]]:
        """
        Get Ellipro epitope predictions for a PDB structure.
        
        HPC-aware implementation:
        1. Try local tool execution first (works on compute nodes, best performance)
        2. Check cache (works on compute nodes, from pre-fetched predictions)
        3. Try web API last (only works on login node, warns if on compute node)
        
        Args:
            pdb_file: Path to PDB file
            cache_dir: Directory to cache results
            
        Returns:
            Dictionary mapping residue index (1-based) to Ellipro score, or None if failed
        """
        # Create cache directory if provided
        cache_file = None
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_file = cache_path / f"{self.pdb_id}_ellipro.json"
        
        # PRIORITY 1: Check cache first (works on compute nodes, from pre-fetched predictions on login node)
        # This is Priority 1 because Ellipro has no local tool - only web API
        # So cache is the primary way to use Ellipro on compute nodes
        if cache_file and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    print(f"Using cached Ellipro predictions for {self.pdb_id}")
                    return {int(k): float(v) for k, v in cached_data['scores'].items()}
            except Exception as e:
                print(f"Warning: Could not read Ellipro cache: {e}")
        
        # PRIORITY 2: Try local tools (if any exist - unlikely for Ellipro)
        try:
            # Check multiple possible command names
            for cmd_name in ['ellipro', 'ellipro-web', 'ellipro-standalone']:
                result = subprocess.run(['which', cmd_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    predictions = self._run_local_ellipro(pdb_file, cmd_name)
                    if predictions:
                        # Cache results for future use
                        if cache_dir and cache_file:
                            cache_data = {'scores': {str(k): float(v) for k, v in predictions.items()}}
                            with open(cache_file, 'w') as f:
                                json.dump(cache_data, f)
                        print(f"✓ Generated Ellipro predictions using local tool for {self.pdb_id}")
                        return predictions
                    break
        except Exception as e:
            print(f"Warning: Local Ellipro execution failed: {e}")
        
        # PRIORITY 3: Try web API (only works on login node, with timeout to prevent hanging)
        is_compute = self._is_compute_node()
        if is_compute:
            print(f"Warning: Running on compute node (no internet). Cannot fetch Ellipro via API.")
            print(f"         Pre-fetch predictions on login node or install local tools.")
        else:
            # Add explicit timeout and quick check to prevent hanging
            try:
                import requests
                import socket
                
                # Quick connectivity check (don't hang if no internet)
                try:
                    socket.create_connection(("tools.iedb.org", 443), timeout=5)
                except (socket.error, OSError):
                    print(f"Warning: Cannot reach IEDB API (no internet or timeout). Skipping Ellipro web API.")
                    return None
                
                # Ellipro web API endpoint
                # Note: IEDB Ellipro may not have a public REST API
                # Trying common IEDB API patterns
                endpoints_to_try = [
                    "https://services.iedb.org/ellipro/rest/submit",
                    "https://tools.iedb.org/ellipro/rest/submit",
                    "https://tools.iedb.org/bcell/rest/ellipro",
                ]
                
                predictions = None
                for url in endpoints_to_try:
                    try:
                        print(f"  → Attempting Ellipro API: {url}...")
                        with open(pdb_file, 'rb') as f:
                            files = {'pdb_file': f, 'file': f}
                            data = {'method': 'ellipro'}
                            # Try POST with file
                            response = requests.post(url, files=files, data=data, timeout=30)
                            
                            if response.status_code == 200:
                                predictions = self._parse_ellipro_response(response.text)
                                if predictions:
                                    break
                            elif response.status_code != 404:
                                print(f"    Status {response.status_code}, trying next endpoint...")
                    except Exception as e:
                        print(f"    Error with {url}: {e}")
                        continue
                
                if predictions:
                    # Cache results for future use (especially for compute nodes)
                    if cache_dir and cache_file:
                        cache_data = {'scores': {str(k): float(v) for k, v in predictions.items()}}
                        with open(cache_file, 'w') as f:
                            json.dump(cache_data, f)
                        print(f"✓ Fetched and cached Ellipro predictions for {self.pdb_id}")
                    
                    return predictions
                else:
                    # If all endpoints fail, Ellipro likely has no public API
                    print(f"Warning: Ellipro API not available (no public REST endpoint found)")
                    print(f"         Ellipro is primarily a web-based tool at https://tools.iedb.org/ellipro/")
                    print(f"         Manual preprocessing may be required, or use BepiPred 3.0 only.")
                    return None
            except requests.exceptions.Timeout:
                print(f"Warning: Ellipro web API timeout (30s). Pre-fetch on login node or use cache.")
            except requests.exceptions.ConnectionError:
                print(f"Warning: Ellipro web API connection failed (no internet). Skipping.")
            except ImportError:
                print("Warning: requests library not available")
            except Exception as e:
                print(f"Warning: Ellipro web API failed: {e}")
        
        print(f"Warning: Could not obtain Ellipro predictions for {self.pdb_id}.")
        return None
    
    def _parse_ellipro_response(self, response_text: str) -> Dict[int, float]:
        """
        Parse Ellipro API response.
        Note: Actual format depends on API - this is a placeholder.
        """
        predictions = {}
        lines = response_text.strip().split('\n')
        for line in lines:
            try:
                # Placeholder parsing - adjust based on actual API format
                parts = line.strip().split()
                if len(parts) >= 2:
                    res_id = int(parts[0])
                    score = float(parts[1])
                    predictions[res_id] = score
            except (ValueError, IndexError):
                continue
        return predictions
    
    def _run_local_ellipro(self, pdb_file: str, cmd_name: str = 'ellipro') -> Optional[Dict[int, float]]:
        """
        Run Ellipro locally if installed.
        
        Args:
            pdb_file: Path to PDB file
            cmd_name: Command name to use (ellipro, ellipro-web, etc.)
        """
        try:
            # Try different argument formats
            for cmd_format in [
                [cmd_name, pdb_file],
                [cmd_name, '-i', pdb_file],
                [cmd_name, '--input', pdb_file],
                [cmd_name, '--pdb', pdb_file]
            ]:
                try:
                    result = subprocess.run(
                        cmd_format,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        return self._parse_ellipro_response(result.stdout)
                except Exception:
                    continue
        except Exception as e:
            print(f"Warning: Local Ellipro execution error: {e}")
        
        return None
    
    def calculate_partial_charges(self, residue, cache_dir: Optional[str] = None):
        """
        Calculate partial charges using PDB2PQR.
        
        HPC-aware implementation:
        1. Check cache first (works on compute nodes, from pre-computed results)
        2. Try local PDB2PQR tool (works on both compute and login nodes if installed)
        3. Cache results for future use (especially important for compute nodes)
        
        Args:
            residue: MDAnalysis residue object
            cache_dir: Optional directory to cache PDB2PQR results (persistent across runs)
            
        Returns:
            float: Total charge for the residue (sum of atomic charges), or 0.0 if calculation fails
        """
        # Initialize charge cache if not already done
        if not hasattr(self, '_charge_cache'):
            self._charge_cache = {}
            self._charge_cache_initialized = False
        
        # Run PDB2PQR and parse results on first call
        if not self._charge_cache_initialized:
            try:
                # PRIORITY 1: Check cache first (works on compute nodes, avoids re-running PDB2PQR)
                cache_file = None
                if cache_dir:
                    cache_path = Path(cache_dir)
                    cache_path.mkdir(parents=True, exist_ok=True)
                    cache_file = cache_path / f"{self.pdb_id}_pdb2pqr_charges.json"
                
                charge_dict = None
                if cache_file and cache_file.exists():
                    try:
                        with open(cache_file, 'r') as f:
                            cached_data = json.load(f)
                            charge_dict = self._load_pdb2pqr_cache_entries(cached_data)
                            if charge_dict:
                                print(f"✓ Using cached PDB2PQR charge results for {self.pdb_id}")
                                self._pdb2pqr_cache_hits = len(charge_dict)
                    except Exception as e:
                        print(f"Warning: Could not read PDB2PQR cache: {e}")
                
                # PRIORITY 2: Run PDB2PQR if cache not available
                if not charge_dict:
                    charge_dict = self._run_pdb2pqr_and_parse(cache_file)
                    if charge_dict:
                        self._pdb2pqr_calculations = len(charge_dict)
                    else:
                        self._pdb2pqr_calculations = 0
                else:
                    # Cache was used, ensure tracking is set
                    if not hasattr(self, '_pdb2pqr_cache_hits'):
                        self._pdb2pqr_cache_hits = len(charge_dict)
                
                if charge_dict:
                    self._charge_cache = charge_dict
                    self._charge_cache_initialized = True
                else:
                    # If PDB2PQR fails, mark as initialized to avoid repeated attempts
                    self._charge_cache_initialized = True
                    print(f"Warning: PDB2PQR calculation failed for {self.pdb_id}, using default charge=0.0")
            except Exception as e:
                print(f"Warning: Error initializing PDB2PQR charge cache for {self.pdb_id}: {e}")
                self._charge_cache_initialized = True
        
        # Look up charge value for this residue
        try:
            # PDB2PQR uses chain ID and residue number
            # Try multiple ways to get chain ID from MDAnalysis residue
            chain_id = 'A'  # Default
            if hasattr(residue, 'segid') and residue.segid:
                chain_id = residue.segid
            elif hasattr(residue, 'segment') and hasattr(residue.segment, 'segid'):
                chain_id = residue.segment.segid
            elif hasattr(residue, 'chainID'):
                chain_id = residue.chainID
            # Try to get from first atom in residue
            elif len(residue.atoms) > 0:
                atom = residue.atoms[0]
                if hasattr(atom, 'segid') and atom.segid:
                    chain_id = atom.segid
                elif hasattr(atom, 'chainID'):
                    chain_id = atom.chainID
            
            resnum = residue.resnum
            
            # Try multiple lookup keys (PDB2PQR format variations)
            lookup_keys = [
                (chain_id, resnum),
                (chain_id.upper(), resnum),
                (chain_id.lower(), resnum),
                (resnum,),  # Some formats only use resnum
            ]
            
            for key in lookup_keys:
                if key in self._charge_cache:
                    return float(self._charge_cache[key])
            
            # If not found, return 0.0 (neutral charge or parsing issue)
            return 0.0
            
        except Exception as e:
            print(f"Warning: Could not retrieve charge for residue {residue.resnum}: {e}")
            return 0.0
    
    def _run_pdb2pqr_and_parse(self, cache_file: Optional[Path] = None) -> Dict:
        """
        Run PDB2PQR on the PDB file and parse the PQR output to extract charges.
        
        HPC-aware: Works on both compute and login nodes (PDB2PQR is a local tool).
        Results are cached to disk if cache_file is provided.
        
        Args:
            cache_file: Optional path to cache file for saving results
        
        Returns:
            dict: Mapping of (chain_id, resnum) -> total residue charge, or empty dict if failed
        """
        import shutil
        
        # Check if PDB2PQR is available
        pdb2pqr_cmd = shutil.which('pdb2pqr')
        if not pdb2pqr_cmd:
            # Try alternative names and common installation paths
            for cmd in ['pdb2pqr', 'PDB2PQR', 'pdb2pqr30', '/usr/bin/pdb2pqr', '/usr/local/bin/pdb2pqr']:
                if os.path.exists(cmd):
                    pdb2pqr_cmd = cmd
                    break
        
        if not pdb2pqr_cmd:
            is_compute = self._is_compute_node()
            print(f"Warning: PDB2PQR not found in PATH.")
            if is_compute:
                print(f"         Running on compute node - PDB2PQR must be pre-installed.")
                print(f"         Install PDB2PQR on login node or contact system administrator.")
            else:
                print(f"         PDB2PQR can be downloaded from: https://github.com/Electrostatics/pdb2pqr")
                print(f"         Install and add to PATH, or pre-compute charges on login node.")
            return {}
        
        try:
            # Create temporary directory for PDB2PQR output
            temp_dir = Path(tempfile.mkdtemp(prefix=f'pdb2pqr_{self.pdb_id}_'))
            pdb_basename = Path(self.pdb_file).stem
            
            # Copy PDB file to temp directory
            temp_pdb = temp_dir / f"{pdb_basename}.pdb"
            shutil.copy2(self.pdb_file, temp_pdb)
            
            # Output PQR file path
            pqr_file = temp_dir / f"{pdb_basename}.pqr"
            
            # Run PDB2PQR
            # PDB2PQR command: pdb2pqr --ff <forcefield> <input_pdb> <output_pqr>
            # Forcefield options: parse, amber, charmm, etc. Using PARSE as default (common for proteins)
            forcefield = 'PARSE'
            result = subprocess.run(
                [pdb2pqr_cmd, '--ff', forcefield, '--keep-chain', str(temp_pdb), str(pqr_file)],
                cwd=str(temp_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600  # 10 minute timeout (PDB2PQR can be slower than FreeSASA)
            )
            
            if result.returncode != 0:
                print(f"Warning: PDB2PQR failed with return code {result.returncode}")
                print(f"         stderr: {result.stderr[:500]}")
                # Try without --keep-chain flag (older versions may not support it)
                print(f"         Retrying without --keep-chain flag...")
                result = subprocess.run(
                    [pdb2pqr_cmd, '--ff', forcefield, str(temp_pdb), str(pqr_file)],
                    cwd=str(temp_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600
                )
            
            if result.returncode != 0 and forcefield != 'AMBER':
                # Retry once more with AMBER forcefield (commonly available)
                print(f"Warning: PDB2PQR retry also failed with PARSE; attempting with AMBER forcefield...")
                result = subprocess.run(
                    [pdb2pqr_cmd, '--ff', 'AMBER', str(temp_pdb), str(pqr_file)],
                    cwd=str(temp_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600
                )
                if result.returncode != 0:
                    print(f"Warning: PDB2PQR retry also failed")
                    return {}
            
            # Parse PQR file
            if not pqr_file.exists():
                print(f"Warning: PDB2PQR output file not found: {pqr_file}")
                return {}
            
            charge_dict = self._parse_pdb2pqr_pqr(pqr_file)
            
            # Cache results to disk for future use (especially important for compute nodes)
            if charge_dict and cache_file:
                try:
                    cache_entries = self._prepare_pdb2pqr_cache_entries(charge_dict)
                    with open(cache_file, 'w') as f:
                        json.dump(cache_entries, f)
                    print(f"✓ Cached PDB2PQR charge results for {self.pdb_id} to {cache_file}")
                except Exception as e:
                    print(f"Warning: Could not cache PDB2PQR results: {e}")
            
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass  # Ignore cleanup errors
            
            return charge_dict
            
        except subprocess.TimeoutExpired:
            print(f"Warning: PDB2PQR timed out after 10 minutes for {self.pdb_id}")
            return {}
        except Exception as e:
            print(f"Warning: Error running PDB2PQR for {self.pdb_id}: {e}")
            return {}
    
    def _parse_pdb2pqr_pqr(self, pqr_file: Path) -> Dict:
        """
        Parse PDB2PQR PQR output file to extract charges per residue.
        
        PQR format:
        ATOM      1  N   MET A   1      20.154  16.967  15.672  0.0000  1.7500
        Format: ATOM serial name resname chain resnum x y z charge radius
        
        Charges are summed per residue to get total residue charge.
        
        Args:
            pqr_file: Path to PQR file
            
        Returns:
            dict: Mapping of (chain_id, resnum) -> total residue charge
        """
        charge_dict = {}
        
        try:
            with open(pqr_file, 'r') as f:
                lines = f.readlines()
            
            # Parse ATOM records
            for line in lines:
                line = line.strip()
                
                # Skip non-ATOM lines
                if not line.startswith('ATOM') and not line.startswith('HETATM'):
                    continue
                
                # Parse ATOM line
                # PDB/PQR format is fixed-width:
                # ATOM  serial name resname chain resnum    x       y       z    charge  radius
                # 0-6   6-11 12-16 17-20  21    22-26   30-38  38-46  46-54  54-62  62-70
                # Example: ATOM      1  N   MET A   1      20.154  16.967  15.672  0.0000  1.7500
                try:
                    # Try fixed-width parsing first (more reliable for PDB format)
                    if len(line) >= 70:
                        # Fixed-width format
                        chain_id = line[21:22].strip() or 'A'
                        resnum_str = line[22:26].strip()
                        charge_str = line[54:62].strip()
                    else:
                        # Fallback to space-separated parsing
                        parts = line.split()
                        if len(parts) < 11:
                            continue
                        chain_id = parts[4] if len(parts) > 4 else 'A'
                        resnum_str = parts[5] if len(parts) > 5 else '1'
                        charge_str = parts[9] if len(parts) > 9 else '0.0'
                    
                    # Parse residue number (handle insertion codes like "1A")
                    resnum = int(''.join(filter(str.isdigit, resnum_str))) if resnum_str else 1
                    
                    # Parse charge
                    charge = float(charge_str) if charge_str else 0.0
                    
                    # Sum charges per residue
                    key = (chain_id.upper(), resnum)
                    if key not in charge_dict:
                        charge_dict[key] = 0.0
                    charge_dict[key] += charge
                    
                    # Also store with lowercase chain ID and resnum-only for flexible lookup
                    charge_dict[(chain_id.lower(), resnum)] = charge_dict[key]
                    charge_dict[(resnum,)] = charge_dict[key]  # Fallback
                    
                except (ValueError, IndexError) as e:
                    # Skip malformed lines
                    continue
            
            if charge_dict:
                # Count unique residues (chain,resnum pairs)
                unique_residues = set()
                for k in charge_dict.keys():
                    if isinstance(k, tuple) and len(k) == 2:
                        chain = str(k[0]).upper()
                        try:
                            resnum = int(k[1])
                        except (TypeError, ValueError):
                            continue
                        unique_residues.add((chain, resnum))
                print(f"✓ Parsed {len(unique_residues)} residues from PDB2PQR PQR file")
            
            return charge_dict
            
        except Exception as e:
            print(f"Warning: Error parsing PDB2PQR PQR file {pqr_file}: {e}")
            return {}
    
    def _prepare_pdb2pqr_cache_entries(self, charge_dict: Dict) -> List[Dict[str, float]]:
        """Convert charge cache dictionary to JSON-serializable list of entries."""
        entries: List[Dict[str, float]] = []
        seen = set()
        
        for key, value in charge_dict.items():
            if isinstance(key, tuple) and len(key) == 2:
                chain_id, resnum = key
                canonical = (str(chain_id).upper(), int(resnum))
                if canonical in seen:
                    continue
                seen.add(canonical)
                entries.append({
                    'chain_id': canonical[0],
                    'resnum': canonical[1],
                    'charge': float(value)
                })
        
        return entries
    
    def _load_pdb2pqr_cache_entries(self, cached_data) -> Dict:
        """Reconstruct charge cache dictionary from cached JSON data."""
        charge_dict: Dict = {}
        
        if isinstance(cached_data, list):
            for entry in cached_data:
                chain_id = str(entry.get('chain_id', 'A')).strip() or 'A'
                resnum = entry.get('resnum')
                charge = entry.get('charge', 0.0)
                try:
                    resnum_int = int(resnum)
                except Exception:
                    continue
                
                canonical = (chain_id.upper(), resnum_int)
                charge_val = float(charge)
                charge_dict[canonical] = charge_val
                charge_dict[(chain_id.lower(), resnum_int)] = charge_val
                charge_dict[(resnum_int,)] = charge_val
        elif isinstance(cached_data, dict):
            # Backwards compatibility with previous cache format
            for key, value in cached_data.items():
                if isinstance(key, list):
                    tuple_key = tuple(key)
                elif isinstance(key, str) and ',' in key:
                    parts = key.split(',')
                    tuple_key = (parts[0], int(parts[1]))
                else:
                    tuple_key = key
                try:
                    charge_dict[tuple_key] = float(value)
                except Exception:
                    continue
        else:
            return {}
        
        return charge_dict
    
    def _load_esm2_model(self):
        """
        Lazy load ESM2 model (only when needed).
        Uses HuggingFace cache to avoid repeated downloads.
        Handles offline mode gracefully.
        """
        if self._esm2_loaded:
            return True
        
        if self._esm2_load_attempted:
            return False  # Already tried and failed
        
        self._esm2_load_attempted = True
        
        try:
            import os
            # Set HuggingFace cache directory (use scratch for faster I/O)
            hf_cache = os.environ.get('HF_HOME', '/leonardo_scratch/fast/EUHPC_D29_035/.cache/huggingface')
            os.environ['HF_HOME'] = hf_cache
            os.makedirs(hf_cache, exist_ok=True)
            
            # Try to load with offline mode first (if model is cached)
            # This prevents hanging on compute nodes
            from transformers import AutoTokenizer, EsmModel
            
            # Check if we're on compute node (no internet)
            is_compute = self._is_compute_node()
            
            if is_compute:
                # On compute node: only use cached model
                print("Running on compute node - using cached ESM2 model only")
                try:
                    # Try loading with local_files_only=True
                    self.esm2_tokenizer = AutoTokenizer.from_pretrained(
                        'facebook/esm2_t30_150M_UR50D',
                        local_files_only=True,
                        cache_dir=hf_cache
                    )
                    self.esm2_model = EsmModel.from_pretrained(
                        'facebook/esm2_t30_150M_UR50D',
                        local_files_only=True,
                        cache_dir=hf_cache
                    )
                    self.esm2_model.eval()
                    self._esm2_loaded = True
                    print("✓ ESM2 model loaded from cache")
                    return True
                except Exception as e:
                    print(f"Warning: ESM2 model not in cache. Cannot download on compute node: {e}")
                    print("         Pre-download model on login node: python -c \"from transformers import AutoTokenizer, EsmModel; AutoTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D'); EsmModel.from_pretrained('facebook/esm2_t30_150M_UR50D')\"")
                    self.esm2_tokenizer = None
                    self.esm2_model = None
                    return False
            else:
                # On login node: can download if needed
                print("Loading ESM2 model (may download if not cached)...")
                self.esm2_tokenizer = AutoTokenizer.from_pretrained(
                    'facebook/esm2_t30_150M_UR50D',
                    cache_dir=hf_cache
                )
                self.esm2_model = EsmModel.from_pretrained(
                    'facebook/esm2_t30_150M_UR50D',
                    cache_dir=hf_cache
                )
                self.esm2_model.eval()
                self._esm2_loaded = True
                print("✓ ESM2 model loaded successfully")
                return True
                
        except Exception as e:
            print(f"Warning: Could not load ESM2 model: {e}")
            self.esm2_tokenizer = None
            self.esm2_model = None
            return False
    
    def extract_sequence_features(self, sequence):
        """
        Extract sequence features using ESM2 (as mentioned in the paper)
        """
        # Lazy load ESM2 model only when needed
        if not self._load_esm2_model():
            # Return zero features if ESM2 is not available
            return np.zeros(1280)  # ESM2-t30 embedding size
        
        try:
            # Tokenize sequence
            inputs = self.esm2_tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.esm2_model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            return embeddings
            
        except Exception as e:
            print(f"Warning: Could not extract sequence features: {e}")
            return np.zeros(1280)
    
    def extract_node_features(self) -> pd.DataFrame:
        """
        Extract node features for all antigen residues according to Epi4Ab methodology.
        Includes logging for FreeSASA/PDB2PQR feature usage.
        """
        print(f"Extracting node features for {self.pdb_id}...")
        
        # Initialize feature usage tracking (must be done before any feature calculations)
        self._rsa_cache_hits = 0
        self._rsa_calculations = 0
        self._pdb2pqr_cache_hits = 0
        self._pdb2pqr_calculations = 0
        self._rsa_values = []
        self._charge_values = []
        
        # Get antigen residues (auto-detected)
        antigen = self.antigen_ca
        
        # Pre-compute antigen center and max raw depth for normalization (use antigen CA atoms only)
        com_all = antigen.center_of_mass() if hasattr(antigen, 'center_of_mass') else antigen.positions.mean(axis=0)
        raw_depths = []
        for residue in antigen.residues:
            ca_atom_tmp = residue.atoms.select_atoms('name CA')
            if len(ca_atom_tmp) == 0:
                continue
            ca_pos_tmp = ca_atom_tmp.positions[0]
            raw_depths.append(float(np.linalg.norm(ca_pos_tmp - com_all)))
        max_raw_depth = max(raw_depths) if raw_depths else 1.0

        features = []
        for residue in antigen.residues:
            res_name = residue.resname
            res_id = residue.resnum
            
            # Basic residue properties from IMGT
            res_weight = self.residue_weights.get(res_name, 0.0)
            res_volume = self.residue_volumes.get(res_name, 0.0)
            hydrophobicity = self.hydrophobicity.get(res_name, 0.0)
            isoelectric_point = self.isoelectric_points.get(res_name, 6.0)
            atom_count = self.atom_counts.get(res_name, 0)
            
            # Calculate structural features
            rsa = self.calculate_rsa(residue, cache_dir=str(self.output_dir / "rsa_cache"))
            partial_charge = self.calculate_partial_charges(residue, cache_dir=str(self.output_dir / "pdb2pqr_cache"))
            
            # Track feature values for statistics (ensure lists are initialized)
            if not hasattr(self, '_rsa_values'):
                self._rsa_values = []
            if not hasattr(self, '_charge_values'):
                self._charge_values = []
            self._rsa_values.append(rsa)
            self._charge_values.append(partial_charge)
            phi, psi, omega, chi = self.calculate_dihedral_angles(residue)
            
            # Calculate relative depth
            ca_atom = residue.atoms.select_atoms('name CA')
            if len(ca_atom) == 0:
                continue
                
            ca_pos = ca_atom.positions[0]
            # Normalize depth to [0, 30] Å range for model compatibility
            raw_depth = float(np.linalg.norm(ca_pos - com_all))
            depth = 30.0 * (raw_depth / max_raw_depth) if max_raw_depth > 0 else 0.0
            depth = float(np.clip(depth, 0.0, 30.0))
            
            # Amino acid composition features
            aac = 1.0 if res_name in ['ALA', 'VAL', 'LEU', 'ILE'] else 0.0  # Hydrophobic
            cc = 1.0 if res_name in ['ARG', 'LYS', 'HIS'] else 0.0  # Charged
            
            # Convert to one-letter code for resShort
            aa_map = {
                'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
            }
            res_short = aa_map.get(res_name, 'X')  # X for unknown
            
            # Antibody features (placeholder - would need proper CDR detection)
            h1_len, h2_len, h3_len = 0, 0, 0
            l1_len, l2_len, l3_len = 0, 0, 0
            h3_score, l1_score = 0.0, 0.0
            
            # VH/VL family features (placeholder - would need SAbDab data)
            vh_family = [0] * 15  # VH1-VH14 + VH_unk
            vl_family = [0] * 15  # VK1-VK14 + VK_others + VLa1-VLa6 + VLa_others
            
            # One-hot encoding for VH families
            vh_onehot = [0] * 15
            vl_onehot = [0] * 15
            
            # Angle NaN flags
            angle_nan = 1 if np.isnan(phi) or np.isnan(psi) or np.isnan(omega) else 0
            chi_nan = 1 if np.isnan(chi) else 0
            
            # Chemical property flags (one-hot encoding)
            negative = 1 if res_name in ['ASP', 'GLU'] else 0
            positive = 1 if res_name in ['ARG', 'LYS', 'HIS'] else 0
            polar = 1 if res_name in ['ASN', 'GLN', 'SER', 'THR', 'TYR', 'CYS'] else 0
            hydrophobic = 1 if res_name in ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP'] else 0
            neutral = 1 if res_name in ['GLY', 'PRO'] else 0
            hydroxyl = 1 if res_name in ['SER', 'THR', 'TYR'] else 0
            sulfur = 1 if res_name in ['CYS', 'MET'] else 0
            carboxyl = 1 if res_name in ['ASP', 'GLU'] else 0
            amino = 1 if res_name in ['LYS', 'ARG'] else 0
            heterocyclic = 1 if res_name in ['HIS', 'TRP', 'PRO'] else 0
            benzene = 1 if res_name in ['PHE', 'TYR', 'TRP'] else 0
            imino = 1 if res_name == 'PRO' else 0
            
            # Create feature row matching Epi4Ab format with all required attributes
            feature_row = {
                'resId': res_id,  # Note: resId not residue_id
                'resShort': res_short,  # One-letter amino acid code
                'sasa': rsa,  # Relative Solvent Accessibility (FreeSASA normalized by Tien et al. maxima)
                'resDepth': depth,
                'caDepth': depth,
                'charge': partial_charge,  # Charge from PDB2PQR
                'psi': psi if not np.isnan(psi) else 0.0,
                'phi': phi if not np.isnan(phi) else 0.0,
                'omega': omega if not np.isnan(omega) else 0.0,
                'chi': chi if not np.isnan(chi) else 0.0,
                'angleNan': angle_nan,
                'chiNan': chi_nan,
                'weight': res_weight,  # Molecular weight
                'volume': res_volume,  # Residue volume
                'hydrophobicity': hydrophobicity,  # Kyte-Doolittle scale
                'atomNumber': atom_count,  # Number of atoms
                'pI': isoelectric_point,  # Isoelectric point
                'negative': negative,  # Chemical property flags
                'positive': positive,
                'polar': polar,
                'hydrophobic': hydrophobic,
                'neutral': neutral,
                'hydroxyl': hydroxyl,
                'sulfur': sulfur,
                'carboxyl': carboxyl,
                'amino': amino,
                'heterocyclic': heterocyclic,
                'benzene': benzene,
                'imino': imino,
                'aac': aac,
                'cc': cc,
                'H1_len': h1_len,
                'H2_len': h2_len,
                'H3_len': h3_len,
                'L1_len': l1_len,
                'L2_len': l2_len,
                'L3_len': l3_len,
                'H3_score': h3_score,
                'L1_score': l1_score
            }
            
            # Add VH family one-hot encoding
            for i in range(15):
                if i < 14:
                    feature_row[f'VH{i+1}'] = vh_onehot[i]
                else:
                    feature_row['VH_unk'] = vh_onehot[i]
            
            # Add VK (kappa) family one-hot encoding
            for i in range(15):
                if i < 14:
                    feature_row[f'VK{i+1}'] = vl_onehot[i]
                elif i == 14:
                    feature_row['VK_others'] = vl_onehot[i]
            
            # Add VLa (lambda) family one-hot encoding
            # VLa1, VLa2, VLa3, VLa6, VLa_others, VL_unk
            feature_row['VLa1'] = 0
            feature_row['VLa2'] = 0
            feature_row['VLa3'] = 0
            feature_row['VLa6'] = 0
            feature_row['VLa_others'] = 0
            feature_row['VL_unk'] = 0
            
            features.append(feature_row)
        
        # Log feature usage statistics
        if len(self._rsa_values) > 0:
            non_zero_rsa = sum(1 for v in self._rsa_values if v > 0.0)
            print(f"  FreeSASA RSA: {non_zero_rsa}/{len(self._rsa_values)} residues with non-zero RSA")
            print(f"    Cache hits: {getattr(self, '_rsa_cache_hits', 0)}, Calculations: {getattr(self, '_rsa_calculations', 0)}")
            print(f"    RSA stats: min={min(self._rsa_values):.4f}, max={max(self._rsa_values):.4f}, mean={np.mean(self._rsa_values):.4f}")
        
        if len(self._charge_values) > 0:
            non_zero_charge = sum(1 for v in self._charge_values if abs(v) > 1e-6)
            print(f"  PDB2PQR Charge: {non_zero_charge}/{len(self._charge_values)} residues with non-zero charge")
            print(f"    Cache hits: {getattr(self, '_pdb2pqr_cache_hits', 0)}, Calculations: {getattr(self, '_pdb2pqr_calculations', 0)}")
            print(f"    Charge stats: min={min(self._charge_values):.4f}, max={max(self._charge_values):.4f}, mean={np.mean(self._charge_values):.4f}")
        
        return pd.DataFrame(features)
    
    def extract_epitope_labels(self) -> pd.DataFrame:
        """
        Extract epitope labels according to Epi4Ab methodology:
        - Label 1: Direct antibody-interacting residues within 5Å (CIPS)
        - Label 2: Potential epitopes (BepiPred 3.0 OR Ellipro consensus)
        - Label 0: Non-epitopes
        """
        print(f"Extracting epitope labels for {self.pdb_id}...")
        
        # Get antigen residues
        antigen = self.antigen_ca
        
        # Get antibody residues
        antibody = self.antibody_ca
        
        # Try to get BepiPred and Ellipro predictions (cache in output directory)
        cache_dir = str(self.output_dir / 'epitope_predictions')
        bepipred_scores = None
        ellipro_scores = None
        
        # Get antigen sequence for BepiPred
        try:
            antigen_sequence, _ = self.extract_antigen_sequence()  # Get sequence string and source
            if antigen_sequence and len(antigen_sequence) > 0:
                bepipred_scores = self.get_bepipred_predictions(antigen_sequence, cache_dir=cache_dir)
        except Exception as e:
            print(f"Warning: Could not get antigen sequence for BepiPred: {e}")
        
        # Get Ellipro predictions (requires PDB file)
        # Ellipro needs preprocessing on login node (web API only)
        # On compute nodes, use cached Ellipro predictions
        ellipro_scores = None
        is_compute = self._is_compute_node()
        if is_compute:
            # On compute node: only use cached Ellipro (from preprocessing on login node)
            # Skip web API calls to avoid hanging
            print(f"Skipping Ellipro web API (running on compute node - no internet access)")
            print(f"Will use cached Ellipro predictions if available")
        else:
            # On login node: can fetch Ellipro via web API
            try:
                ellipro_scores = self.get_ellipro_predictions(self.pdb_file, cache_dir=cache_dir)
            except Exception as e:
                print(f"Warning: Could not get Ellipro predictions: {e}")
                ellipro_scores = None
        
        # Label 2 uses consensus: BepiPred 3.0 OR Ellipro
        # Either tool predicting epitope = Label 2
        use_real_predictions = (bepipred_scores is not None) or (ellipro_scores is not None)
        if not use_real_predictions:
            print(f"Warning: BepiPred 3.0 and Ellipro predictions not available. Label 2 will be set to 0 (non-epitope).")
        else:
            if bepipred_scores:
                print(f"  ✓ BepiPred 3.0: {len(bepipred_scores)} predictions available")
            if ellipro_scores:
                print(f"  ✓ Ellipro: {len(ellipro_scores)} predictions available")
        
        labels = []
        for residue in antigen.residues:
            res_id = residue.resnum
            
            # Get CA atom
            ca_atom = residue.atoms.select_atoms('name CA')
            if len(ca_atom) == 0:
                continue
                
            ca_pos = ca_atom.positions[0]
            
            # Calculate distances to all antibody residues
            if len(antibody) > 0:
                distances = distance_array(ca_pos.reshape(1, -1), antibody.positions)
                min_distance = np.min(distances)
                
                # Label 1: Direct antibody-interacting residues within 5Å (CIPS)
                if min_distance <= 5.0:
                    label = 1
                else:
                    # Label 2: Potential epitope (use BepiPred 3.0 OR Ellipro consensus)
                    if use_real_predictions:
                        # Get scores from both tools (use 0.0 if not available)
                        bepipred_score = bepipred_scores.get(res_id, 0.0) if bepipred_scores else 0.0
                        ellipro_score = ellipro_scores.get(res_id, 0.0) if ellipro_scores else 0.0
                        
                        # Consensus: residue is predicted epitope if EITHER tool predicts it
                        # BepiPred 3.0 scores typically range 0.0-0.4, max around 0.38
                        # Since BepiPred is supplementary to CIPS (Label 1), use conservative threshold
                        # Threshold 0.3 keeps Label 2 low (~0-20% depending on structure) - fewer false positives
                        bepipred_threshold = 0.3  # BepiPred 3.0 threshold (conservative, low epitope count)
                        ellipro_threshold = 0.5   # Ellipro default threshold
                        
                        if bepipred_score >= bepipred_threshold or ellipro_score >= ellipro_threshold:
                            label = 2  # Predicted epitope (consensus)
                        else:
                            label = 0  # Non-epitope
                    else:
                        # No predictions available: set to non-epitope (Label 0)
                        label = 0  # Non-epitope
            else:
                label = 0  # No antibody found
            
            labels.append({
                'resId': res_id,  # Note: resId not residue_id
                'isInterface': label  # Note: isInterface not label
            })
        
        # Log label distribution
        label_counts = {}
        for label_row in labels:
            label_val = label_row['isInterface']
            label_counts[label_val] = label_counts.get(label_val, 0) + 1
        
        total_labels = len(labels)
        if total_labels > 0:
            print(f"  Label distribution for {self.pdb_id}:")
            print(f"    Label 0 (Non-epitope):     {label_counts.get(0, 0):4d} ({label_counts.get(0, 0)/total_labels*100:5.2f}%)")
            print(f"    Label 1 (CIPS):            {label_counts.get(1, 0):4d} ({label_counts.get(1, 0)/total_labels*100:5.2f}%)")
            print(f"    Label 2 (BepiPred/Ellipro): {label_counts.get(2, 0):4d} ({label_counts.get(2, 0)/total_labels*100:5.2f}%)")
            
            if label_counts.get(1, 0) == 0:
                print(f"    ⚠️  Warning: No CIPS labels found - check antibody chain detection")
        
        return pd.DataFrame(labels)
    
    def extract_graph_connectivity(self) -> pd.DataFrame:
        """
        Extract graph connectivity using Cα-Cα distance cut-off of 10Å
        """
        print(f"Extracting graph connectivity for {self.pdb_id}...")
        
        # Get antigen residues
        antigen = self.antigen_ca
        
        edges = []
        positions = antigen.positions
        
        # Calculate pairwise distances
        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if i != j:
                    distance = np.linalg.norm(pos_i - pos_j)
                    # Edge if distance <= 10Å (Cα-Cα cut-off)
                    if distance <= 10.0:
                        edges.append({
                            'source': i,
                            'target': j,
                            'distance': distance
                        })
        
        return pd.DataFrame(edges)
    
    def extract_antigen_sequence(self) -> tuple[str, str]:
        """
        Extract antigen sequence using modular sequence fetcher.
        Priority: FASTA → RCSB API → PDB extraction
        
        Returns:
            Tuple of (sequence string, source used)
        """
        sequence, source = get_sequence_auto(
            pdb_id=self.pdb_id,
            universe=self.universe,
            fasta_dir=self.fasta_dir,
            chain_id=self.antigen_chain,
            sequence_source=self.sequence_source
        )
        # CRITICAL FIX: If we used PDB extraction, rebuild sequence from deduplicated antigen_ca
        # Problem: PDB file may have duplicate residues (e.g., 6B0N has duplicate residue 321)
        # - _deduplicate_residues() removes duplicates from antigen_ca → node features use 609 residues
        # - But PDB sequence extraction pulls raw sequence → includes duplicates → 610 residues  
        # Solution: Rebuild sequence from deduplicated antigen_ca.residues to match node feature count
        if source == 'pdb' and self.antigen_ca is not None and len(self.antigen_ca) > 0:
            raw_count = len(self.universe.select_atoms(f'chainid {self.antigen_chain} and name CA'))
            residues = self.antigen_ca.residues  # This is already deduplicated
            sequence = ''.join(AA_MAP.get(res.resname, 'X') for res in residues)
            dedup_count = len(sequence)
            if raw_count != dedup_count:
                print(f"  ✓ Rebuilt PDB sequence from deduplicated antigen_ca: {dedup_count} residues (was {raw_count} in raw PDB, removed {raw_count - dedup_count} duplicate(s))")
        return sequence, source
    
    def extract_cdr_sequences(self) -> dict:
        """
        Extract CDR sequences from antibody heavy and light chains.
        Uses Chothia numbering scheme for CDR boundaries.
        
        Returns:
            Dictionary with H1_seq, H2_seq, H3_seq, L1_seq, L2_seq, L3_seq
        """
        cdr_sequences = {
            'H1_seq': '',
            'H2_seq': '',
            'H3_seq': '',
            'L1_seq': '',
            'L2_seq': '',
            'L3_seq': ''
        }
        
        if self.antibody_ca is None or len(self.antibody_ca) == 0:
            print(f"  Warning: No antibody chains found for {self.pdb_id}, using empty CDR sequences")
            return cdr_sequences
        
        try:
            # Get all antibody residues
            ab_residues = self.antibody_ca.residues
            
            # Identify heavy and light chains by finding conserved cysteines
            # Heavy chain typically has Cys around position ~22 and ~104
            # Light chain typically has Cys around position ~23 and ~88
            
            # Group residues by chain
            chain_groups = {}
            for atom in self.antibody_ca:
                try:
                    chain_id = atom.segment.segid if hasattr(atom.segment, 'segid') else 'X'
                    if chain_id not in chain_groups:
                        chain_groups[chain_id] = []
                    # Store residue by index
                    if atom not in [a for r in chain_groups[chain_id] for a in r.atoms]:
                        chain_groups[chain_id].append(atom.residue)
                except:
                    continue
            
            # Try to identify heavy and light chains
            heavy_chain = None
            light_chain = None
            
            for chain_id, residues in chain_groups.items():
                if len(residues) == 0:
                    continue
                # Get sequence
                seq = ''.join(AA_MAP.get(res.resname, 'X') for res in residues)
                # Look for characteristic cysteines
                # Heavy chain: usually has C around position 22-23 (0-indexed ~20-21)
                # Light chain: usually has C around position 23-24 (0-indexed ~22-23)
                if len(seq) > 20:
                    if seq[20:23].count('C') > 0 and len(seq) > 100:
                        # Likely heavy chain (longer, C around pos 22)
                        heavy_chain = (chain_id, residues, seq)
                    elif seq[22:25].count('C') > 0 and len(seq) < 120:
                        # Likely light chain (shorter, C around pos 23)
                        light_chain = (chain_id, residues, seq)
            
            # Fallback: if can't identify by length/sequence, assume first longer chain is heavy
            if heavy_chain is None and chain_groups:
                sorted_chains = sorted(chain_groups.items(), key=lambda x: len(x[1]), reverse=True)
                if len(sorted_chains) >= 1:
                    chain_id, residues = sorted_chains[0]
                    seq = ''.join(AA_MAP.get(res.resname, 'X') for res in residues)
                    if len(seq) > 90:  # Heavy chain is typically longer
                        heavy_chain = (chain_id, residues, seq)
                if len(sorted_chains) >= 2:
                    chain_id, residues = sorted_chains[1]
                    seq = ''.join(AA_MAP.get(res.resname, 'X') for res in residues)
                    light_chain = (chain_id, residues, seq)
            
            # Extract CDR sequences using Chothia numbering
            # H1: residues 26-32 (0-indexed: 25-31), but approximate
            if heavy_chain is not None:
                _, residues, seq = heavy_chain
                # H1: ~26-32 (Chothia)
                if len(seq) > 32:
                    cdr_sequences['H1_seq'] = seq[25:32] if len(seq) > 32 else seq[25:]
                # H2: ~52-56 (Chothia)
                if len(seq) > 56:
                    cdr_sequences['H2_seq'] = seq[51:57] if len(seq) > 57 else seq[51:]
                # H3: variable, typically from ~95 onwards to end of VH (or ~105 in Chothia)
                # For simplicity, take from position 95 to end (or adjust based on length)
                if len(seq) > 95:
                    # H3 usually extends to end or ~110
                    cdr_sequences['H3_seq'] = seq[94:min(110, len(seq))]
            
            # Light chain CDRs
            if light_chain is not None:
                _, residues, seq = light_chain
                # L1: ~24-34 (Chothia)
                if len(seq) > 34:
                    cdr_sequences['L1_seq'] = seq[23:35] if len(seq) > 35 else seq[23:]
                # L2: ~50-56 (Chothia)
                if len(seq) > 56:
                    cdr_sequences['L2_seq'] = seq[49:57] if len(seq) > 57 else seq[49:]
                # L3: ~89-97 (Chothia)
                if len(seq) > 97:
                    cdr_sequences['L3_seq'] = seq[88:98] if len(seq) > 98 else seq[88:]
            
            # Remove any 'X' characters (unknown residues) and validate
            for cdr_name in cdr_sequences:
                seq_clean = cdr_sequences[cdr_name].replace('X', '')
                if len(seq_clean) > 0:
                    cdr_sequences[cdr_name] = seq_clean
                else:
                    cdr_sequences[cdr_name] = ''  # Empty if all X
            
            print(f"  ✓ Extracted CDR sequences: H1={len(cdr_sequences['H1_seq'])}, H2={len(cdr_sequences['H2_seq'])}, H3={len(cdr_sequences['H3_seq'])}, L1={len(cdr_sequences['L1_seq'])}, L2={len(cdr_sequences['L2_seq'])}, L3={len(cdr_sequences['L3_seq'])}")
            
        except Exception as e:
            print(f"  Warning: Could not extract CDR sequences for {self.pdb_id}: {e}")
            # Return empty sequences as fallback
        
        return cdr_sequences
    
    def extract_edge_attributes(self, edges_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract edge attributes using the exact formulas from the paper:
        - Bond potential: p_b = 1/d_{i,j}
        - Lennard-Jones potential: p_{LJ} = V(d_{i,j})
        - Charge potential: p_c = q_i * q_j / d_{i,j}
        """
        print(f"Extracting edge attributes for {self.pdb_id}...")
        
        # Distance attributes
        distance_attrs = []
        charge_attrs = []
        
        # Cache antigen residues for indexing and add bounds checks
        antigen_atoms = self.antigen_ca
        antigen_residues = antigen_atoms.residues if antigen_atoms is not None else []

        for _, edge in edges_df.iterrows():
            source_idx = int(edge['source'])  # Ensure integer indices
            target_idx = int(edge['target'])  # Ensure integer indices
            distance = edge['distance']
            
            # Bond potential: p_b = 1/d_{i,j}
            bond_potential = 1.0 / distance if distance > 0 else 0.0
            
            # Lennard-Jones potential: p_{LJ} = V(d_{i,j})
            # Using the formula from the paper with B = 3.4Å
            B = 3.4  # Parameter from the paper
            sigma = B
            epsilon = 0.1  # Typical LJ parameter
            
            if distance > 0:
                lj_potential = 4 * epsilon * ((sigma/distance)**12 - (sigma/distance)**6)
            else:
                lj_potential = 0.0
            
            # Distance attributes - just the distance (bond/LJ calculations done in CalculateAttribute)
            distance_attrs.append({
                'source': source_idx,
                'target': target_idx,
                'dist': distance
            })
            
            # Charge attributes - just the charge values (charge potential calculated in CalculateAttribute)
            # Get partial charges for source and target residues with bounds checking
            if source_idx < 0 or target_idx < 0:
                continue
            if len(antigen_residues) == 0:
                continue
            if source_idx >= len(antigen_residues) or target_idx >= len(antigen_residues):
                # Skip out-of-range indices (observed in 6B0P)
                continue
            source_residue = antigen_residues[source_idx]
            target_residue = antigen_residues[target_idx]
            source_charge = self.calculate_partial_charges(source_residue, cache_dir=str(self.output_dir / "pdb2pqr_cache"))
            target_charge = self.calculate_partial_charges(target_residue, cache_dir=str(self.output_dir / "pdb2pqr_cache"))
            
            charge_attrs.append({
                'source': source_idx,
                'target': target_idx,
                'charge': int(source_charge * target_charge)  # Note: charge as int
            })
        
        return pd.DataFrame(distance_attrs), pd.DataFrame(charge_attrs)
    
    def process_pdb(self):
        """
        Main processing function that converts PDB to all required parquet files
        """
        print(f"Processing {self.pdb_id}...")
        
        try:
            # Extract all features
            node_features = self.extract_node_features()
            epitope_labels = self.extract_epitope_labels()
            graph_edges = self.extract_graph_connectivity()
            antigen_sequence, seq_source = self.extract_antigen_sequence()
            cdr_sequences = self.extract_cdr_sequences()  # Extract CDR sequences for AntiBERTy
            
            # Check if we have valid data
            if len(node_features) == 0:
                print(f"Warning: No node features extracted for {self.pdb_id} - skipping")
                return False
                
            if len(epitope_labels) == 0:
                print(f"Warning: No epitope labels extracted for {self.pdb_id} - skipping")
                return False
            
            # Handle empty graph edges
            if len(graph_edges) == 0:
                print(f"Warning: No graph edges found for {self.pdb_id} - creating empty edge files")
                # Create empty edge files
                empty_edges = pd.DataFrame(columns=['source', 'target'])
                empty_dist_attrs = pd.DataFrame(columns=['source', 'target', 'dist'])
                empty_charge_attrs = pd.DataFrame(columns=['source', 'target', 'charge'])
                
                # Save to parquet files
                node_features.to_parquet(self.output_dir / 'node_feature.parquet', index=False)
                epitope_labels.to_parquet(self.output_dir / 'node_label_pi.parquet', index=False)
                empty_edges.to_parquet(self.output_dir / 'edge_index.parquet', index=False)
                empty_dist_attrs.to_parquet(self.output_dir / 'edge_attribute_dist.parquet', index=False)
                empty_charge_attrs.to_parquet(self.output_dir / 'edge_attribute_charge.parquet', index=False)
                
                # Save sequence files for ESM2 inference and AntiBERTy
                sequence_dir = self.output_dir / 'sequence'
                sequence_dir.mkdir(exist_ok=True)
                sequence_data = {'pdb_sequence': antigen_sequence}
                with open(sequence_dir / 'antigen_sequence.json', 'w') as f:
                    json.dump(sequence_data, f)
                # Save CDR sequences for AntiBERTy
                with open(sequence_dir / 'cdr_sequence.json', 'w') as f:
                    json.dump(cdr_sequences, f)
                
                print(f"Successfully processed {self.pdb_id} (no edges)")
                print(f"  - Node features: {len(node_features)} residues")
                print(f"  - Epitope labels: {len(epitope_labels)} residues")
                print(f"  - Graph edges: 0 connections")
                print(f"  - Antigen sequence: {len(antigen_sequence)} residues (from {seq_source})")
                
                # Feature usage summary
                if hasattr(self, '_rsa_values') and len(self._rsa_values) > 0:
                    non_zero_rsa = sum(1 for v in self._rsa_values if v > 0.0)
                    print(f"  - FreeSASA RSA: {non_zero_rsa}/{len(self._rsa_values)} non-zero values")
                if hasattr(self, '_charge_values') and len(self._charge_values) > 0:
                    non_zero_charge = sum(1 for v in self._charge_values if abs(v) > 1e-6)
                    print(f"  - PDB2PQR Charge: {non_zero_charge}/{len(self._charge_values)} non-zero values")
                
                return True
            else:
                # Normal processing with edges
                distance_attrs, charge_attrs = self.extract_edge_attributes(graph_edges)
                
                # Save to parquet files
                node_features.to_parquet(self.output_dir / 'node_feature.parquet', index=False)
                epitope_labels.to_parquet(self.output_dir / 'node_label_pi.parquet', index=False)
                graph_edges[['source', 'target']].to_parquet(self.output_dir / 'edge_index.parquet', index=False)
                distance_attrs.to_parquet(self.output_dir / 'edge_attribute_dist.parquet', index=False)
                charge_attrs.to_parquet(self.output_dir / 'edge_attribute_charge.parquet', index=False)
                
                # Save sequence files for ESM2 inference and AntiBERTy
                sequence_dir = self.output_dir / 'sequence'
                sequence_dir.mkdir(exist_ok=True)
                sequence_data = {'pdb_sequence': antigen_sequence}
                with open(sequence_dir / 'antigen_sequence.json', 'w') as f:
                    json.dump(sequence_data, f)
                # Save CDR sequences for AntiBERTy
                with open(sequence_dir / 'cdr_sequence.json', 'w') as f:
                    json.dump(cdr_sequences, f)
                
                print(f"Successfully processed {self.pdb_id}")
                print(f"  - Node features: {len(node_features)} residues")
                print(f"  - Epitope labels: {len(epitope_labels)} residues")
                print(f"  - Graph edges: {len(graph_edges)} connections")
                print(f"  - Antigen sequence: {len(antigen_sequence)} residues (from {seq_source})")
                
                # Feature usage summary
                if hasattr(self, '_rsa_values') and len(self._rsa_values) > 0:
                    non_zero_rsa = sum(1 for v in self._rsa_values if v > 0.0)
                    print(f"  - FreeSASA RSA: {non_zero_rsa}/{len(self._rsa_values)} non-zero values")
                if hasattr(self, '_charge_values') and len(self._charge_values) > 0:
                    non_zero_charge = sum(1 for v in self._charge_values if abs(v) > 1e-6)
                    print(f"  - PDB2PQR Charge: {non_zero_charge}/{len(self._charge_values)} non-zero values")
                
                return True
            
        except Exception as e:
            print(f"Error processing {self.pdb_id}: {str(e)}")
            print(f"Skipping {self.pdb_id} and continuing with next PDB...")
            return False

def main():
    parser = argparse.ArgumentParser(description='Convert PDB files to Epi4Ab parquet format')
    parser.add_argument('--pdb_dir', help='Directory containing PDB files')
    parser.add_argument('--output_dir', help='Output directory for parquet files')
    parser.add_argument('--pdb_file', help='Single PDB file (alternative to pdb_dir)')
    parser.add_argument('--fasta_dir', help='Directory containing FASTA files (optional, named {pdb_id}.fasta)')
    parser.add_argument('--sequence_source', 
                       choices=['auto', 'pdb', 'fasta', 'rcsb'],
                       default='auto',
                       help='Sequence source: auto (FASTA→RCSB→PDB), pdb (extract from PDB), fasta (use FASTA only), rcsb (fetch from RCSB only)')
    parser.add_argument('--antigen_chain', default='A', help='Antigen chain ID (default: A)')
    
    args = parser.parse_args()
    
    if args.pdb_dir:
        # Process multiple PDB files
        pdb_dir = Path(args.pdb_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all PDB files
        pdb_files = list(pdb_dir.glob('*.pdb'))
        if not pdb_files:
            print(f"No PDB files found in {pdb_dir}")
            return
        
        print(f"Found {len(pdb_files)} PDB files to process")
        
        successful = 0
        failed = 0
        
        for pdb_file in pdb_files:
            try:
                # Create output subdirectory for this PDB
                pdb_output_dir = output_dir / pdb_file.stem
                pdb_output_dir.mkdir(exist_ok=True)
                
                # Process the PDB
                processor = Epi4AbDataProcessor(
                    str(pdb_file), 
                    str(pdb_output_dir),
                    fasta_dir=args.fasta_dir,
                    sequence_source=args.sequence_source,
                    antigen_chain=args.antigen_chain
                )
                if processor.process_pdb():
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"Failed to process {pdb_file}: {e}")
                failed += 1
                continue
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful} PDB files")
        print(f"Failed: {failed} PDB files")
        
    elif args.pdb_file:
        # Process single PDB file
        processor = Epi4AbDataProcessor(
            args.pdb_file, 
            args.output_dir,
            fasta_dir=args.fasta_dir,
            sequence_source=args.sequence_source,
            antigen_chain=args.antigen_chain
        )
        processor.process_pdb()
    else:
        print("Error: Please provide either --pdb_dir or --pdb_file")
        return

if __name__ == '__main__':
    main()
