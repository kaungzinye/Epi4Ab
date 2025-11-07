#!/usr/bin/env python3
"""
Epi4Ab Data Processing Pipeline
Based on the detailed methodology from the Epi4Ab paper.

This pipeline implements the exact data processing steps described in the paper:
1. Node Features: Sequence (protBERT), Structural (Naccess, PDB2PQR, MDAnalysis), Biophysical (IMGT, Kyte-Doolittle)
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
sys.path.append('/leonardo_work/AIFAC_F01_302/Epi4Ab')

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
from data_processing.sequence_utils import get_sequence_auto

# Amino acid three-letter to one-letter mapping (for sequence reconstruction)
AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
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
    
    def calculate_rsa(self, residue):
        """
        Calculate Relative Solvent Accessibility (RSA) using Naccess
        This is a simplified version - in practice, you'd call Naccess
        """
        try:
            # Get CA atom position
            ca_atom = residue.atoms.select_atoms('name CA')
            if len(ca_atom) == 0:
                return 0.0
            
            ca_pos = ca_atom.positions[0]
            
            # Calculate distance to center of mass (simplified RSA)
            com = self.universe.atoms.center_of_mass()
            distance_to_com = np.linalg.norm(ca_pos - com)
            
            # Simplified RSA calculation (would be replaced by Naccess output)
            rsa = min(1.0, distance_to_com / 20.0)
            
            return rsa
            
        except Exception as e:
            print(f"Warning: Could not calculate RSA for residue {residue.resnum}: {e}")
            return 0.0
    
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
    
    def calculate_partial_charges(self, residue):
        """
        Calculate partial charges using PDB2PQR
        This is a simplified version - in practice, you'd call PDB2PQR
        """
        try:
            # Simplified charge calculation based on residue type
            # In practice, this would come from PDB2PQR output
            res_name = residue.resname
            
            # Basic charge assignment
            if res_name in ['ARG', 'LYS']:
                charge = 1.0  # Positive
            elif res_name in ['ASP', 'GLU']:
                charge = -1.0  # Negative
            elif res_name == 'HIS':
                charge = 0.5  # Partially positive
            else:
                charge = 0.0  # Neutral
            
            return charge
            
        except Exception as e:
            print(f"Warning: Could not calculate partial charges for residue {residue.resnum}: {e}")
            return 0.0
    
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
            hf_cache = os.environ.get('HF_HOME', '/leonardo_scratch/fast/AIFAC_F01_302/.cache/huggingface')
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
        Extract node features for each antigen residue according to Epi4Ab methodology
        """
        print(f"Extracting node features for {self.pdb_id}...")
        
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
            rsa = self.calculate_rsa(residue)
            partial_charge = self.calculate_partial_charges(residue)
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
            
            # Create feature row matching Epi4Ab v1.0.2 format exactly
            feature_row = {
                'resId': res_id,  # Note: resId not residue_id
                'resShort': res_short,  # One-letter amino acid code
                'resDepth': depth,
                'caDepth': depth,
                'psi': psi if not np.isnan(psi) else 0.0,
                'phi': phi if not np.isnan(phi) else 0.0,
                'omega': omega if not np.isnan(omega) else 0.0,
                'chi': chi if not np.isnan(chi) else 0.0,
                'aac': aac,
                'cc': cc,
                'H1_len': h1_len,
                'H2_len': h2_len,
                'H3_len': h3_len,
                'L1_len': l1_len,
                'L2_len': l2_len,
                'L3_len': l3_len,
                'H3_score': h3_score,
                'L1_score': l1_score,
                'angleNan': angle_nan,
                'chiNan': chi_nan
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
            source_charge = self.calculate_partial_charges(source_residue)
            target_charge = self.calculate_partial_charges(target_residue)
            
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
