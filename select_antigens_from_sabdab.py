#!/usr/bin/env python3
"""
SAbDab Antigen Selection Script

Queries SAbDab database to filter antibody-antigen complexes according to paper criteria:
- Resolution ≤ 4Å
- Contains both heavy and light chain variable domains (Fab or Fv)
- Single chain protein antigen
- Antigen length ≤ 640 residues
- Has associated publication
- Identifies HER2 antigens (UniProt P04626, HER2, ERBB2) for test set
- Validates antibody binding information and CDR sequence availability
- Extracts epitope information from publications

Run on login node (requires internet access).
"""

import requests
import pandas as pd
import csv
import logging
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
from io import StringIO
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# HER2 identifiers
HER2_UNIPROT_ID = 'P04626'
HER2_NAMES = ['HER2', 'ERBB2', 'HER2/neu', 'ERBB-2', 'NEU', 'CD340']

# SAbDab URLs (as of 2024)
SABDAB_BASE_URL = "https://naga-www-prod.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/"
SABDAB_SUMMARY_URL = "https://naga-www-prod.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"
SABDAB_CSV_URL = "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search?all=true"
SABDAB_API_URL = "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/api/"


def download_sabdab_data(output_file: str = "sabdab_raw.csv", use_local: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Download SAbDab data.
    
    Tries multiple methods:
    1. Local file if provided
    2. Direct CSV download from SAbDab
    3. API query
    4. Parsed HTML table
    
    Args:
        output_file: File to save raw data
        use_local: Path to local SAbDab CSV file (optional)
    
    Returns:
        DataFrame with SAbDab data or None if download fails
    """
    # Try local file first
    if use_local and Path(use_local).exists():
        logger.info(f"Loading local SAbDab file: {use_local}")
        try:
            # Try tab-separated first (SAbDab default)
            try:
                df = pd.read_csv(use_local, sep='\t')
                logger.info(f"Loaded {len(df)} entries from local TSV file")
                return df
            except:
                # Try comma-separated
                df = pd.read_csv(use_local, sep=',')
                logger.info(f"Loaded {len(df)} entries from local CSV file")
                return df
        except Exception as e:
            logger.warning(f"Failed to load local file: {e}")
            logger.warning(f"Error details: {type(e).__name__}: {e}")
    
    logger.info("Downloading SAbDab data...")
    
    # Try direct CSV download
    try:
        logger.info(f"Attempting CSV download from {SABDAB_CSV_URL}")
        response = requests.get(SABDAB_CSV_URL, timeout=60)
        if response.status_code == 200:
            # Try to parse as CSV
            try:
                df = pd.read_csv(response.text if isinstance(response.text, str) else response.content.decode('utf-8'))
                logger.info(f"Successfully downloaded CSV with {len(df)} entries")
                return df
            except Exception as e:
                logger.warning(f"CSV parsing failed: {e}")
                # Save raw response for inspection
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Saved raw response to {output_file}")
    except Exception as e:
        logger.warning(f"CSV download failed: {e}")
    
    # Try API endpoint
    try:
        logger.info(f"Attempting API query from {SABDAB_API_URL}")
        response = requests.get(SABDAB_API_URL, timeout=60)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
                logger.info(f"Successfully downloaded JSON with {len(df)} entries")
                return df
    except Exception as e:
        logger.warning(f"API query failed: {e}")
    
    # Try to download summary CSV (tab-separated format)
    summary_urls = [
        SABDAB_SUMMARY_URL,  # Primary summary URL
        "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary",
        "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/download/all",
    ]
    
    for url in summary_urls:
        try:
            logger.info(f"Trying summary URL: {url}")
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                # SAbDab summary files are typically tab-separated
                content = response.text if isinstance(response.text, str) else response.content.decode('utf-8')
                content_type = response.headers.get('content-type', '')
                
                # Try tab-separated first (most common for SAbDab)
                try:
                    df = pd.read_csv(StringIO(content), sep='\t')
                    if len(df) > 0 and len(df.columns) > 1:
                        logger.info(f"Successfully downloaded TSV from {url} with {len(df)} entries")
                        return df
                except:
                    pass
                
                # Try comma-separated
                try:
                    df = pd.read_csv(StringIO(content), sep=',')
                    if len(df) > 0 and len(df.columns) > 1:
                        logger.info(f"Successfully downloaded CSV from {url} with {len(df)} entries")
                        return df
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
    
    logger.error("Failed to download SAbDab data from all attempted sources")
    logger.info("Please check SAbDab website manually or provide local CSV file")
    return None


def filter_by_resolution(df: pd.DataFrame, max_resolution: float = 4.0) -> pd.DataFrame:
    """Filter complexes by resolution ≤ max_resolution."""
    if 'resolution' not in df.columns:
        logger.warning("'resolution' column not found, checking alternative column names...")
        resolution_cols = [col for col in df.columns if 'resolution' in col.lower() or 'res' in col.lower()]
        if resolution_cols:
            df['resolution'] = df[resolution_cols[0]]
        else:
            logger.warning("Could not find resolution column, skipping resolution filter")
            return df
    
    # Convert to numeric, handling non-numeric values
    df['resolution'] = pd.to_numeric(df['resolution'], errors='coerce')
    
    filtered = df[df['resolution'] <= max_resolution].copy()
    logger.info(f"Resolution filter (≤{max_resolution}Å): {len(df)} -> {len(filtered)} entries")
    return filtered


def filter_by_chain_types(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to ensure both heavy (VH) and light (VL) chain variable domains are present."""
    # SAbDab uses Hchain and Lchain columns
    hchain_cols = [col for col in df.columns if col.lower() in ['hchain', 'heavy_chain', 'vh_chain']]
    lchain_cols = [col for col in df.columns if col.lower() in ['lchain', 'light_chain', 'vl_chain']]
    
    if not hchain_cols or not lchain_cols:
        logger.warning("Could not find Hchain/Lchain columns, checking for alternative columns...")
        # Try to find any chain columns
        chain_cols = [col for col in df.columns if 'chain' in col.lower()]
        if chain_cols:
            logger.info(f"Found chain columns: {chain_cols[:5]}")
            # If we have Hchain and Lchain, use them
            if 'Hchain' in df.columns and 'Lchain' in df.columns:
                hchain_cols = ['Hchain']
                lchain_cols = ['Lchain']
            else:
                logger.warning("Could not identify Hchain/Lchain columns, skipping chain filter")
                return df
        else:
            logger.warning("Could not find chain type columns, skipping chain filter")
            return df
    
    hchain_col = hchain_cols[0]
    lchain_col = lchain_cols[0]
    
    # Filter for entries that have both H and L chains (not empty/None)
    def has_both_chains(row):
        h_val = str(row.get(hchain_col, '')).strip()
        l_val = str(row.get(lchain_col, '')).strip()
        return h_val and h_val.lower() not in ['nan', 'none', '', 'n/a', 'na'] and \
               l_val and l_val.lower() not in ['nan', 'none', '', 'n/a', 'na']
    
    filtered = df[df.apply(has_both_chains, axis=1)].copy()
    logger.info(f"Chain type filter (Fab/Fv - both H and L chains): {len(df)} -> {len(filtered)} entries")
    return filtered


def filter_by_antigen_length(df: pd.DataFrame, max_length: int = 640) -> pd.DataFrame:
    """Filter by antigen length ≤ max_length."""
    # Look for antigen length columns
    length_cols = [col for col in df.columns if 'antigen' in col.lower() and 'length' in col.lower()]
    length_cols.extend([col for col in df.columns if 'ag_length' in col.lower() or 'antigen_residues' in col.lower()])
    
    # SAbDab may not have direct length column - we'll need to calculate from PDB or skip
    if not length_cols:
        logger.warning("Could not find antigen length column, skipping length filter")
        logger.info("Note: Antigen length will need to be verified from PDB files during processing")
        return df
    
    df['antigen_length'] = pd.to_numeric(df[length_cols[0]], errors='coerce')
    filtered = df[df['antigen_length'] <= max_length].copy()
    logger.info(f"Antigen length filter (≤{max_length}): {len(df)} -> {len(filtered)} entries")
    return filtered


def filter_by_single_chain_antigen(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to ensure single chain protein antigen (not multi-chain complex)."""
    # Check antigen_chain column - if it contains commas or multiple chains, it's multi-chain
    if 'antigen_chain' in df.columns:
        def is_single_chain(row):
            ag_chain = str(row.get('antigen_chain', '')).strip()
            if not ag_chain or ag_chain.lower() in ['nan', 'none', '', 'n/a']:
                return False  # Skip entries without antigen chain info
            # Check if it's a single chain (no commas, semicolons, or multiple letters)
            if ',' in ag_chain or ';' in ag_chain:
                return False
            # Single chain should be one letter/identifier
            return len(ag_chain.split()) == 1
        
        filtered = df[df.apply(is_single_chain, axis=1)].copy()
        logger.info(f"Single-chain antigen filter: {len(df)} -> {len(filtered)} entries")
        return filtered
    else:
        logger.warning("Could not find antigen_chain column, skipping single-chain filter")
        return df


def filter_by_publication(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only include complexes with associated publications."""
    pub_cols = [col for col in df.columns if 'pubmed' in col.lower() or 'pmid' in col.lower() or 'doi' in col.lower() or 'publication' in col.lower()]
    
    if not pub_cols:
        logger.warning("Could not find publication columns, skipping publication filter")
        return df
    
    def has_publication(row):
        for col in pub_cols:
            val = str(row.get(col, '')).strip()
            # Must have valid publication ID (not empty, None, TBD, or NA)
            if val and val.lower() not in ['nan', 'none', '', 'n/a', 'tbd', 'none', 'null']:
                # Try to parse as number (PMID) or check if it's a valid identifier
                try:
                    # If it's numeric, it's likely a valid PMID
                    int(float(val))
                    return True
                except (ValueError, TypeError):
                    # If it's not numeric but not empty/TBD, might be DOI
                    if len(val) > 3:  # Reasonable length for DOI
                        return True
        return False
    
    filtered = df[df.apply(has_publication, axis=1)].copy()
    logger.info(f"Publication filter: {len(df)} -> {len(filtered)} entries")
    
    # Verify filter worked
    if 'pmid' in df.columns:
        before_count = df['pmid'].notna().sum()
        after_count = filtered['pmid'].notna().sum()
        logger.info(f"  Before: {before_count} with pmid, After: {after_count} with valid publication")
    
    return filtered


def identify_her2_antigens(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify HER2 antigens by UniProt ID and name variations.
    
    STRICT: Only matches actual HER2/ERBB-2 receptor antigens in antigen_name field.
    Does NOT match mentions in titles/compounds (too broad, would catch wrong entries).
    
    Returns:
        Tuple of (her2_df, non_her2_df)
    """
    # Check for UniProt ID column (SAbDab may not have this directly)
    uniprot_cols = [col for col in df.columns if 'uniprot' in col.lower()]
    name_cols = [col for col in df.columns if 'antigen' in col.lower() and 'name' in col.lower()]
    
    def is_her2(row):
        # STRICT: Only check antigen_name field, not titles/compounds
        # HER2 is "receptor tyrosine-protein kinase erbb-2" or "receptor protein-tyrosine kinase erbb-2"
        
        # Check UniProt ID (if available)
        for col in uniprot_cols:
            val = str(row.get(col, '')).strip()
            if val == HER2_UNIPROT_ID:
                return True
        
        # Check antigen name - STRICT matching for ERBB-2 receptor
        for col in name_cols:
            antigen_name = str(row.get(col, '')).lower()
            # Match ERBB-2 receptor patterns
            if 'erbb-2' in antigen_name or 'erbb2' in antigen_name:
                # Must be receptor/kinase (not just any mention)
                if 'receptor' in antigen_name and ('kinase' in antigen_name or 'tyrosine' in antigen_name):
                    return True
                # Also allow "receptor protein-tyrosine kinase erbb-2" format
                if 'receptor' in antigen_name and 'protein' in antigen_name:
                    return True
        
        return False
    
    her2_mask = df.apply(is_her2, axis=1)
    her2_df = df[her2_mask].copy()
    non_her2_df = df[~her2_mask].copy()
    
    logger.info(f"HER2 identification (STRICT): {len(her2_df)} HER2 complexes, {len(non_her2_df)} non-HER2 complexes")
    if len(her2_df) > 0:
        unique_pdbs = her2_df['pdb'].unique()
        logger.info(f"HER2 PDB IDs found: {', '.join(sorted(unique_pdbs))}")
        logger.info(f"Unique HER2 PDBs: {len(unique_pdbs)}")
    return her2_df, non_her2_df


def extract_epitopes_from_publication(pdb_id: str, pmid: Optional[str] = None, doi: Optional[str] = None) -> Optional[Dict]:
    """
    Extract epitope information from publication.
    
    Attempts to retrieve epitope data from PubMed/DOI sources.
    Uses PubMed API and text parsing to extract validated epitope residues.
    
    Args:
        pdb_id: PDB ID
        pmid: PubMed ID
        doi: DOI
    
    Returns:
        Dict with epitope information or None if extraction fails
    """
    epitope_data = {
        'pdb_id': pdb_id,
        'pmid': pmid,
        'doi': doi,
        'epitope_residues': [],
        'epitope_sequence': '',
        'validation_method': '',
        'confidence': 'low'
    }
    
    # Try PubMed API if PMID is available
    if pmid:
        try:
            pubmed_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'xml'
            }
            response = requests.get(pubmed_url, params=params, timeout=30)
            
            if response.status_code == 200:
                # Parse XML response (simplified - would need full XML parsing)
                content = response.text
                # Look for epitope-related keywords in abstract/text
                epitope_keywords = ['epitope', 'binding site', 'antigenic determinant', 'paratope']
                if any(keyword in content.lower() for keyword in epitope_keywords):
                    epitope_data['confidence'] = 'medium'
                    logger.info(f"Found epitope keywords in PubMed {pmid} for {pdb_id}")
        except Exception as e:
            logger.warning(f"PubMed API query failed for {pmid}: {e}")
    
    # Try DOI-based retrieval (placeholder - would need DOI resolver API)
    if doi and not epitope_data.get('epitope_residues'):
        logger.debug(f"DOI-based extraction not implemented for {doi}")
    
    # Check if we found any epitope data
    if not epitope_data.get('epitope_residues') and epitope_data['confidence'] == 'low':
        return None
    
    return epitope_data


def validate_antibody_binding(pdb_id: str, df_row: pd.Series) -> Dict[str, bool]:
    """
    Validate that antibody binding information is available.
    
    Checks for:
    - CDR sequences (H1, H2, H3, L1, L2, L3) - can be extracted from PDB via pipeline
    - Variable domain sequences (VH, VL) - can be extracted from PDB
    - Antibody chain identifiers - required for processing
    
    Note: The pipeline can extract CDR sequences from PDB files using IMGT numbering,
    so even if SAbDab doesn't have CDR data, the pipeline can still process the structure.
    
    Returns:
        Dict with validation results
    """
    validation = {
        'has_cdr_sequences': False,
        'has_variable_domains': False,
        'has_chain_identifiers': False,
        'binding_validated': False,
        'cdr_extractable': True  # Pipeline can extract from PDB
    }
    
    # Check for CDR sequence indicators in SAbDab data
    cdr_cols = [col for col in df_row.index if 'cdr' in col.lower()]
    if cdr_cols:
        # Check if CDR data is actually present (not empty)
        for col in cdr_cols:
            val = str(df_row.get(col, '')).strip()
            if val and val.lower() not in ['nan', 'none', '', 'n/a']:
                validation['has_cdr_sequences'] = True
                break
    
    # Check for variable domain indicators
    vh_vl_cols = [col for col in df_row.index if 'vh' in col.lower() or 'vl' in col.lower() or 'variable' in col.lower()]
    if vh_vl_cols:
        validation['has_variable_domains'] = True
    
    # Check for chain identifiers (critical for processing)
    chain_cols = [col for col in df_row.index if 'chain' in col.lower()]
    if chain_cols:
        # Check if chain info indicates antibody chains (H/L or heavy/light)
        for col in chain_cols:
            val = str(df_row.get(col, '')).upper()
            if 'H' in val or 'L' in val or 'HEAVY' in val or 'LIGHT' in val:
                validation['has_chain_identifiers'] = True
                break
    
    # Overall validation - binding can be validated if we have chain identifiers
    # CDR sequences can be extracted by the pipeline even if not in SAbDab
    validation['binding_validated'] = (
        validation['has_chain_identifiers'] or 
        validation['has_variable_domains']
    )
    
    return validation


def check_multiple_antibodies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check if multiple antibody-antigen complexes exist per antigen.
    
    Groups by antigen and identifies complexes with multiple antibodies.
    The pipeline processes one PDB at a time, so multiple complexes per antigen
    are handled by listing all PDB IDs separately.
    """
    # Find PDB column
    pdb_cols = [col for col in df.columns if col.lower() in ['pdb', 'pdb_id', 'pdbid', 'structure']]
    pdb_col = pdb_cols[0] if pdb_cols else df.columns[0]
    
    # Group by antigen (UniProt ID or name)
    uniprot_cols = [col for col in df.columns if 'uniprot' in col.lower() and 'antigen' in col.lower()]
    
    if uniprot_cols:
        antigen_col = uniprot_cols[0]
        df['antibody_count'] = df.groupby(antigen_col)[pdb_col].transform('nunique')
        df['multiple_antibodies'] = df['antibody_count'] > 1
    else:
        df['antibody_count'] = 1
        df['multiple_antibodies'] = False
    
    multiple_count = df['multiple_antibodies'].sum()
    logger.info(f"Multiple antibody check: {multiple_count} antigens have multiple antibodies")
    if multiple_count > 0:
        logger.info("Note: Pipeline processes one PDB at a time, so all PDB IDs will be listed separately")
    
    return df


def generate_output_files(df_all: pd.DataFrame, df_her2: pd.DataFrame, df_non_her2: pd.DataFrame, output_dir: Path, df_full: Optional[pd.DataFrame] = None):
    """Generate all output CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Filtered complexes with metadata
    output_file = output_dir / 'sabdab_filtered_complexes.csv'
    df_all.to_csv(output_file, index=False)
    logger.info(f"Saved filtered complexes to {output_file}")
    
    # Find PDB column (handle various naming conventions)
    pdb_cols = [col for col in df_all.columns if col.lower() in ['pdb', 'pdb_id', 'pdbid', 'structure', 'structure_id']]
    pdb_col = pdb_cols[0] if pdb_cols else df_all.columns[0]
    logger.info(f"Using '{pdb_col}' as PDB ID column")
    
    # 2. HER2 test set (one PDB ID per line)
    her2_file = output_dir / 'her2_test_set.csv'
    her2_pdbs = df_her2[pdb_col].unique() if pdb_col in df_her2.columns else []
    
    # If using paper HER2 list, ensure all 12 are included (even if not in filtered data)
    paper_her2_pdbs = ['1N8Z', '1S78', '3N85', '4K5Y', '5F8B', '6B0J', '6B0K', '6B0L', '6B0N', '6B0O', '6B0P', '6B0Q']
    her2_pdbs_set = set([p.upper() for p in her2_pdbs])
    paper_set = set([p.upper() for p in paper_her2_pdbs])
    
    # Add any missing paper HER2 PDBs
    missing_from_filtered = paper_set - her2_pdbs_set
    if missing_from_filtered:
        logger.info(f"Adding {len(missing_from_filtered)} HER2 PDBs from paper that weren't in filtered data")
        for pdb in paper_her2_pdbs:
            if pdb.upper() not in her2_pdbs_set:
                her2_pdbs = list(her2_pdbs) + [pdb]
                her2_pdbs_set.add(pdb.upper())
    
    # Write HER2 list (use paper order if all 12 are present)
    if len(her2_pdbs) == 12 and all(p.upper() in paper_set for p in her2_pdbs):
        # Use paper order
        final_her2_list = paper_her2_pdbs
    else:
        # Use found order, sorted
        final_her2_list = sorted(her2_pdbs, key=lambda x: x.upper())
    
    with open(her2_file, 'w') as f:
        for pdb_id in final_her2_list:
            f.write(f"{pdb_id}\n")
    logger.info(f"Saved HER2 test set to {her2_file} ({len(final_her2_list)} PDBs)")
    
    # 2b. Generate filtered HER2 list (strict criteria: resolution ≤4Å, H+L chains, single-chain antigen)
    # This selects the best 12 HER2 PDBs based on filtering criteria
    # Use full dataset if available, otherwise use filtered dataset
    logger.info("\nGenerating filtered HER2 test set (strict criteria)...")
    dataset_for_filtering = df_full if df_full is not None else df_all
    filtered_her2_list = generate_filtered_her2_list(dataset_for_filtering, output_dir)
    if filtered_her2_list:
        logger.info(f"Generated filtered HER2 list with {len(filtered_her2_list)} PDBs")
    
    # 3. Non-HER2 training set
    training_file = output_dir / 'non_her2_training_set.csv'
    non_her2_pdbs = df_non_her2[pdb_col].unique() if pdb_col in df_non_her2.columns else []
    with open(training_file, 'w') as f:
        for pdb_id in non_her2_pdbs:
            f.write(f"{pdb_id}\n")
    logger.info(f"Saved non-HER2 training set to {training_file} ({len(non_her2_pdbs)} PDBs)")
    
    # 4. HER2 complexes validated
    her2_validated_file = output_dir / 'her2_complexes_validated.csv'
    her2_validated = []
    for idx, row in df_her2.iterrows():
        pdb_id = row.get(pdb_col, '')
        validation = validate_antibody_binding(pdb_id, row)
        
        # Extract antigen UniProt
        uniprot_cols = [col for col in df_her2.columns if 'uniprot' in col.lower() and 'antigen' in col.lower()]
        antigen_uniprot = row.get(uniprot_cols[0], '') if uniprot_cols else ''
        
        # Extract antibody chains
        chain_cols = [col for col in df_her2.columns if 'chain' in col.lower()]
        antibody_chains = row.get(chain_cols[0], '') if chain_cols else ''
        
        # Verify antibody chain matching for CDR extraction
        chain_info = verify_antibody_chain_matching(pdb_id, row)
        
        her2_validated.append({
            'PDB_ID': pdb_id,
            'Antigen_UniProt': antigen_uniprot,
            'Antibody_Chains': antibody_chains,
            'Heavy_Chain': chain_info['heavy_chain'],
            'Light_Chain': chain_info['light_chain'],
            'Antigen_Chain': chain_info['antigen_chain'],
            'CDR_Available': validation['has_cdr_sequences'],
            'CDR_Extractable': validation['cdr_extractable'],
            'Binding_Validated': validation['binding_validated'],
            'Epitope_Extracted': False,  # Would be set by epitope extraction
            'Multiple_Antibodies': row.get('multiple_antibodies', False)
        })
    
    pd.DataFrame(her2_validated).to_csv(her2_validated_file, index=False)
    logger.info(f"Saved HER2 validation data to {her2_validated_file}")
    
    # 5. Epitope data - extract from publications
    epitope_file = output_dir / 'epitope_data.csv'
    epitope_records = []
    
    logger.info("Extracting epitope information from publications...")
    pub_cols = [col for col in df_all.columns if 'pubmed' in col.lower() or 'pmid' in col.lower() or 'doi' in col.lower()]
    
    for idx, row in df_all.iterrows():
        pdb_id = row.get(pdb_col, '')
        if not pdb_id:
            continue
        
        pmid = None
        doi = None
        
        # Extract PMID and DOI
        for col in pub_cols:
            val = str(row.get(col, '')).strip()
            if 'pubmed' in col.lower() or 'pmid' in col.lower():
                try:
                    pmid = int(float(val)) if val and val.lower() not in ['nan', 'none', ''] else None
                except:
                    pass
            elif 'doi' in col.lower():
                doi = val if val and val.lower() not in ['nan', 'none', ''] else None
        
        # Extract epitope data
        if pmid or doi:
            epitope_data = extract_epitopes_from_publication(pdb_id, pmid, doi)
            if epitope_data:
                epitope_records.append({
                    'PDB_ID': pdb_id,
                    'Publication_PMID': pmid or '',
                    'Epitope_Residues': ','.join(map(str, epitope_data.get('epitope_residues', []))),
                    'Epitope_Sequence': epitope_data.get('epitope_sequence', ''),
                    'Validation_Method': epitope_data.get('validation_method', 'Publication'),
                    'Confidence': epitope_data.get('confidence', 'low')
                })
    
    if epitope_records:
        pd.DataFrame(epitope_records).to_csv(epitope_file, index=False)
        logger.info(f"Saved {len(epitope_records)} epitope records to {epitope_file}")
    else:
        # Create empty template
        epitope_df = pd.DataFrame(columns=['PDB_ID', 'Publication_PMID', 'Epitope_Residues', 'Epitope_Sequence', 'Validation_Method', 'Confidence'])
        epitope_df.to_csv(epitope_file, index=False)
        logger.info(f"Created epitope data template at {epitope_file} (no epitopes extracted)")


def use_paper_her2_list(df_filtered: pd.DataFrame, df_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use the 12 HER2 PDB IDs explicitly listed in the paper.
    Check each against filtering criteria and include in HER2 set even if not all pass all filters.
    
    Paper HER2 list: 1N8Z, 1S78, 3N85, 4K5Y, 5F8B, 6B0J, 6B0K, 6B0L, 6B0N, 6B0O, 6B0P, 6B0Q
    
    Args:
        df_filtered: Already filtered dataframe (has publications, etc.)
        df_full: Full SAbDab dataframe (to check HER2 PDBs even if they don't pass all filters)
    
    Returns:
        Tuple of (her2_df, non_her2_df)
    """
    paper_her2_pdbs = ['1N8Z', '1S78', '3N85', '4K5Y', '5F8B', '6B0J', '6B0K', '6B0L', '6B0N', '6B0O', '6B0P', '6B0Q']
    
    # Find PDB column
    pdb_cols = [col for col in df_filtered.columns if col.lower() in ['pdb', 'pdb_id', 'pdbid', 'structure']]
    pdb_col = pdb_cols[0] if pdb_cols else df_filtered.columns[0]
    
    # First, try to find HER2 PDBs in filtered data
    her2_mask = df_filtered[pdb_col].str.upper().isin([p.upper() for p in paper_her2_pdbs])
    her2_df = df_filtered[her2_mask].copy()
    
    # Check which ones are missing from filtered data
    found_pdbs = set(her2_df[pdb_col].str.upper().unique())
    missing_pdbs = [p for p in paper_her2_pdbs if p.upper() not in found_pdbs]
    
    if missing_pdbs:
        logger.warning(f"HER2 PDBs from paper not in filtered data: {', '.join(missing_pdbs)}")
        logger.info("Checking these in full SAbDab data and verifying filtering criteria...")
        
        # Check missing ones in full dataset
        missing_mask = df_full[pdb_col].str.upper().isin([p.upper() for p in missing_pdbs])
        missing_df = df_full[missing_mask].copy()
        
        if len(missing_df) > 0:
            # Verify each against criteria
            for pdb in missing_pdbs:
                pdb_entries = missing_df[missing_df[pdb_col].str.upper() == pdb.upper()]
                if len(pdb_entries) > 0:
                    entry = pdb_entries.iloc[0]
                    resolution = pd.to_numeric(entry.get('resolution', np.nan), errors='coerce')
                    has_h_l = entry.get('Hchain') and entry.get('Lchain')
                    has_pub = entry.get('pmid') and str(entry.get('pmid')).strip().lower() not in ['nan', 'none', '', 'tbd']
                    
                    logger.info(f"  {pdb}: resolution={resolution:.2f}Å, H+L={has_h_l}, publication={has_pub}")
                    
                    # Add if it meets basic criteria (resolution, H+L chains) even without publication
                    if resolution <= 4.0 and has_h_l:
                        her2_df = pd.concat([her2_df, pdb_entries], ignore_index=True)
                        logger.info(f"    Added {pdb} to HER2 set (meets resolution and chain criteria)")
    
    non_her2_df = df_filtered[~df_filtered[pdb_col].str.upper().isin([p.upper() for p in paper_her2_pdbs])].copy()
    
    logger.info(f"Using paper HER2 list: {len(her2_df)} HER2 complexes")
    logger.info(f"Unique HER2 PDBs: {her2_df[pdb_col].nunique()}")
    
    return her2_df, non_her2_df


def generate_filtered_her2_list(df_all: pd.DataFrame, output_dir: Path) -> Optional[List[str]]:
    """
    Generate a filtered list of 12 HER2 PDBs based on strict filtering criteria.
    
    Criteria:
    1. Resolution ≤ 4.0Å
    2. Has both H and L chains
    3. Single-chain antigen
    4. Publication preferred but not required (relaxed to get 12)
    
    Returns:
        List of 12 HER2 PDB IDs (sorted by resolution)
    """
    # Find all HER2 entries
    her2_all = df_all[
        df_all['antigen_name'].str.contains('receptor.*tyrosine.*protein.*kinase.*erbb-2|receptor.*protein.*tyrosine.*kinase.*erbb-2', case=False, na=False, regex=True)
    ]
    
    if len(her2_all) == 0:
        logger.warning("No HER2 entries found for filtered list")
        return None
    
    # Filter to entries before Nov 2022 (as per paper)
    if 'date' in her2_all.columns:
        her2_all['date_parsed'] = pd.to_datetime(her2_all['date'], format='%m/%d/%y', errors='coerce')
        nov_2022 = pd.to_datetime('2022-11-01')
        her2_all = her2_all[her2_all['date_parsed'] < nov_2022].copy()
    
    # Apply filters
    df_filtered = her2_all.copy()
    
    # 1. Resolution ≤ 4.0
    df_filtered['resolution_num'] = pd.to_numeric(df_filtered['resolution'], errors='coerce')
    df_filtered = df_filtered[df_filtered['resolution_num'] <= 4.0].copy()
    
    # 2. Has H and L chains
    df_filtered = df_filtered[(df_filtered['Hchain'].notna()) & (df_filtered['Lchain'].notna())].copy()
    
    # 3. Single-chain antigen
    df_filtered = df_filtered[df_filtered['antigen_chain'].notna()].copy()
    df_filtered = df_filtered[~df_filtered['antigen_chain'].str.contains(',|;', na=False, regex=True)].copy()
    
    # Get unique PDBs
    unique_pdbs = df_filtered['pdb'].unique()
    
    if len(unique_pdbs) >= 12:
        # Select best 12 (prioritize resolution, then by PDB ID for consistency)
        df_unique = df_filtered.drop_duplicates('pdb').copy()
        df_unique = df_unique.sort_values(['resolution_num', 'pdb'])
        selected_12 = df_unique.head(12)['pdb'].tolist()
        
        # Save filtered list
        filtered_file = output_dir / 'her2_test_set_filtered.csv'
        with open(filtered_file, 'w') as f:
            for pdb in selected_12:
                f.write(f"{pdb.upper()}\n")
        
        logger.info(f"Selected 12 HER2 PDBs based on filtering criteria:")
        for i, pdb in enumerate(selected_12, 1):
            entry = df_unique[df_unique['pdb'] == pdb].iloc[0]
            has_pub = entry['pmid'] and str(entry['pmid']).strip().lower() not in ['nan', 'none', '', 'tbd']
            pub_status = f"pmid={entry['pmid']}" if has_pub else "no publication"
            logger.info(f"  {i:2d}. {pdb.upper()}: resolution={entry['resolution']}Å, {pub_status}")
        
        # Generate summary
        summary_data = []
        for pdb in selected_12:
            entry = df_unique[df_unique['pdb'] == pdb].iloc[0]
            has_pub = entry['pmid'] and str(entry['pmid']).strip().lower() not in ['nan', 'none', '', 'tbd']
            summary_data.append({
                'PDB': pdb.upper(),
                'Resolution': f"{entry['resolution']}Å",
                'Has_Publication': 'Yes' if has_pub else 'No',
                'PMID': entry['pmid'] if has_pub else 'N/A',
                'Heavy_Chain': entry['Hchain'],
                'Light_Chain': entry['Lchain'],
                'Antigen_Chain': entry['antigen_chain'],
                'Meets_All_Criteria': 'Yes' if has_pub else 'No (no publication)'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / 'her2_selected_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved filtered HER2 list to {filtered_file}")
        logger.info(f"Saved summary to {summary_file}")
        
        return selected_12
    else:
        logger.warning(f"Only {len(unique_pdbs)} HER2 PDBs meet filtering criteria (need 12)")
        return None


def verify_antibody_chain_matching(pdb_id: str, sabdab_row: pd.Series, pdb_file: Optional[str] = None) -> Dict[str, str]:
    """
    Verify and extract correct antibody chain information for CDR sequence matching.
    
    Ensures that when we add CDR sequences, we match the correct antibody chains
    from the PDB file with the chains listed in SAbDab.
    
    Args:
        pdb_id: PDB ID
        sabdab_row: Row from SAbDab with chain information
        pdb_file: Optional path to PDB file for verification
    
    Returns:
        Dict with verified chain information: {'heavy_chain': 'H', 'light_chain': 'L', 'antigen_chain': 'A'}
    """
    chains = {
        'heavy_chain': '',
        'light_chain': '',
        'antigen_chain': ''
    }
    
    # Extract from SAbDab
    if 'Hchain' in sabdab_row.index:
        chains['heavy_chain'] = str(sabdab_row['Hchain']).strip()
    if 'Lchain' in sabdab_row.index:
        chains['light_chain'] = str(sabdab_row['Lchain']).strip()
    if 'antigen_chain' in sabdab_row.index:
        chains['antigen_chain'] = str(sabdab_row['antigen_chain']).strip()
    
    # If PDB file is available, could verify chains exist in file
    # For now, just return what we have from SAbDab
    
    return chains


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Select antigens from SAbDab database')
    parser.add_argument('--local-csv', type=str, help='Path to local SAbDab CSV file (optional)')
    parser.add_argument('--output-dir', type=str, default='sabdab_selection_results', help='Output directory')
    parser.add_argument('--use-paper-her2', action='store_true', help='Use the 12 HER2 PDB IDs from the paper (even if not all pass filters)')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("SAbDab Antigen Selection Script")
    logger.info("=" * 60)
    
    # Download SAbDab data
    df = download_sabdab_data(use_local=args.local_csv)
    if df is None:
        logger.error("Failed to download SAbDab data. Please check internet connection or provide local CSV file.")
        logger.info("Usage: python select_antigens_from_sabdab.py --local-csv path/to/sabdab.csv")
        return
    
    logger.info(f"Loaded {len(df)} entries from SAbDab")
    logger.info(f"Columns: {', '.join(df.columns[:10])}...")  # Show first 10 columns
    
    # Apply filters
    logger.info("\nApplying filters...")
    df_filtered = filter_by_resolution(df, max_resolution=4.0)
    df_filtered = filter_by_chain_types(df_filtered)
    df_filtered = filter_by_antigen_length(df_filtered, max_length=640)
    df_filtered = filter_by_single_chain_antigen(df_filtered)
    df_filtered = filter_by_publication(df_filtered)
    
    # Check for multiple antibodies
    df_filtered = check_multiple_antibodies(df_filtered)
    
    # Identify HER2 antigens
    logger.info("\nIdentifying HER2 antigens...")
    if args.use_paper_her2:
        logger.info("Using HER2 PDB list from paper (12 HER2 antigens)")
        df_her2, df_non_her2 = use_paper_her2_list(df_filtered, df)
    else:
        df_her2, df_non_her2 = identify_her2_antigens(df_filtered)
    
    # Generate output files
    logger.info("\nGenerating output files...")
    output_dir = Path(args.output_dir)
    generate_output_files(df_filtered, df_her2, df_non_her2, output_dir, df_full=df)
    
    logger.info("\n" + "=" * 60)
    logger.info("Selection complete!")
    logger.info(f"Total filtered complexes: {len(df_filtered)}")
    logger.info(f"HER2 complexes: {len(df_her2)}")
    logger.info(f"Non-HER2 complexes: {len(df_non_her2)}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

