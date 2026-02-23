#!/usr/bin/env python3
"""
Pre-cache BepiPred 2.0 predictions from IEDB API.

This script must be run on a LOGIN NODE with internet access.
It fetches BepiPred predictions for antigen sequences and caches them
for later use by the label generation script on compute nodes.

Usage:
    cd /leonardo_work/EUHPC_D29_035/Epi4Ab
    source venv/bin/activate
    python scripts/cache_bepipred.py
"""

import os
import sys
import json
import time
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

# Configuration
NODES_EDGES_DIR = "/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/nodes_edges"
CACHE_DIR = "/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/bepipred_cache"
IEDB_URL = "https://tools.iedb.org/bcell/"

# 19 PDBs that need labels (excluding 1n8z_BAC which already has labels)
PDB_LIST = [
    "1s78_DCA", "3be1_HLA", "3n85_HLA", "3wlw_CDA", "3wsq_HLA",
    "4lst_HLG", "4mwf_ABD", "4ywg_HLG", "5o4g_BAC",
    "6att_HLA", "6j6y_EFD", "6mug_HLG", "6nms_HLS", "6nmu_BAC",
    "6urm_DEC", "6wo5_HLE", "7l7r_DCG", "7lf7_ABM", "7mn8_DCB"
]


def check_internet():
    """Check if we have internet access (login node only)."""
    try:
        response = requests.get("https://tools.iedb.org", timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def get_sequence_from_node_feature(pdb_id: str) -> str:
    """Extract antigen sequence from node_feature.parquet."""
    node_feature_file = os.path.join(NODES_EDGES_DIR, pdb_id, "node_feature.parquet")
    
    if not os.path.exists(node_feature_file):
        print(f"  Warning: {node_feature_file} not found")
        return None
    
    df = pd.read_parquet(node_feature_file)
    
    if 'resShort' not in df.columns:
        print(f"  Warning: 'resShort' column not found in {pdb_id}")
        return None
    
    sequence = ''.join(df['resShort'].tolist())
    return sequence


def fetch_bepipred_prediction(sequence: str, session: requests.Session, max_retries: int = 3) -> dict:
    """
    Fetch BepiPred 2.0 prediction from IEDB API.
    
    Returns dict with:
        - scores: {position: score} mapping
        - assignments: {position: 'E' or '.'} mapping
        - sequence: original sequence
    """
    # Use separate connect and read timeouts
    connect_timeout = 30
    read_timeout = 300  # 5 minutes for long sequences
    
    # Step 1: Get CSRF token
    try:
        response = session.get(IEDB_URL, timeout=(connect_timeout, read_timeout))
        soup = BeautifulSoup(response.text, 'html.parser')
        csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'})
        
        if not csrf_token:
            raise ValueError("Could not find CSRF token on IEDB page")
        
        csrf_token = csrf_token['value']
    except requests.exceptions.Timeout as e:
        raise ValueError(f"IEDB connection timeout: {e}")
    except Exception as e:
        raise ValueError(f"Failed to get CSRF token: {e}")
    
    # Step 2: Submit prediction request (with retries)
    data = {
        'csrfmiddlewaretoken': csrf_token,
        'pred_tool': 'bcell',
        'source': 'html',
        'form_name': 'submission_form',
        'swissprot': '',
        'sequence_text': sequence,
        'method': 'Bepipred2'
    }
    
    headers = {
        'Referer': IEDB_URL,
        'Origin': 'https://tools.iedb.org',
    }
    
    # IEDB can be slow for long sequences - use longer timeout and retries
    response = None
    last_exception = None
    for attempt in range(max_retries):
        try:
            print(f"      Submitting request (attempt {attempt + 1}/{max_retries})...")
            sys.stdout.flush()
            response = session.post(IEDB_URL, data=data, headers=headers, timeout=(connect_timeout, read_timeout))
            if response.status_code == 200:
                print(f"      ✓ Form submission successful")
                sys.stdout.flush()
                break
            else:
                # Non-200 status code - treat as error but don't retry (server error)
                raise ValueError(f"IEDB returned status {response.status_code}")
        except requests.exceptions.ConnectTimeout as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 15
                print(f"      Connection timeout on attempt {attempt + 1}, retrying in {wait_time}s...")
                sys.stdout.flush()
                time.sleep(wait_time)
                # Get fresh CSRF token for retry
                try:
                    get_response = session.get(IEDB_URL, timeout=(connect_timeout, 30))
                    soup = BeautifulSoup(get_response.text, 'html.parser')
                    csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'})['value']
                    data['csrfmiddlewaretoken'] = csrf_token
                except Exception as e:
                    print(f"      Warning: Could not refresh CSRF token: {e}")
            else:
                # Last attempt failed - will raise after loop
                pass
        except requests.exceptions.ReadTimeout as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 15
                print(f"      Read timeout on attempt {attempt + 1} (IEDB server slow), retrying in {wait_time}s...")
                sys.stdout.flush()
                time.sleep(wait_time)
                # Get fresh CSRF token for retry
                try:
                    get_response = session.get(IEDB_URL, timeout=(connect_timeout, 30))
                    soup = BeautifulSoup(get_response.text, 'html.parser')
                    csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'})['value']
                    data['csrfmiddlewaretoken'] = csrf_token
                except Exception as e:
                    print(f"      Warning: Could not refresh CSRF token: {e}")
            else:
                # Last attempt failed - will raise after loop
                pass
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 15
                print(f"      Error on attempt {attempt + 1}: {e}, retrying in {wait_time}s...")
                sys.stdout.flush()
                time.sleep(wait_time)
                # Get fresh CSRF token for retry
                try:
                    get_response = session.get(IEDB_URL, timeout=(connect_timeout, 30))
                    soup = BeautifulSoup(get_response.text, 'html.parser')
                    csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'})['value']
                    data['csrfmiddlewaretoken'] = csrf_token
                except Exception:
                    pass
            else:
                # Last attempt failed - will raise after loop
                pass
    
    # Check if we got a successful response
    if response is None:
        if last_exception:
            if isinstance(last_exception, requests.exceptions.ConnectTimeout):
                raise ValueError(f"IEDB connection timeout after {max_retries} attempts: {last_exception}")
            elif isinstance(last_exception, requests.exceptions.ReadTimeout):
                raise ValueError(f"IEDB read timeout after {max_retries} attempts (server may be overloaded): {last_exception}")
            else:
                raise ValueError(f"IEDB request failed after {max_retries} attempts: {last_exception}")
        else:
            raise ValueError("IEDB form submission failed: No response received")
    elif response.status_code != 200:
        raise ValueError(f"IEDB form submission failed with status {response.status_code}")
    
    # Step 3: Get CSV results (also can be slow, with retries)
    csv_response = None
    csv_last_exception = None
    for attempt in range(max_retries):
        try:
            print(f"      Fetching CSV results (attempt {attempt + 1}/{max_retries})...")
            sys.stdout.flush()
            csv_response = session.get("https://tools.iedb.org/bcell/result_in_csv/", timeout=(connect_timeout, read_timeout))
            if csv_response.status_code == 200:
                print(f"      ✓ CSV download successful")
                sys.stdout.flush()
                break
            else:
                raise ValueError(f"IEDB CSV returned status {csv_response.status_code}")
        except requests.exceptions.ConnectTimeout as e:
            csv_last_exception = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 15
                print(f"      CSV connection timeout on attempt {attempt + 1}, retrying in {wait_time}s...")
                sys.stdout.flush()
                time.sleep(wait_time)
            else:
                pass  # Will raise after loop
        except requests.exceptions.ReadTimeout as e:
            csv_last_exception = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 15
                print(f"      CSV read timeout on attempt {attempt + 1}, retrying in {wait_time}s...")
                sys.stdout.flush()
                time.sleep(wait_time)
            else:
                pass  # Will raise after loop
        except Exception as e:
            csv_last_exception = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 15
                print(f"      CSV error on attempt {attempt + 1}: {e}, retrying in {wait_time}s...")
                sys.stdout.flush()
                time.sleep(wait_time)
            else:
                pass  # Will raise after loop
    
    # Check if we got a successful CSV response
    if csv_response is None:
        if csv_last_exception:
            if isinstance(csv_last_exception, requests.exceptions.ConnectTimeout):
                raise ValueError(f"IEDB CSV connection timeout after {max_retries} attempts: {csv_last_exception}")
            elif isinstance(csv_last_exception, requests.exceptions.ReadTimeout):
                raise ValueError(f"IEDB CSV read timeout after {max_retries} attempts: {csv_last_exception}")
            else:
                raise ValueError(f"IEDB CSV request failed after {max_retries} attempts: {csv_last_exception}")
        else:
            raise ValueError("IEDB CSV download failed: No response received")
    elif csv_response.status_code != 200:
        raise ValueError(f"IEDB CSV download failed with status {csv_response.status_code}")
    
    # Step 4: Parse CSV
    scores = {}
    assignments = {}
    
    lines = csv_response.text.strip().split('\n')
    for line in lines[1:]:  # Skip header
        parts = line.split(',')
        if len(parts) >= 4:
            try:
                position = int(parts[0])
                score = float(parts[2])
                assignment = parts[3]
                scores[position] = score
                assignments[position] = assignment
            except (ValueError, IndexError):
                continue
    
    return {
        'sequence': sequence,
        'scores': scores,
        'assignments': assignments,
        'method': 'Bepipred2',
        'threshold': 0.5
    }


def main():
    print("=" * 60)
    print("BepiPred 2.0 Cache Script")
    print("=" * 60)
    print()
    sys.stdout.flush()
    
    # Check internet connectivity
    print("Checking internet connectivity...")
    sys.stdout.flush()
    if not check_internet():
        print("ERROR: No internet access!")
        print("  This script must be run on a LOGIN NODE.")
        print("  Run from: login01.leonardo.local or login02.leonardo.local")
        sys.exit(1)
    print("  ✓ Internet access confirmed")
    print()
    sys.stdout.flush()
    
    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Cache directory: {CACHE_DIR}")
    print()
    sys.stdout.flush()
    
    # Process each PDB
    session = requests.Session()
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    print(f"Processing {len(PDB_LIST)} PDBs...")
    print()
    sys.stdout.flush()
    
    for pdb_id in PDB_LIST:
        cache_file = os.path.join(CACHE_DIR, f"{pdb_id}_bepipred.json")
        
        # Check if already cached
        if os.path.exists(cache_file):
            print(f"  {pdb_id}: Already cached, skipping")
            sys.stdout.flush()
            skip_count += 1
            continue
        
        print(f"  {pdb_id}: Processing...")
        sys.stdout.flush()
        
        # Get sequence
        sequence = get_sequence_from_node_feature(pdb_id)
        if not sequence:
            print(f"    ERROR: Could not extract sequence")
            sys.stdout.flush()
            fail_count += 1
            continue
        
        print(f"    Sequence length: {len(sequence)}")
        sys.stdout.flush()
        
        # Fetch prediction
        try:
            print(f"    Fetching BepiPred prediction...")
            sys.stdout.flush()
            result = fetch_bepipred_prediction(sequence, session)
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Count epitopes
            epitope_count = sum(1 for a in result['assignments'].values() if a == 'E')
            print(f"    ✓ Cached: {epitope_count}/{len(sequence)} predicted epitopes")
            sys.stdout.flush()
            success_count += 1
            
            # Rate limiting - be nice to IEDB servers
            time.sleep(3)  # Increased from 2 to 3 seconds
            
        except (ValueError, requests.exceptions.RequestException) as e:
            print(f"    ERROR: {e}")
            sys.stdout.flush()
            fail_count += 1
            # Continue with next PDB instead of stopping
        except Exception as e:
            print(f"    ERROR: Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            fail_count += 1
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Success: {success_count}")
    print(f"  Skipped (cached): {skip_count}")
    print(f"  Failed: {fail_count}")
    print()
    
    if fail_count > 0:
        print("WARNING: Some PDBs failed. Check output above for details.")
        print("You can re-run this script to retry failed PDBs (already cached ones will be skipped).")
    else:
        print("✓ All PDBs processed successfully!")
    
    print("BepiPred caching complete!")


if __name__ == "__main__":
    main()

