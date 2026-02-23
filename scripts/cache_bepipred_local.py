#!/usr/bin/env python3
"""
BepiPred 2.0 Cache Script - Local Version

This script fetches BepiPred predictions from IEDB API and caches them locally.
You can run this on your local machine for debugging.

Usage:
    # Install dependencies first:
    pip install requests beautifulsoup4 pandas pyarrow fastparquet
    
    # Option 1: Provide sequences directly
    python cache_bepipred_local.py --sequences sequences.txt
    
    # Option 2: Use PDB IDs and fetch sequences from parquet files
    python cache_bepipred_local.py --pdb_list 1s78_DCA,3be1_HLA --parquet_dir /path/to/nodes_edges
    
    # Option 3: Use a JSON file with PDB ID -> sequence mapping
    python cache_bepipred_local.py --json sequences.json
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

# IEDB API URL
IEDB_URL = "https://tools.iedb.org/bcell/"

# Default cache directory (current directory)
DEFAULT_CACHE_DIR = "./bepipred_cache"


def check_internet():
    """Check if we have internet access."""
    try:
        response = requests.get("https://tools.iedb.org", timeout=10)
        return response.status_code == 200
    except Exception:
        return False


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
                # Non-200 status code - retry (server might be temporarily down)
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 15
                    print(f"      IEDB returned status {response.status_code}, retrying in {wait_time}s...")
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
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 15
                    print(f"      CSV returned status {csv_response.status_code}, retrying in {wait_time}s...")
                    sys.stdout.flush()
                    time.sleep(wait_time)
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
                pass
        except requests.exceptions.ReadTimeout as e:
            csv_last_exception = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 15
                print(f"      CSV read timeout on attempt {attempt + 1}, retrying in {wait_time}s...")
                sys.stdout.flush()
                time.sleep(wait_time)
            else:
                pass
        except Exception as e:
            csv_last_exception = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 15
                print(f"      CSV error on attempt {attempt + 1}: {e}, retrying in {wait_time}s...")
                sys.stdout.flush()
                time.sleep(wait_time)
            else:
                pass
    
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
                assignment = parts[3].strip()
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


def get_sequence_from_parquet(pdb_id: str, parquet_dir: str) -> str:
    """Extract antigen sequence from node_feature.parquet."""
    node_feature_file = os.path.join(parquet_dir, pdb_id, "node_feature.parquet")
    
    if not os.path.exists(node_feature_file):
        print(f"  Warning: {node_feature_file} not found")
        return None
    
    df = pd.read_parquet(node_feature_file)
    
    if 'resShort' not in df.columns:
        print(f"  Warning: 'resShort' column not found in {pdb_id}")
        return None
    
    sequence = ''.join(df['resShort'].tolist())
    return sequence


def main():
    parser = argparse.ArgumentParser(description="Cache BepiPred 2.0 predictions locally")
    parser.add_argument('--cache_dir', type=str, default=DEFAULT_CACHE_DIR,
                       help='Cache directory (default: ./bepipred_cache)')
    parser.add_argument('--sequences', type=str,
                       help='Text file with PDB_ID:SEQUENCE pairs (one per line)')
    parser.add_argument('--pdb_list', type=str,
                       help='Comma-separated list of PDB IDs (requires --parquet_dir)')
    parser.add_argument('--parquet_dir', type=str,
                       help='Directory containing node_feature.parquet files')
    parser.add_argument('--json', type=str,
                       help='JSON file with {"pdb_id": "sequence"} mapping')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BepiPred 2.0 Cache Script (Local Version)")
    print("=" * 60)
    print()
    
    # Check internet connectivity
    print("Checking internet connectivity...")
    if not check_internet():
        print("ERROR: No internet access!")
        sys.exit(1)
    print("  ✓ Internet access confirmed")
    print()
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"Cache directory: {args.cache_dir}")
    print()
    
    # Load sequences
    sequences_dict = {}
    
    if args.json:
        # Load from JSON file
        with open(args.json, 'r') as f:
            sequences_dict = json.load(f)
    elif args.sequences:
        # Load from text file (format: PDB_ID:SEQUENCE)
        with open(args.sequences, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    pdb_id, sequence = line.split(':', 1)
                    sequences_dict[pdb_id.strip()] = sequence.strip()
    elif args.pdb_list and args.parquet_dir:
        # Load from parquet files
        pdb_ids = [p.strip() for p in args.pdb_list.split(',')]
        for pdb_id in pdb_ids:
            sequence = get_sequence_from_parquet(pdb_id, args.parquet_dir)
            if sequence:
                sequences_dict[pdb_id] = sequence
    else:
        print("ERROR: Must provide one of: --sequences, --json, or --pdb_list with --parquet_dir")
        parser.print_help()
        sys.exit(1)
    
    if not sequences_dict:
        print("ERROR: No sequences found!")
        sys.exit(1)
    
    print(f"Processing {len(sequences_dict)} sequences...")
    print()
    
    # Process each sequence
    session = requests.Session()
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for pdb_id, sequence in sequences_dict.items():
        cache_file = os.path.join(args.cache_dir, f"{pdb_id}_bepipred.json")
        
        # Check if already cached
        if os.path.exists(cache_file):
            print(f"  {pdb_id}: Already cached, skipping")
            skip_count += 1
            continue
        
        print(f"  {pdb_id}: Processing...")
        print(f"    Sequence length: {len(sequence)}")
        
        # Fetch prediction
        try:
            print(f"    Fetching BepiPred prediction...")
            result = fetch_bepipred_prediction(sequence, session)
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Count epitopes
            epitope_count = sum(1 for a in result['assignments'].values() if a == 'E')
            print(f"    ✓ Cached: {epitope_count}/{len(sequence)} predicted epitopes")
            success_count += 1
            
            # Rate limiting
            time.sleep(3)
            
        except (ValueError, requests.exceptions.RequestException) as e:
            print(f"    ERROR: {e}")
            fail_count += 1
        except Exception as e:
            print(f"    ERROR: Unexpected error: {e}")
            import traceback
            traceback.print_exc()
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
        print("WARNING: Some sequences failed. Check output above for details.")
        print("You can re-run this script to retry failed ones (already cached will be skipped).")
    else:
        print("✓ All sequences processed successfully!")
    
    print(f"\nCache files saved to: {args.cache_dir}")


if __name__ == "__main__":
    main()

