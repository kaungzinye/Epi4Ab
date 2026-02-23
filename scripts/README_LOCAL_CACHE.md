# Local BepiPred Caching Instructions

This guide helps you run BepiPred caching on your local machine for debugging.

## Step 1: Extract Sequences from HPC

On the HPC system, run:

```bash
cd /leonardo_work/EUHPC_D29_035/Epi4Ab
source venv/bin/activate
python scripts/extract_sequences_for_local.py > sequences.txt
```

This creates `sequences.txt` with format:
```
1s78_DCA:MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH
3be1_HLA:MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH
...
```

## Step 2: Copy Files to Local Machine

Copy these files from HPC to your local machine:

**Note:** Replace `HOSTNAME` below with your actual Leonardo hostname. Common options:
- `leonardo.cineca.it` (external access)
- `login.leonardo.cineca.it` (external access)
- `login05.leonardo.local` (internal, if you're already on Leonardo network)

**To find your hostname, on HPC run:**
```bash
hostname -f
# or check how you normally SSH in
```

**On your local machine, run:**

```bash
# Copy the local cache script
scp knaung00@HOSTNAME:/leonardo_work/EUHPC_D29_035/Epi4Ab/scripts/cache_bepipred_local.py ./

# Copy the sequences file
scp knaung00@HOSTNAME:/leonardo_work/EUHPC_D29_035/Epi4Ab/sequences.txt ./
```

**Or copy both at once:**

```bash
# Copy both files
scp knaung00@HOSTNAME:/leonardo_work/EUHPC_D29_035/Epi4Ab/scripts/cache_bepipred_local.py \
   knaung00@HOSTNAME:/leonardo_work/EUHPC_D29_035/Epi4Ab/sequences.txt ./
```

**Alternative: Copy from HPC (if you're on HPC):**

```bash
# On HPC, create a tarball
cd /leonardo_work/EUHPC_D29_035/Epi4Ab
tar -czf local_cache_files.tar.gz scripts/cache_bepipred_local.py sequences.txt

# Then copy the tarball to your local machine
# On local machine:
scp knaung00@HOSTNAME:/leonardo_work/EUHPC_D29_035/Epi4Ab/local_cache_files.tar.gz ./
tar -xzf local_cache_files.tar.gz
```

**Files needed:**
- `cache_bepipred_local.py` - The local caching script
- `sequences.txt` - Sequences extracted in Step 1

## Step 3: Install Dependencies Locally

On your local machine:

```bash
pip install requests beautifulsoup4 pandas pyarrow fastparquet lxml
```

## Step 4: Run Local Caching

On your local machine:

```bash
python cache_bepipred_local.py --sequences sequences.txt --cache_dir ./bepipred_cache
```

This will:
- Fetch BepiPred predictions for each sequence
- Cache results in `./bepipred_cache/` directory
- Retry on failures (up to 3 attempts)
- Skip already cached files

## Step 5: Copy Cache Back to HPC

Once complete, copy the cache directory back to HPC:

```bash
# On local machine (replace HOSTNAME with your Leonardo hostname)
scp -r bepipred_cache/ knaung00@HOSTNAME:/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/bepipred_cache/
```

Or copy individual files:
```bash
scp bepipred_cache/*_bepipred.json knaung00@HOSTNAME:/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/bepipred_cache/
```

## Alternative: Use JSON Format

Instead of sequences.txt, you can use a JSON file:

```json
{
  "1s78_DCA": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH",
  "3be1_HLA": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH"
}
```

Then run:
```bash
python cache_bepipred_local.py --json sequences.json
```

## Troubleshooting

- **Connection timeouts**: IEDB API can be slow. The script retries automatically.
- **500 errors**: IEDB server may be temporarily down. Wait and retry.
- **Rate limiting**: Script includes 3-second delays between requests.

## Output Format

Each cached file (`{pdb_id}_bepipred.json`) contains:
```json
{
  "sequence": "MKTAYIAK...",
  "scores": {
    "0": 0.239,
    "1": 0.302,
    ...
  },
  "assignments": {
    "0": ".",
    "5": "E",
    ...
  },
  "method": "Bepipred2",
  "threshold": 0.5
}
```

Where:
- `scores`: BepiPred score for each position (0-based)
- `assignments`: "E" = epitope (score >= 0.5), "." = non-epitope

