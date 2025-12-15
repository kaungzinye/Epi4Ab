# VS Code Setup for Epi4Ab Workspace

## Workspace Location

**HPC Path:**
```
/leonardo_work/AIFAC_F01_302/Epi4Ab
```

**Git Repository:**
- Origin (Your Fork): `git@github.com:kaungzinye/Epi4Ab.git`
- Upstream (Official): `https://github.com/AMPMgroup/Epi4Ab.git`

**Current Branch:** `upstream-preprocess-test`

---

## Option 1: VS Code Remote SSH (Recommended)

### Prerequisites
1. VS Code installed on your local machine
2. SSH access to Leonardo HPC configured
3. Remote - SSH extension installed in VS Code

### Steps

1. **Install Remote - SSH Extension**
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Remote - SSH"
   - Install the extension by Microsoft

2. **Configure SSH Connection**
   
   Add to your `~/.ssh/config` file (on your local machine):
   ```
   Host leonardo
       HostName login07.leonardo.local
       User knaung00
       ForwardAgent yes
       ServerAliveInterval 60
       ServerAliveCountMax 3
   ```

3. **Connect to HPC**
   - Press `F1` (or `Cmd+Shift+P` on Mac)
   - Type "Remote-SSH: Connect to Host"
   - Select "leonardo" (or enter `knaung00@login07.leonardo.local`)
   - Enter your SSH password (or use SSH key if configured)

4. **Open Workspace**
   - Once connected, click "Open Folder"
   - Navigate to: `/leonardo_work/AIFAC_F01_302/Epi4Ab`
   - Click "OK"

5. **Install Python Extension (on Remote)**
   - VS Code will prompt to install extensions on the remote
   - Install "Python" extension by Microsoft
   - Install "Pylance" for better IntelliSense

---

## Option 2: Clone Locally (If You Have Local Access)

If you want to work locally and sync changes:

1. **Clone Your Fork**
   ```bash
   git clone git@github.com:kaungzinye/Epi4Ab.git
   cd Epi4Ab
   ```

2. **Add Upstream Remote**
   ```bash
   git remote add upstream https://github.com/AMPMgroup/Epi4Ab.git
   git fetch upstream
   ```

3. **Checkout Your Branch**
   ```bash
   git checkout upstream-preprocess-test
   # Or create a new branch from the remote
   git checkout -b upstream-preprocess-test origin/upstream-preprocess-test
   ```

4. **Open in VS Code**
   ```bash
   code .
   ```

**Note:** This creates a local copy. You'll need to push/pull to sync with HPC.

---

## Option 3: VS Code in Browser (Code Server / GitHub Codespaces)

If you have access to a web-based IDE:

1. **Use JupyterLab/VS Code on HPC** (if available)
   - Some HPC systems provide web-based IDEs
   - Check with your HPC admin

2. **GitHub Codespaces** (if repository is public/accessible)
   - Go to your GitHub repository
   - Click "Code" → "Codespaces"
   - Create a new codespace
   - Clone the repository in the codespace

---

## Recommended VS Code Extensions

Once connected, install these extensions:

### Essential
- **Python** (Microsoft) - Python language support
- **Pylance** (Microsoft) - Fast Python language server
- **GitLens** - Enhanced Git capabilities

### Optional but Useful
- **Remote - SSH** (Microsoft) - Already installed
- **Jupyter** (Microsoft) - For notebook support
- **Markdown All in One** - For documentation
- **YAML** - For configuration files
- **Docker** (if using containers)

---

## Workspace Configuration

Create `.vscode/settings.json` in the workspace:

```json
{
    "python.defaultInterpreterPath": "/leonardo_work/AIFAC_F01_302/Epi4Ab/.venv/bin/python",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/source_code",
        "${workspaceFolder}/data_processing",
        "${workspaceFolder}/preprocess"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/logs": true,
        "**/output_inference": true,
        "**/.git": false
    },
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true,
    "python.formatting.provider": "black"
}
```

---

## Troubleshooting

### Connection Issues
- **SSH timeout**: Increase `ServerAliveInterval` in SSH config
- **Permission denied**: Check SSH key permissions (`chmod 600 ~/.ssh/id_rsa`)
- **Host key verification**: Add HPC host to known_hosts

### Python Interpreter Issues
- **Wrong interpreter**: Press `Ctrl+Shift+P` → "Python: Select Interpreter"
- **Missing packages**: Activate your venv and install requirements
  ```bash
  source /leonardo_work/AIFAC_F01_302/Epi4Ab/.venv/bin/activate
  pip install -r requirements.txt
  ```

### Git Issues
- **Remote not found**: Check remotes with `git remote -v`
- **Branch not found**: Fetch from remote first: `git fetch origin`

---

## Quick Start Commands

Once connected in VS Code:

```bash
# Check current branch
git branch --show-current

# View recent commits
git log --oneline -10

# Check status
git status

# Switch branches (if needed)
git checkout master
git checkout upstream-preprocess-test
```

---

## SLURM Workflow (Required for Production)

### Overview

**IMPORTANT:** All preprocessing and inference tasks MUST be submitted via SLURM batch jobs. Direct script execution is NOT recommended for production use.

**Why SLURM?**
- Proper resource allocation and job scheduling
- Automatic logging and error tracking
- Isolation from login node (no internet needed on compute nodes)
- Prevents timeout issues on long-running jobs

### Login Node vs Compute Node

| Feature | Login Node | Compute Node |
|---------|-----------|--------------|
| Internet Access | ✅ Yes | ❌ No |
| Time Limit | Short (~minutes) | Long (hours) |
| Resources | Limited | Full allocation |
| Use Case | Download, submit jobs, quick tasks | Heavy computation |

### Complete SLURM Workflow

#### Step 1: Download PDBs (Login Node)

Run on login node (has internet access):

```bash
cd /leonardo_work/AIFAC_F01_302/Epi4Ab
./scripts/download_pdbs_login.sh
```

This downloads .cif files and creates lig.pdb files needed for preprocessing.

#### Step 2: Submit Preprocessing Job (SLURM)

Once downloads complete, submit preprocessing job:

```bash
sbatch slurm/preprocess_test3A.sbatch
```

Monitor job:
```bash
# Check job status
squeue -u $USER

# View live output
tail -f logs/epi4ab-preprocess-test3A-JOBID.out

# View errors
tail -f logs/epi4ab-preprocess-test3A-JOBID.err
```

**Job Details:**
- Time limit: 8 hours
- Runs Steps 2-4: PDB2PQR, feature extraction, nodes/edges, fill edges
- Output: Preprocessed data in `/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/upstream_preprocess/`

#### Step 3: Submit Inference Job (SLURM)

After preprocessing completes, submit inference job:

```bash
sbatch slurm/inference_test3A.sbatch
```

**Job Details:**
- Time limit: 2 hours
- Runs inference on all preprocessed PDBs
- Output: Results in `output_inference/` with timestamped directory

#### Step 4: Generate Dashboard (Login Node)

After inference completes, generate visualization dashboard:

```bash
# Find the latest inference output directory
LATEST_OUTPUT=$(ls -td output_inference/20* | head -1)

# Generate dashboard
./scripts/generate_dashboard.sh $LATEST_OUTPUT/test_record
```

Dashboard will be saved as `$LATEST_OUTPUT/dashboard.html`

### Available SLURM Scripts

| Script | Purpose | Time Limit | Resources |
|--------|---------|------------|-----------|
| `slurm/preprocess_test3A.sbatch` | Preprocessing (Steps 2-4) | 8 hours | 8 CPUs, 32GB RAM |
| `slurm/inference_test3A.sbatch` | Model inference | 2 hours | 4 CPUs, 16GB RAM |

### Useful SLURM Commands

```bash
# Submit a job
sbatch slurm/preprocess_test3A.sbatch

# Check your jobs
squeue -u $USER

# Cancel a job
scancel JOBID

# View job details
scontrol show job JOBID

# View completed job info
sacct -j JOBID --format=JobID,JobName,Partition,State,ExitCode,Elapsed

# Check your account usage
sacct -u $USER --starttime $(date -d '7 days ago' +%Y-%m-%d)
```

### Monitoring Job Progress

```bash
# Watch job queue
watch -n 5 'squeue -u $USER'

# Tail output logs (replace JOBID)
tail -f logs/epi4ab-preprocess-test3A-JOBID.out

# Check if preprocessing completed successfully
ls /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/upstream_preprocess/nodes_edges/

# Count completed PDBs
ls -d /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/upstream_preprocess/nodes_edges/*/ | wc -l
```

### Warning about Direct Script Execution

The following scripts have warnings and 5-second delays:
- `run_preprocess.sh`
- `run_preprocess_1n8z.sh`
- `run_preprocess_test3A.sh`
- `run_inference.sh`
- `run_inference_1n8z.sh`
- `run_inference_test3A.sh`

**These are for development/testing only.** For production, always use SLURM.

---

## Current Workspace State

**Branch:** `upstream-preprocess-test`  
**Latest Commit:** `70b375c` - "Add comprehensive pipeline comparison and upstream HPC adaptations"

**Key Files:**
- `docs/PIPELINE_COMPARISON.md` - Comprehensive comparison document
- `preprocess/scripts/extract_depth_freesasa.py` - FreeSASA replacement
- `scripts/visualize_results.py` - Visualization tool
- `.env.upstream_template` - Environment configuration template

---

## Need Help?

- Check git remotes: `git remote -v`
- View branch history: `git log --oneline --graph --all`
- Check workspace path: `pwd` (should be `/leonardo_work/AIFAC_F01_302/Epi4Ab`)

