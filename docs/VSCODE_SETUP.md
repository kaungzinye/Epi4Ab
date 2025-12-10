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

