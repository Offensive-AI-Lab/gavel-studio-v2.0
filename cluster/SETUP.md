# Cluster training — setup from scratch

Follow these steps once per machine to enable cluster training on
`slurm.bgu.ac.il`. After this, training runs automatically the next
time you click Train in the UI.

## Prerequisites

- BGU VPN active
- A cluster account on `slurm.bgu.ac.il`
- Git clone of this repo on your local machine

Reachability check:

| OS | Command |
|----|---------|
| macOS / Linux | `nc -z -v slurm.bgu.ac.il 22` |
| Windows (PowerShell) | `Test-NetConnection slurm.bgu.ac.il -Port 22` |

---

## Option A — automated (recommended for macOS / Linux / WSL / Git Bash)

From the project root, run these **two separate commands** (the first
makes the script executable, the second runs it):

```bash
chmod +x cluster/setup_local.sh
```

```bash
./cluster/setup_local.sh
```

The script asks for three things — for any prompt with a value in
`[brackets]`, just press **Enter** to accept the default:

1. **Cluster username** — your BGU SLURM login (e.g. `avigofek`)
2. **Repo path** — auto-detected when run from the project root; press
   Enter
3. **SSH key path** — default `~/.ssh/id_ed25519_cluster`; press Enter

Then it does all of this for you:

- Generates an SSH key if you don't already have one
- Copies the public key to the cluster (asks for your BGU password
  once — it's not stored)
- Verifies passwordless login works
- Pushes `cluster/`, `backend/classifier_engine/`, `backend/utils/` to
  `~/gavel_code/` on the cluster
- Creates an isolated `gavel` conda environment on the cluster and
  installs PyTorch + the other Python packages the trainer needs
- Adds `SLURM_HOST` and `SLURM_USER` to `backend/.env` (asks first)

Every step is idempotent — re-run the script any time you want to
re-push code or re-verify the setup.

On Windows: run it from Git Bash or WSL. Native PowerShell users
should follow Option B instead.

---

## Option B — manual

### 1. SSH key

Generate a key dedicated to the cluster:

| OS | Command |
|----|---------|
| macOS / Linux | `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_cluster -N ''` |
| Windows (PowerShell) | `ssh-keygen -t ed25519 -f $HOME\.ssh\id_ed25519_cluster -N '""'` |

Copy the public half to the cluster (asks for your cluster password once):

| OS | Command |
|----|---------|
| macOS / Linux | `ssh-copy-id -i ~/.ssh/id_ed25519_cluster.pub <cluster-user>@slurm.bgu.ac.il` |
| Windows (PowerShell) | `type $HOME\.ssh\id_ed25519_cluster.pub | ssh <cluster-user>@slurm.bgu.ac.il "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"` |

Verify passwordless login:

| OS | Command |
|----|---------|
| macOS / Linux | `ssh -i ~/.ssh/id_ed25519_cluster <cluster-user>@slurm.bgu.ac.il "hostname"` |
| Windows (PowerShell) | `ssh -i $HOME\.ssh\id_ed25519_cluster <cluster-user>@slurm.bgu.ac.il "hostname"` |

### 2. Backend `.env`

Add these two lines to `backend/.env`:

```dotenv
SLURM_HOST=slurm.bgu.ac.il
SLURM_USER=<your-cluster-username>
```

### 3. Push code to the cluster

**Run these from the project root** (the directory that contains the
`cluster/` and `backend/` folders).

macOS / Linux:

```bash
ssh -i ~/.ssh/id_ed25519_cluster <cluster-user>@slurm.bgu.ac.il "mkdir -p ~/gavel_code"
scp -i ~/.ssh/id_ed25519_cluster -r cluster <cluster-user>@slurm.bgu.ac.il:~/gavel_code/
scp -i ~/.ssh/id_ed25519_cluster -r backend/classifier_engine <cluster-user>@slurm.bgu.ac.il:~/gavel_code/
scp -i ~/.ssh/id_ed25519_cluster -r backend/utils <cluster-user>@slurm.bgu.ac.il:~/gavel_code/
```

Windows (PowerShell):

```powershell
ssh -i $HOME\.ssh\id_ed25519_cluster <cluster-user>@slurm.bgu.ac.il "mkdir -p ~/gavel_code"
scp -i $HOME\.ssh\id_ed25519_cluster -r cluster <cluster-user>@slurm.bgu.ac.il:~/gavel_code/
scp -i $HOME\.ssh\id_ed25519_cluster -r backend\classifier_engine <cluster-user>@slurm.bgu.ac.il:~/gavel_code/
scp -i $HOME\.ssh\id_ed25519_cluster -r backend\utils <cluster-user>@slurm.bgu.ac.il:~/gavel_code/
```

Re-run these any time you edit `cluster/`, `classifier_engine/`, or
`utils/`.

### 4. Create the conda environment

SSH in and run the setup script (~10 min the first time):

```bash
ssh -i ~/.ssh/id_ed25519_cluster <cluster-user>@slurm.bgu.ac.il
```

(replace `~` with `$HOME` on PowerShell.) On the cluster:

```bash
cd ~/gavel_code/cluster
chmod +x setup_cluster_env.sh
./setup_cluster_env.sh
```

Verify the env loads:

```bash
module load anaconda
conda activate gavel
python -c "import torch"
```

### 5. Smoke test

Still on the cluster, submit a short job to confirm SLURM allocates a GPU:

```bash
sbatch --partition=main --qos=normal --gpus=rtx_4090:1 --time=00:01:00 \
       --wrap="nvidia-smi -L"
```

`squeue -u $USER` shows it queued. When it finishes, the
`slurm-<jobid>.out` file in your home directory should list a GPU.

Setup is complete.
