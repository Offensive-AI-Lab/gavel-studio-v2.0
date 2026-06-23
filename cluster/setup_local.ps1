# Interactive cluster setup for WINDOWS - PowerShell port of setup_local.sh.
#
# For Windows users who DON'T have Git Bash / WSL. Uses only what ships with
# Windows: PowerShell + the built-in OpenSSH client (ssh / scp / ssh-keygen).
#
# Run it (from anywhere) with:
#   powershell -ExecutionPolicy Bypass -File cluster\setup_local.ps1
#   powershell -ExecutionPolicy Bypass -File cluster\setup_local.ps1 -RecreateEnv
#
# Does the same 6 steps as the .sh:
#   1. Prereq check (ssh/scp/ssh-keygen, reachability)
#   2. SSH key generation (skipped if one already exists)
#   3. Authorize the key on the cluster (asks for your cluster password once)
#   4. Clean-sync cluster/ + backend/classifier_engine/ + backend/utils/ via scp
#   5. Run the cluster-side conda env setup (versioned; safe to re-run)
#   6. Write SLURM config into backend\.env (the single config file, native + docker)
#
# Requires the BGU VPN to be ON.

[CmdletBinding()]
param(
    [switch]$RecreateEnv,   # wipe + rebuild the cluster conda env from scratch
    [switch]$Fresh          # alias for -RecreateEnv
)

$ErrorActionPreference = 'Stop'
$recreate = '0'
if ($RecreateEnv -or $Fresh) { $recreate = '1' }

$ClusterHost   = 'slurm.bgu.ac.il'
$SshKeyDefault = Join-Path $HOME '.ssh\id_ed25519_cluster'

# Where the GAVEL code lives ON THE CLUSTER - matches the backend's
# SLURM_CODE_DIR default ("~/gavel_code"). If you change it, change that too.
$ClusterCodeDir = '~/gavel_code'

# --- pretty output ---------------------------------------------------------
function Say  ($m) { Write-Host $m }
function OK   ($m) { Write-Host "OK  $m"   -ForegroundColor Green }
function Warn ($m) { Write-Host "!   $m"   -ForegroundColor Yellow }
function ErrM ($m) { Write-Host "x   $m"   -ForegroundColor Red }
function Step ($m) { Write-Host ""; Write-Host ">> $m" -ForegroundColor Cyan }
function Abort($m) { ErrM $m; exit 1 }

function Ask($prompt, $default = '') {
    if ($default) {
        $r = Read-Host "$prompt [$default]"
        if ([string]::IsNullOrWhiteSpace($r)) { return $default } else { return $r }
    }
    return (Read-Host $prompt)
}
function Confirm($q) { return ((Read-Host "$q (y/N)") -match '^[Yy]$') }

# Common ssh options used everywhere.
$SshOpts = @('-o', 'StrictHostKeyChecking=no')

# --- 0. Banner + prereqs ---------------------------------------------------
Write-Host ""
Write-Host "GAVEL cluster setup (Windows / PowerShell)" -ForegroundColor White
Write-Host ""
Say "This configures your machine + the BGU SLURM cluster so the GAVEL"
Say "backend can submit training jobs over SSH."

Step "Checking prerequisites"
foreach ($cmd in 'ssh', 'scp', 'ssh-keygen') {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        ErrM "$cmd not found."
        Say  "Install the Windows OpenSSH client, then re-run:"
        Say  "  Settings > Apps > Optional Features > Add > 'OpenSSH Client'"
        Say  "  ...or in an admin PowerShell:"
        Say  "  Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0"
        exit 1
    }
}
try {
    $reach = Test-NetConnection -ComputerName $ClusterHost -Port 22 -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($reach) { OK "${ClusterHost}:22 is reachable" }
    else { Abort "Cannot reach ${ClusterHost}:22. Is your BGU VPN active?" }
} catch {
    Warn "Reachability check unavailable - make sure your BGU VPN is on."
}

# --- 1. Inputs -------------------------------------------------------------
Step "Configuration"
Say "Three things. Press Enter to accept a [bracketed] default."
Write-Host ""

Say "1) Cluster username  (the name in: ssh <name>@slurm.bgu.ac.il)"
$ClusterUser = Ask "   Cluster username"
if ([string]::IsNullOrWhiteSpace($ClusterUser)) { Abort "Cluster username is required." }
Write-Host ""

# Auto-detect the repo root: cwd if it looks right, else this script's parent.
$DefaultRepo = (Get-Location).Path
if (-not ((Test-Path (Join-Path $DefaultRepo 'cluster')) -and (Test-Path (Join-Path $DefaultRepo 'backend\classifier_engine')))) {
    $cand = Split-Path $PSScriptRoot -Parent
    if ((Test-Path (Join-Path $cand 'cluster')) -and (Test-Path (Join-Path $cand 'backend\classifier_engine'))) {
        $DefaultRepo = $cand
    }
}
Say "2) Path to this repo (the folder containing 'cluster\' and 'backend\')"
$RepoDir = Ask "   Repo root" $DefaultRepo
if (-not ((Test-Path (Join-Path $RepoDir 'cluster')) -and (Test-Path (Join-Path $RepoDir 'backend\classifier_engine')))) {
    Abort "Repo not found at $RepoDir (expected cluster\ and backend\classifier_engine\ inside)"
}
Write-Host ""

Say "3) SSH key path (default is fine for almost everyone)"
$SshKey = Ask "   SSH key path" $SshKeyDefault
Write-Host ""

Say "Summary:"
Say "  Cluster user : $ClusterUser@$ClusterHost"
Say "  Repo root    : $RepoDir"
Say "  SSH key      : $SshKey"
Write-Host ""
if (-not (Confirm "Continue with this configuration?")) { Abort "Aborted." }

# --- 2. SSH key ------------------------------------------------------------
Step "SSH key"
if (Test-Path $SshKey) {
    OK "Key already exists at $SshKey - reusing it"
} else {
    Say "Generating a new ed25519 key at $SshKey"
    $keyDir = Split-Path $SshKey -Parent
    if ($keyDir -and -not (Test-Path $keyDir)) { New-Item -ItemType Directory -Force -Path $keyDir | Out-Null }
    # Generate with an EMPTY passphrase. Go through cmd /c so the empty-string
    # arg ( -N "" ) survives - PowerShell otherwise drops it and ssh-keygen
    # would block on an interactive passphrase prompt.
    cmd /c "ssh-keygen -t ed25519 -f `"$SshKey`" -N `"`" -q"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $SshKey)) { Abort "ssh-keygen failed." }
    OK "Generated"
}

# --- 3. Authorize key on the cluster ---------------------------------------
Step "Authorize key on the cluster"
& ssh -o BatchMode=yes -o ConnectTimeout=5 @SshOpts -i $SshKey "$ClusterUser@$ClusterHost" "true" 2>$null
if ($LASTEXITCODE -eq 0) {
    OK "Passwordless SSH already works - skipping copy"
} else {
    Say "Copying the public key to the cluster. You'll be prompted for your BGU"
    Say "cluster password ONCE. Nothing is stored."
    Write-Host ""
    # No ssh-copy-id on Windows - append the pubkey over a single ssh session.
    Get-Content "$SshKey.pub" -Raw | & ssh @SshOpts "$ClusterUser@$ClusterHost" `
        "umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys"
    if ($LASTEXITCODE -ne 0) { Abort "Key copy failed." }

    & ssh -o BatchMode=yes -o ConnectTimeout=5 @SshOpts -i $SshKey "$ClusterUser@$ClusterHost" "true" 2>$null
    if ($LASTEXITCODE -eq 0) { OK "Passwordless SSH confirmed" }
    else { Abort "Key copied but passwordless login still fails. Check ~/.ssh on the cluster." }
}

# --- 4. Push code (clean sync) ---------------------------------------------
Step "Pushing code to the cluster"
& ssh @SshOpts -i $SshKey "$ClusterUser@$ClusterHost" `
    "mkdir -p $ClusterCodeDir && rm -rf $ClusterCodeDir/cluster $ClusterCodeDir/compute_jobs $ClusterCodeDir/classifier_engine $ClusterCodeDir/utils"
if ($LASTEXITCODE -ne 0) { Abort "Failed to prepare $ClusterCodeDir on the cluster." }

foreach ($src in 'cluster', 'backend\compute_jobs', 'backend\classifier_engine', 'backend\utils') {
    $leaf = Split-Path $src -Leaf
    Say "Uploading $src -> $ClusterCodeDir/$leaf ..."
    & scp @SshOpts -i $SshKey -q -r (Join-Path $RepoDir $src) "$ClusterUser@${ClusterHost}:$ClusterCodeDir/"
    if ($LASTEXITCODE -ne 0) { Abort "scp of $src failed." }
}
OK "Code uploaded to $ClusterCodeDir (clean sync - no stale files left behind)"

# Strip CR from the uploaded scripts so the cluster's bash doesn't choke on a
# "\r" in the shebang (/bin/bash^M: bad interpreter) - Windows clones carry CRLF.
$normCmd = "find $ClusterCodeDir -type f \( -name '*.sh' -o -name '*.sbatch' \) -exec sed -i 's/\r`$//' {} + 2>/dev/null; true"
& ssh @SshOpts -i $SshKey "$ClusterUser@$ClusterHost" $normCmd
OK "Normalized script line endings to LF on the cluster"

# --- 5. Conda env on the cluster -------------------------------------------
Step "Conda environment on the cluster"
if ($recreate -eq '1') {
    Say "  - -RecreateEnv given: the env will be wiped and rebuilt cleanly."
} else {
    Say "  - an up-to-date env is left as-is (skips the slow reinstall);"
    Say "  - it's rebuilt automatically only if the env spec changed."
}
Write-Host ""
& ssh @SshOpts -i $SshKey -t "$ClusterUser@$ClusterHost" `
    "chmod +x $ClusterCodeDir/cluster/setup_cluster_env.sh && $ClusterCodeDir/cluster/setup_cluster_env.sh gavel $recreate"
if ($LASTEXITCODE -ne 0) { Abort "Conda env setup failed on the cluster. Check the output above." }
OK "Conda env ready"

# --- 6. Write SLURM config into backend\.env (the ONLY env file) -----------
Step "Backend configuration"

# Set KEY=value in an env file: replace the line in place if present, else
# append. Writes LF, UTF-8 without BOM (what python-dotenv / docker expect).
function Set-EnvKv($file, $key, $val) {
    $lines = @()
    if (Test-Path $file) { $lines = @([System.IO.File]::ReadAllLines($file)) }
    $found = $false
    $out = foreach ($line in $lines) {
        if ($line -match ('^' + [regex]::Escape($key) + '=')) { $found = $true; "$key=$val" }
        else { $line }
    }
    if (-not $found) { $out = @($out) + "$key=$val" }
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($file, (($out -join "`n") + "`n"), $enc)
}

# Ensure backend\.env exists (from its .example if present, else empty), then
# set/refresh the SLURM keys - leaving every other value untouched. Writes BOTH
# SLURM_SSH_KEY (native reads it directly) and SLURM_SSH_KEY_FILE (docker
# bind-mounts the host key) into the SAME file, so there is only one env file.
function Write-SlurmEnv($envPath, $keyVal) {
    if (-not (Test-Path $envPath)) {
        $dir = Split-Path $envPath -Parent
        if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        if (Test-Path "$envPath.example") { Copy-Item "$envPath.example" $envPath; OK "created $envPath (from .example)" }
        else { New-Item -ItemType File -Path $envPath | Out-Null; OK "created $envPath" }
    }
    Set-EnvKv $envPath 'SLURM_HOST' $ClusterHost
    Set-EnvKv $envPath 'SLURM_USER' $ClusterUser
    Set-EnvKv $envPath 'SLURM_SSH_KEY'      $keyVal   # native: read directly
    Set-EnvKv $envPath 'SLURM_SSH_KEY_FILE' $keyVal   # docker: bind-mounted in
    OK "wrote cluster config to $envPath (SLURM_HOST, SLURM_USER, SLURM_SSH_KEY, SLURM_SSH_KEY_FILE)"
}

# Use forward slashes for the key path in .env - Python and Docker Desktop both
# accept them on Windows, and they avoid backslash-escaping surprises.
$SshKeyForEnv = ($SshKey -replace '\\', '/')

Say "Writing cluster config to backend\.env - the single config file"
Say "(used by both native and Docker). No second repo-root .env is created."
Write-Host ""
Write-SlurmEnv (Join-Path $RepoDir 'backend\.env') $SshKeyForEnv

Step "Setup complete"
Write-Host ""
