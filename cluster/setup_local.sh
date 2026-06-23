#!/usr/bin/env bash
# Interactive cluster setup — runs on Mac, Linux, WSL, and Git Bash on Windows.
#
# Walks the user through:
#   1. Prereq check (ssh, scp, network reachability)
#   2. SSH key generation (skipped if one already exists)
#   3. Copying the public key to the cluster (asks for cluster password once)
#   4. Pushing cluster/ + backend/classifier_engine/ + backend/utils/ via scp
#   5. Running the cluster-side conda env setup
#   6. Writing the SLURM config into backend/.env — the SINGLE config file used
#      by both native (run.sh) and Docker — so cluster training works whichever
#      way the project is launched. Nothing is written to a second repo-root .env.
#
# Re-runnable & SAFE to re-run after we change this setup:
#   * the code push is a CLEAN sync — stale/renamed files from a previous run
#     are removed first, so the cluster never imports leftover old code;
#   * the conda env is versioned — an unchanged env is left as-is (the slow
#     pip install is skipped), and it's recreated cleanly only when the env
#     spec changed (or you pass --recreate-env). No package "drift"/collisions.
#
# Usage:
#   ./setup_local.sh                 # normal (idempotent) setup
#   ./setup_local.sh --recreate-env  # force a clean rebuild of the conda env

set -u

# --recreate-env / --fresh → wipe and rebuild the cluster conda env from scratch
# (use after an upgrade if you suspect a half-old / broken env).
RECREATE_ENV=0
for _arg in "$@"; do
    case "$_arg" in
        --recreate-env|--fresh) RECREATE_ENV=1 ;;
    esac
done

CLUSTER_HOST="slurm.bgu.ac.il"
SSH_KEY_DEFAULT="$HOME/.ssh/id_ed25519_cluster"

# Where the GAVEL code lives ON THE CLUSTER. Defined ONCE here so all four uses
# below stay in sync. This is the SAME path the backend reads as SLURM_CODE_DIR
# (its default is "~/gavel_code"), so the two agree by that shared default —
# nothing extra goes in .env. If you ever change this, change the backend's
# SLURM_CODE_DIR default to match (services/cluster_direct.py).
CLUSTER_CODE_DIR="~/gavel_code"

c_red()    { printf '\033[31m%s\033[0m' "$1"; }
c_green()  { printf '\033[32m%s\033[0m' "$1"; }
c_yellow() { printf '\033[33m%s\033[0m' "$1"; }
c_blue()   { printf '\033[34m%s\033[0m' "$1"; }
c_bold()   { printf '\033[1m%s\033[0m' "$1"; }

say()  { printf '%s\n' "$*"; }
ok()   { printf '%s %s\n' "$(c_green '✓')" "$*"; }
warn() { printf '%s %s\n' "$(c_yellow '!')" "$*"; }
err()  { printf '%s %s\n' "$(c_red '✗')" "$*" >&2; }
step() { printf '\n%s %s\n' "$(c_blue '▶')" "$(c_bold "$*")"; }

ask() {
    # ask "prompt" [default]
    local prompt="$1"
    local default="${2-}"
    local reply
    if [[ -n "$default" ]]; then
        read -r -p "$prompt [$default]: " reply
        printf '%s' "${reply:-$default}"
    else
        read -r -p "$prompt: " reply
        printf '%s' "$reply"
    fi
}

confirm() {
    # confirm "question" — returns 0 for yes, 1 for no. Default no.
    local reply
    read -r -p "$1 (y/N) " reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

abort() {
    err "$1"
    exit 1
}

# ---------------------------------------------------------------------------
# 0. Banner + prereq check
# ---------------------------------------------------------------------------
printf '\n'
c_bold "GAVEL cluster setup"
printf '\n\n'
say "This script will configure your machine and the BGU SLURM cluster"
say "so the GAVEL backend can submit training jobs over SSH."
printf '\n'

step "Checking prerequisites"

for cmd in ssh scp ssh-keygen ssh-copy-id; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        if [[ "$cmd" == "ssh-copy-id" ]]; then
            warn "ssh-copy-id not found — we'll fall back to a manual copy via cat | ssh"
            HAVE_SSH_COPY_ID=0
        else
            abort "$cmd is not installed. Install OpenSSH and re-run this script."
        fi
    fi
done
HAVE_SSH_COPY_ID=${HAVE_SSH_COPY_ID:-1}

if command -v nc >/dev/null 2>&1; then
    if nc -z -w 5 "$CLUSTER_HOST" 22 >/dev/null 2>&1; then
        ok "$CLUSTER_HOST:22 is reachable"
    else
        abort "Cannot reach $CLUSTER_HOST:22. Is your BGU VPN active?"
    fi
else
    warn "nc not available — skipping reachability check. Make sure your BGU VPN is on."
fi

# ---------------------------------------------------------------------------
# 1. Collect inputs
# ---------------------------------------------------------------------------
step "Configuration"

say "We need three things to continue. For any prompt that shows a value"
say "in [brackets], just press Enter to accept the default."
printf '\n'

say "$(c_bold '1) Cluster username')"
say "   The login name you use on slurm.bgu.ac.il — the same one"
say "   you'd type for: ssh <name>@slurm.bgu.ac.il"
CLUSTER_USER="$(ask "   Cluster username")"
[[ -z "$CLUSTER_USER" ]] && abort "Cluster username is required."
printf '\n'

DEFAULT_REPO="$(pwd)"
if [[ ! -d "$DEFAULT_REPO/cluster" || ! -d "$DEFAULT_REPO/backend/classifier_engine" ]]; then
    # Try the parent of this script — useful if user runs the script from inside cluster/
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -d "$SCRIPT_DIR/../cluster" && -d "$SCRIPT_DIR/../backend/classifier_engine" ]]; then
        DEFAULT_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
    fi
fi
say "$(c_bold '2) Path to this repo on your machine')"
say "   The folder that contains 'cluster/' and 'backend/'."
say "   We auto-detected the path below — if it's correct, just press Enter."
say "   If it's wrong, type the correct path and then press Enter."
REPO_DIR="$(ask "   Repo root" "$DEFAULT_REPO")"
REPO_DIR="${REPO_DIR/#\~/$HOME}"
[[ -d "$REPO_DIR/cluster" && -d "$REPO_DIR/backend/classifier_engine" ]] \
    || abort "Repo not found at $REPO_DIR (expected cluster/ and backend/classifier_engine/ inside)"
printf '\n'

say "$(c_bold '3) SSH key path')"
say "   Where to store (or read) the dedicated cluster SSH key."
say "   The default is fine for almost everyone — just press Enter."
SSH_KEY="$(ask "   SSH key path" "$SSH_KEY_DEFAULT")"
SSH_KEY="${SSH_KEY/#\~/$HOME}"

printf '\n'
say "Summary:"
say "  Cluster user : $(c_bold "$CLUSTER_USER@$CLUSTER_HOST")"
say "  Repo root    : $(c_bold "$REPO_DIR")"
say "  SSH key      : $(c_bold "$SSH_KEY")"
printf '\n'
confirm "Continue with this configuration?" || abort "Aborted."

# ---------------------------------------------------------------------------
# 2. SSH key
# ---------------------------------------------------------------------------
step "SSH key"

if [[ -f "$SSH_KEY" ]]; then
    ok "Key already exists at $SSH_KEY — reusing it"
else
    say "Generating a new ed25519 key at $SSH_KEY"
    mkdir -p "$(dirname "$SSH_KEY")"
    chmod 700 "$(dirname "$SSH_KEY")"
    ssh-keygen -t ed25519 -f "$SSH_KEY" -N '' >/dev/null
    ok "Generated"
fi

# ---------------------------------------------------------------------------
# 3. Copy public key to cluster
# ---------------------------------------------------------------------------
step "Authorize key on the cluster"

# Already authorized? Test silently with BatchMode (no prompts).
if ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
       -i "$SSH_KEY" "$CLUSTER_USER@$CLUSTER_HOST" "true" 2>/dev/null; then
    ok "Passwordless SSH already works — skipping copy"
else
    say "Need to copy the public key to the cluster. You will be prompted for"
    say "your BGU cluster password $(c_bold 'once'). Nothing is stored."
    printf '\n'
    if [[ "$HAVE_SSH_COPY_ID" == "1" ]]; then
        ssh-copy-id -i "$SSH_KEY.pub" -o StrictHostKeyChecking=no \
                    "$CLUSTER_USER@$CLUSTER_HOST" \
            || abort "ssh-copy-id failed."
    else
        cat "$SSH_KEY.pub" | ssh -o StrictHostKeyChecking=no \
            "$CLUSTER_USER@$CLUSTER_HOST" \
            "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys" \
            || abort "Manual key copy failed."
    fi

    # Re-verify.
    if ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
           -i "$SSH_KEY" "$CLUSTER_USER@$CLUSTER_HOST" "true" 2>/dev/null; then
        ok "Passwordless SSH confirmed"
    else
        abort "Key copied but passwordless login still fails. Check ~/.ssh on the cluster."
    fi
fi

# ---------------------------------------------------------------------------
# 4. Push code
# ---------------------------------------------------------------------------
step "Pushing code to the cluster"

# CLEAN sync: remove the three dirs WE manage before re-uploading, so files
# renamed/deleted in the repo don't linger on the cluster and get imported.
# Only these managed subdirs are touched — ~/gavel_code itself (and anything
# else in it, e.g. job outputs) is left alone.
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$CLUSTER_USER@$CLUSTER_HOST" \
    "mkdir -p $CLUSTER_CODE_DIR && rm -rf $CLUSTER_CODE_DIR/cluster $CLUSTER_CODE_DIR/compute_jobs $CLUSTER_CODE_DIR/classifier_engine $CLUSTER_CODE_DIR/utils" \
    || abort "Failed to prepare $CLUSTER_CODE_DIR on the cluster."

for src in "cluster" "backend/compute_jobs" "backend/classifier_engine" "backend/utils"; do
    say "Uploading $src -> $CLUSTER_CODE_DIR/$(basename "$src") ..."
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -q -r \
        "$REPO_DIR/$src" "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_CODE_DIR/" \
        || abort "scp of $src failed."
done
ok "Code uploaded to $CLUSTER_CODE_DIR (clean sync — no stale files left behind)"

# Normalize line endings on the cluster. A Windows clone may carry CRLF, and
# the cluster's bash rejects a "\r" in a shebang (/bin/bash^M: bad interpreter),
# which breaks setup_cluster_env.sh and the .sbatch job scripts. Strip CR from
# every shell + sbatch script we just uploaded so they run regardless of the
# user's OS / git autocrlf setting.
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$CLUSTER_USER@$CLUSTER_HOST" \
    "find $CLUSTER_CODE_DIR -type f \( -name '*.sh' -o -name '*.sbatch' \) -exec sed -i 's/\r\$//' {} + 2>/dev/null; true" \
    || warn "Could not normalize line endings (continuing) — re-run if the next step fails."
ok "Normalized script line endings to LF on the cluster"

# ---------------------------------------------------------------------------
# 5. Conda env on the cluster
# ---------------------------------------------------------------------------
step "Conda environment on the cluster"

say "Creating an isolated 'gavel' conda environment on the cluster and"
say "installing PyTorch and the other Python packages the trainer needs."
say "This is fully sandboxed — it doesn't touch anything else on your"
say "cluster account, and it's safe to re-run:"
if [[ "$RECREATE_ENV" == "1" ]]; then
    say "  • --recreate-env given → the env will be wiped and rebuilt cleanly."
else
    say "  • an up-to-date env is left as-is (skips the slow reinstall);"
    say "  • it's rebuilt automatically only if the env spec changed."
fi
printf '\n'

# Pass the env name + recreate flag. The cluster script keeps a version marker
# so it can skip an unchanged env and rebuild cleanly when the spec changes.
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -t \
    "$CLUSTER_USER@$CLUSTER_HOST" \
    "chmod +x $CLUSTER_CODE_DIR/cluster/setup_cluster_env.sh && $CLUSTER_CODE_DIR/cluster/setup_cluster_env.sh gavel $RECREATE_ENV" \
    || abort "Conda env setup failed on the cluster. Check the output above."
ok "Conda env ready"

# ---------------------------------------------------------------------------
# 6. Done — write the cluster config into backend/.env (the ONLY env file)
# ---------------------------------------------------------------------------
step "Backend configuration"

# ONE config file: backend/.env — used by BOTH run modes:
#   * native (run.sh + uvicorn) reads SLURM_SSH_KEY (the key path) directly;
#   * docker reads SLURM_SSH_KEY_FILE (the host key path) to bind-mount the key
#     into the container — via `docker compose --env-file backend/.env up`.
# Both hold the SAME host path, so we just write both into backend/.env. Nothing
# goes into a separate repo-root .env anymore.
# Set KEY=value in an env file: replace the line in place if present, else
# append. Uses awk so values containing / : @ ~ (paths, URLs) need no escaping.
_set_kv() {
    local file="$1" key="$2" val="$3"
    if grep -qE "^${key}=" "$file" 2>/dev/null; then
        awk -v k="$key" -v v="$val" '
            $0 ~ "^"k"=" { print k"="v; found=1; next }
            { print }
            END { if (!found) print k"="v }
        ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    else
        printf '%s=%s\n' "$key" "$val" >> "$file"
    fi
}

# Ensure the env file EXISTS, then set/refresh the SLURM keys in it. Creates
# the file when missing — from its sibling .example when present, otherwise a
# fresh file — and UPDATES existing values in place, so re-running the script
# always lands the latest cluster config rather than leaving stale lines.
_write_slurm_env() {           # $1 = env file (backend/.env)
    local env_path="$1"
    if [[ ! -f "$env_path" ]]; then
        mkdir -p "$(dirname "$env_path")"
        if [[ -f "${env_path}.example" ]]; then
            cp "${env_path}.example" "$env_path"
            ok "created $env_path (from ${env_path##*/}.example)"
        else
            : > "$env_path"
            ok "created $env_path"
        fi
    fi
    _set_kv "$env_path" SLURM_HOST "$CLUSTER_HOST"
    _set_kv "$env_path" SLURM_USER "$CLUSTER_USER"
    _set_kv "$env_path" SLURM_SSH_KEY      "$SSH_KEY"   # native: read directly
    _set_kv "$env_path" SLURM_SSH_KEY_FILE "$SSH_KEY"   # docker: bind-mounted in
    ok "wrote cluster config to $env_path (SLURM_HOST, SLURM_USER, SLURM_SSH_KEY, SLURM_SSH_KEY_FILE)"
}

say "Writing cluster config to $(c_bold 'backend/.env') — the single config file"
say "(used by both native and Docker). No second repo-root .env is created."
printf '\n'
_write_slurm_env "$REPO_DIR/backend/.env"

step "Setup complete"
printf '\n'
