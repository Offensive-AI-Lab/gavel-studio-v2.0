"""Backfill ratings.creator_username for EXISTING rows (run on the central server).

The creator_username column was added so that un-rating an asset decrements
exactly the creator that rating incremented. Rows created before the column exists
have it NULL; this fills them.

Ground truth (per ofek): every published rule/CE was created by 'gavel' EXCEPT:
    rule : medical_dosage_prescription      -> ofek
    ce   : substance_dosage_directive       -> ofek
    ce   : healthcare                        -> ofek

ratings store OPAQUE uuid public_ids (rule_<hex> / ce_<hex>), not names, so we
resolve name -> public_id from the HF manifest (rule_names / ce_names sections).

Run on the central server (its venv + env: HF_TOKEN, HF_REPO_ID, DATABASE_URL):
    python backfill_rating_creators.py            # apply
    python backfill_rating_creators.py --dry-run  # preview only

Idempotent: safe to re-run. Only NULL creator_username rows are defaulted to
'gavel'; the three ofek assets are then flipped to 'ofek' by their public_id.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Make `app.*` importable when run from the central-server/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load central-server/.env so DATABASE_URL / HF_TOKEN are available even when run
# from a plain shell (best-effort; harmless if python-dotenv isn't installed).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

GAVEL = "gavel"
OFEK = "ofek"
OFEK_RULES = ["medical_dosage_prescription"]
OFEK_CES = ["substance_dosage_directive", "healthcare"]


def _load_manifest() -> dict:
    """Download + parse manifest.json from the shared HF dataset repo."""
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_REPO_ID", "GavelPublicData/public-library")
    repo_type = os.getenv("HF_REPO_TYPE", "dataset")
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo_id=repo_id, filename="manifest.json",
                            repo_type=repo_type, token=token)
    with open(local, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Resolve + report, write nothing.")
    args = ap.parse_args()

    from app.utils.db import execute, execute_dict

    manifest = _load_manifest()
    rule_names = manifest.get("rule_names", {}) or {}   # {name: public_id}
    ce_names = manifest.get("ce_names", {}) or {}

    # Resolve the ofek-created assets to their public_ids.
    ofek_targets = []   # (asset_type, public_id, name)
    for name in OFEK_RULES:
        pid = rule_names.get(name)
        print(f"rule  {name:<32} -> {pid or 'NOT FOUND in manifest (skipped)'}")
        if pid:
            ofek_targets.append(("rule", pid, name))
    for name in OFEK_CES:
        pid = ce_names.get(name)
        print(f"ce    {name:<32} -> {pid or 'NOT FOUND in manifest (skipped)'}")
        if pid:
            ofek_targets.append(("ce", pid, name))

    total = execute_dict("SELECT COUNT(*) AS n FROM ratings")[0]["n"]
    nulls = execute_dict("SELECT COUNT(*) AS n FROM ratings WHERE creator_username IS NULL")[0]["n"]
    print(f"\nratings rows: total={total}, with NULL creator_username={nulls}")

    if args.dry_run:
        print("\n[dry-run] would:")
        print(f"  1) set creator_username='gavel' on the {nulls} NULL row(s)")
        for atype, pid, name in ofek_targets:
            c = execute_dict(
                "SELECT COUNT(*) AS n FROM ratings WHERE asset_type=%s AND asset_public_id=%s",
                (atype, pid),
            )[0]["n"]
            print(f"  2) set creator_username='ofek' on {c} rating(s) of {atype} '{name}'")
        return

    # 1) Default every still-unattributed (pre-column) row to gavel.
    execute("UPDATE ratings SET creator_username = %s WHERE creator_username IS NULL", (GAVEL,))
    # 2) Flip the ofek-created assets (always ofek — authoritative by public_id).
    flipped = 0
    for atype, pid, name in ofek_targets:
        execute(
            "UPDATE ratings SET creator_username = %s WHERE asset_type = %s AND asset_public_id = %s",
            (OFEK, atype, pid),
        )
        c = execute_dict(
            "SELECT COUNT(*) AS n FROM ratings WHERE asset_type=%s AND asset_public_id=%s",
            (atype, pid),
        )[0]["n"]
        flipped += c
        print(f"set creator='ofek' on {atype} '{name}': {c} rating(s)")

    remaining = execute_dict(
        "SELECT COUNT(*) AS n FROM ratings WHERE creator_username IS NULL")[0]["n"]
    print(f"\nDone. NULLs filled with 'gavel', {flipped} row(s) flipped to 'ofek'. "
          f"Remaining NULL creator_username: {remaining} (should be 0).")


if __name__ == "__main__":
    main()
