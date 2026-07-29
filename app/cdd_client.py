"""
app/cdd_client.py
=================
Minimal CDD Vault REST API client — look up a compound's SMILES by GEN-ID.

Uses the documented Molecules endpoint:
    GET https://app.collaborativedrug.com/api/v1/vaults/<vault_id>/molecules?names=<GEN-ID>
    header: X-CDD-Token: <token>
The matching molecule object's "smiles" field is returned.

SECURITY: the API token is a secret supplied by the caller (entered in the app
session). It is sent only to CDD over HTTPS and is never logged or persisted here.
"""

from __future__ import annotations

import requests

CDD_BASE = "https://app.collaborativedrug.com/api/v1"
DEFAULT_TIMEOUT = 30


class CDDError(Exception):
    """Raised for any CDD lookup failure, with a user-facing message."""


def fetch_smiles_by_name(gen_id: str, vault_id: str, token: str, *,
                         base_url: str = CDD_BASE,
                         timeout: int = DEFAULT_TIMEOUT) -> tuple[str, str]:
    """
    Look up a molecule by name/synonym (the GEN-ID) and return (smiles, matched_name).

    Raises CDDError with a clear, user-facing message on any failure.
    """
    gen_id = (gen_id or "").strip()
    vault_id = str(vault_id or "").strip()
    token = (token or "").strip()
    if not gen_id:
        raise CDDError("Enter a GEN-ID to look up.")
    if not vault_id or not token:
        raise CDDError("Connect to CDD first (Vault ID + API token required).")

    url = f"{base_url}/vaults/{vault_id}/molecules"
    try:
        r = requests.get(
            url,
            headers={"X-CDD-Token": token},
            params={"names": gen_id, "no_structures": "false"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise CDDError(f"Could not reach CDD Vault: {exc}") from exc

    if r.status_code in (401, 403):
        raise CDDError("CDD rejected the credentials — check the Vault ID and API token.")
    if r.status_code == 404:
        raise CDDError(f"Vault {vault_id} not found — check the Vault ID.")
    if r.status_code != 200:
        raise CDDError(f"CDD returned HTTP {r.status_code}: {r.text[:200]}")

    try:
        data = r.json()
    except ValueError as exc:
        raise CDDError("CDD response was not valid JSON.") from exc

    objs = data.get("objects")
    if objs is None:
        objs = [data] if data.get("smiles") else []
    if not objs:
        raise CDDError(f"No molecule named '{gen_id}' found in vault {vault_id}.")

    # Prefer an exact name/synonym match; otherwise take the first hit.
    best = None
    for o in objs:
        names = [o.get("name", "")] + list(o.get("synonyms", []) or [])
        if any(n and n.strip().lower() == gen_id.lower() for n in names):
            best = o
            break
    best = best or objs[0]

    smiles = best.get("smiles") or best.get("cxsmiles")
    if not smiles:
        raise CDDError(f"'{gen_id}' was found but has no SMILES structure in CDD.")
    return smiles, best.get("name", gen_id)
