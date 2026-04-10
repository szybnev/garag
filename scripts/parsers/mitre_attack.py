"""Parse the MITRE ATT&CK Enterprise STIX bundle into `Document`s.

Pulls the five content-bearing object types: attack-pattern (techniques),
x-mitre-tactic, intrusion-set (groups), malware + tool (software),
course-of-action (mitigations). Skips relationships, identities, markings,
and any deprecated objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas import Document

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "mitre_attack"
BUNDLE = RAW / "enterprise-attack.json"

ATTACK_SITE = "https://attack.mitre.org"

_URL_PATHS = {
    "attack-pattern": "techniques",
    "x-mitre-tactic": "tactics",
    "intrusion-set": "groups",
    "malware": "software",
    "tool": "software",
    "course-of-action": "mitigations",
}


def _external_id(obj: dict[str, Any]) -> str | None:
    for ref in obj.get("external_references") or []:
        if ref.get("source_name") in ("mitre-attack", "mitre-atlas") and ref.get("external_id"):
            return str(ref["external_id"])
    return None


def _canonical_url(obj_type: str, ext_id: str) -> str:
    path = _URL_PATHS.get(obj_type)
    if path is None:
        return ATTACK_SITE
    if obj_type == "attack-pattern" and "." in ext_id:
        # sub-technique T1059.001 → /techniques/T1059/001/
        parent, sub = ext_id.split(".", 1)
        return f"{ATTACK_SITE}/{path}/{parent}/{sub}/"
    return f"{ATTACK_SITE}/{path}/{ext_id}/"


def _compose_text(obj: dict[str, Any]) -> str:
    parts: list[str] = []
    desc = obj.get("description")
    if desc:
        parts.append(desc.strip())

    # Enrich techniques with detection / platforms / data sources
    if obj.get("type") == "attack-pattern":
        platforms = obj.get("x_mitre_platforms") or []
        if platforms:
            parts.append("Platforms: " + ", ".join(platforms))
        data_sources = obj.get("x_mitre_data_sources") or []
        if data_sources:
            parts.append("Data sources: " + ", ".join(data_sources))
        detection = obj.get("x_mitre_detection")
        if detection:
            parts.append("Detection:\n" + detection.strip())

    # Enrich groups with aliases
    if obj.get("type") == "intrusion-set":
        aliases = obj.get("aliases") or obj.get("x_mitre_aliases") or []
        if aliases:
            parts.append("Aliases: " + ", ".join(aliases))

    return "\n\n".join(parts).strip()


def _metadata(obj: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {"stix_type": obj["type"]}
    for key in (
        "x_mitre_platforms",
        "x_mitre_is_subtechnique",
        "x_mitre_version",
        "x_mitre_domains",
        "x_mitre_data_sources",
        "aliases",
    ):
        if obj.get(key) is not None:
            meta[key.removeprefix("x_mitre_")] = obj[key]

    phases = obj.get("kill_chain_phases") or []
    if phases:
        meta["kill_chain_phases"] = [p.get("phase_name") for p in phases if p.get("phase_name")]

    if obj.get("x_mitre_deprecated") or obj.get("revoked"):
        meta["deprecated"] = True
    return meta


def parse() -> list[Document]:
    if not BUNDLE.exists():
        raise FileNotFoundError(
            f"{BUNDLE} not found — run `uv run python -m scripts.fetch_mitre_attack` first"
        )

    with BUNDLE.open() as fh:
        bundle = json.load(fh)

    docs: list[Document] = []
    seen_ids: set[str] = set()

    for obj in bundle.get("objects", []):
        obj_type = obj.get("type")
        if obj_type not in _URL_PATHS:
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        ext_id = _external_id(obj)
        if not ext_id:
            continue

        text = _compose_text(obj)
        if not text:
            continue

        title = obj.get("name")
        if not title:
            continue

        doc_id = f"mitre_attack:{ext_id}"
        if doc_id in seen_ids:
            # duplicate across minor STIX objects — keep first
            continue
        seen_ids.add(doc_id)

        docs.append(
            Document(
                doc_id=doc_id,
                source="mitre_attack",
                url=_canonical_url(obj_type, ext_id),
                title=title,
                text=text,
                metadata=_metadata(obj),
            )
        )

    return docs


if __name__ == "__main__":
    docs = parse()
    print(f"[mitre_attack] {len(docs)} documents")
    if docs:
        print(f"  sample: {docs[0].doc_id} — {docs[0].title}")
        print(f"  url:    {docs[0].url}")
