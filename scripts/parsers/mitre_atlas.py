"""Parse the MITRE ATLAS compiled YAML into `Document`s.

ATLAS publishes a single `dist/ATLAS.yaml` with a matrix containing tactics,
techniques, mitigations, plus top-level case-studies. We turn each of those
into a separate Document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.schemas import Document

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "mitre_atlas"
BUNDLE = RAW / "ATLAS.yaml"

ATLAS_SITE = "https://atlas.mitre.org"


def _url(object_type: str, obj_id: str) -> str:
    if object_type == "tactic":
        return f"{ATLAS_SITE}/tactics/{obj_id}"
    if object_type == "technique":
        return f"{ATLAS_SITE}/techniques/{obj_id}"
    if object_type == "mitigation":
        return f"{ATLAS_SITE}/mitigations/{obj_id}"
    if object_type == "case-study":
        return f"{ATLAS_SITE}/studies/{obj_id}"
    return ATLAS_SITE


def _technique_text(obj: dict[str, Any]) -> str:
    parts: list[str] = []
    if obj.get("description"):
        parts.append(str(obj["description"]).strip())
    if obj.get("tactics"):
        parts.append("Tactics: " + ", ".join(obj["tactics"]))
    if obj.get("subtechnique-of"):
        parts.append(f"Sub-technique of: {obj['subtechnique-of']}")
    if obj.get("ATT&CK-reference"):
        ref = obj["ATT&CK-reference"]
        if isinstance(ref, dict) and ref.get("id"):
            parts.append(f"ATT&CK reference: {ref['id']}")
        elif isinstance(ref, str):
            parts.append(f"ATT&CK reference: {ref}")
    return "\n\n".join(parts).strip()


def _mitigation_text(obj: dict[str, Any]) -> str:
    parts: list[str] = []
    if obj.get("description"):
        parts.append(str(obj["description"]).strip())
    if obj.get("techniques"):
        # list of {id, use} dicts
        tech_lines: list[str] = []
        for t in obj["techniques"]:
            if isinstance(t, dict) and t.get("id"):
                use = (t.get("use") or "").strip()
                tech_lines.append(f"- {t['id']}: {use}" if use else f"- {t['id']}")
            elif isinstance(t, str):
                tech_lines.append(f"- {t}")
        if tech_lines:
            parts.append("Addresses techniques:\n" + "\n".join(tech_lines))
    if obj.get("category"):
        parts.append(f"Category: {obj['category']}")
    return "\n\n".join(parts).strip()


def _case_study_text(obj: dict[str, Any]) -> str:
    parts: list[str] = []
    if obj.get("summary"):
        parts.append(str(obj["summary"]).strip())
    if obj.get("target"):
        parts.append(f"Target: {obj['target']}")
    if obj.get("actor"):
        parts.append(f"Actor: {obj['actor']}")
    if obj.get("procedure"):
        proc = obj["procedure"]
        if isinstance(proc, list):
            steps = []
            for i, step in enumerate(proc, 1):
                if isinstance(step, dict):
                    tid = step.get("technique", "")
                    desc = step.get("description", "")
                    steps.append(f"{i}. [{tid}] {desc}".strip())
                else:
                    steps.append(f"{i}. {step}")
            if steps:
                parts.append("Procedure:\n" + "\n".join(steps))
    return "\n\n".join(parts).strip()


def parse() -> list[Document]:
    if not BUNDLE.exists():
        raise FileNotFoundError(
            f"{BUNDLE} not found — run `uv run python -m scripts.fetch_mitre_atlas` first"
        )

    with BUNDLE.open() as fh:
        data = yaml.safe_load(fh)

    docs: list[Document] = []
    matrix = (data.get("matrices") or [{}])[0]

    for tactic in matrix.get("tactics", []) or []:
        text = str(tactic.get("description", "")).strip()
        if not text or not tactic.get("name") or not tactic.get("id"):
            continue
        docs.append(
            Document(
                doc_id=f"mitre_atlas:{tactic['id']}",
                source="mitre_atlas",
                url=_url("tactic", tactic["id"]),
                title=tactic["name"],
                text=text,
                metadata={"object_type": "tactic"},
            )
        )

    for tech in matrix.get("techniques", []) or []:
        text = _technique_text(tech)
        if not text or not tech.get("name") or not tech.get("id"):
            continue
        docs.append(
            Document(
                doc_id=f"mitre_atlas:{tech['id']}",
                source="mitre_atlas",
                url=_url("technique", tech["id"]),
                title=tech["name"],
                text=text,
                metadata={
                    "object_type": "technique",
                    "tactics": tech.get("tactics") or [],
                    "subtechnique_of": tech.get("subtechnique-of"),
                    "maturity": tech.get("maturity"),
                },
            )
        )

    for mit in matrix.get("mitigations", []) or []:
        text = _mitigation_text(mit)
        if not text or not mit.get("name") or not mit.get("id"):
            continue
        docs.append(
            Document(
                doc_id=f"mitre_atlas:{mit['id']}",
                source="mitre_atlas",
                url=_url("mitigation", mit["id"]),
                title=mit["name"],
                text=text,
                metadata={
                    "object_type": "mitigation",
                    "category": mit.get("category"),
                    "ml_lifecycle": mit.get("ml-lifecycle"),
                },
            )
        )

    for study in data.get("case-studies", []) or []:
        text = _case_study_text(study)
        if not text or not study.get("name") or not study.get("id"):
            continue
        docs.append(
            Document(
                doc_id=f"mitre_atlas:{study['id']}",
                source="mitre_atlas",
                url=_url("case-study", study["id"]),
                title=study["name"],
                text=text,
                metadata={
                    "object_type": "case-study",
                    "incident_date": str(study.get("incident-date") or ""),
                    "case_study_type": study.get("case-study-type"),
                },
            )
        )

    return docs


if __name__ == "__main__":
    docs = parse()
    print(f"[mitre_atlas] {len(docs)} documents")
    by_type: dict[str, int] = {}
    for d in docs:
        key = d.metadata.get("object_type", "?")
        by_type[key] = by_type.get(key, 0) + 1
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {v}")
    if docs:
        print(f"  sample: {docs[0].doc_id} — {docs[0].title}")
