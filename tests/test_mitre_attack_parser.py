"""MITRE ATT&CK parser invariants."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import scripts.parsers.mitre_attack as mitre_attack
from scripts.parsers.mitre_attack import _append_tactic_technique_list, _techniques_by_phase, parse

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_tactic_docs_include_techniques_from_kill_chain_phase() -> None:
    objects = [
        {
            "type": "attack-pattern",
            "name": "Exfiltration Over C2 Channel",
            "external_references": [
                {"source_name": "mitre-attack", "external_id": "T1041"},
            ],
            "kill_chain_phases": [{"phase_name": "exfiltration"}],
        },
        {
            "type": "attack-pattern",
            "name": "Scheduled Transfer",
            "external_references": [
                {"source_name": "mitre-attack", "external_id": "T1029"},
            ],
            "kill_chain_phases": [{"phase_name": "exfiltration"}],
        },
    ]
    tactic = {
        "type": "x-mitre-tactic",
        "name": "Exfiltration",
        "x_mitre_shortname": "exfiltration",
        "external_references": [
            {"source_name": "mitre-attack", "external_id": "TA0010"},
        ],
    }

    rendered = _append_tactic_technique_list(
        "The adversary is trying to steal data.",
        tactic,
        _techniques_by_phase(objects),
    )

    assert "Techniques in Exfiltration (TA0010):" in rendered
    assert "T1029: Scheduled Transfer" in rendered
    assert "T1041: Exfiltration Over C2 Channel" in rendered


def test_parse_exfiltration_tactic_lists_related_techniques(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "enterprise-attack.json"
    bundle.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "type": "x-mitre-tactic",
                        "name": "Exfiltration",
                        "description": "The adversary is trying to steal data.",
                        "x_mitre_shortname": "exfiltration",
                        "external_references": [
                            {"source_name": "mitre-attack", "external_id": "TA0010"},
                        ],
                    },
                    {
                        "type": "attack-pattern",
                        "name": "Exfiltration Over C2 Channel",
                        "description": "Exfiltrate data over an existing C2 channel.",
                        "external_references": [
                            {"source_name": "mitre-attack", "external_id": "T1041"},
                        ],
                        "kill_chain_phases": [{"phase_name": "exfiltration"}],
                    },
                    {
                        "type": "attack-pattern",
                        "name": "Exfiltration to Cloud Storage",
                        "description": "Exfiltrate data to cloud storage.",
                        "external_references": [
                            {"source_name": "mitre-attack", "external_id": "T1567.002"},
                        ],
                        "kill_chain_phases": [{"phase_name": "exfiltration"}],
                    },
                ],
            }
        )
    )
    monkeypatch.setattr(mitre_attack, "BUNDLE", bundle)

    docs = {doc.doc_id: doc for doc in parse()}

    text = docs["mitre_attack:TA0010"].text

    assert "Techniques in Exfiltration (TA0010):" in text
    assert "T1041: Exfiltration Over C2 Channel" in text
    assert "T1567.002: Exfiltration to Cloud Storage" in text
