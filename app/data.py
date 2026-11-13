"""Read-only access to the fixture data.

Every claim lookup takes the verified party id, so there is no call that
returns one policyholder's claim to a session verified as another.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any


class Repository:
    def __init__(self, fixtures_dir: Path) -> None:
        self._policyholders: list[dict[str, Any]] = _load(fixtures_dir / "policyholders.json")
        self._claims: list[dict[str, Any]] = _load(fixtures_dir / "claims.json")
        self._representatives: list[dict[str, Any]] = _load(fixtures_dir / "representatives.json")
        self._guidelines: dict[str, Any] = _load(fixtures_dir / "required_document_guideline.json")
        self._consent: dict[str, Any] = _load(fixtures_dir / "consent_scenarios.json")
        self._schema: dict[str, Any] = _load(fixtures_dir / "claim_schema.json")

    # -- policyholders -------------------------------------------------------

    def policyholders(self) -> list[dict[str, Any]]:
        return self._policyholders

    def policyholder(self, party_id: str) -> dict[str, Any]:
        for record in self._policyholders:
            if record["party_id"] == party_id:
                return record
        raise KeyError(party_id)

    def representatives(self) -> list[dict[str, Any]]:
        return self._representatives

    def authorised_representative(self, party_id: str, rep_name: str) -> dict[str, Any] | None:
        wanted = _name_key(rep_name)
        for rep in self._representatives:
            if rep["buyer_party_id"] == party_id and _name_key(rep["rep_name"]) == wanted:
                return rep
        return None

    # -- claims --------------------------------------------------------------

    def claims_for(self, party_id: str) -> list[dict[str, Any]]:
        return [c for c in self._claims if c["party_id"] == party_id]

    def claim(self, party_id: str, case_id: str) -> dict[str, Any] | None:
        wanted = normalise_case_id(case_id)
        for claim in self.claims_for(party_id):
            if claim["case_id"] == wanted:
                return claim
        return None

    def claims_index(self, party_id: str) -> list[dict[str, Any]]:
        return [
            {
                "case_id": c["case_id"],
                "case_type": c["case_type"],
                "filed_on": c["created_at"],
                "status": c["status"],
                "summary": c["summary"],
            }
            for c in self.claims_for(party_id)
        ]

    def case_file(self, party_id: str, case_id: str, today: date) -> dict[str, Any] | None:
        """The claim record plus facts the harness derives so the model does no date arithmetic."""
        claim = self.claim(party_id, case_id)
        if claim is None:
            return None
        record = {k: v for k, v in claim.items() if k != "party_id"}
        record["filed_on"] = record.pop("created_at")
        deadline = claim.get("appeal_deadline")
        if deadline:
            days = (date.fromisoformat(deadline) - today).days
            record["days_until_appeal_deadline"] = days
            record["appeal_deadline_has_passed"] = days < 0
        return record

    def field_glossary(self) -> dict[str, str]:
        return {name: spec["description"] for name, spec in self._schema["field_descriptions"].items()}

    # -- guidance ------------------------------------------------------------

    def document_guidance(self, claim: dict[str, Any], document: str) -> dict[str, Any]:
        g = self._guidelines
        key = _match_document(document, list(g["document_guidance"]))
        result: dict[str, Any] = {
            "document": document,
            "general_guidance": g["default_guidance"]["en"],
        }
        case_type_note = g["case_type_guidance"].get(claim["case_type"])
        if case_type_note:
            result["claim_type_guidance"] = case_type_note["en"]
        if key is not None:
            result["matched_guideline"] = key
            result["requirements"] = g["document_guidance"][key]["en"]
            result["if_unavailable"] = g["document_alternative_guidance"][key]["en"]
        else:
            result["requirements"] = "There is no document-specific guideline for this item; the general guidance applies."
            result["if_unavailable"] = g["document_alternative_guidance"]["default"]["en"]
        result["if_no_alternative_exists"] = g["claim_followup_settings"][
            "human_review_after_document_alternatives_exhausted"
        ]["en"]
        return result

    def followup_guidance(self, claim: dict[str, Any], topic: str) -> dict[str, Any]:
        g = self._guidelines
        documents = claim.get("documents_needed") or []
        entry = next((e for e in g["claim_followup_guidance"] if e["topic"] == topic), None)
        if entry is None or (entry.get("requires_documents") and not documents):
            return {
                "topic": topic,
                "applies": False,
                "guidance": g["claim_followup_fallback"]["en"],
                "note": "No claim-specific rule covers this; the text above is the general expectation.",
            }
        text = entry["en"].format(
            case_id=claim["case_id"],
            documents=_join(documents),
            average_processing_time_after_submission=g["claim_followup_settings"][
                "average_processing_time_after_submission"
            ]["en"],
        )
        result: dict[str, Any] = {"topic": topic, "applies": True, "guidance": text}
        if topic == "file_format_requirements":
            result["per_document_requirements"] = {
                doc: self.document_guidance(claim, doc)["requirements"] for doc in documents
            }
        return result

    def followup_phrases(self) -> dict[str, list[str]]:
        """Trigger phrases shipped with the fixture; only the offline stub uses them."""
        return {e["topic"]: e.get("match_any", []) for e in self._guidelines["claim_followup_guidance"]}

    # -- consent simulator ---------------------------------------------------

    def consent_scenarios(self) -> list[str]:
        return list(self._consent)

    def consent_status(self, scenario: str, poll_index: int) -> str:
        sequence = self._consent.get(scenario, self._consent["default"])["status_sequence"]
        return sequence[min(poll_index, len(sequence) - 1)]


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _name_key(name: str) -> tuple[str, ...]:
    return tuple(sorted(re.sub(r"[^a-z\s]", " ", name.lower()).split()))


def normalise_case_id(raw: str) -> str:
    digits = re.sub(r"\D", "", raw or "")
    return f"CL-{digits}" if digits else (raw or "").strip().upper()


def _match_document(document: str, keys: list[str]) -> str | None:
    """Match a claim's document name to a guideline key ("office note" -> "treating provider office note")."""
    wanted = set(re.findall(r"[a-z]+", document.lower()))
    if not wanted:
        return None
    best: tuple[int, str] | None = None
    for key in keys:
        tokens = set(re.findall(r"[a-z]+", key.lower()))
        if wanted <= tokens or tokens <= wanted:
            overlap = len(wanted & tokens)
            if best is None or overlap > best[0]:
                best = (overlap, key)
    return best[1] if best else None


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    if len(local) <= 2:
        return local[:1] + "*@" + domain
    return local[0] + "*" * (len(local) - 2) + local[-1] + "@" + domain
