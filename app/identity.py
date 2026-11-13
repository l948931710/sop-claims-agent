"""Identity verification: pure functions, no model involved.

The model extracts what the caller said; this module decides whether it
matches. Values on file never leave this module except as booleans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .sop import IDENTITY_FIELDS

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def canonical(field_name: str, raw: Any) -> str | None:
    """Normalise a caller-supplied value. None means it is unusable and should be asked for again."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if field_name == "full_name":
        cleaned = " ".join(re.sub(r"[^\w\s'\-]|[\d_]", " ", text).split())
        return cleaned or None
    if field_name == "dob":
        return _canonical_date(text)
    if field_name == "phone":
        digits = re.sub(r"\D", "", text)
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        return digits if len(digits) == 10 else None
    if field_name == "email":
        lowered = text.lower().strip(".,;")
        return lowered if _EMAIL_RE.match(lowered) else None
    if field_name == "id_last4":
        digits = re.sub(r"\D", "", text)
        # A caller may read out the whole number; only the last four are ever kept.
        return digits[-4:] if len(digits) in (4, 9) else None
    if field_name == "policy_number":
        digits = re.sub(r"\D", "", text)
        return f"POL-{digits}" if digits else None
    raise KeyError(field_name)


def _canonical_date(text: str) -> str | None:
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text, flags=re.I).replace(",", " ")
    cleaned = " ".join(cleaned.split())
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
        if 1900 <= parsed.year <= date.today().year:
            return parsed.isoformat()
    return None


def _name_key(name: str) -> tuple[str, ...]:
    return tuple(sorted(re.sub(r"[^\w\s]|[\d_]", " ", name.lower()).split()))


def _phone_key(phone: str) -> str:
    return re.sub(r"\D", "", phone)[-10:]


def field_matches(field_name: str, value: str, record: dict[str, Any]) -> bool:
    if field_name == "full_name":
        names = [record["name"], *record.get("name_aliases", [])]
        return _name_key(value) in {_name_key(n) for n in names}
    if field_name == "dob":
        return value == record["dob"]
    if field_name == "phone":
        phones = [record["phone"], *record.get("phone_aliases", [])]
        return value in {_phone_key(p) for p in phones}
    if field_name == "email":
        emails = [record["email"], *record.get("email_aliases", [])]
        return value in {e.lower() for e in emails}
    if field_name == "id_last4":
        return value == record["id_last4"]
    if field_name == "policy_number":
        return value == record["policy_number"].upper()
    raise KeyError(field_name)


@dataclass
class Verification:
    party_id: str | None = None  # the single record the provided details point at
    matched: list[str] = field(default_factory=list)  # counted fields that match it
    mismatched: list[str] = field(default_factory=list)  # counted fields provided that do not
    missing: list[str] = field(default_factory=list)  # counted fields not provided yet
    policy_number_matched: bool | None = None
    ambiguous: bool = False
    required: int = 3

    @property
    def verified(self) -> bool:
        return self.party_id is not None and len(self.matched) >= self.required

    @property
    def record_located(self) -> bool:
        return self.party_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_located": self.record_located,
            "matched": self.matched,
            "mismatched": self.mismatched,
            "missing": self.missing,
            "policy_number_matched": self.policy_number_matched,
            "ambiguous": self.ambiguous,
            "required": self.required,
            "verified": self.verified,
        }


def evaluate(provided: dict[str, str], records: list[dict[str, Any]], required: int = 3) -> Verification:
    """Compare what the caller has said so far against every policyholder record.

    The candidate is the record the most provided details agree with. Fewer
    than `required` matching counted fields means not verified, whatever else
    is true.
    """
    counted = {f: v for f, v in provided.items() if f in IDENTITY_FIELDS}
    policy = provided.get("policy_number")
    result = Verification(required=required, missing=[f for f in IDENTITY_FIELDS if f not in counted])
    if not counted and not policy:
        return result

    scored: list[tuple[int, dict[str, Any]]] = []
    for record in records:
        score = sum(field_matches(f, v, record) for f, v in counted.items())
        if policy and field_matches("policy_number", policy, record):
            score += 1
        scored.append((score, record))
    best = max(score for score, _ in scored)
    if best == 0:
        result.mismatched = list(counted)
        result.policy_number_matched = False if policy else None
        return result
    leaders = [record for score, record in scored if score == best]
    if len(leaders) > 1:
        result.ambiguous = True
        return result

    record = leaders[0]
    result.party_id = record["party_id"]
    result.matched = [f for f, v in counted.items() if field_matches(f, v, record)]
    result.mismatched = [f for f in counted if f not in result.matched]
    if policy:
        result.policy_number_matched = field_matches("policy_number", policy, record)
    return result
