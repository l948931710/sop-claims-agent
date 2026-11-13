"""Output guards: deterministic checks on a reply before the caller sees it.

The first line of defence is structural (before verification the model is
given no claim data at all, so there is nothing to leak). These checks are the
second line: every claim number, dollar amount and calendar date in a reply
must be traceable to something the harness supplied this turn or the caller
said themselves, and no identifier held on file may appear unless the caller
supplied it. A reply that fails is regenerated once with the violations as
feedback, then replaced by the harness's own script.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]  # fmt: skip

_CLAIM_ID_RE = re.compile(r"\bCL[-\s]?(\d{3,6})\b", re.I)
_MONEY_RE = re.compile(r"\$\s?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)|\b((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)\s+dollars\b", re.I)
_PLAIN_DECIMAL_RE = re.compile(r"(?<![\w.,-])((?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2})(?![\w]|\.\d)")
_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b")
_MONTH_DAY_RE = re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}})\b)?", re.I)
_DAY_MONTH_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH})\b\.?(?:,?\s+(\d{{4}})\b)?", re.I)
_MONTH_YEAR_RE = re.compile(rf"\b({_MONTH})\.?,?\s+(\d{{4}})\b", re.I)

DateKey = tuple[int | None, int, int | None]  # (year, month, day); year or day may be unknown


def claim_ids(text: str) -> set[str]:
    return {f"CL-{m.group(1)}" for m in _CLAIM_ID_RE.finditer(text)}


def money_amounts(text: str, *, include_plain_decimals: bool = False) -> set[Decimal]:
    found: set[Decimal] = set()
    raw = [m.group(1) or m.group(2) for m in _MONEY_RE.finditer(text)]
    if include_plain_decimals:
        raw += [m.group(1) for m in _PLAIN_DECIMAL_RE.finditer(text)]
    for item in raw:
        try:
            found.add(Decimal(item.replace(",", "")))
        except InvalidOperation:
            continue
    return found


def dates(text: str) -> set[DateKey]:
    found: set[DateKey] = set()
    for m in _ISO_RE.finditer(text):
        found.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in _SLASH_RE.finditer(text):
        year = int(m.group(3))
        found.add((year + 2000 if year < 100 else year, int(m.group(1)), int(m.group(2))))
    for m in _MONTH_DAY_RE.finditer(text):
        found.add((int(m.group(3)) if m.group(3) else None, _MONTHS[m.group(1).lower()], int(m.group(2))))
    for m in _DAY_MONTH_RE.finditer(text):
        found.add((int(m.group(3)) if m.group(3) else None, _MONTHS[m.group(2).lower()], int(m.group(1))))
    for m in _MONTH_YEAR_RE.finditer(text):
        found.add((int(m.group(2)), _MONTHS[m.group(1).lower()], None))
    return {d for d in found if 1 <= d[1] <= 12 and (d[2] is None or 1 <= d[2] <= 31)}


def _date_grounded(candidate: DateKey, known: set[DateKey]) -> bool:
    year, month, day = candidate
    for k_year, k_month, k_day in known:
        if month != k_month:
            continue
        if day is not None and k_day is not None and day != k_day:
            continue
        if day is not None and k_day is None:
            continue  # a known month-year does not ground a specific day
        if year is not None and k_year is not None and year != k_year:
            continue
        if year is not None and k_year is None:
            continue
        return True
    return False


def _format_date(d: DateKey) -> str:
    year, month, day = d
    parts = [_MONTH_NAMES[month - 1]]
    if day is not None:
        parts.append(str(day))
    return " ".join(parts) + (f", {year}" if year is not None else "")


def _squash(text: str) -> str:
    return re.sub(r"[\s\-().+]", "", text)


def _sensitive_values(records: Iterable[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """(label, kind, value) for every identifier held on file."""
    out: list[tuple[str, str, str]] = []
    for record in records:
        for phone in [record["phone"], *record.get("phone_aliases", [])]:
            out.append(("a phone number on file", "phone", re.sub(r"\D", "", phone)[-10:]))
        for email in [record["email"], *record.get("email_aliases", [])]:
            out.append(("an email address on file", "email", email.lower()))
        out.append(("a date of birth on file", "dob", record["dob"]))
        out.append(("ID digits on file", "last4", record["id_last4"]))
    return out


def _contains(kind: str, value: str, text: str) -> bool:
    if kind == "phone":
        return value in _squash(text)
    if kind == "email":
        return value in text.lower()
    if kind == "dob":
        year, month, day = (int(p) for p in value.split("-"))
        return (year, month, day) in dates(text)
    return re.search(rf"(?<!\d){re.escape(value)}(?!\d)", text) is not None


def check_reply(reply: str, *, corpus: list[str], policyholders: Iterable[dict[str, Any]]) -> list[str]:
    """Return a list of violations; empty means the reply may be sent.

    `corpus` is everything the reply is allowed to draw on: the directive's
    facts and script, tool results from this turn, and the transcript so far.
    """
    grounded = "\n".join(corpus)
    violations: list[str] = []

    unknown_ids = claim_ids(reply) - claim_ids(grounded)
    for cid in sorted(unknown_ids):
        violations.append(f"mentions claim number {cid}, which is not in the facts for this turn")

    known_money = money_amounts(grounded, include_plain_decimals=True)
    derived = {a + b for a in known_money for b in known_money} | {abs(a - b) for a in known_money for b in known_money}
    for amount in sorted(money_amounts(reply) - known_money - derived):
        violations.append(f"states the amount ${amount}, which is not in the facts for this turn")

    known_dates = dates(grounded)
    for d in sorted(dates(reply), key=lambda k: (k[0] or 0, k[1], k[2] or 0)):
        if not _date_grounded(d, known_dates):
            violations.append(f"states the date {_format_date(d)}, which is not in the facts for this turn")

    for label, kind, value in _sensitive_values(policyholders):
        if _contains(kind, value, reply) and not _contains(kind, value, grounded):
            violations.append(f"contains {label} that the caller did not supply")

    return violations
