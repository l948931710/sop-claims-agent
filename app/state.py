"""Session state: everything the harness knows about one conversation.

The model never holds state. Each turn it is handed a view of this object that
the current phase allows, and anything it wants to change goes through the
engine or the tool gateway.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from .sop import ALLOWED_TRANSITIONS, Phase


class IllegalTransition(RuntimeError):
    pass


@dataclass
class MemoryItem:
    key: str
    value: Any
    turn: int
    phase: str  # the phase the conversation was in when the caller said it
    source: str  # the caller's words
    used: bool = False  # set once a later phase has acted on it

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "turn": self.turn,
            "phase": self.phase,
            "source": self.source,
            "used": self.used,
        }


class Memory:
    """Cross-phase slots. Written every turn, in every phase.

    Keys are namespaced: identity.*, caller.*, rep.*, intent.*, case.*, prefs.*
    Latest value wins; provenance (turn, phase, caller's words) is kept so a
    later phase can say where it learned something.
    """

    def __init__(self) -> None:
        self.slots: dict[str, MemoryItem] = {}
        self.notes: list[MemoryItem] = []

    def remember(self, key: str, value: Any, *, turn: int, phase: Phase, source: str) -> bool:
        """Store a slot. Returns True when the value is new or changed."""
        existing = self.slots.get(key)
        if existing is not None and existing.value == value:
            return False
        self.slots[key] = MemoryItem(key, value, turn, phase.value, _excerpt(source))
        return True

    def note(self, text: str, *, turn: int, phase: Phase, source: str) -> bool:
        if any(n.value == text for n in self.notes):
            return False
        self.notes.append(MemoryItem("note", text, turn, phase.value, _excerpt(source)))
        return True

    def get(self, key: str, default: Any = None) -> Any:
        item = self.slots.get(key)
        return item.value if item is not None else default

    def group(self, prefix: str) -> dict[str, Any]:
        """All slots under a namespace, with the namespace stripped."""
        start = prefix + "."
        return {k[len(start):]: v.value for k, v in self.slots.items() if k.startswith(start)}

    def drop_group(self, prefix: str) -> None:
        start = prefix + "."
        for key in [k for k in self.slots if k.startswith(start)]:
            del self.slots[key]

    def mark_used(self, prefix: str) -> None:
        start = prefix + "."
        for key, item in self.slots.items():
            if key.startswith(start):
                item.used = True

    def group_used(self, prefix: str) -> bool:
        start = prefix + "."
        items = [v for k, v in self.slots.items() if k.startswith(start)]
        return bool(items) and all(i.used for i in items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": [item.to_dict() for item in self.slots.values()],
            "notes": [item.to_dict() for item in self.notes],
        }


def _excerpt(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str
    phase: str


@dataclass
class TraceEvent:
    kind: str  # nlu | memory | gate | policy | transition | directive | tool | guard | error | email
    title: str
    detail: dict[str, Any] = field(default_factory=dict)
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "title": self.title, "detail": self.detail, "ok": self.ok}


@dataclass
class TurnTrace:
    turn: int
    phase_before: str
    phase_after: str = ""
    seconds: float = 0.0
    events: list[TraceEvent] = field(default_factory=list)

    def add(self, kind: str, title: str, detail: dict[str, Any] | None = None, ok: bool = True) -> None:
        self.events.append(TraceEvent(kind, title, detail or {}, ok))

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "phase_before": self.phase_before,
            "phase_after": self.phase_after,
            "seconds": round(self.seconds, 1),
            "events": [e.to_dict() for e in self.events],
        }


@dataclass
class ConsentState:
    scenario: str = "default"
    status: str = "not_required"  # not_required | pending | approved | timed_out
    polls: int = 0
    last_poll_turn: int = -1


@dataclass
class EmailState:
    status: str = "not_offered"  # not_offered | offered | sent | skipped | unavailable
    decided_turn: int = -1


@dataclass
class Email:
    to_masked: str
    subject: str
    body: str
    turn: int
    drafted_by: str  # "model" | "template"

    def to_dict(self) -> dict[str, Any]:
        return {
            "to": self.to_masked,
            "subject": self.subject,
            "body": self.body,
            "turn": self.turn,
            "drafted_by": self.drafted_by,
        }


@dataclass
class Handoff:
    reason: str
    reason_text: str
    detail: str
    summary: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "reason_text": self.reason_text,
            "detail": self.detail,
            "summary": self.summary,
        }


@dataclass
class CaseLog:
    """One claim the caller was helped with, for the wrap-up summary."""

    case_id: str
    intent: str
    topics: list[str] = field(default_factory=list)  # guidance looked up while handling it


class Session:
    def __init__(self, consent_scenario: str = "default") -> None:
        self.id = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.phase = Phase.VERIFY_ID
        self.turn = 0
        self.transcript: list[Turn] = []
        self.memory = Memory()
        self.traces: list[TurnTrace] = []

        # VERIFY_ID
        self.verified = False
        self.party_id: str | None = None
        self.verification: dict[str, Any] = {}
        self.failed_attempts = 0
        self.counted_mismatches: set[tuple[str, str]] = set()
        self.stall_count = 0
        self.consent = ConsentState(scenario=consent_scenario)

        # cross-phase policies
        self.oos_strikes = 0
        self.heat_streak = 0
        self.emotions: list[dict[str, Any]] = []

        # RESOLVE_INTENT / PROCESS_CASE
        self.selected_case_id: str | None = None
        self.selected_intent: str | None = None
        self.cases: list[CaseLog] = []

        # POST_PROCESS
        self.email = EmailState()
        self.outbox: list[Email] = []

        self.handoff: Handoff | None = None

    # -- transitions ---------------------------------------------------------

    def transition(self, target: Phase, reason: str) -> None:
        if target not in ALLOWED_TRANSITIONS[self.phase]:
            raise IllegalTransition(f"{self.phase.value} -> {target.value} is not allowed by the SOP ({reason})")
        if target is Phase.RESOLVE_INTENT and not self.verified:
            raise IllegalTransition("cannot leave VERIFY_ID before identity is verified")
        if target is Phase.PROCESS_CASE and not (self.verified and self.selected_case_id):
            raise IllegalTransition("PROCESS_CASE needs a verified caller and a selected claim")
        before = self.phase
        self.phase = target
        if self.traces:
            self.traces[-1].add(
                "transition",
                f"{before.value} → {target.value}",
                {"reason": reason, "rule": ALLOWED_TRANSITIONS[before][target]},
            )

    # -- transcript ----------------------------------------------------------

    def add_turn(self, role: str, text: str) -> None:
        self.transcript.append(Turn(role, text, self.phase.value))

    def caller_texts(self) -> list[str]:
        return [t.text for t in self.transcript if t.role == "user"]

    def last_assistant_text(self) -> str:
        for turn in reversed(self.transcript):
            if turn.role == "assistant":
                return turn.text
        return ""

    def current_case_log(self) -> CaseLog | None:
        for log in reversed(self.cases):
            if log.case_id == self.selected_case_id:
                return log
        return None
