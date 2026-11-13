"""The boundary between the harness and whatever does the language work.

A Brain does three things and nothing else:

  understand  read one caller message into a structured record
  respond     write the reply for a directive, optionally calling gated tools
  draft_email write the wrap-up email from facts the harness supplies

It holds no state and makes no workflow decisions. The harness can run on a
real model (claude.py), an offline rule-based stand-in (stub.py), or a test
double that misbehaves on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..directive import Directive
    from ..tools import ToolGateway


class BrainError(RuntimeError):
    """The model call failed or returned something unusable. The harness falls back to its own script."""

    def __init__(self, message: str, *, user_fixable: bool = False) -> None:
        super().__init__(message)
        self.user_fixable = user_fixable  # e.g. missing or rejected API key


# -- understand ---------------------------------------------------------------

IntentName = Literal[
    "status_inquiry",
    "denial_question",
    "document_submission",
    "next_steps",
    "general_claim_question",
    "none",
]


class IdentityClaims(BaseModel):
    full_name: Optional[str] = Field(description="The policyholder's full name, if stated in the latest message.")
    date_of_birth: Optional[str] = Field(description="The policyholder's date of birth as YYYY-MM-DD.")
    phone: Optional[str] = Field(description="A phone number offered as an identifier.")
    email: Optional[str] = Field(description="An email address offered as an identifier.")
    id_last4: Optional[str] = Field(
        description="Last four digits of an SSN or national ID. If the caller gave a longer number, keep only the last four."
    )
    policy_number: Optional[str] = Field(description="A policy number such as POL-9921.")


class CaseHints(BaseModel):
    case_id: Optional[str] = Field(description="A claim number such as CL-2048, if the caller gave one.")
    case_type: Optional[str] = Field(description="Kind of claim in one word: healthcare, dental, auto, or the caller's word.")
    status: Optional[str] = Field(description="A status the caller attributes to the claim: denied, open, closed.")
    time_reference: Optional[str] = Field(
        description="When the claim is from, in the caller's terms: 'January', 'January 2026', 'last month', '2025-11'."
    )
    description: Optional[str] = Field(
        description="Short paraphrase of which claim or what need the caller described, in their own terms."
    )


class TurnUnderstanding(BaseModel):
    """What the caller's latest message contains. Extraction only; no decisions."""

    identity: IdentityClaims
    caller_role: Literal["policyholder", "representative", "unknown"] = Field(
        description="'representative' when the caller says they are calling on someone else's behalf."
    )
    representative_name: Optional[str] = Field(description="The caller's own name when they are a representative.")
    representative_relationship: Optional[str] = Field(description="Their relationship to the policyholder, e.g. 'son'.")
    intent: IntentName = Field(description="The workflow path the caller's stated need fits, or 'none'.")
    case_hints: CaseHints
    scope: Literal["in_scope", "out_of_scope", "mixed", "social"]
    out_of_scope_topic: Optional[str] = Field(description="A few words naming the off-topic subject, if any.")
    emotion: Literal["neutral", "frustrated", "angry", "anxious", "confused", "sad"]
    emotion_intensity: Literal["low", "medium", "high"]
    refuses_to_provide_info: bool = Field(description="Declines to give identifying details or to take a required step.")
    asks_why_needed: bool = Field(description="Asks why verification, consent or another required step is needed.")
    demands_bypass: bool = Field(description="Insists on claim information or action without completing a required step.")
    requests_human: bool = Field(description="Clearly asks to speak with a human, a person, a supervisor.")
    email_decision: Literal["send", "skip", "none"] = Field(
        description="'send' or 'skip' when the caller accepts, requests or declines an email summary; else 'none'."
    )
    wants_to_end: bool = Field(description="Signals they are finished: nothing else, that's all, goodbye.")
    has_open_request: bool = Field(
        description=(
            "The message asks a question or makes a request about claims that still needs an answer "
            "(not counting a yes/no to the email summary, a goodbye, or supplying identity details)."
        )
    )
    notes: list[str] = Field(
        description="Other facts worth remembering later (e.g. 'already mailed the pathology report on Feb 20'). Usually empty."
    )

    @classmethod
    def blank(cls, **overrides: Any) -> "TurnUnderstanding":
        base: dict[str, Any] = {
            "identity": IdentityClaims(
                full_name=None, date_of_birth=None, phone=None, email=None, id_last4=None, policy_number=None
            ),
            "caller_role": "unknown",
            "representative_name": None,
            "representative_relationship": None,
            "intent": "none",
            "case_hints": CaseHints(case_id=None, case_type=None, status=None, time_reference=None, description=None),
            "scope": "in_scope",
            "out_of_scope_topic": None,
            "emotion": "neutral",
            "emotion_intensity": "low",
            "refuses_to_provide_info": False,
            "asks_why_needed": False,
            "demands_bypass": False,
            "requests_human": False,
            "email_decision": "none",
            "wants_to_end": False,
            "has_open_request": False,
            "notes": [],
        }
        base.update(overrides)
        return cls(**base)


@dataclass
class UnderstandContext:
    phase: str
    latest_message: str
    recent_turns: list[tuple[str, str]]  # (role, text), oldest first, excluding the latest message
    last_agent_message: str
    email_offer_pending: bool
    selected_case: str | None  # e.g. "CL-2048 (healthcare, denied)"; None before a case is selected
    today: date


# -- respond ------------------------------------------------------------------


@dataclass
class RespondContext:
    directive: "Directive"
    transcript: list[tuple[str, str]]  # (role, text) including the latest caller message
    today: date
    company_name: str
    feedback: list[str] = field(default_factory=list)  # guard violations from a rejected first attempt


@dataclass
class RespondResult:
    text: str | None = None
    transitioned: bool = False  # a tool call moved the conversation to another phase; text is discarded


# -- email --------------------------------------------------------------------


class EmailDraft(BaseModel):
    subject: str
    body: str = Field(description="Plain text. Sections: what we discussed, claim status, next steps.")


@dataclass
class EmailContext:
    caller_first_name: str
    company_name: str
    today: date
    cases: list[dict[str, Any]]  # case files of the claims discussed, with intent and guidance looked up
    transcript: list[tuple[str, str]]
    template: str  # the harness's own deterministic draft, as reference


class Brain(Protocol):
    name: str

    def understand(self, ctx: UnderstandContext) -> TurnUnderstanding: ...

    def respond(self, ctx: RespondContext, tools: "ToolGateway") -> RespondResult: ...

    def draft_email(self, ctx: EmailContext) -> EmailDraft: ...
