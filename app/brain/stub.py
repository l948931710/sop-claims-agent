"""An offline Brain: keyword rules instead of a model.

It exists so the harness can be run and tested with no API key. It reads
messages with regular expressions and replies with the harness's own script,
so it sounds stilted and misses anything phrased unusually. What it shows is
that every SOP guarantee holds without a model in the loop: the gates,
memory, tool permissions and guards are all the engine's.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..guards import dates as find_dates
from ..tools import ToolGateway
from .base import (
    CaseHints,
    EmailContext,
    EmailDraft,
    IdentityClaims,
    RespondContext,
    RespondResult,
    TurnUnderstanding,
    UnderstandContext,
)

_NAME = r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2})"
_SELF_NAME_RE = re.compile(rf"(?i:\bmy name is|\bname is|\bi am|\bi'm|\bthis is)\s+{_NAME}")
_BEHALF_RE = re.compile(
    rf"(?i:on behalf of|calling for|calling about)\s+(?i:my\s+)?(?i:(mother|father|mom|dad|wife|husband|son|daughter|"
    rf"partner|brother|sister|client|parent)[,\s]+)?{_NAME}"
)
_RELATIONSHIP_RE = re.compile(r"\b(?:i am|i'm)\s+(?:his|her|their)\s+(son|daughter|wife|husband|spouse|partner|brother|sister)\b", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"(?<![\d-])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_POLICY_RE = re.compile(r"\bPOL[-\s]?\d+\b", re.I)
_LAST4_RE = re.compile(r"(?:ssn|social|last\s*(?:four|4)|national id|id number|ends? in)[^\d.!?]{0,25}?(\d{4})(?![\d/-])", re.I)
_CLAIM_ID_RE = re.compile(r"\bCL[-\s]?\d{3,6}\b", re.I)
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|june|july|august|september|october|november|december)\b(?:\s+(20\d{2}))?", re.I
)

_IN_SCOPE_WORDS = (
    "claim", "policy", "insur", "appeal", "denied", "denial", "document", "verify", "verif", "identity",
    "ssn", "social security", "date of birth", "dob", "policyholder", "representative", "human", "agent",
    "supervisor", "deductible", "coverage", "reimburs", "payment", "paid", "deadline", "submit", "upload",
    "portal", "pathology", "office note", "report", "email", "summary", "consent", "approved", "status",
)  # fmt: skip
_SOCIAL_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|sure|yes|yeah|yep|no|nope|great|got it|alright|bye|goodbye|"
    r"good (morning|afternoon|evening))\b[\s\w,.!']{0,40}$",
    re.I,
)
_QUESTION_RE = re.compile(r"\?|^\s*(what|who|how|why|when|where|tell me|explain|write|give me|can you|could you)\b", re.I)

_INTENT_RULES = [
    ("document_submission", r"submit|upload|how do i send|where do i send|portal|format|pdf|scan|don't have the|do not have the|can't get|cannot get"),
    ("denial_question", r"denied|denial|rejected|why was"),
    ("next_steps", r"appeal|next step|what do i do|what should i do|what happens next|how long|how soon"),
    ("status_inquiry", r"status|where is|update on|progress"),
    ("general_claim_question", r"how much|amount|paid|reimburse|allowed"),
]  # fmt: skip
_TOPIC_RULES = [
    ("missing_required_material_alternatives", r"don't have|do not have|can't get|cannot get|lost|missing|alternative|instead"),
    ("submission_method", r"how do i submit|where do i submit|how do i send|where should i send|how should i send|portal|upload"),
    ("file_format_requirements", r"format|file type|pdf|scan|photo|image|clear enough"),
    ("receipt_confirmation", r"confirm|how do i know|got it|received"),
    ("processing_time_after_submission", r"how long|processing time|review time|once i submit|after i submit|after i send"),
    ("submission_timing", r"how soon|when do i need|when should i|deadline to submit"),
]  # fmt: skip
_TYPE_WORDS = {"healthcare": r"health|medical|hospital", "dental": r"dental|dentist", "auto": r"\bauto\b|\bcar\b|vehicle"}
_EMOTION_RULES = [
    ("angry", "high", r"ridiculous|unacceptable|outrageous|furious|sick of|worst|!!"),
    ("frustrated", "medium", r"frustrat|annoy|already told you|again\?|waste of|come on|seriously"),
    ("anxious", "medium", r"worried|scared|anxious|nervous|afraid|stress"),
    ("confused", "medium", r"confus|don't understand|do not understand|what do you mean|lost me"),
    ("sad", "medium", r"devastat|heartbroken|so sad|upset"),
]  # fmt: skip


class StubBrain:
    name = "stub"

    # -- understand ----------------------------------------------------------

    def understand(self, ctx: UnderstandContext) -> TurnUnderstanding:
        text = ctx.latest_message
        low = text.lower()

        behalf = _BEHALF_RE.search(text)
        self_name = _SELF_NAME_RE.search(text)
        representative = bool(behalf) or bool(re.search(r"on (?:his|her|their) behalf|i'm (?:his|her|their)\b", low))
        relationship = _RELATIONSHIP_RE.search(text)

        identity = IdentityClaims(
            full_name=(behalf.group(2) if behalf else self_name.group(1) if self_name else None),
            date_of_birth=_birth_date(text),
            phone=_first(_PHONE_RE, text),
            email=_first(_EMAIL_RE, text),
            id_last4=(m.group(1) if (m := _LAST4_RE.search(text)) else None),
            policy_number=_first(_POLICY_RE, text),
        )
        # A bare four-digit reply right after being asked for the last four digits.
        if identity.id_last4 is None and re.fullmatch(r"\s*\d{4}\s*\.?", text) and "last four" in ctx.last_agent_message.lower():
            identity.id_last4 = text.strip(" .")
        has_identity = any(v for v in identity.model_dump().values())

        intent = next((name for name, pattern in _INTENT_RULES if re.search(pattern, low)), "none")
        if intent == "none" and "claim" in low and _QUESTION_RE.search(text):
            intent = "general_claim_question"

        hints = _case_hints(text, low)
        in_scope_hit = any(word in low for word in _IN_SCOPE_WORDS)
        asks_something = bool(_QUESTION_RE.search(text))
        if has_identity and asks_something and not in_scope_hit:
            scope = "mixed"
        elif has_identity or in_scope_hit:
            scope = "in_scope"
        elif _SOCIAL_RE.match(text) or not asks_something:
            scope = "social"
        else:
            scope = "out_of_scope"

        emotion, intensity = next(
            ((name, level) for name, level, pattern in _EMOTION_RULES if re.search(pattern, low)), ("neutral", "low")
        )
        refuses = bool(re.search(r"i won't|i will not|i'm not (?:giving|telling|going to)|not comfortable|i refuse|none of your business|don't want to (?:give|share|tell)", low))
        asks_why = bool(re.search(r"why do (?:you|i) (?:need|have)|why is that|why should i", low))
        bypass = bool(re.search(r"just tell me|already told you|skip (?:this|that|the)|don't need to verify", low))
        wants_human = bool(
            re.search(r"\b(?:speak|talk|transfer|connect)\b.*\b(?:human|person|representative|agent|supervisor|someone)\b", low)
            or re.search(r"\b(?:human|real person|supervisor)\b.*\bplease\b", low)
        )

        yes = bool(re.match(r"\s*(yes|yeah|yep|sure|please do|ok|okay|go ahead|that would be great)\b", low))
        no = bool(re.match(r"\s*(no|nope|skip|don't|do not|not necessary|no need)\b", low))
        email_decision = "none"
        if re.search(r"(?:don't|do not|no need to) (?:email|send)|no email|skip the (?:email|summary)", low):
            email_decision = "skip"
        elif re.search(r"email me|send (?:me )?(?:the|a|that) (?:summary|email)|yes.*email", low):
            email_decision = "send"
        elif ctx.email_offer_pending and yes:
            email_decision = "send"
        elif ctx.email_offer_pending and no:
            email_decision = "skip"

        offered_human = "human representative" in ctx.last_agent_message.lower() and "connect you" in ctx.last_agent_message.lower()
        if offered_human and yes and not ctx.email_offer_pending:
            wants_human = True

        wants_to_end = bool(
            re.search(r"that's all|that is all|that's it|nothing else|no thanks|no thank you|i'm done|all set|goodbye|\bbye\b", low)
        ) or (no and not ctx.email_offer_pending and "anything else" in ctx.last_agent_message.lower())
        open_request = (intent != "none" or bool(hints.case_id or hints.case_type)) and not (wants_to_end and intent == "none")
        if ctx.phase == "VERIFY_ID":
            open_request = intent != "none"

        return TurnUnderstanding(
            identity=identity,
            caller_role="representative" if representative else ("policyholder" if re.search(r"\bpolicyholder\b|\bmy (?:own )?(?:claim|policy)\b", low) else "unknown"),
            representative_name=(self_name.group(1) if representative and self_name else None),
            representative_relationship=(relationship.group(1) if relationship else (_implied_relationship(behalf) if behalf else None)),
            intent=intent,
            case_hints=hints,
            scope=scope,
            out_of_scope_topic=(text.strip()[:60] if scope in ("out_of_scope", "mixed") else None),
            emotion=emotion,
            emotion_intensity=intensity,
            refuses_to_provide_info=refuses,
            asks_why_needed=asks_why,
            demands_bypass=bypass,
            requests_human=wants_human,
            email_decision=email_decision,
            wants_to_end=wants_to_end,
            has_open_request=open_request,
            notes=[],
        )

    # -- respond -------------------------------------------------------------

    def respond(self, ctx: RespondContext, tools: ToolGateway) -> RespondResult:
        d = ctx.directive
        latest = ctx.transcript[-1][1].lower() if ctx.transcript else ""

        if "select_case" in tools.allowed:
            choice = self._choose_case(d.facts)
            if choice is not None:
                outcome = tools.call("select_case", choice)
                if not outcome.is_error:
                    return RespondResult(transitioned=True)

        if "switch_case" in tools.allowed:
            claim = d.facts["open_claim"]
            other_id = [c.upper().replace(" ", "-") for c in _CLAIM_ID_RE.findall(latest) if c.upper().replace(" ", "-") != claim["case_id"]]
            other_type = [t for t, pattern in _TYPE_WORDS.items() if re.search(pattern, latest) and t != claim["case_type"]]
            if (other_id or other_type) and not d.facts.get("claim_selected_this_turn"):
                tools.call("switch_case", {"reason": "caller mentioned a different claim"})
                return RespondResult(transitioned=True)

            for topic, pattern in _TOPIC_RULES:
                if re.search(pattern, latest):
                    outcome = tools.call("get_followup_guidance", {"topic": topic})
                    if not outcome.is_error:
                        return RespondResult(text=_guidance_text(outcome.content))
            for document in claim.get("documents_needed") or []:
                if document in latest:
                    outcome = tools.call("get_document_guidance", {"document": document})
                    if not outcome.is_error:
                        return RespondResult(text=_requirements_text(outcome.content))

        return RespondResult(text=d.script)

    @staticmethod
    def _choose_case(facts: dict[str, Any]) -> dict[str, Any] | None:
        index = facts.get("claims_on_file") or []
        candidates = facts.get("claims_consistent_with_what_they_said") or []
        remembered = facts.get("remembered_from_earlier_in_this_call") or {}
        if len(candidates) == 1 and remembered.get("which_claim"):
            case_id = candidates[0]
        elif len(index) == 1:
            case_id = index[0]["case_id"]
        else:
            return None
        status = next(c["status"] for c in index if c["case_id"] == case_id)
        intent = remembered.get("what_they_want") or ("denial_question" if status == "denied" else "status_inquiry")
        return {"case_id": case_id, "intent": intent, "reason": "matched what the caller said earlier"}

    # -- email ---------------------------------------------------------------

    def draft_email(self, ctx: EmailContext) -> EmailDraft:
        return EmailDraft(subject=f"Summary of your conversation with {ctx.company_name} claims support", body=ctx.template)


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0) if match else None


def _birth_date(text: str) -> str | None:
    """A full date that could be a birthday: explicitly labelled, or simply too old to be a claim date."""
    labelled = re.search(r"(?:dob|date of birth|born|birthday)\D{0,15}([\w/,\-\s]{6,22})", text, re.I)
    for scope in ([labelled.group(1)] if labelled else []) + [text]:
        for year, month, day in sorted(find_dates(scope), key=lambda d: d[0] or 9999):
            if year and day and (labelled or year < 2012):
                return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _case_hints(text: str, low: str) -> CaseHints:
    case_id = _first(_CLAIM_ID_RE, text)
    case_type = next((name for name, pattern in _TYPE_WORDS.items() if re.search(pattern, low)), None)
    status = "denied" if re.search(r"denied|denial|rejected", low) else "open" if re.search(r"\bopen\b|pending|in progress", low) else "closed" if re.search(r"closed|settled", low) else None
    month = _MONTH_RE.search(low)
    when = month.group(0) if month else ("last month" if "last month" in low else "last year" if "last year" in low else None)
    description = None
    if "claim" in low:
        sentence = next((s for s in re.split(r"(?<=[.!?])\s+", text) if "claim" in s.lower()), text)
        description = sentence.strip().rstrip(".")[:140]
    if not any([case_id, case_type, status, when]):
        description = None
    return CaseHints(case_id=case_id, case_type=case_type, status=status, time_reference=when, description=description)


def _implied_relationship(behalf: re.Match[str]) -> str | None:
    return {"mother": "child", "father": "child", "mom": "child", "dad": "child", "wife": "husband", "husband": "wife"}.get(
        (behalf.group(1) or "").lower()
    )


def _guidance_text(content: str) -> str:
    data = json.loads(content)
    text = data["guidance"]
    for document, requirement in (data.get("per_document_requirements") or {}).items():
        text += f" For the {document}: {requirement}"
    return text


def _requirements_text(content: str) -> str:
    data = json.loads(content)
    return f"{data['requirements']} {data['if_unavailable']}"
