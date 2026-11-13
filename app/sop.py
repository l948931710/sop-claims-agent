"""The Standard Operating Procedure, as data.

Everything a business owner would want to read or change lives here: the
phases, how much freedom the model gets in each one, which tools it may call,
which transitions are legal, the numeric limits, and the wording the SOP
itself authors (why verification matters, how to respond to each emotion).

The engine (engine.py) interprets this file. Nothing in here calls a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"
    CLOSED = "CLOSED"
    ESCALATED = "ESCALATED"


TERMINAL_PHASES = frozenset({Phase.CLOSED, Phase.ESCALATED})


class Autonomy(str, Enum):
    """How much of the reply the model decides."""

    SCRIPTED = "scripted"  # the harness decides what is said; the model only phrases it
    GUIDED = "guided"  # the model picks one of a few bounded actions
    OPEN = "open"  # the model reasons freely, but only over data the harness hands it


@dataclass(frozen=True)
class PhaseSpec:
    phase: Phase
    step: int
    title: str
    autonomy: Autonomy
    purpose: str
    tools: tuple[str, ...]
    data_visible_to_model: str


PHASES: dict[Phase, PhaseSpec] = {
    Phase.VERIFY_ID: PhaseSpec(
        phase=Phase.VERIFY_ID,
        step=1,
        title="Verify identity",
        autonomy=Autonomy.SCRIPTED,
        purpose="Confirm the caller is the policyholder (or an authorised representative with consent).",
        tools=(),
        data_visible_to_model="Which identity fields matched or did not. Never the values on file, never claim data.",
    ),
    Phase.RESOLVE_INTENT: PhaseSpec(
        phase=Phase.RESOLVE_INTENT,
        step=2,
        title="Resolve intent",
        autonomy=Autonomy.GUIDED,
        purpose="Work out which claim the caller means and which workflow path fits their need.",
        tools=("select_case", "request_human_handoff"),
        data_visible_to_model="An index of the verified policyholder's own claims, plus everything remembered so far.",
    ),
    Phase.PROCESS_CASE: PhaseSpec(
        phase=Phase.PROCESS_CASE,
        step=3,
        title="Process case",
        autonomy=Autonomy.OPEN,
        purpose="Answer the caller's questions about the selected claim from grounded data.",
        tools=("get_document_guidance", "get_followup_guidance", "switch_case", "request_human_handoff"),
        data_visible_to_model="The selected claim record and guidance retrieved through tools.",
    ),
    Phase.POST_PROCESS: PhaseSpec(
        phase=Phase.POST_PROCESS,
        step=4,
        title="Post-process",
        autonomy=Autonomy.SCRIPTED,
        purpose="Offer an email summary; send it only if the caller says yes; then close.",
        tools=(),
        data_visible_to_model="The masked email address on file and the outcome of the email decision.",
    ),
    Phase.CLOSED: PhaseSpec(
        phase=Phase.CLOSED,
        step=5,
        title="Closed",
        autonomy=Autonomy.SCRIPTED,
        purpose="Say goodbye.",
        tools=(),
        data_visible_to_model="Nothing beyond the caller's first name.",
    ),
    Phase.ESCALATED: PhaseSpec(
        phase=Phase.ESCALATED,
        step=5,
        title="Handed to a human",
        autonomy=Autonomy.SCRIPTED,
        purpose="Explain the handoff and stop.",
        tools=(),
        data_visible_to_model="The reason for the handoff.",
    ),
}

# Phase order is enforced here and nowhere else. A transition that is not in
# this table raises, whoever asked for it.
ALLOWED_TRANSITIONS: dict[Phase, dict[Phase, str]] = {
    Phase.VERIFY_ID: {
        Phase.RESOLVE_INTENT: "at least 3 identity fields match one policyholder (and consent is approved for a representative)",
        Phase.ESCALATED: "caller asks for a human, lockout, repeated refusal, consent timeout, or repeated off-topic requests",
    },
    Phase.RESOLVE_INTENT: {
        Phase.PROCESS_CASE: "a claim owned by the verified policyholder and a known intent are selected",
        Phase.POST_PROCESS: "caller has nothing (further) to resolve",
        Phase.ESCALATED: "handoff",
    },
    Phase.PROCESS_CASE: {
        Phase.RESOLVE_INTENT: "caller raises a different claim",
        Phase.POST_PROCESS: "caller is done with the case",
        Phase.ESCALATED: "handoff",
    },
    Phase.POST_PROCESS: {
        Phase.PROCESS_CASE: "caller has another question about the same claim",
        Phase.RESOLVE_INTENT: "caller raises something new",
        Phase.CLOSED: "the email decision is made and the caller has nothing else",
        Phase.ESCALATED: "handoff",
    },
    Phase.CLOSED: {},
    Phase.ESCALATED: {},
}


@dataclass(frozen=True)
class Limits:
    min_identity_matches: int = 3
    max_failed_verifications: int = 3  # turns that introduce a value which does not match
    max_verification_stalls: int = 3  # consecutive refusals / bypass demands with no progress
    max_out_of_scope_strikes: int = 3  # consecutive off-topic turns before handing off
    max_consent_polls: int = 3
    heated_turns_before_offering_human: int = 2
    max_hops_per_turn: int = 5  # phase steps allowed inside one caller turn
    max_tool_rounds: int = 4


# The five fields the SOP counts toward verification, with caller-facing labels.
IDENTITY_FIELDS: dict[str, str] = {
    "full_name": "full name",
    "dob": "date of birth",
    "phone": "phone number on file",
    "email": "email address on file",
    "id_last4": "last four digits of the SSN or national ID",
}
# Helps locate the record but does not count toward the three.
LOCATOR_FIELDS: dict[str, str] = {"policy_number": "policy number"}


@dataclass(frozen=True)
class IntentSpec:
    label: str
    when: str
    playbook: str


# The bounded workflow paths. RESOLVE_INTENT must land on exactly one of these.
INTENTS: dict[str, IntentSpec] = {
    "status_inquiry": IntentSpec(
        label="Claim status",
        when="The caller wants to know where a claim stands.",
        playbook=(
            "State the claim's current status, when it was filed and the one-line summary. "
            "If it was denied, give the reason and offer to go through what is needed next. "
            "If it is open, say it is still in progress. If it is closed, say what was paid."
        ),
    ),
    "denial_question": IntentSpec(
        label="Why a claim was denied",
        when="The caller asks why a claim was denied or disputes a denial.",
        playbook=(
            "Explain the denial reason in plain language, list the documents still needed, and give the appeal "
            "deadline. Offer to walk through how to submit the documents."
        ),
    ),
    "document_submission": IntentSpec(
        label="Submitting documents",
        when="The caller asks what to send, how to send it, in what format, or what to do if they cannot get a document.",
        playbook=(
            "Say which documents are needed. Use the guidance tools for how to submit, format requirements, "
            "alternatives when a document is unavailable, and how receipt is confirmed."
        ),
    ),
    "next_steps": IntentSpec(
        label="What happens next",
        when="The caller asks what they should do now, about appealing, or about timing.",
        playbook=(
            "Lay out the next steps in order: what the caller needs to do, by when, and what happens after. "
            "Use the guidance tools for timing and processing expectations."
        ),
    ),
    "general_claim_question": IntentSpec(
        label="General question about a claim",
        when="Any other question about a specific claim, such as amounts paid or allowed.",
        playbook=(
            "Answer from the claim record. For money questions use the field glossary to explain what each "
            "amount means."
        ),
    ),
}

FOLLOWUP_TOPICS: dict[str, str] = {
    "missing_required_material_alternatives": "The caller cannot get a requested document and asks what else is acceptable.",
    "submission_timing": "How soon the documents need to be submitted.",
    "processing_time_after_submission": "How long review takes once documents are received.",
    "submission_method": "How or where to submit documents (portal, upload link, fax, mail).",
    "file_format_requirements": "File type, scan or photo quality, format requirements.",
    "receipt_confirmation": "How the caller will know the documents were received.",
}

# Wording the SOP authors. The model rephrases these; it does not invent them.
RATIONALE = {
    "verification": (
        "Claim records contain protected health and financial information. Confirming at least three "
        "identifying details makes sure we only share them with the policyholder or someone they have authorised."
    ),
    "consent": (
        "When someone calls on a policyholder's behalf, the policyholder has to approve it first. That keeps "
        "them in control of who can see their claim information."
    ),
    "scope": "This line is only for insurance claims support.",
    "email_only_on_file": (
        "For security the summary can only go to the email address already on file for the policy."
    ),
}

TONE_PLAYBOOK = {
    "frustrated": (
        "The caller is frustrated. Open by acknowledging that in your own words (one sentence, specific to what "
        "they said, no stock apology). Then give the reason for the step, then the options. Keep it short."
    ),
    "angry": (
        "The caller is angry. Stay calm and respectful, do not argue or defend. Acknowledge the anger and what "
        "caused it first, briefly. Then the reason for the step, then the options. Do not match their intensity."
    ),
    "anxious": (
        "The caller is anxious. Reassure first: you are going to help and this is a normal step. Be concrete "
        "about what happens next so the uncertainty goes down."
    ),
    "confused": (
        "The caller is confused. Slow down, use plain words, explain one thing at a time, and check they are "
        "with you before moving on."
    ),
    "sad": (
        "The caller is upset. Acknowledge it gently and sincerely in one sentence before anything procedural."
    ),
}

ESCALATION_REASONS = {
    "caller_requested_human": "The caller asked to speak with a human representative.",
    "verification_failed": "Identity could not be verified after the allowed number of attempts.",
    "verification_declined": "The caller declined to complete identity verification.",
    "representative_not_authorised": "The caller is not listed as an authorised representative on the policy.",
    "consent_timeout": "The policyholder did not approve the consent request in time.",
    "repeated_out_of_scope": "The caller repeatedly asked for things outside claims support.",
    "agent_requested": "The assistant judged that a human representative is needed.",
}
