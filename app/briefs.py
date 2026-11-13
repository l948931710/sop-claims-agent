"""Directive builders: one per situation the SOP recognises.

Each builder returns a Directive with two parallel things: instructions for
the model (`required_points`) and a plain caller-facing version the harness
wrote itself (`script`). They are built side by side from the same conditions
so they cannot drift apart.

No builder reads session state or calls a model; the engine passes in exactly
what the phase is allowed to show.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from .directive import Directive
from .identity import Verification
from .sop import ESCALATION_REASONS, IDENTITY_FIELDS, INTENTS, PHASES, RATIONALE, Limits, Phase

EMPATHY = {
    "frustrated": "I'm sorry for the frustration, and I do want to get you an answer.",
    "angry": "I hear you, and I'm sorry this has been so aggravating.",
    "anxious": "I understand this is worrying, and I'm here to help you through it.",
    "confused": "No problem at all, let me make this clearer.",
    "sad": "I'm sorry you're dealing with this.",
}


class _Parts:
    """Collects (instruction for the model, sentence for the script) pairs."""

    def __init__(self) -> None:
        self.points: list[str] = []
        self.sentences: list[str] = []

    def add(self, point: str, sentence: str) -> None:
        self.points.append(point)
        self.sentences.append(sentence)

    @property
    def script(self) -> str:
        return " ".join(self.sentences)


def _directive(phase: Phase, objective: str, parts: _Parts, **kwargs: Any) -> Directive:
    spec = PHASES[phase]
    return Directive(
        phase=phase,
        autonomy=spec.autonomy,
        objective=objective,
        script=parts.script,
        required_points=parts.points,
        **kwargs,
    )


# -- small formatting helpers -------------------------------------------------


def human_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def money(raw: str) -> str:
    return f"${Decimal(raw):,.2f}"


def join_and(items: list[str]) -> str:
    return _join(items, "and")


def join_or(items: list[str]) -> str:
    return _join(items, "or")


def _join(items: list[str], word: str) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} {word} {items[1]}"
    return ", ".join(items[:-1]) + f", {word} {items[-1]}"


def _comma_name(first_name: str | None) -> str:
    return f", {first_name}" if first_name else ""


# -- greeting and scope -------------------------------------------------------


def greeting(company: str) -> str:
    return (
        f"Thank you for contacting {company} claims support. I'm a virtual assistant. Before I can discuss any "
        f"claim, I need to verify your identity. Could you give me any three of these: "
        f"{join_or(list(IDENTITY_FIELDS.values()))}?"
    )


def decline_line(mention_human: bool) -> str:
    line = "I'm sorry, but that's outside what I can help with here. I can only assist with insurance claims."
    if mention_human:
        line += " If you need help with something else, I can connect you with a human representative."
    return line


def scope_note(topic: str | None, *, mention_human: bool, mixed: bool) -> str:
    subject = f" ({topic})" if topic else ""
    note = (
        f"Part of the caller's message{subject} is outside insurance claims support. "
        if mixed
        else f"The caller asked about something outside insurance claims support{subject}. "
    )
    note += (
        "Decline that part in one friendly sentence. Do not answer it, not even briefly, partially or as an aside, "
        "and do not suggest where else to find the answer. Then continue with the objective."
    )
    if mention_human:
        note += (
            " This is not the first off-topic request in a row: also say that if they need help with something "
            "other than claims, you can connect them with a human representative."
        )
    return note


# -- VERIFY_ID ----------------------------------------------------------------

VERIFY_PROHIBITIONS = [
    "Do not confirm or deny that any claim exists, and do not share any claim, policy or account detail.",
    "Do not reveal, confirm or hint at any value held on file.",
    "Do not tell the caller they are verified.",
    "Never ask for a full SSN; only the last four digits.",
]


def verify_brief(
    *,
    v: Verification,
    limits: Limits,
    failed_attempts: int,
    representative: bool,
    caller_reason: str | None,
    reason_is_new: bool,
    unusable: list[str],
    pushback: bool,
    offer_human: bool,
    newly_matched: list[str],
) -> Directive:
    labels = IDENTITY_FIELDS
    whose = "the policyholder's" if representative else "your"
    matched = [labels[f] for f in v.matched]
    just_confirmed = [labels[f] for f in newly_matched]
    mismatched = [labels[f] for f in v.mismatched]
    missing = [labels[f] for f in v.missing]
    provided_any = bool(v.matched or v.mismatched or v.ambiguous or v.policy_number_matched is not None)
    still_needed = max(limits.min_identity_matches - len(v.matched), 0)
    attempts_left = max(limits.max_failed_verifications - failed_attempts, 0)

    parts = _Parts()
    if reason_is_new and caller_reason:
        parts.add(
            "Let the caller know you have noted what they are calling about and will get to it as soon as "
            "verification is done. Say nothing about the claim itself.",
            "I've noted what you're calling about, and we'll get to it as soon as you're verified.",
        )
    if pushback:
        parts.add(
            "Explain in a sentence or two why verification comes first (see why_verification_is_required) and "
            "that you cannot discuss claim details until it is done. Then make clear they can pick whichever "
            "accepted identifiers they are comfortable sharing.",
            "Claim records hold protected health and financial information, so I can only discuss them once I've "
            "confirmed at least three identifying details. You can choose whichever of them you're comfortable sharing.",
        )
    if just_confirmed:
        parts.add(
            f"Acknowledge what was just confirmed, naming the fields without repeating the values: {join_and(just_confirmed)}.",
            f"Thank you, I've confirmed {whose} {join_and(just_confirmed)}.",
        )
    if mismatched and v.record_located:
        parts.add(
            f"Tell the caller the {join_and(mismatched)} did not match what is on file. Invite them to re-check "
            "it or to use a different identifier instead.",
            f"The {join_and(mismatched)} you gave didn't match what we have on file. You're welcome to double-check "
            "it or use a different detail.",
        )
    elif provided_any and not v.record_located:
        if v.ambiguous:
            parts.add(
                "Say you need one more identifying detail to locate the right record.",
                "I need one more detail to locate the right record.",
            )
        else:
            parts.add(
                "Say you could not find a record matching those details, without saying which detail was the "
                "problem. Suggest re-checking them or giving a different identifier.",
                "I wasn't able to find a record matching those details. Could you double-check them, or give me a "
                "different identifier?",
            )
    if unusable:
        unclear = join_and([labels.get(f, "policy number") for f in unusable])
        parts.add(
            f"Ask the caller to give the {unclear} again; the value was not in a usable form.",
            f"I didn't quite catch the {unclear}. Could you give it to me again?",
        )
    if failed_attempts and attempts_left == 1:
        parts.add(
            "Let the caller know, kindly, that for their security one more detail that does not match means you "
            "will need to connect them with a human representative.",
            "For your security, if the next detail doesn't match I'll need to connect you with a human representative.",
        )
    if still_needed:
        if not provided_any:
            parts.add(
                f"Ask for any three of the accepted identifiers ({whose}).",
                f"Could you give me any three of these: {whose} {join_or(list(labels.values()))}?",
            )
        elif missing:
            parts.add(
                f"Ask for {still_needed} more identifier(s). Not yet provided: {join_or(missing)}.",
                f"To finish verifying, I need {still_needed} more: {whose} {join_or(missing)}.",
            )
    if offer_human:
        parts.add(
            "Offer, as a real alternative and without pressure, to connect them with a human representative, and "
            "let them choose.",
            "If you'd prefer, I can connect you with a human representative instead.",
        )

    facts: dict[str, Any] = {
        "accepted_identifiers": list(labels.values()),
        "matches_required": limits.min_identity_matches,
        "confirmed_so_far": matched,
        "provided_but_did_not_match": mismatched,
        "not_yet_provided": missing,
        "record_located": v.record_located,
        "why_verification_is_required": RATIONALE["verification"],
        "caller_is_a_representative": representative,
    }
    if caller_reason:
        facts["callers_own_words_about_why_they_are_calling"] = caller_reason
    return _directive(
        Phase.VERIFY_ID,
        "Collect and confirm the caller's identity. Nothing about any claim may be discussed yet.",
        parts,
        prohibitions=list(VERIFY_PROHIBITIONS),
        facts=facts,
    )


def representative_identity_brief() -> Directive:
    parts = _Parts()
    parts.add(
        "Thank the caller: the policyholder's details check out. Because they are calling on someone else's "
        "behalf, ask for their own full name and their relationship to the policyholder.",
        "Thank you, those details check out. Since you're calling on the policyholder's behalf, could I have your "
        "own full name and your relationship to them?",
    )
    return _directive(
        Phase.VERIFY_ID,
        "Identify the representative before any consent step.",
        parts,
        prohibitions=list(VERIFY_PROHIBITIONS),
        facts={"why_consent_is_required": RATIONALE["consent"]},
    )


def representative_not_authorised_brief(offer_human: bool) -> Directive:
    parts = _Parts()
    parts.add(
        "Explain that they are not listed as an authorised representative on this policy, so you cannot discuss "
        "it with them. The policyholder is welcome to call directly.",
        "I'm not able to discuss this policy with you because you're not listed as an authorised representative "
        "on it. The policyholder is welcome to contact us directly.",
    )
    parts.add(
        "Offer to connect them with a human representative who can explain how to be added as an authorised "
        "representative." + (" Make this offer the focus now." if offer_human else ""),
        "If you'd like, I can connect you with a human representative who can explain how to be added.",
    )
    return _directive(
        Phase.VERIFY_ID,
        "The caller is not an authorised representative; do not proceed.",
        parts,
        prohibitions=list(VERIFY_PROHIBITIONS),
        facts={"why_consent_is_required": RATIONALE["consent"]},
    )


def consent_pending_brief(*, just_requested: bool, checks_left: int, pushback: bool) -> Directive:
    parts = _Parts()
    if just_requested:
        parts.add(
            "Tell the caller the policyholder's details check out and that, because they are calling on someone "
            "else's behalf, a consent request has just been sent to the policyholder's phone number on file.",
            "Thank you, those details check out. Because you're calling on the policyholder's behalf, I've just "
            "sent a consent request to their phone number on file.",
        )
    else:
        parts.add(
            "Tell the caller you checked again and the policyholder has not approved the consent request yet.",
            "I've checked again, and the policyholder hasn't approved the consent request yet.",
        )
    if just_requested or pushback:
        parts.add(
            "Explain briefly why consent is needed (see why_consent_is_required).",
            "The policyholder has to approve before I can share their claim information with anyone else, which "
            "keeps them in control of it.",
        )
    parts.add(
        "Ask them to let you know once the policyholder has approved, and you will check again.",
        "Please let me know once they've approved it and I'll check again.",
    )
    return _directive(
        Phase.VERIFY_ID,
        "Wait for the policyholder's consent. Nothing about any claim may be discussed until it is approved.",
        parts,
        prohibitions=list(VERIFY_PROHIBITIONS),
        facts={
            "consent_status": "pending",
            "checks_remaining_before_handoff": checks_left,
            "why_consent_is_required": RATIONALE["consent"],
        },
    )


# -- RESOLVE_INTENT -----------------------------------------------------------


def describe_claim(entry: dict[str, Any]) -> str:
    article = "an" if entry["case_type"][:1].lower() in "aeiou" else "a"
    return f"{article} {entry['case_type']} claim filed {human_date(entry['filed_on'])} ({entry['case_id']}, {entry['status']})"


def resolve_brief(
    *,
    first_name: str | None,
    caller: dict[str, Any],
    claims_index: list[dict[str, Any]],
    remembered: dict[str, Any],
    candidates: list[str],
    just_verified: bool,
    today: date,
) -> Directive:
    parts = _Parts()
    if just_verified:
        parts.add(
            "Let the caller know, briefly, that they are verified.",
            f"Thank you{_comma_name(first_name)}, you're verified.",
        )
    facts: dict[str, Any] = {
        "caller": caller,
        "today": today.isoformat(),
        "claims_on_file": claims_index,
        "remembered_from_earlier_in_this_call": remembered,
        "workflow_paths": {name: spec.when for name, spec in INTENTS.items()},
    }
    if not claims_index:
        parts.add(
            "Tell the caller there are no claims on file under this policy. If they think that is wrong, offer "
            "to connect them with a human representative. Ask if there is anything else.",
            "I don't see any claims on file under this policy. If that doesn't sound right, I can connect you with "
            "a human representative. Is there anything else I can help with?",
        )
        return _directive(
            Phase.RESOLVE_INTENT,
            "There is nothing to select; tell the caller and offer a human representative.",
            parts,
            facts=facts,
            tools=("request_human_handoff",),
            prohibitions=["Do not invent or guess at claims that are not on file."],
        )

    facts["claims_consistent_with_what_they_said"] = candidates
    narrowed = [c for c in claims_index if c["case_id"] in candidates]
    if 1 < len(narrowed) < len(claims_index):
        listing = "; ".join(describe_claim(c) for c in narrowed)
        parts.sentences.append(
            f"I see {len(narrowed)} claims that could be the one you mean: {listing}. Which one are you calling about?"
        )
    else:
        listing = "; ".join(describe_claim(c) for c in claims_index)
        parts.sentences.append(
            f"I see {len(claims_index)} claim{'s' if len(claims_index) != 1 else ''} on file: {listing}. "
            "Which one are you calling about, and what would you like to know?"
        )
    return _directive(
        Phase.RESOLVE_INTENT,
        "Work out which claim the caller means and which workflow path fits, using everything they have already "
        "said in this call. If exactly one claim on file fits, call select_case now rather than asking them to "
        "repeat themselves. If more than one fits, ask one short question that tells those claims apart. If none "
        "fits, say what is on file and ask.",
        parts,
        facts=facts,
        tools=PHASES[Phase.RESOLVE_INTENT].tools,
        resume="Which of your claims are you calling about today?",
        prohibitions=[
            "Do not go into a claim's details (reasons, amounts, documents, deadlines) before it is selected; "
            "use the index only to tell claims apart.",
            "Do not invent or guess at claims that are not on file.",
        ],
    )


# -- PROCESS_CASE -------------------------------------------------------------


def case_summary(case: dict[str, Any], whose: str = "your") -> str:
    """A plain, fully grounded description of where a claim stands."""
    lead = f"I have {whose} {case['case_type']} claim {case['case_id']}, filed {human_date(case['filed_on'])}, in front of me."
    status = case["status"]
    if status == "denied":
        text = f"It was denied because {case.get('denial_reason', 'of the reason noted on the file')}."
        documents = case.get("documents_needed") or []
        if documents:
            text += f" To have it reconsidered, we need the {join_and(documents)}."
        if case.get("appeal_deadline"):
            days = case["days_until_appeal_deadline"]
            when = "which has already passed" if days < 0 else f"which is {days} day{'s' if days != 1 else ''} from today"
            text += f" The appeal deadline is {human_date(case['appeal_deadline'])}, {when}."
    elif status == "open":
        text = (
            f"It is open and still in progress. The expected reimbursement is "
            f"{money(case['expected_reimbursement_amount'])}, and nothing has been paid yet."
        )
    else:
        text = (
            f"It is {status}. The amount paid was {money(case['net_pay'])}, against an allowed maximum of "
            f"{money(case['allowed_max_amount'])}."
        )
    return f"{lead} {text}"


def process_brief(
    *,
    first_name: str | None,
    caller: dict[str, Any],
    case: dict[str, Any],
    intent: str,
    glossary: dict[str, str],
    notes: list[str],
    just_selected: bool,
    just_verified: bool,
    today: date,
) -> Directive:
    parts = _Parts()
    if just_verified:
        parts.add(
            "Let the caller know, briefly, that they are verified.",
            f"Thank you{_comma_name(first_name)}, you're verified.",
        )
    if just_selected:
        parts.points.append(
            "Say which claim you have pulled up (type, filing date, claim number) so the caller can correct you "
            "if it is the wrong one, then go straight on to answer the need they already stated."
        )
    parts.sentences += [case_summary(case, caller["claims_belong_to"]), "What else would you like to know about this claim?"]
    spec = INTENTS[intent]
    return _directive(
        Phase.PROCESS_CASE,
        f"Help the caller with claim {case['case_id']}. Answer what they asked from the claim record below and "
        "the guidance tools, in a natural conversational way.",
        parts,
        facts={
            "caller": caller,
            "today": today.isoformat(),
            "workflow_path": spec.label,
            "claim_selected_this_turn": just_selected,
            "open_claim": case,
            "what_the_amount_fields_mean": glossary,
            "notes_the_caller_mentioned_earlier": notes,
        },
        tools=PHASES[Phase.PROCESS_CASE].tools,
        playbook=spec.playbook,
        resume=f"Is there anything else you'd like to know about claim {case['case_id']}?",
        prohibitions=[
            "Do not state any amount, date, deadline, document requirement or process detail that is not in the "
            "facts or a tool result from this turn. If it is not there, say you do not have that information and "
            "offer a human representative.",
            "Do not work out new dates or amounts yourself; use the ones provided.",
            "Do not promise an outcome (approval, payment, or timing beyond what the guidance says).",
            "Do not give medical or legal advice.",
        ],
    )


# -- POST_PROCESS -------------------------------------------------------------


def email_offer_brief(*, masked_email: str, remembered_wish: bool, reask: bool, representative: bool) -> Directive:
    parts = _Parts()
    on_file = "the policyholder's address on file" if representative else "the address on file"
    if reask:
        parts.add(
            "Their answer about the email summary was not clear. Ask again, simply: send it, or skip it?",
            f"Sorry, I want to be sure I have that right: shall I email the summary to {masked_email}, or would "
            "you rather skip it?",
        )
    elif remembered_wish:
        parts.add(
            "The caller said earlier in the call that they would like a summary by email. Mention that you "
            f"remember, and confirm you should send it to {on_file} ({masked_email}). They can still say no.",
            f"You mentioned earlier that you'd like a summary by email. Shall I send it to {on_file}, "
            f"{masked_email}?",
        )
    else:
        parts.add(
            "Offer to email a summary covering what was discussed, the claim status and the next steps, to "
            f"{on_file} ({masked_email}). Make it clear it is their choice: yes, or skip it.",
            "Before we wrap up, would you like me to email a summary of what we discussed, including the "
            f"claim status and next steps? It would go to {on_file}, {masked_email}. You're welcome to skip it.",
        )
    return _directive(
        Phase.POST_PROCESS,
        "Get a clear yes or no on the email summary. Nothing is sent without a yes.",
        parts,
        facts={"email_on_file_masked": masked_email, "why_only_that_address": RATIONALE["email_only_on_file"]},
        prohibitions=[
            "Do not say or imply the email has been sent.",
            "Do not offer to send it to any address other than the one on file.",
        ],
    )


def email_outcome_brief(*, outcome: str, masked_email: str, remembered_skip: bool) -> Directive:
    parts = _Parts()
    if outcome == "sent":
        parts.add(
            f"Confirm the summary has been sent to {masked_email}.",
            f"Done. I've sent the summary to {masked_email}.",
        )
    elif outcome == "unavailable":
        parts.add(
            "Explain there is no email address on file, so a summary cannot be sent.",
            "There's no email address on file for this policy, so I'm not able to send a summary.",
        )
    elif remembered_skip:
        parts.add(
            "The caller said earlier they did not want an email summary. Tell them you are skipping it as they "
            "asked, and that they can change their mind.",
            "You mentioned earlier that you'd rather not receive an email summary, so I'll skip that. Just say so "
            "if you've changed your mind.",
        )
    else:
        parts.add("Confirm that no email will be sent.", "No problem, I won't send an email.")
    parts.add("Ask whether there is anything else you can help with.", "Is there anything else I can help you with today?")
    return _directive(
        Phase.POST_PROCESS,
        "Confirm the outcome of the email decision and check whether the caller needs anything else.",
        parts,
        facts={"email_summary": outcome, "email_on_file_masked": masked_email},
        prohibitions=["Do not claim anything was sent other than what email_summary says."],
    )


def anything_else_brief() -> Directive:
    parts = _Parts()
    parts.add(
        "Ask whether there is anything else about their claims you can help with.",
        "Is there anything else about your claims I can help you with?",
    )
    return _directive(Phase.POST_PROCESS, "Check whether the caller needs anything else before closing.", parts)


# -- terminal -----------------------------------------------------------------


def goodbye_brief(first_name: str | None, company: str, email_outcome: str | None = None, masked_email: str = "") -> Directive:
    parts = _Parts()
    if email_outcome == "sent":
        parts.add(f"Confirm the summary has been sent to {masked_email}.", f"I've sent the summary to {masked_email}.")
    elif email_outcome == "skipped":
        parts.add("Confirm that no email will be sent.", "I won't send an email.")
    parts.add(
        "Thank the caller and say goodbye warmly, in one or two sentences.",
        f"Thank you for contacting {company}{_comma_name(first_name)}. Take care, and goodbye.",
    )
    facts = {"caller_first_name": first_name}
    if email_outcome:
        facts.update({"email_summary": email_outcome, "email_on_file_masked": masked_email})
    return _directive(
        Phase.CLOSED,
        "Close the conversation.",
        parts,
        facts=facts,
        prohibitions=["Do not ask another question; the caller has said they are finished."],
    )


_HANDOFF_SCRIPTS = {
    "caller_requested_human": "Of course. I'm connecting you with a human representative now.",
    "verification_failed": (
        "I'm sorry, I wasn't able to verify your identity with the details provided, so for your security I "
        "can't continue here. I'm connecting you with a human representative who can verify you another way."
    ),
    "verification_declined": (
        "I understand, and I respect that. Because I can't share claim details without verifying identity, I'm "
        "connecting you with a human representative who can talk through other options with you."
    ),
    "representative_not_authorised": (
        "Since I can't discuss this policy with you directly, I'm connecting you with a human representative "
        "who can explain how to be added as an authorised representative."
    ),
    "consent_timeout": (
        "The policyholder hasn't approved the consent request, so I can't share claim information on this "
        "call. I'm passing this to a human representative who can arrange consent another way."
    ),
    "repeated_out_of_scope": (
        "I'm only able to help with insurance claims here, so I'm not the right resource for those questions. "
        "I'm connecting you with a human representative who can point you in the right direction."
    ),
    "agent_requested": "This needs a human representative to review, so I'm connecting you with one now.",
}


def handoff_brief(*, reason: str, detail: str, verified: bool) -> Directive:
    parts = _Parts()
    parts.add(
        f"Tell the caller you are connecting them with a human representative, and why: {ESCALATION_REASONS[reason]}"
        + (f" ({detail})" if detail else ""),
        _HANDOFF_SCRIPTS[reason],
    )
    if verified:
        parts.add(
            "Reassure them that what has been covered so far is passed along, so they will not have to repeat it.",
            "I've passed along what we've covered so you won't have to repeat yourself.",
        )
    else:
        parts.add(
            "Let them know the representative will need to verify their identity before discussing any claim.",
            "They'll need to verify your identity before discussing any claim details.",
        )
    prohibitions = ["Do not keep trying to persuade the caller; the decision to hand off is made."]
    if not verified:
        prohibitions.append("Do not share or hint at any claim, policy or account detail.")
    return _directive(
        Phase.ESCALATED,
        "Explain the handoff, kindly and briefly, and stop.",
        parts,
        facts={"handoff_reason": ESCALATION_REASONS[reason], "caller_is_verified": verified},
        prohibitions=prohibitions,
    )


def session_over_notice(phase: Phase) -> str:
    if phase is Phase.ESCALATED:
        return "This conversation has been handed to a human representative. Start a new conversation to try another scenario."
    return "This conversation has ended. Start a new conversation to continue."


# -- email --------------------------------------------------------------------


def email_template(
    *, first_name: str, company: str, today: date, cases: list[dict[str, Any]], on_behalf_by: str | None = None
) -> tuple[str, str]:
    """The harness's own summary email: fully grounded, used as reference and as fallback."""
    when = human_date(today.isoformat())
    subject = f"Summary of your conversation with {company} claims support"
    opening = (
        f"{on_behalf_by} contacted {company} claims support on your behalf on {when}. Here is a summary of that conversation."
        if on_behalf_by
        else f"Thank you for contacting {company} claims support on {when}. Here is a summary of our conversation."
    )
    lines = [f"Hello {first_name},", "", opening, "", "What we discussed"]
    if not cases:
        lines.append("- We verified your identity. No specific claim was reviewed in detail.")
    for item in cases:
        case = item["claim"]
        line = (
            f"- {INTENTS[item['intent']].label} for your {case['case_type']} claim {case['case_id']} "
            f"(filed {human_date(case['filed_on'])})."
        )
        if item["topics"]:
            line += f" We also went over: {join_and(item['topics'])}."
        lines.append(line)

    if cases:
        lines += ["", "Claim status"]
        for item in cases:
            case = item["claim"]
            if case["status"] == "denied":
                lines.append(f"- {case['case_id']}: denied because {case.get('denial_reason', 'of the reason on file')}.")
            elif case["status"] == "open":
                lines.append(
                    f"- {case['case_id']}: open and in progress. Expected reimbursement "
                    f"{money(case['expected_reimbursement_amount'])}; nothing paid yet."
                )
            else:
                lines.append(f"- {case['case_id']}: {case['status']}. Amount paid: {money(case['net_pay'])}.")

        lines += ["", "Next steps"]
        for item in cases:
            case = item["claim"]
            documents = case.get("documents_needed") or []
            if case["status"] == "denied" and documents:
                step = f"- {case['case_id']}: submit the {join_and(documents)}"
                if case.get("appeal_deadline"):
                    step += f" before the appeal deadline of {human_date(case['appeal_deadline'])}"
                lines.append(step + ". The member portal or claim upload link is the best place to start.")
            elif case["status"] == "open":
                lines.append(f"- {case['case_id']}: no action is needed from you right now.")
            else:
                lines.append(f"- {case['case_id']}: no further action is needed.")

    lines += [
        "",
        "If anything here looks wrong, or you have more questions, please contact claims support.",
        "",
        f"{company} Claims Support",
    ]
    return subject, "\n".join(lines)
