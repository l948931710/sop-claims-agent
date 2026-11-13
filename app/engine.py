"""The harness: a deterministic turn loop around a stateless Brain.

One caller message goes through six stages, always in this order:

  1. understand   the Brain reads the message into a structured record
  2. remember     everything useful is written to memory, whatever the phase
  3. policies     human request, scope strikes, emotion: phase-independent rules
  4. phase step   the current phase's gate runs in code and may move the phase
  5. respond      the Brain writes the reply for the directive, with gated tools
  6. guard        the reply is checked; a failing reply is retried, then replaced

Stages 4 and 5 repeat when the phase changes inside a turn, so one message can
carry the caller from VERIFY_ID through RESOLVE_INTENT into PROCESS_CASE.

The Brain never changes state. Phase changes happen only in Session.transition,
which enforces the SOP's transition table.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from . import briefs
from .brain.base import (
    Brain,
    BrainError,
    EmailContext,
    RespondContext,
    TurnUnderstanding,
    UnderstandContext,
)
from .config import Settings
from .data import Repository, mask_email, normalise_case_id
from .directive import Directive
from .guards import check_reply
from .identity import Verification, canonical, evaluate
from .sop import (
    ESCALATION_REASONS,
    IDENTITY_FIELDS,
    INTENTS,
    PHASES,
    TERMINAL_PHASES,
    TONE_PLAYBOOK,
    Autonomy,
    Limits,
    Phase,
)
from .state import Email, Handoff, Session, TurnTrace
from .tools import ToolGateway

_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]  # fmt: skip
_CASE_TYPE_WORDS = {
    "healthcare": {"healthcare", "health", "medical", "hospital", "doctor"},
    "dental": {"dental", "dentist", "teeth", "tooth"},
    "auto": {"auto", "car", "vehicle", "accident", "collision"},
}
_STATUS_WORDS = {
    "denied": {"denied", "denial", "rejected", "declined", "refused"},
    "open": {"open", "pending", "in progress", "processing", "ongoing"},
    "closed": {"closed", "settled", "paid", "completed", "resolved", "finished"},
}


@dataclass
class TurnContext:
    """Scratch space for one caller turn."""

    u: TurnUnderstanding
    text: str
    trace: TurnTrace
    start_phase: Phase
    brain: Brain
    reading_seconds: float = 0.0
    hop: int = 0
    events: list[str] = field(default_factory=list)
    tone: str | None = None
    scope_note: str | None = None
    decline_prefix: str | None = None
    pure_out_of_scope: bool = False
    offer_human: bool = False
    identity_supplied: bool = False
    unusable_fields: list[str] = field(default_factory=list)
    hint_captured: bool = False
    just_verified: bool = False
    just_selected: bool = False
    closing_email: tuple[str, str] | None = None  # (outcome, masked address) when the call closes on the email decision


class Harness:
    def __init__(self, repo: Repository, settings: Settings, limits: Limits | None = None) -> None:
        self.repo = repo
        self.settings = settings
        self.limits = limits or Limits()

    # -- public API ----------------------------------------------------------

    def new_session(self, consent_scenario: str = "default") -> Session:
        session = Session(consent_scenario=consent_scenario)
        session.add_turn("assistant", briefs.greeting(self.settings.company_name))
        return session

    def handle(self, session: Session, text: str, brain: Brain) -> str:
        """Process one caller message and return the reply. Raises BrainError only when the caller's
        setup is at fault (for example a missing API key) and nothing has been changed."""
        text = text.strip()
        if session.phase in TERMINAL_PHASES:
            return briefs.session_over_notice(session.phase)

        started = time.monotonic()
        try:
            understanding = brain.understand(self._understand_context(session, text))
        except BrainError as err:
            if err.user_fixable:
                raise
            # Without a reading of the message the harness cannot safely act on it. Nothing advances.
            session.turn += 1
            trace = TurnTrace(session.turn, session.phase.value, session.phase.value)
            trace.add("error", "could not read the caller's message; nothing was changed", {"error": str(err)}, ok=False)
            session.traces.append(trace)
            session.add_turn("user", text)
            reply = "I'm sorry, I had trouble with that on my end. Could you say that again?"
            session.add_turn("assistant", reply)
            return reply

        session.turn += 1
        trace = TurnTrace(session.turn, session.phase.value)
        session.traces.append(trace)
        session.add_turn("user", text)
        ctx = TurnContext(u=understanding, text=text, trace=trace, start_phase=session.phase, brain=brain)

        ctx.reading_seconds = time.monotonic() - started
        self._remember(session, ctx)
        self._apply_policies(session, ctx)
        reply = self._drive(session, ctx)

        session.add_turn("assistant", reply)
        trace.phase_after = session.phase.value
        trace.seconds = time.monotonic() - started
        return reply

    # -- stage 1: understand -------------------------------------------------

    def _understand_context(self, s: Session, text: str) -> UnderstandContext:
        selected = None
        if s.selected_case_id and s.party_id:
            claim = self.repo.claim(s.party_id, s.selected_case_id)
            if claim:
                selected = f"{claim['case_id']} ({claim['case_type']}, {claim['status']})"
        return UnderstandContext(
            phase=s.phase.value,
            latest_message=text,
            recent_turns=[(t.role, t.text) for t in s.transcript[-8:]],
            last_agent_message=s.last_assistant_text(),
            email_offer_pending=s.phase is Phase.POST_PROCESS and s.email.status == "offered",
            selected_case=selected,
            today=self.settings.today,
        )

    # -- stage 2: remember ---------------------------------------------------

    def _remember(self, s: Session, ctx: TurnContext) -> None:
        """Write whatever the caller said to memory. Runs in every phase, before any gate."""
        u, mem = ctx.u, s.memory
        updates: list[dict[str, Any]] = []

        def put(key: str, value: Any) -> None:
            if mem.remember(key, value, turn=s.turn, phase=s.phase, source=ctx.text):
                updates.append({"key": key, "value": value})

        raw_identity = {
            "full_name": u.identity.full_name,
            "dob": u.identity.date_of_birth,
            "phone": u.identity.phone,
            "email": u.identity.email,
            "id_last4": u.identity.id_last4,
            "policy_number": u.identity.policy_number,
        }
        for name, raw in raw_identity.items():
            if raw is None or not str(raw).strip():
                continue
            value = canonical(name, raw)
            if value is None:
                ctx.unusable_fields.append(name)
                continue
            ctx.identity_supplied = True
            put(f"identity.{name}", value)

        # The role decides whether the consent gate applies, so it is only taken during verification and
        # never downgraded: a caller who said they are a representative cannot talk their way out of consent.
        if s.phase is Phase.VERIFY_ID and u.caller_role != "unknown" and mem.get("caller.role") != "representative":
            put("caller.role", u.caller_role)
        if u.representative_name and u.representative_name.strip():
            put("rep.name", " ".join(u.representative_name.split()))
        if u.representative_relationship and u.representative_relationship.strip():
            put("rep.relationship", u.representative_relationship.strip().lower())

        hints = {k: v.strip() for k, v in u.case_hints.model_dump().items() if isinstance(v, str) and v.strip()}
        if "case_id" in hints:
            hints["case_id"] = normalise_case_id(hints["case_id"])
        if hints and mem.group_used("case"):
            # Hints that already led to a selected claim belong to that claim, not the next one.
            mem.drop_group("case")
        for key, value in hints.items():
            put(f"case.{key}", value)
        if u.intent != "none":
            put("intent.name", u.intent)
        ctx.hint_captured = any(item["key"].startswith(("case.", "intent.")) for item in updates)

        # An email wish stated before the offer is remembered, not acted on.
        offer_open = s.phase is Phase.POST_PROCESS and s.email.status in ("offered", "skipped")
        if u.email_decision != "none" and not offer_open:
            put("prefs.email_summary", u.email_decision)
        for note in u.notes:
            if note.strip() and mem.note(note.strip(), turn=s.turn, phase=s.phase, source=ctx.text):
                updates.append({"key": "note", "value": note.strip()})

        # A message that carries identity details is on-task, however the reader labelled the rest of it:
        # never purely off-topic, and not mere small talk either.
        if ctx.identity_supplied and u.scope == "out_of_scope":
            u.scope = "mixed"
        elif ctx.identity_supplied and u.scope == "social":
            u.scope = "in_scope"

        ctx.trace.add("nlu", f"understood the message ({ctx.reading_seconds:.1f}s)", _compact(u))
        if updates:
            ctx.trace.add("memory", f"remembered {len(updates)} item(s) during {s.phase.value}", {"items": updates})

    # -- stage 3: phase-independent policies ---------------------------------

    def _apply_policies(self, s: Session, ctx: TurnContext) -> None:
        u, lim = ctx.u, self.limits

        # Emotion: recognise it and set the tone before anything procedural.
        s.emotions.append({"turn": s.turn, "emotion": u.emotion, "intensity": u.emotion_intensity})
        if u.emotion != "neutral":
            ctx.tone = TONE_PLAYBOOK[u.emotion]
        heated = u.emotion in ("angry", "frustrated") and u.emotion_intensity == "high"
        s.heat_streak = s.heat_streak + 1 if heated else 0
        if s.heat_streak >= lim.heated_turns_before_offering_human:
            ctx.offer_human = True
        if u.emotion != "neutral":
            ctx.trace.add(
                "policy",
                f"emotion: {u.emotion} ({u.emotion_intensity})",
                {"heated_turns_in_a_row": s.heat_streak, "offer_human": ctx.offer_human},
            )

        # An explicit request for a person is honoured immediately, in any phase.
        if u.requests_human:
            self._escalate(s, ctx, "caller_requested_human")
            return

        # Scope: decline, count consecutive off-topic turns, hand off when they keep coming.
        if u.scope == "out_of_scope":
            s.oos_strikes += 1
            ctx.pure_out_of_scope = True
            ctx.trace.add(
                "policy",
                f"out of scope: strike {s.oos_strikes} of {lim.max_out_of_scope_strikes}",
                {"topic": u.out_of_scope_topic},
                ok=False,
            )
            if s.oos_strikes >= lim.max_out_of_scope_strikes:
                self._escalate(s, ctx, "repeated_out_of_scope")
                return
            mention_human = s.oos_strikes >= lim.max_out_of_scope_strikes - 1
            ctx.scope_note = briefs.scope_note(u.out_of_scope_topic, mention_human=mention_human, mixed=False)
            ctx.decline_prefix = briefs.decline_line(mention_human)
        elif u.scope == "mixed":
            ctx.scope_note = briefs.scope_note(u.out_of_scope_topic, mention_human=False, mixed=True)
            ctx.decline_prefix = briefs.decline_line(False)
            ctx.trace.add("policy", "partly out of scope: declining that part only", {"topic": u.out_of_scope_topic})
        elif u.scope == "in_scope":
            s.oos_strikes = 0

    # -- stages 4-6: drive the phases ----------------------------------------

    def _drive(self, s: Session, ctx: TurnContext) -> str:
        steps = {
            Phase.VERIFY_ID: self._verify_step,
            Phase.RESOLVE_INTENT: self._resolve_step,
            Phase.PROCESS_CASE: self._process_step,
            Phase.POST_PROCESS: self._post_step,
            Phase.CLOSED: self._terminal_step,
            Phase.ESCALATED: self._terminal_step,
        }
        directive: Directive | None = None
        for hop in range(self.limits.max_hops_per_turn):
            ctx.hop = hop
            directive = steps[s.phase](s, ctx)
            if directive is None:
                continue  # the gate moved the conversation on; run the next phase's step
            self._decorate(directive, ctx)
            reply = self._respond(s, directive, ctx)
            if reply is not None:
                return reply
            # a tool call moved the conversation on; run the next phase's step
        ctx.trace.add("error", "too many phase changes in one turn; using a safe reply", ok=False)
        if directive is not None:
            return directive.script
        return "Thank you for your patience. Could you tell me a little more about what you need?"

    def _decorate(self, d: Directive, ctx: TurnContext) -> None:
        """Add what is true of the whole turn, whatever the phase: events, tone, scope."""
        d.events = list(ctx.events)
        d.tone = ctx.tone
        d.scope_note = ctx.scope_note
        off_topic_only = ctx.pure_out_of_scope and d.phase not in TERMINAL_PHASES
        prefix: list[str] = []
        if ctx.u.emotion != "neutral" and d.phase is not Phase.CLOSED:
            prefix.append(briefs.EMPATHY[ctx.u.emotion])
        if ctx.decline_prefix and d.phase not in TERMINAL_PHASES:
            prefix.append(ctx.decline_prefix)
        if prefix:
            body = d.resume if off_topic_only and d.resume else d.script
            d.script = " ".join([*prefix, body])
        if ctx.offer_human and d.phase in (Phase.RESOLVE_INTENT, Phase.PROCESS_CASE, Phase.POST_PROCESS):
            d.required_points.append(
                "The caller has been upset for several turns. Offer, without pressure, to connect them with a "
                "human representative if they would prefer that."
            )
            d.script += " If you'd prefer, I can connect you with a human representative instead."
        if off_topic_only:
            # An off-topic turn gets no tools and no latitude: decline, then pick the SOP back up.
            d.tools = ()
            d.autonomy = Autonomy.SCRIPTED
            d.objective = "Decline the off-topic request (see SCOPE), then pick up where the call left off: " + d.objective

    def _respond(self, s: Session, d: Directive, ctx: TurnContext) -> str | None:
        """Ask the Brain for the reply and guard it. Returns None if a tool call changed the phase."""
        ctx.trace.add("directive", f"{d.phase.value} · {d.autonomy.value}", d.summary())
        transcript = [(t.role, t.text) for t in s.transcript]
        feedback: list[str] = []
        for attempt in (1, 2):
            gateway = ToolGateway(self.repo, s, d.tools, ctx.trace, self.settings.today)
            started = time.monotonic()
            try:
                result = ctx.brain.respond(
                    RespondContext(
                        directive=d,
                        transcript=transcript,
                        today=self.settings.today,
                        company_name=self.settings.company_name,
                        feedback=feedback,
                    ),
                    gateway,
                )
            except BrainError as err:
                ctx.trace.add("error", "model call failed; replied with the SOP script", {"error": str(err)}, ok=False)
                return d.script
            if gateway.pending_transition is not None:
                self._apply_tool_transition(s, ctx, gateway)
                return None
            reply = (result.text or "").strip()
            corpus = [
                json.dumps(d.facts, ensure_ascii=False),
                d.script,
                self.settings.today.isoformat(),
                *d.events,
                *gateway.outputs,
                *(t.text for t in s.transcript),
            ]
            violations = (
                check_reply(reply, corpus=corpus, policyholders=self.repo.policyholders())
                if reply
                else ["the reply was empty"]
            )
            if not violations:
                seconds = round(time.monotonic() - started, 1)
                ctx.trace.add("guard", f"reply written and checked ({seconds}s)", {"attempt": attempt, "seconds": seconds})
                return reply
            ctx.trace.add("guard", "reply rejected", {"attempt": attempt, "violations": violations}, ok=False)
            feedback = violations
        ctx.trace.add("guard", "replied with the SOP script instead", {})
        return d.script

    def _apply_tool_transition(self, s: Session, ctx: TurnContext, gateway: ToolGateway) -> None:
        target, code, detail = gateway.pending_transition  # type: ignore[misc]
        if target is Phase.ESCALATED:
            self._escalate(s, ctx, code, detail)
        elif target is Phase.PROCESS_CASE:
            s.transition(Phase.PROCESS_CASE, code)
            ctx.just_selected = True
            label = INTENTS[s.selected_intent].label if s.selected_intent else ""
            # The model's stated reason stays in the trace. Events feed the grounding check, so they
            # carry only what the harness itself knows.
            ctx.events.append(f"Claim {s.selected_case_id} was just selected (path: {label}) from what the caller said.")
        elif target is Phase.RESOLVE_INTENT:
            s.selected_case_id = None
            s.selected_intent = None
            s.transition(Phase.RESOLVE_INTENT, code)
            ctx.events.append("The caller asked about a different claim, so the call returned to claim selection.")

    # -- VERIFY_ID -----------------------------------------------------------

    def _verify_step(self, s: Session, ctx: TurnContext) -> Directive | None:
        u, lim = ctx.u, self.limits
        provided = s.memory.group("identity")
        v = evaluate(provided, self.repo.policyholders(), required=lim.min_identity_matches)
        previously_matched = s.verification.get("matched", []) if s.verification else []
        s.verification = v.to_dict()

        # A turn that introduces a value which does not match counts as one failed attempt.
        fresh = [(f, provided[f]) for f in v.mismatched if (f, provided[f]) not in s.counted_mismatches]
        if fresh:
            s.counted_mismatches.update(fresh)
            s.failed_attempts += 1
        ctx.trace.add(
            "gate",
            f"identity: {len(v.matched)} of {lim.min_identity_matches} required fields match",
            {**v.to_dict(), "failed_attempts": s.failed_attempts},
            ok=v.verified,
        )

        representative = s.memory.get("caller.role") == "representative"
        newly_matched = [f for f in v.matched if f not in previously_matched]
        if v.verified:
            if representative:
                passed, waiting = self._consent_gate(s, ctx, v)
                if not passed:
                    return waiting  # a directive while consent is outstanding, None after a handoff
            s.verified = True
            s.party_id = v.party_id
            s.stall_count = 0
            s.transition(Phase.RESOLVE_INTENT, "identity_verified")
            ctx.just_verified = True
            ctx.events.append(
                "The caller's identity was just verified (matched: "
                + ", ".join(IDENTITY_FIELDS[f] for f in v.matched)
                + ")."
            )
            return None

        if s.failed_attempts >= lim.max_failed_verifications:
            self._escalate(s, ctx, "verification_failed")
            return None

        pushback = u.refuses_to_provide_info or u.demands_bypass or u.asks_why_needed
        if (u.refuses_to_provide_info or u.demands_bypass) and not ctx.identity_supplied:
            s.stall_count += 1
            ctx.trace.add(
                "policy",
                f"caller pushed back without progress: {s.stall_count} of {lim.max_verification_stalls}",
                {},
                ok=False,
            )
        elif ctx.identity_supplied:
            s.stall_count = 0
        if s.stall_count >= lim.max_verification_stalls:
            # Stop persuading. Three explanations is enough; the caller gets a person.
            self._escalate(s, ctx, "verification_declined")
            return None

        return briefs.verify_brief(
            v=v,
            limits=lim,
            failed_attempts=s.failed_attempts,
            representative=representative,
            caller_reason=_caller_reason(s),
            reason_is_new=ctx.hint_captured,
            unusable=ctx.unusable_fields,
            pushback=pushback,
            offer_human=ctx.offer_human or s.stall_count >= lim.max_verification_stalls - 1,
            newly_matched=newly_matched,
        )

    def _consent_gate(self, s: Session, ctx: TurnContext, v: Verification) -> tuple[bool, Directive | None]:
        """Extra gate for a representative: (passed, directive to send while waiting).
        (False, None) means the gate handed the call to a human."""
        lim = self.limits
        rep_name = s.memory.get("rep.name")
        if not rep_name:
            return False, briefs.representative_identity_brief()
        if self.repo.authorised_representative(v.party_id or "", rep_name) is None:
            s.stall_count += 1
            ctx.trace.add("gate", "representative is not listed on the policy", {"stalls": s.stall_count}, ok=False)
            if s.stall_count >= lim.max_verification_stalls:
                self._escalate(s, ctx, "representative_not_authorised")
                return False, None
            return False, briefs.representative_not_authorised_brief(s.stall_count >= lim.max_verification_stalls - 1)

        consent = s.consent
        just_requested = consent.status == "not_required"
        if just_requested:
            consent.status = "pending"
            ctx.trace.add("gate", "consent request sent to the policyholder", {"scenario": consent.scenario})
        if consent.last_poll_turn != s.turn:
            status = self.repo.consent_status(consent.scenario, consent.polls)
            consent.polls += 1
            consent.last_poll_turn = s.turn
            ctx.trace.add(
                "gate",
                f"consent check {consent.polls} of {lim.max_consent_polls}: {status}",
                {},
                ok=status == "approved",
            )
            if status == "approved":
                consent.status = "approved"
        if consent.status == "approved":
            ctx.events.append("The policyholder approved the consent request.")
            return True, None
        if consent.polls >= lim.max_consent_polls:
            consent.status = "timed_out"
            self._escalate(s, ctx, "consent_timeout")
            return False, None
        u = ctx.u
        return False, briefs.consent_pending_brief(
            just_requested=just_requested,
            checks_left=lim.max_consent_polls - consent.polls,
            pushback=u.refuses_to_provide_info or u.demands_bypass or u.asks_why_needed,
        )

    # -- RESOLVE_INTENT ------------------------------------------------------

    def _resolve_step(self, s: Session, ctx: TurnContext) -> Directive | None:
        u = ctx.u
        if ctx.hop == 0 and u.wants_to_end and not u.has_open_request:
            s.transition(Phase.POST_PROCESS, "caller_has_nothing_to_resolve")
            return None
        index = self.repo.claims_index(s.party_id or "")
        hints = s.memory.group("case")
        remembered = {
            "what_they_want": s.memory.get("intent.name"),
            "which_claim": hints,
            "notes": [n.value for n in s.memory.notes],
        }
        return briefs.resolve_brief(
            first_name=self._first_name(s),
            caller=self._caller(s),
            claims_index=index,
            remembered=remembered,
            candidates=match_claims(index, hints, self.settings.today),
            just_verified=ctx.just_verified,
            today=self.settings.today,
        )

    # -- PROCESS_CASE --------------------------------------------------------

    def _process_step(self, s: Session, ctx: TurnContext) -> Directive | None:
        u = ctx.u
        if ctx.hop == 0 and u.wants_to_end and not u.has_open_request:
            s.transition(Phase.POST_PROCESS, "caller_is_done_with_the_case")
            return None
        case = self.repo.case_file(s.party_id or "", s.selected_case_id or "", self.settings.today)
        if case is None:  # cannot happen through the gateway; fail safe rather than guess
            s.selected_case_id = None
            s.transition(Phase.RESOLVE_INTENT, "no_open_claim")
            return None
        return briefs.process_brief(
            first_name=self._first_name(s),
            caller=self._caller(s),
            case=case,
            intent=s.selected_intent or "general_claim_question",
            glossary=self.repo.field_glossary(),
            notes=[n.value for n in s.memory.notes],
            just_selected=ctx.just_selected,
            just_verified=ctx.just_verified,
            today=self.settings.today,
        )

    # -- POST_PROCESS --------------------------------------------------------

    def _post_step(self, s: Session, ctx: TurnContext) -> Directive | None:
        u, email = ctx.u, s.email
        arrived_this_turn = ctx.start_phase is not Phase.POST_PROCESS

        if not arrived_this_turn and ctx.hop == 0 and u.has_open_request and u.scope in ("in_scope", "mixed"):
            if email.status == "offered":
                email.status = "not_offered"  # the offer is made again at the next wrap-up
            target = Phase.PROCESS_CASE if s.selected_case_id else Phase.RESOLVE_INTENT
            s.transition(target, "caller_has_another_request")
            ctx.events.append("The caller has another question, so the call went back to the claim.")
            return None

        address = self.repo.policyholder(s.party_id or "").get("email")
        masked = mask_email(address) if address else ""
        remembered = s.memory.get("prefs.email_summary")

        decision = u.email_decision if u.email_decision != "none" else None
        remembered_skip = False
        if decision is None and email.status == "not_offered" and remembered == "skip":
            decision, remembered_skip = "skip", True
        if email.status == "sent" and decision == "send":
            decision = None  # already sent this conversation

        decided_now = False
        if not address and email.status == "not_offered":
            email.status, email.decided_turn, decided_now = "unavailable", s.turn, True
        elif decision == "send" and email.status in ("not_offered", "offered", "skipped"):
            self._send_summary(s, ctx, address or "")
            email.decided_turn, decided_now = s.turn, True
            ctx.trace.add("gate", "email consent: yes, summary sent", {"to": masked})
        elif decision == "skip" and email.status in ("not_offered", "offered"):
            email.status, email.decided_turn, decided_now = "skipped", s.turn, True
            ctx.trace.add("gate", "email consent: no, nothing sent", {"remembered": remembered_skip})

        if decided_now:
            if u.email_decision != "none" and u.wants_to_end and not u.has_open_request:
                # "That's all, and don't email me": the decision and the goodbye came together, so close.
                ctx.closing_email = (email.status, masked)
                s.transition(Phase.CLOSED, "wrap_up_complete")
                return None
            return briefs.email_outcome_brief(outcome=email.status, masked_email=masked, remembered_skip=remembered_skip)

        if email.status in ("not_offered", "offered"):
            reask = email.status == "offered"
            email.status = "offered"
            ctx.trace.add("gate", "email consent: waiting for a clear yes or no", {}, ok=False)
            return briefs.email_offer_brief(
                masked_email=masked,
                remembered_wish=remembered == "send" and not reask,
                reask=reask,
                representative=self._is_representative(s),
            )

        if u.scope in ("out_of_scope", "mixed"):
            return briefs.anything_else_brief()
        # The email decision was made on an earlier turn and the caller has nothing new.
        s.transition(Phase.CLOSED, "wrap_up_complete")
        return None

    def _send_summary(self, s: Session, ctx: TurnContext, address: str) -> None:
        # The summary goes to the policyholder's address, so it is written to the policyholder.
        first_name = self._policyholder_first_name(s) or "there"
        cases = self._summary_cases(s)
        subject, body = briefs.email_template(
            first_name=first_name,
            company=self.settings.company_name,
            today=self.settings.today,
            cases=cases,
            on_behalf_by=s.memory.get("rep.name") if self._is_representative(s) else None,
        )
        drafted_by = "template"
        # Only the conversation after verification: the identity details the caller gave are not
        # material for an email, and leaving them out means the guard rejects a draft that has them.
        transcript = [(t.role, t.text) for t in s.transcript if t.phase != Phase.VERIFY_ID.value]
        try:
            draft = ctx.brain.draft_email(
                EmailContext(
                    caller_first_name=first_name,
                    company_name=self.settings.company_name,
                    today=self.settings.today,
                    cases=cases,
                    transcript=transcript,
                    template=body,
                )
            )
            corpus = [json.dumps(cases, ensure_ascii=False), body, self.settings.today.isoformat(), *(t for _, t in transcript)]
            violations = check_reply(
                f"{draft.subject}\n{draft.body}", corpus=corpus, policyholders=self.repo.policyholders()
            )
            if draft.body.strip() and draft.subject.strip() and not violations:
                subject, body, drafted_by = draft.subject.strip(), draft.body.strip(), ctx.brain.name
            else:
                ctx.trace.add("guard", "email draft rejected; sent the template instead", {"violations": violations}, ok=False)
        except BrainError as err:
            ctx.trace.add("error", "email draft failed; sent the template instead", {"error": str(err)}, ok=False)
        s.outbox.append(Email(mask_email(address), subject, body, s.turn, drafted_by))
        s.email.status = "sent"
        ctx.trace.add("email", "summary email sent", {"to": mask_email(address), "drafted_by": drafted_by})

    def _summary_cases(self, s: Session) -> list[dict[str, Any]]:
        items = []
        for log in s.cases:
            case = self.repo.case_file(s.party_id or "", log.case_id, self.settings.today)
            if case is not None:
                items.append({"claim": case, "intent": log.intent, "topics": list(log.topics)})
        return items

    # -- terminal phases -----------------------------------------------------

    def _terminal_step(self, s: Session, ctx: TurnContext) -> Directive:
        if s.phase is Phase.ESCALATED and s.handoff is not None:
            return briefs.handoff_brief(reason=s.handoff.reason, detail=s.handoff.detail, verified=s.verified)
        outcome, masked = ctx.closing_email or (None, "")
        return briefs.goodbye_brief(self._first_name(s), self.settings.company_name, outcome, masked)

    def _escalate(self, s: Session, ctx: TurnContext, reason: str, detail: str = "") -> None:
        s.handoff = Handoff(reason, ESCALATION_REASONS[reason], detail, self._handoff_summary(s))
        s.transition(Phase.ESCALATED, reason)

    def _handoff_summary(self, s: Session) -> list[str]:
        lines: list[str] = []
        if s.verified:
            lines.append(f"Identity: verified ({self._first_name(s)}, party {s.party_id}).")
        else:
            matched = len(s.verification.get("matched", []))
            lines.append(f"Identity: NOT verified ({matched} of {self.limits.min_identity_matches} fields matched).")
        if s.memory.get("caller.role") == "representative":
            lines.append(f"Caller is a representative; consent status: {s.consent.status}.")
        reason = _caller_reason(s)
        if reason:
            lines.append(f"Caller's stated need: {reason}.")
        if s.selected_case_id:
            lines.append(f"Claim in focus: {s.selected_case_id} ({s.selected_intent}).")
        moods = [e["emotion"] for e in s.emotions if e["emotion"] != "neutral"]
        if moods:
            lines.append("Emotion during the call: " + ", ".join(dict.fromkeys(moods)) + ".")
        if s.oos_strikes:
            lines.append(f"Off-topic requests in a row: {s.oos_strikes}.")
        return lines

    # -- helpers -------------------------------------------------------------

    def _is_representative(self, s: Session) -> bool:
        return s.memory.get("caller.role") == "representative"

    def _policyholder_first_name(self, s: Session) -> str | None:
        if s.verified and s.party_id:
            return self.repo.policyholder(s.party_id)["name"].split()[0]
        return None

    def _first_name(self, s: Session) -> str | None:
        """How to address the person on the line: the representative, when there is one."""
        if not s.verified:
            return None
        if self._is_representative(s) and s.memory.get("rep.name"):
            return str(s.memory.get("rep.name")).split()[0]
        return self._policyholder_first_name(s)

    def _caller(self, s: Session) -> dict[str, Any]:
        """Who is on the line and whose claims these are, for the model-led phases."""
        policyholder = self._policyholder_first_name(s)
        if self._is_representative(s):
            return {
                "first_name": self._first_name(s),
                "role": "authorised representative (" + str(s.memory.get("rep.relationship") or "relationship not stated") + ")",
                "policyholder_first_name": policyholder,
                "claims_belong_to": f"{policyholder}'s",
            }
        return {"first_name": policyholder, "role": "policyholder", "claims_belong_to": "your"}

    def snapshot(self, s: Session) -> dict[str, Any]:
        """Everything the test UI's inspector shows. Contains nothing from the records on file before
        verification, and afterwards only the verified policyholder's first name and masked email."""
        lim = self.limits
        verification = dict(s.verification) if s.verification else evaluate({}, [], lim.min_identity_matches).to_dict()
        verification["labels"] = IDENTITY_FIELDS
        return {
            "session_id": s.id,
            "phase": s.phase.value,
            "turn": s.turn,
            "phases": [
                {"phase": spec.phase.value, "title": spec.title, "autonomy": spec.autonomy.value, "tools": list(spec.tools)}
                for spec in PHASES.values()
                if spec.phase not in TERMINAL_PHASES
            ],
            "verified": s.verified,
            "caller_first_name": self._first_name(s),
            "caller_role": s.memory.get("caller.role", "policyholder"),
            "verification": verification,
            "counters": {
                "failed_attempts": [s.failed_attempts, lim.max_failed_verifications],
                "stalls": [s.stall_count, lim.max_verification_stalls],
                "out_of_scope_strikes": [s.oos_strikes, lim.max_out_of_scope_strikes],
                "heated_turns": [s.heat_streak, lim.heated_turns_before_offering_human],
            },
            "consent": {"status": s.consent.status, "scenario": s.consent.scenario, "polls": s.consent.polls},
            "memory": s.memory.to_dict(),
            "selected_case_id": s.selected_case_id,
            "selected_intent": s.selected_intent,
            "email": {"status": s.email.status},
            "outbox": [e.to_dict() for e in s.outbox],
            "handoff": s.handoff.to_dict() if s.handoff else None,
            "ended": s.phase in TERMINAL_PHASES,
            "traces": [t.to_dict() for t in s.traces],
        }


def _caller_reason(s: Session) -> str | None:
    """The caller's own description of why they are calling, if they have given one."""
    description = s.memory.get("case.description")
    if description:
        return description
    intent = s.memory.get("intent.name")
    return INTENTS[intent].label.lower() if intent in INTENTS else None


def _compact(u: TurnUnderstanding) -> dict[str, Any]:
    """The understanding with empty fields dropped, for the trace."""
    data = u.model_dump()
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if v not in (None, "", [])}
        if value in (None, "", [], {}, False, "none", "unknown"):
            continue
        out[key] = value
    return out


def match_claims(index: list[dict[str, Any]], hints: dict[str, Any], today: date) -> list[str]:
    """Claims consistent with what the caller has said. A simple, conservative filter: a hint it does
    not understand is ignored rather than guessed at. The model sees this as a suggestion, and
    select_case is validated either way."""
    if not hints:
        return []
    if hints.get("case_id"):
        return [c["case_id"] for c in index if c["case_id"] == hints["case_id"]]

    def consistent(claim: dict[str, Any]) -> bool:
        text = " ".join(str(hints.get(k, "")) for k in ("case_type", "status", "description")).lower()
        types = {name for name, words in _CASE_TYPE_WORDS.items() if any(_has_word(text, w) for w in words)}
        if types and claim["case_type"] not in types:
            return False
        statuses = {name for name, words in _STATUS_WORDS.items() if any(_has_word(text, w) for w in words)}
        if statuses and claim["status"] not in statuses:
            return False
        when = _when_text(hints)
        filed = date.fromisoformat(claim["filed_on"])
        months = _months_named(hints)
        if months and filed.month not in months:
            return False
        years = {int(y) for y in re.findall(r"\b(20\d{2})\b", when)}
        if years and filed.year not in years:
            return False
        if "last year" in when and filed.year != today.year - 1:
            return False
        if "this year" in when and filed.year != today.year:
            return False
        return True

    matches = [c for c in index if consistent(c)]
    if len(matches) > 1 and _months_named(hints) and not _names_a_year(hints):
        # "from January" with no year most naturally means the most recent January.
        latest = max(date.fromisoformat(c["filed_on"]).year for c in matches)
        recent = [c for c in matches if date.fromisoformat(c["filed_on"]).year == latest]
        if len(recent) == 1:
            matches = recent
    return [c["case_id"] for c in matches]


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _when_text(hints: dict[str, Any]) -> str:
    return " ".join(str(hints.get(k, "")) for k in ("time_reference", "description")).lower()


def _months_named(hints: dict[str, Any]) -> set[int]:
    """Months the caller named. 'may' counts only in the time reference, where it cannot be the verb."""
    time_reference = str(hints.get("time_reference", "")).lower()
    description = str(hints.get("description", "")).lower()
    found: set[int] = set()
    for number, name in enumerate(_MONTHS, start=1):
        if _has_word(time_reference, name) or _has_word(time_reference, name[:3]):
            found.add(number)
        elif name != "may" and _has_word(description, name):
            found.add(number)
    return found


def _names_a_year(hints: dict[str, Any]) -> bool:
    when = _when_text(hints)
    return bool(re.search(r"\b20\d{2}\b", when)) or "last year" in when or "this year" in when
