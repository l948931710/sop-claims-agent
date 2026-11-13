"""The tool gateway: the only way the model can read claim guidance or change state.

Three checks happen here, in code, on every call:

  1. the tool is one the current phase allows (not merely one the model was shown)
  2. the arguments are valid
  3. anything about a claim is looked up under the verified party id, so a
     claim belonging to someone else cannot be selected or read

A tool that changes phase does not change it directly. It records a request
and the engine applies it through Session.transition, which enforces the SOP's
transition table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from .data import Repository, normalise_case_id
from .sop import FOLLOWUP_TOPICS, INTENTS, PHASES, Phase
from .state import CaseLog, Session, TurnTrace


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    properties: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": self.properties,
                "required": list(self.properties),
                "additionalProperties": False,
            },
        }


TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "select_case",
            "Commit to the claim the caller is asking about and the workflow path that fits their need. "
            "Call this as soon as the caller's words (including what they said earlier in the call) identify "
            "exactly one claim on file. If more than one claim fits, ask the caller instead of guessing.",
            {
                "case_id": {"type": "string", "description": "Claim number from the claims on file, e.g. CL-2048."},
                "intent": {
                    "type": "string",
                    "enum": list(INTENTS),
                    "description": "The workflow path that best matches what the caller wants.",
                },
                "reason": {"type": "string", "description": "One sentence: why this claim and this path."},
            },
        ),
        ToolSpec(
            "get_document_guidance",
            "Look up what a required document must contain and what to do if the caller cannot get it. "
            "Call this before describing requirements or substitutes for any document; do not answer those "
            "from general knowledge.",
            {"document": {"type": "string", "description": "Document name as listed on the claim, e.g. 'pathology report'."}},
        ),
        ToolSpec(
            "get_followup_guidance",
            "Look up the claim-specific rule for a follow-up question about submitting documents: timing, "
            "method, format, processing time, receipt confirmation, or alternatives when a document is missing. "
            "Call this before answering any of those.",
            {
                "topic": {
                    "type": "string",
                    "enum": list(FOLLOWUP_TOPICS),
                    "description": "; ".join(f"{name}: {desc}" for name, desc in FOLLOWUP_TOPICS.items()),
                }
            },
        ),
        ToolSpec(
            "switch_case",
            "Call this when the caller starts asking about a different claim than the one currently open. "
            "It returns the conversation to claim selection.",
            {"reason": {"type": "string", "description": "What the caller asked for, in a few words."}},
        ),
        ToolSpec(
            "request_human_handoff",
            "Hand the conversation to a human representative. Call this when the guidance says a human must "
            "review (for example every document alternative is exhausted), or when the caller needs something "
            "the claim data and guidance cannot answer and they want it resolved.",
            {"reason": {"type": "string", "description": "Why a human is needed, in one sentence."}},
        ),
    )
}


@dataclass
class ToolOutcome:
    content: str
    is_error: bool = False


class ToolGateway:
    def __init__(
        self,
        repo: Repository,
        session: Session,
        allowed: tuple[str, ...],
        trace: TurnTrace,
        today: date,
    ) -> None:
        self._repo = repo
        self._session = session
        self._allowed = allowed
        self._trace = trace
        self._today = today
        self.outputs: list[str] = []  # successful results, for the grounding check
        self.pending_transition: tuple[Phase, str, str] | None = None  # target, reason code, detail
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "select_case": self._select_case,
            "get_document_guidance": self._get_document_guidance,
            "get_followup_guidance": self._get_followup_guidance,
            "switch_case": self._switch_case,
            "request_human_handoff": self._request_human_handoff,
        }

    def schemas(self) -> list[dict[str, Any]]:
        return [TOOL_SPECS[name].schema() for name in self._allowed]

    @property
    def allowed(self) -> tuple[str, ...]:
        return self._allowed

    def call(self, name: str, args: Any) -> ToolOutcome:
        phase = self._session.phase
        if name not in self._handlers:
            return self._deny(name, args, f"There is no tool named {name}.")
        if name not in PHASES[phase].tools or name not in self._allowed:
            return self._deny(name, args, f"{name} is not available at this point in the call ({phase.value}).")
        if self.pending_transition is not None:
            return self._deny(name, args, "The conversation is already moving to its next step; no further tool calls.")
        if not isinstance(args, dict):
            return self._deny(name, args, "Arguments must be an object.")
        missing = [key for key in TOOL_SPECS[name].properties if not str(args.get(key, "")).strip()]
        if missing:
            return self._deny(name, args, f"Missing argument(s): {', '.join(missing)}.")
        try:
            result = self._handlers[name](args)
        except ToolRefused as refusal:
            return self._deny(name, args, str(refusal))
        content = json.dumps(result, ensure_ascii=False)
        self.outputs.append(content)
        self._trace.add("tool", name, {"args": args, "result": result})
        return ToolOutcome(content)

    def _deny(self, name: str, args: Any, message: str) -> ToolOutcome:
        self._trace.add("tool", f"{name} — refused", {"args": args, "error": message}, ok=False)
        return ToolOutcome(json.dumps({"error": message}), is_error=True)

    # -- handlers ------------------------------------------------------------

    def _open_claim(self) -> dict[str, Any]:
        s = self._session
        claim = self._repo.claim(s.party_id, s.selected_case_id) if s.party_id and s.selected_case_id else None
        if claim is None:
            raise ToolRefused("No claim is open.")
        return claim

    def _select_case(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._session
        if not s.verified or s.party_id is None:
            raise ToolRefused("The caller is not verified.")
        if args["intent"] not in INTENTS:
            raise ToolRefused(f"Unknown intent. Choose one of: {', '.join(INTENTS)}.")
        case_id = normalise_case_id(str(args["case_id"]))
        claim = self._repo.claim(s.party_id, case_id)
        if claim is None:
            # Same message whether the claim does not exist or belongs to someone else.
            on_file = ", ".join(c["case_id"] for c in self._repo.claims_for(s.party_id)) or "none"
            raise ToolRefused(f"{case_id} is not a claim on this policy. Claims on file: {on_file}.")
        s.selected_case_id = claim["case_id"]
        s.selected_intent = args["intent"]
        log = s.current_case_log()
        if log is None:
            s.cases.append(CaseLog(claim["case_id"], args["intent"]))
        else:
            log.intent = args["intent"]
        s.memory.mark_used("case")
        s.memory.mark_used("intent")
        self.pending_transition = (Phase.PROCESS_CASE, "case_selected", str(args["reason"]))
        return {"selected": claim["case_id"], "intent": args["intent"]}

    def _get_document_guidance(self, args: dict[str, Any]) -> dict[str, Any]:
        claim = self._open_claim()
        result = self._repo.document_guidance(claim, str(args["document"]))
        self._log_topic(f"requirements for the {args['document']}")
        return result

    def _get_followup_guidance(self, args: dict[str, Any]) -> dict[str, Any]:
        claim = self._open_claim()
        topic = str(args["topic"])
        if topic not in FOLLOWUP_TOPICS:
            raise ToolRefused(f"Unknown topic. Choose one of: {', '.join(FOLLOWUP_TOPICS)}.")
        result = self._repo.followup_guidance(claim, topic)
        self._log_topic(topic.replace("_", " "))
        return result

    def _switch_case(self, args: dict[str, Any]) -> dict[str, Any]:
        self._open_claim()
        self.pending_transition = (Phase.RESOLVE_INTENT, "caller_changed_claim", str(args["reason"]))
        return {"ok": True}

    def _request_human_handoff(self, args: dict[str, Any]) -> dict[str, Any]:
        self.pending_transition = (Phase.ESCALATED, "agent_requested", str(args["reason"]))
        return {"ok": True}

    def _log_topic(self, topic: str) -> None:
        log = self._session.current_case_log()
        if log is not None and topic not in log.topics:
            log.topics.append(topic)


class ToolRefused(Exception):
    pass
