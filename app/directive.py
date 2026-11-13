"""The directive: what the harness tells the model for one reply.

Each turn the engine works out, deterministically, what has to happen next and
hands the model a directive. The model's freedom is whatever the directive
leaves open:

  scripted  every content point is listed; the model chooses wording and tone
  guided    the model chooses between a few named actions
  open      the model answers freely from `facts` and tool results

`script` is a plain reply the harness wrote itself that already satisfies the
SOP. It is what the caller gets if the model call fails or its reply is
rejected by a guard, so the conversation never depends on the model behaving.
`resume` is the short form used after declining an off-topic request, where
repeating the whole script would be noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .sop import Autonomy, Phase


@dataclass
class Directive:
    phase: Phase
    autonomy: Autonomy
    objective: str
    script: str
    required_points: list[str] = field(default_factory=list)
    prohibitions: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    tools: tuple[str, ...] = ()
    playbook: str | None = None
    resume: str | None = None  # one line that picks the call back up after declining an off-topic request
    # filled in by the engine after the phase step
    events: list[str] = field(default_factory=list)  # what the harness did this turn
    tone: str | None = None
    scope_note: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "autonomy": self.autonomy.value,
            "objective": self.objective,
            "required_points": self.required_points,
            "prohibitions": self.prohibitions,
            "tools": list(self.tools),
            "events": self.events,
            "tone": self.tone,
            "scope_note": self.scope_note,
            "fact_keys": sorted(self.facts),
        }
