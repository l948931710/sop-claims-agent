"""Runtime settings, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The starter fixtures describe a moment in early March 2026 (claims filed late
# February, appeal deadlines in mid March). Pinning the clock keeps "from
# January" and "days until the appeal deadline" coherent no matter when the
# demo is run. Set DEMO_TODAY=real to use the wall clock instead.
DEFAULT_DEMO_TODAY = "2026-03-02"

# Two models, two jobs. The writer needs judgement and tone. The reader fills a fixed record from one
# message, which the smaller model does as well in the scenarios tried and 1.5 to 3 seconds faster a turn.
DEFAULT_WRITER_MODEL = "claude-opus-5-5"
DEFAULT_READER_MODEL = "claude-haiku-5-5"

# Models that accept the server-side refusal fallback parameter.
FALLBACK_CAPABLE_MODELS = frozenset(
    {
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
        "claude-fable-5-2",
        "claude-fable-5-1",
    }
)


@dataclass(frozen=True)
class Settings:
    provider: str  # "anthropic" | "stub"
    model: str  # writes the replies
    nlu_model: str  # reads each caller message into a structured record
    effort: str
    fallbacks_enabled: bool
    fixtures_dir: Path
    today: date
    company_name: str

    @property
    def server_key_configured(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _env(name: str, default: str) -> str:
    """An environment value, treating unset and blank the same (compose passes blanks through)."""
    return os.environ.get(name, "").strip() or default


def _demo_today() -> date:
    raw = _env("DEMO_TODAY", DEFAULT_DEMO_TODAY)
    if raw.lower() == "real":
        return date.today()
    return date.fromisoformat(raw)


def load_settings() -> Settings:
    return Settings(
        provider=_env("LLM_PROVIDER", "anthropic").lower(),
        model=_env("SOP_MODEL", DEFAULT_WRITER_MODEL),
        nlu_model=_env("SOP_NLU_MODEL", DEFAULT_READER_MODEL),
        effort=_env("SOP_EFFORT", "low").lower(),
        fallbacks_enabled=_env("ANTHROPIC_FALLBACKS", "default").lower() != "off",
        fixtures_dir=Path(_env("FIXTURES_DIR", str(ROOT / "fixtures" / "insurance_claims"))),
        today=_demo_today(),
        company_name=_env("COMPANY_NAME", "Acme Insurance"),
    )
