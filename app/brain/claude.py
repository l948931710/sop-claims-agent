"""The Brain backed by the Claude API.

Three calls, each narrow:

  understand   structured output (the reader prompt); no tools, no caller-facing text
  respond      the writer prompt plus the turn's brief; a short tool loop through the gateway
  draft_email  structured output from facts the harness supplies

The tool loop is hand-written rather than left to the SDK's tool runner because
the harness needs to stop the loop the moment a tool changes the phase.
"""

from __future__ import annotations

import threading
from typing import Any

import anthropic

from ..config import FALLBACK_CAPABLE_MODELS, Settings
from ..sop import Limits
from ..tools import ToolGateway
from . import prompts
from .base import (
    BrainError,
    EmailContext,
    EmailDraft,
    RespondContext,
    RespondResult,
    TurnUnderstanding,
    UnderstandContext,
)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 8000
REQUEST_TIMEOUT_SECONDS = 60.0  # a chat turn should fail over to the SOP script, not hang

def make_client(api_key: str | None = None) -> anthropic.Anthropic:
    """A client with chat-sized limits. With no key it resolves credentials from the environment
    (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile)."""
    options: dict[str, Any] = {"timeout": REQUEST_TIMEOUT_SECONDS, "max_retries": 1}
    if api_key:
        options["api_key"] = api_key
    return anthropic.Anthropic(**options)


# Set the first time the API rejects the fallback parameter (a key or platform without that beta),
# so later calls skip it instead of paying for a rejected request every turn.
_fallbacks_rejected = threading.Event()


class ClaudeBrain:
    name = "claude"

    def __init__(
        self,
        settings: Settings,
        api_key: str | None = None,
        client: Any = None,
        limits: Limits | None = None,
    ) -> None:
        self._settings = settings
        self._limits = limits or Limits()
        self._client = client if client is not None else make_client(api_key)

    # -- the three roles -----------------------------------------------------

    def understand(self, ctx: UnderstandContext) -> TurnUnderstanding:
        response = self._call(
            "parse",
            self._settings.nlu_model,
            system=prompts.READER_SYSTEM.format(company=self._settings.company_name),
            messages=[{"role": "user", "content": prompts.reader_input(ctx)}],
            output_format=TurnUnderstanding,
        )
        parsed = response.parsed_output
        if parsed is None:
            raise BrainError("The model returned no structured reading of the message.")
        return parsed

    def respond(self, ctx: RespondContext, tools: ToolGateway) -> RespondResult:
        system = [
            {"type": "text", "text": prompts.WRITER_SYSTEM.format(company=ctx.company_name)},
            {"type": "text", "text": prompts.render_brief(ctx.directive, ctx)},
        ]
        messages: list[dict[str, Any]] = list(prompts.writer_messages(ctx.transcript))
        schemas = tools.schemas()
        extra: dict[str, Any] = {"tools": schemas} if schemas else {}

        for _ in range(self._limits.max_tool_rounds + 1):
            response = self._call("create", self._settings.model, system=system, messages=messages, **extra)
            calls = [block for block in response.content if block.type == "tool_use"]
            if not calls:
                text = "\n".join(block.text for block in response.content if block.type == "text").strip()
                return RespondResult(text=text)

            # Thinking and tool_use blocks go back exactly as received.
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for call in calls:
                outcome = tools.call(call.name, call.input)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": outcome.content,
                        "is_error": outcome.is_error,
                    }
                )
            if tools.pending_transition is not None:
                # The phase is changing; the engine re-briefs the model for the new phase.
                return RespondResult(transitioned=True)
            messages.append({"role": "user", "content": results})

        raise BrainError("The model kept calling tools without replying.")

    def draft_email(self, ctx: EmailContext) -> EmailDraft:
        response = self._call(
            "parse",
            self._settings.model,
            system=prompts.EMAIL_SYSTEM.format(company=ctx.company_name),
            messages=[{"role": "user", "content": prompts.email_input(ctx)}],
            output_format=EmailDraft,
        )
        parsed = response.parsed_output
        if parsed is None:
            raise BrainError("The model returned no email draft.")
        return parsed

    # -- one place that talks to the API -------------------------------------

    def _call(self, method: str, model: str, **kwargs: Any) -> Any:
        kwargs["max_tokens"] = MAX_TOKENS
        if self._settings.effort not in ("", "off", "none"):
            kwargs["output_config"] = {"effort": self._settings.effort}

        with_fallbacks = (
            self._settings.fallbacks_enabled and model in FALLBACK_CAPABLE_MODELS and not _fallbacks_rejected.is_set()
        )
        try:
            try:
                response = self._send(method, model, with_fallbacks, kwargs)
            except anthropic.BadRequestError as err:
                if not (with_fallbacks and "fallback" in str(err).lower()):
                    raise
                _fallbacks_rejected.set()
                response = self._send(method, model, False, kwargs)
        except anthropic.AuthenticationError as err:
            raise BrainError("The API key was rejected. Check the key and try again.", user_fixable=True) from err
        except anthropic.PermissionDeniedError as err:
            raise BrainError(f"This API key is not allowed to use {model}.", user_fixable=True) from err
        except anthropic.NotFoundError as err:
            raise BrainError(f"The model '{model}' was not found. Check SOP_MODEL.", user_fixable=True) from err
        except anthropic.RateLimitError as err:
            raise BrainError("The model API is rate limiting requests. Try again shortly.") from err
        except anthropic.APIStatusError as err:
            raise BrainError(f"The model API returned an error ({err.status_code}): {err.message}") from err
        except anthropic.APIConnectionError as err:
            raise BrainError("Could not reach the model API.") from err
        except anthropic.AnthropicError as err:
            # Raised before any request when no credentials can be resolved.
            raise BrainError(f"The model client is not configured: {err}", user_fixable=True) from err

        if response.stop_reason == "refusal":
            raise BrainError("The model declined to respond to this turn.")
        if response.stop_reason == "max_tokens":
            raise BrainError("The model's reply was cut off.")
        return response

    def _send(self, method: str, model: str, with_fallbacks: bool, kwargs: dict[str, Any]) -> Any:
        if with_fallbacks:
            # If a safety classifier declines the request, the API retries it on the model Anthropic
            # recommends for that category, in the same call.
            call = getattr(self._client.beta.messages, method)
            return call(model=model, betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        return getattr(self._client.messages, method)(model=model, **kwargs)
