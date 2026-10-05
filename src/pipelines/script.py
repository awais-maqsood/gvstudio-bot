"""Deterministic script LLM — STT keywords → fixed TTS + tools (no cloud/local model)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from ..logging_config import get_logger

from .base import LLMComponent, LLMResponse

logger = get_logger(__name__)

_YES_RE = re.compile(
    r"\b("
    r"yes|yeah|yep|yup|yah|yea|ya|"
    r"sure|correct|right|absolutely|definitely|"
    r"i\s*do|i\s*have|i\s*am|i'?m|"
    r"uh[\s\-]?huh|mm[\s\-]?hmm|mhm|affirmative|"
    r"ok|okay|of\s*course"
    r")\b",
    re.IGNORECASE,
)
_NO_RE = re.compile(
    r"\b("
    r"no|nope|nah|negative|"
    r"i\s*don'?t|do\s*not|don'?t|"
    r"not\s+(really|at\s*all|yet)|"
    r"neither|never"
    r")\b",
    re.IGNORECASE,
)
# Hostile / abusive language → immediate hangup (word-boundary; STT-friendly variants)
_ABUSE_RE = re.compile(
    r"\b("
    r"f+u+c+k+(?:ing|ed|er|ers)?|"
    r"mother\s*fuck(?:er|ing)?|"
    r"bull\s*shit|bullshit|"
    r"shit(?:ty|head)?|"
    r"ass(?:hole|hat)?|"
    r"bitch(?:es|ing)?|"
    r"bastard|"
    r"cunt|"
    r"dick(?:head)?|"
    r"cock(?:sucker)?|"
    r"piss\s*off|"
    r"screw\s*(?:you|off)|"
    r"go\s*to\s*hell|"
    r"shut\s*(?:the\s*fuck\s*)?up|"
    r"son\s*of\s*a\s*bitch|"
    r"piece\s*of\s*shit|"
    r"dumb\s*ass|dumbass|"
    r"idiot|moron|"
    r"kill\s*(?:yourself|you)|"
    r"hate\s*you|"
    r"fuck\s*(?:off|you|this|that)"
    r")\b",
    re.IGNORECASE,
)

AGE_QUESTION = "Are you above the age of sixty-five?"
DIABETES_QUESTION = "Are you diabetic?"
MEDICARE_REASK = (
    "Just to confirm — do you currently have Medicare Part A and Part B?"
)
DISQUALIFY = "Ok, thanks."
ABUSE_HANGUP = "Ok, thanks."
TRANSFER_LINE = (
    "Perfect — it looks like you qualify. I’m going to transfer your call "
    "to a specialist who will assist you further. Please hold."
)


@dataclass
class _CallScriptState:
    step: str = "medicare"  # medicare | age | diabetes | done
    reasks: int = 0
    history: List[str] = field(default_factory=list)


class ScriptLLMAdapter(LLMComponent):
    """Keyword router for Clara CGM eligibility — no neural LLM."""

    supports_streaming: bool = False

    def __init__(self, component_key: str = "script_llm", options: Optional[Dict[str, Any]] = None):
        self.component_key = component_key
        self._options = dict(options or {})
        self._states: Dict[str, _CallScriptState] = {}
        self._pending_tool_calls_by_call: Dict[str, List[Dict[str, Any]]] = {}

    async def start(self) -> None:
        logger.info("Script LLM adapter ready (deterministic Clara CGM flow)")

    async def stop(self) -> None:
        self._states.clear()
        self._pending_tool_calls_by_call.clear()

    async def open_call(self, call_id: str, options: Dict[str, Any]) -> None:
        self._states[call_id] = _CallScriptState()
        self._pending_tool_calls_by_call[call_id] = []
        logger.info("Script LLM call opened", call_id=call_id, step="medicare")

    async def close_call(self, call_id: str) -> None:
        self._states.pop(call_id, None)
        self._pending_tool_calls_by_call.pop(call_id, None)

    async def validate_connectivity(self, options: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "healthy": True,
            "error": None,
            "details": {"component": self.component_key, "mode": "deterministic_script"},
        }

    @staticmethod
    def _is_abuse(text: str) -> bool:
        return bool(_ABUSE_RE.search(text or ""))

    @staticmethod
    def _classify(text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return "unclear"
        # Prefer explicit no when both match (e.g. "no yes" rare; "yes no" → no)
        has_yes = bool(_YES_RE.search(raw))
        has_no = bool(_NO_RE.search(raw))
        if has_no and not has_yes:
            return "no"
        if has_yes and not has_no:
            return "yes"
        if has_no and has_yes:
            # Last-mentioned wins for short replies
            yes_pos = max(m.end() for m in _YES_RE.finditer(raw))
            no_pos = max(m.end() for m in _NO_RE.finditer(raw))
            return "no" if no_pos >= yes_pos else "yes"
        return "unclear"

    def _hangup(self, farewell: str) -> LLMResponse:
        return LLMResponse(
            text=farewell,
            tool_calls=[
                {
                    "id": "script_hangup",
                    "name": "hangup_call",
                    "parameters": {"farewell_message": farewell},
                    "type": "function",
                }
            ],
        )

    def _transfer(self, line: str, destination: str = "specialist") -> LLMResponse:
        return LLMResponse(
            text=line,
            tool_calls=[
                {
                    "id": "script_transfer",
                    "name": "blind_transfer",
                    "parameters": {"destination": destination},
                    "type": "function",
                }
            ],
        )

    def _next(self, call_id: str, text: str) -> LLMResponse:
        state = self._states.setdefault(call_id, _CallScriptState())

        # Abuse / hostile language → hang up immediately (any step)
        if self._is_abuse(text):
            state.step = "done"
            state.history.append(f"abuse:{(text or '')[:80]}")
            logger.warning(
                "Script LLM abuse detected — hanging up",
                call_id=call_id,
                transcript_preview=(text or "")[:80],
            )
            return self._hangup(ABUSE_HANGUP)

        answer = self._classify(text)
        state.history.append(f"{state.step}:{answer}:{text[:80]}")
        logger.info(
            "Script LLM turn",
            call_id=call_id,
            step=state.step,
            answer=answer,
            transcript_preview=(text or "")[:80],
        )

        if state.step == "done":
            return self._hangup("Thanks again. Goodbye.")

        if state.step == "medicare":
            if answer == "yes":
                state.step = "age"
                state.reasks = 0
                return LLMResponse(text=AGE_QUESTION)
            if answer == "no":
                state.step = "done"
                return self._hangup(DISQUALIFY)
            if state.reasks < 1:
                state.reasks += 1
                return LLMResponse(text=MEDICARE_REASK)
            state.step = "done"
            return self._hangup(DISQUALIFY)

        if state.step == "age":
            if answer == "yes":
                state.step = "diabetes"
                state.reasks = 0
                return LLMResponse(text=DIABETES_QUESTION)
            if answer == "no":
                state.step = "done"
                return self._hangup(DISQUALIFY)
            if state.reasks < 1:
                state.reasks += 1
                return LLMResponse(text=AGE_QUESTION)
            state.step = "done"
            return self._hangup(DISQUALIFY)

        if state.step == "diabetes":
            if answer == "yes":
                state.step = "done"
                return self._transfer(TRANSFER_LINE, destination="specialist")
            if answer == "no":
                state.step = "done"
                return self._hangup(DISQUALIFY)
            if state.reasks < 1:
                state.reasks += 1
                return LLMResponse(text=DIABETES_QUESTION)
            state.step = "done"
            return self._hangup(DISQUALIFY)

        state.step = "done"
        return self._hangup(DISQUALIFY)

    async def generate(
        self,
        call_id: str,
        transcript: str,
        context: Dict[str, Any],
        options: Dict[str, Any],
    ) -> LLMResponse:
        response = self._next(call_id, transcript or "")
        # Keep streaming-compat bucket in sync if engine inspects it
        self._pending_tool_calls_by_call[call_id] = list(response.tool_calls or [])
        logger.info(
            "Script LLM response",
            call_id=call_id,
            preview=(response.text or "")[:80],
            tools=[tc.get("name") for tc in (response.tool_calls or [])],
        )
        return response

    async def generate_stream(
        self,
        call_id: str,
        transcript: str,
        context: Dict[str, Any],
        options: Dict[str, Any],
    ) -> AsyncIterator[str]:
        result = await self.generate(call_id, transcript, context, options)
        if result.text:
            yield result.text
