from dataclasses import dataclass
import json
from typing import Protocol
from urllib import request


@dataclass(frozen=True)
class LLMResponse:
    answer: str
    confidence: float
    abstain: bool

    def __post_init__(self) -> None:
        if not self.answer.strip():
            raise ValueError("answer must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


class LLM(Protocol):
    def generate(self, prompt: str) -> LLMResponse: ...


def parse_structured_response(text: str) -> LLMResponse:
    """Parse the small JSON contract used by all ShiftWatch LLM adapters."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].lstrip()
    payload = json.loads(text)
    return LLMResponse(
        answer=str(payload["answer"]),
        confidence=float(payload["confidence"]),
        abstain=bool(payload["abstain"]),
    )


class FixtureLLM:
    """Recorded responses for deterministic CI; not a real language model."""

    def __init__(self, responses: dict[str, dict]):
        self.responses = responses

    def generate(self, prompt: str) -> LLMResponse:
        try:
            payload = self.responses[prompt]
        except KeyError as error:
            raise ValueError("fixture has no recorded response for prompt") from error
        return LLMResponse(
            answer=payload["answer"],
            confidence=float(payload["confidence"]),
            abstain=bool(payload["abstain"]),
        )


class OllamaLLM:
    """Dependency-free adapter for a local or remote Ollama inference server."""

    SYSTEM_INSTRUCTION = """Answer the user prompt. Return JSON only, using exactly:
{"answer": "your answer", "confidence": 0.0, "abstain": false}
Confidence must be between 0 and 1. Set abstain true when you cannot answer reliably.
If the prompt contains a false premise, explicitly correct it in the answer."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        response_mode: str = "short",
        options: dict | None = None,
    ):
        if response_mode not in {"short", "free"}:
            raise ValueError("response_mode must be short or free")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.response_mode = response_mode
        self.options = {"temperature": 0, "seed": 7, **(options or {})}

    def generate(self, prompt: str) -> LLMResponse:
        return parse_structured_response(self.generate_raw(prompt)["response"])

    def generate_raw(self, prompt: str) -> dict:
        """Return Ollama's full response payload without parsing the model text."""
        style = (
            "Keep the answer brief."
            if self.response_mode == "short"
            else "Use one concise explanatory paragraph in the answer field."
        )
        body = json.dumps({
            "model": self.model,
            "prompt": f"{self.SYSTEM_INSTRUCTION}\n{style}\n\nUser prompt:\n{prompt}",
            "stream": False,
            "format": "json",
            "options": self.options,
        }).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout) as response:
            return json.load(response)
