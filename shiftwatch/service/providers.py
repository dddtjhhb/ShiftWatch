"""Provider adapters that return raw output plus a retryable/fatal error classification."""
from dataclasses import dataclass, field
import http.client
import json
import socket
import time
from urllib import error as urlerror

from ..llm import OllamaLLM, parse_structured_response


class ProviderError(Exception):
    def __init__(self, kind: str, message: str, retryable: bool):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


@dataclass
class ProviderOutput:
    raw_text: str
    latency_ms: int
    usage: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        """Parse the model's JSON contract. A malformed reply is a *model result*, not an
        infrastructure failure: it is stored with parse_error and scored as incorrect,
        rather than retried (retrying a deterministic model would just pay again)."""
        record = {"raw_text": self.raw_text, "latency_ms": self.latency_ms, "usage": self.usage}
        try:
            parsed = parse_structured_response(self.raw_text)
        except (ValueError, KeyError, TypeError) as error:
            record["parse_error"] = f"{type(error).__name__}: {error}"[:500]
        else:
            record.update(
                answer=parsed.answer, confidence=parsed.confidence, abstain=parsed.abstain
            )
        return record


def classify_exception(error: Exception) -> ProviderError:
    if isinstance(error, ProviderError):
        return error
    if isinstance(error, urlerror.HTTPError):
        if error.code == 429:
            return ProviderError("rate_limited", "provider returned 429", True)
        if error.code >= 500:
            return ProviderError("server_error", f"provider returned {error.code}", True)
        # 400/404 etc.: bad model name or request; retrying cannot help.
        return ProviderError("client_error", f"provider returned {error.code}", False)
    if isinstance(error, (TimeoutError, socket.timeout)):
        return ProviderError("timeout", str(error) or "timed out", True)
    if isinstance(error, (urlerror.URLError, ConnectionError, http.client.HTTPException)):
        return ProviderError("connection_error", str(error)[:500], True)
    if isinstance(error, (json.JSONDecodeError, KeyError)):
        # The HTTP envelope (not the model text) was malformed: treat as transient.
        return ProviderError("bad_envelope", str(error)[:500], True)
    return ProviderError("unexpected", f"{type(error).__name__}: {error}"[:500], False)


class OllamaProvider:
    def __init__(self, params: dict):
        options = {
            key: params[key] for key in ("temperature", "seed", "top_p", "num_predict")
            if key in params
        }
        self.client = OllamaLLM(
            model=params["model"],
            base_url=params.get("base_url", "http://localhost:11434"),
            timeout=float(params.get("timeout", 120)),
            response_mode=params.get("response_mode", "short"),
            options=options,
        )

    def generate(self, prompt: str, case_payload: dict, condition: str) -> ProviderOutput:
        started = time.monotonic()
        try:
            payload = self.client.generate_raw(prompt)
            text = payload["response"]
        except Exception as error:  # noqa: BLE001 - classified below
            raise classify_exception(error) from error
        usage = {
            key: payload[key]
            for key in ("prompt_eval_count", "eval_count", "total_duration")
            if key in payload
        }
        return ProviderOutput(text, int((time.monotonic() - started) * 1000), usage)


class FixtureProvider:
    """Replays fixture_responses embedded in the dataset. Deterministic; not a model."""

    def __init__(self, params: dict):
        self.params = params

    def generate(self, prompt: str, case_payload: dict, condition: str) -> ProviderOutput:
        try:
            response = case_payload["fixture_responses"][condition]
        except KeyError as error:
            raise ProviderError("client_error", "no fixture response for task", False) from error
        return ProviderOutput(json.dumps(response), 0, {})


def build_provider(provider: str, params: dict):
    if provider == "ollama":
        return OllamaProvider(params)
    if provider == "fixture":
        return FixtureProvider(params)
    raise ProviderError("client_error", f"unknown provider {provider}", False)
