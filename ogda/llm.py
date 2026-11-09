"""Chat Completions and Responses transports, and an offline scripted client.

Only YAML selects credentials, endpoint, models and reasoning options.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import litellm
import yaml
from openai import APIConnectionError, AsyncOpenAI, DefaultAsyncHttpxClient, OpenAIError
from openai.types.responses import Response


REJECTED_STATUSES = {400, 401, 402, 403, 404}
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


class LLMRequestRejected(LLMError):
    """A permanent endpoint/configuration error; cannot yield benchmark evidence."""


class BudgetExceeded(LLMError):
    pass


@dataclass
class Generation:
    """Agent model output: text, and native tool calls as {"name": str, "arguments": raw JSON str}."""

    text: str
    tool_calls: list[dict] = field(default_factory=list)
    truncated: bool = False  # The response stopped at the output token limit.


class InvalidResponse(LLMError):
    """The model returned text that is not the requested JSON object."""


def empty_usage() -> dict:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }


def add_usage(usage: dict, delta: dict) -> None:
    for key in usage.keys() & delta.keys():
        usage[key] += delta[key]


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge_settings(base: dict, overrides: dict) -> dict:
    settings = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(settings.get(key), dict):
            settings[key] = merge_settings(settings[key], value)
        else:
            settings[key] = value
    return settings


def normalize_reasoning(reasoning: dict) -> dict:
    """Accept quoted YAML booleans and the legacy enabled-as-effort shorthand."""
    reasoning = dict(reasoning)
    for key in ("enabled", "exclude"):
        value = reasoning.get(key)
        if not isinstance(value, str):
            continue
        value = value.strip().lower()
        if value in ("true", "false"):
            reasoning[key] = value == "true"
        elif key == "enabled" and value in (
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        ):
            if reasoning.get("effort", value) != value:
                raise LLMRequestRejected(
                    "YAML reasoning.enabled and reasoning.effort conflict"
                )
            reasoning.update(enabled=True, effort=value)
    return reasoning


def request_id(role: str, system: str, prompt: str) -> str:
    return hashlib.sha256(
        json.dumps([role, system, prompt], ensure_ascii=False).encode()
    ).hexdigest()


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise InvalidResponse("Model returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise InvalidResponse("Model response must be a JSON object")
    return result


class OpenAICompatibleClient:
    """Synchronous interface with opt-in Responses and default LiteLLM Chat Completions."""

    def __init__(self, config: str | Path, record_path: str | Path | None = None):
        document = yaml.safe_load(Path(config).read_text()) or {}
        self.config = document.get("llm", document)
        self.usage = empty_usage()
        self.record_path = Path(record_path) if record_path else None
        # Optional provider safety caps. The construction token budget is the normal limit.
        self.max_calls = float(self.config.get("max_calls", math.inf))
        self.max_total_tokens = float(self.config.get("max_total_tokens", math.inf))
        self.max_cost_usd = float(self.config.get("max_cost_usd", math.inf))
        self.max_output_tokens = int(
            self.config.get("max_output_tokens", self.config.get("max_tokens", 3000))
        )
        self.timeout = float(self.config.get("timeout_seconds", 120))
        self._runner = asyncio.Runner()
        if self.record_path:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            if self.record_path.exists():
                for record in read_records(self.record_path):
                    add_usage(self.usage, record.get("usage", {}))

    def close(self) -> None:
        self._runner.close()

    def role_config(self, role: str) -> dict:
        settings = {k: v for k, v in self.config.items() if k != "roles"}
        roles = self.config.get("roles", {})
        fallback = {"experience": "proposer", "data_agent": "analyst"}.get(role)
        if fallback:
            settings = merge_settings(settings, roles.get(fallback, {}) or {})
        overrides = roles.get(role, {}) or {}
        settings = merge_settings(settings, overrides)
        model = settings.get("model", "")
        endpoint = settings.get("base_url", settings.get("api_base", ""))
        if not model or model == "provider/model-name":
            raise LLMError(f"YAML must supply a working model for {role}")
        if not endpoint:
            raise LLMError(f"YAML must supply base_url for {role}")
        credential = settings.get("api_key")
        if not credential and settings.get("api_key_env"):
            credential = os.environ.get(settings["api_key_env"])
        if not credential:
            raise LLMError(f"YAML credential is unavailable for {role}")
        settings["credential"] = credential
        settings["endpoint"] = endpoint.rstrip("/")
        return settings

    def model_name(self, role: str) -> str:
        return self.role_config(role)["model"]

    def _request_kwargs(
        self, settings: dict, system: str, prompt: str, cap: int
    ) -> dict:
        # Some JSON-mode endpoints require an explicit JSON instruction.
        if "json" not in (system + "\n" + prompt).lower():
            system += "\nReturn a valid JSON object."
        api = settings.get("api", "chat_completions")
        if api == "responses":
            return self._responses_request_kwargs(settings, system, prompt, cap)
        if api != "chat_completions":
            raise LLMRequestRejected("YAML llm.api must be chat_completions or responses")
        kwargs = dict(settings.get("model_kwargs", {}) or {})
        kwargs.update(kwargs.pop("extra_body", {}) or {})
        for key in ("temperature", "reasoning", "reasoning_effort", "provider", "seed"):
            if key in settings:
                kwargs[key] = settings[key]
        if isinstance(kwargs.get("reasoning"), dict):
            kwargs["reasoning"] = normalize_reasoning(kwargs["reasoning"])
        for key in (
            "base_url",
            "max_completion_tokens",
            "request_timeout",
            "custom_llm_provider",
        ):
            kwargs.pop(key, None)
        model = settings["model"]
        modern_openai = re.match(r"(?:gpt-(?:[5-9]|\d{2,})(?:[.-]|$)|o[134](?:-|$))", model.rsplit("/", 1)[-1])
        token_key = "max_completion_tokens" if modern_openai else "max_tokens"
        return {
            **kwargs,
            "model": model if model.startswith("openai/") else "openai/" + model,
            "api_base": settings["endpoint"],
            "api_key": settings["credential"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            token_key: cap,
            "response_format": {"type": "json_object"},
            "timeout": float(settings.get("timeout_seconds", self.timeout)),
            "stream": False,
            # Count every attempt against the application budget, including retries.
            "num_retries": 0,
            "max_retries": 0,
        }

    def _responses_request_kwargs(self, settings: dict, system: str, prompt: str, cap: int) -> dict:
        options = dict(settings.get("model_kwargs", {}) or {})
        options.update(options.pop("extra_body", {}) or {})
        for key in ("temperature", "reasoning", "reasoning_effort"):
            if key in settings:
                options[key] = settings[key]
        reasoning = normalize_reasoning(options.get("reasoning") or {})
        effort = options.get("reasoning_effort", reasoning.get("effort"))
        if reasoning.get("enabled") is False:
            if effort not in (None, "none", "no"):
                raise LLMRequestRejected("YAML reasoning.enabled and reasoning effort conflict")
            effort = "none"
        if effort == "no":
            effort = "none"
        # Omission lets the provider choose its default; never send a null effort.
        native_reasoning = {"effort": effort} if effort not in (None, "default") else {}
        for key in ("summary", "generate_summary"):
            if reasoning.get(key) is not None:
                native_reasoning[key] = reasoning[key]
        kwargs = {key: options[key] for key in ("temperature", "top_p", "parallel_tool_calls") if key in options}
        if native_reasoning:
            kwargs["reasoning"] = native_reasoning
        return {
            **kwargs,
            "api": "responses",
            "api_base": settings["endpoint"],
            "api_key": settings["credential"],
            "timeout": float(settings.get("timeout_seconds", self.timeout)),
            "model": settings["model"].removeprefix("openai/"),
            "input": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_output_tokens": cap,
            "text": {"format": {"type": "json_object"}},
            "store": False,
            "stream": False,
        }

    @staticmethod
    def _responses_input(messages: list[dict]) -> list[dict]:
        # CodeAgent keeps calls as assistant text and observations as user text.
        result = []
        for message in messages:
            content = message.get("content") or ""
            if (message.get("role") not in {"system", "developer", "user", "assistant"}
                    or not isinstance(content, str) or message.get("tool_calls")):
                raise LLMError("Responses transport requires text messages; native tool-call history is unsupported")
            result.append({"role": message["role"], "content": content})
        return result

    async def _request(self, kwargs: dict) -> litellm.ModelResponse | Response:
        if kwargs.get("api") == "responses":
            options = {key: value for key, value in kwargs.items()
                       if key not in {"api", "api_base", "api_key", "timeout"}}
            async with AsyncOpenAI(api_key=kwargs["api_key"], base_url=kwargs["api_base"],
                                   timeout=kwargs["timeout"], max_retries=0,
                                   http_client=DefaultAsyncHttpxClient(trust_env=True)) as client:
                return await self._request_attempts(lambda: client.responses.create(**options), kwargs["timeout"])
        return await self._request_attempts(lambda: litellm.acompletion(**kwargs), kwargs["timeout"])

    async def _request_attempts(self, request, timeout: float):
        for attempt in range(3):
            if self.usage["calls"] >= self.max_calls:
                raise BudgetExceeded("LLM call cap exhausted")
            self.usage["calls"] += 1
            try:
                # An async deadline also bounds responses with whitespace keepalives.
                return await asyncio.wait_for(request(), timeout=timeout)
            except (OpenAIError, TimeoutError) as exc:
                transport_error = isinstance(
                    exc, (APIConnectionError, TimeoutError)
                )
                retryable = (
                    transport_error
                    or getattr(exc, "status_code", None) in RETRYABLE_STATUSES
                )
                if attempt == 2 or not retryable:
                    raise
                await asyncio.sleep(2**attempt)

    def complete(self, role: str, system: str, prompt: str) -> dict:
        generation = self._complete(role, system, prompt)
        if generation.truncated:
            raise InvalidResponse("Model response was cut off at the output token limit")
        return parse_json(generation.text)

    def generate(self, role: str, messages: list[dict], stop_sequences=None, *,
                 tools: list[dict] | None = None, tool_choice: str | None = None) -> Generation:
        """Generate CodeAgent output without JSON response mode, optionally with native tool calls."""
        prompt = json.dumps({"messages": messages, "stop": stop_sequences, "tools": tools, "tool_choice": tool_choice},
                            ensure_ascii=False, sort_keys=True)
        return self._complete(role, "codeagent_v1", prompt, messages=messages, stop=stop_sequences,
                              tools=tools, tool_choice=tool_choice)

    def _complete(self, role, system, prompt, *, messages=None, stop=None, tools=None, tool_choice=None) -> Generation:
        settings = self.role_config(role)
        estimated_input = max(1, (len(system) + len(prompt)) // 3)
        remaining = self.max_total_tokens - self.usage["total_tokens"] - estimated_input
        if (
            remaining <= 0
            or self.usage["calls"] >= self.max_calls
            or self.usage["cost_usd"] >= self.max_cost_usd
        ):
            raise BudgetExceeded("Overall LLM budget exhausted")
        cap = int(min(int(settings.get("max_output_tokens", self.max_output_tokens)), self.max_output_tokens, remaining))
        kwargs = self._request_kwargs(settings, system, prompt, cap)
        responses = kwargs.get("api") == "responses"
        if messages is not None:
            if responses:
                kwargs["input"] = self._responses_input(messages)
                kwargs.pop("text", None)
            else:
                kwargs["messages"] = messages
                kwargs.pop("response_format", None)
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
            if tools:
                if responses:
                    if any(tool.get("type") != "function" for tool in tools):
                        raise LLMError("Responses transport supports function tools only")
                    tools = [{"type": "function", "strict": False, **tool["function"]} for tool in tools]
                kwargs.update(tools=tools, tool_choice=tool_choice or "auto")
            if stop and not responses:
                kwargs["stop"] = stop
            else:
                kwargs.pop("stop", None)
        request = (role, system, prompt)
        started = time.monotonic()
        calls_before = self.usage["calls"]
        try:
            response = self._runner.run(self._request(kwargs))
        except BudgetExceeded:
            self._record(
                request,
                started,
                error="budget_exceeded",
                usage={"calls": self.usage["calls"] - calls_before},
            )
            raise
        except (OpenAIError, TimeoutError) as exc:
            transport_error = isinstance(
                exc, (APIConnectionError, TimeoutError)
            )
            status = None if transport_error else getattr(exc, "status_code", None)
            error = f"http_{status}" if status is not None else "transport"
            self._record(
                request,
                started,
                error=error,
                usage={"calls": self.usage["calls"] - calls_before},
            )
            if status in REJECTED_STATUSES:
                raise LLMRequestRejected(
                    f"LLM endpoint rejected request: HTTP {status}"
                ) from exc
            message = (
                f"LLM endpoint returned HTTP {status}"
                if status is not None
                else "LLM transport failed"
            )
            raise LLMError(message) from exc
        provider_usage = response.usage.model_dump() if response.usage else {}
        delta = {
            "input_tokens": int(provider_usage.get("input_tokens" if responses else "prompt_tokens") or estimated_input),
            "output_tokens": int(provider_usage.get("output_tokens" if responses else "completion_tokens", 0)),
            "cost_usd": float(provider_usage.get("cost", 0) or 0),
        }
        delta["total_tokens"] = int(
            provider_usage.get("total_tokens")
            or delta["input_tokens"] + delta["output_tokens"]
        )
        add_usage(self.usage, delta)
        delta["calls"] = self.usage["calls"] - calls_before
        if responses:
            truncated = (response.status == "incomplete" and response.incomplete_details is not None
                         and response.incomplete_details.reason == "max_output_tokens")
            generation = Generation(response.output_text, [
                {"name": item.name, "arguments": item.arguments or ""}
                for item in response.output if item.type == "function_call"
            ], truncated=truncated)
            finish_reason = "length" if truncated else "tool_calls" if generation.tool_calls else response.status
        else:
            message = response.choices[0].message
            generation = Generation(message.content or "", [
                {"name": call.function.name, "arguments": call.function.arguments or ""}
                for call in message.tool_calls or []
            ], truncated=response.choices[0].finish_reason == "length")
            finish_reason = response.choices[0].finish_reason
        self._record(
            request,
            started,
            model=settings["model"].removeprefix("openai/"),
            response_text=generation.text,
            tool_calls=generation.tool_calls,
            finish_reason=finish_reason,
            usage=delta,
        )
        if (
            self.usage["total_tokens"] > self.max_total_tokens
            or self.usage["cost_usd"] > self.max_cost_usd
        ):
            raise BudgetExceeded("LLM response exhausted overall budget")
        return generation

    def _record(self, request: tuple[str, str, str], started: float, **fields) -> None:
        if self.record_path:
            record = {
                "request_id": request_id(*request),
                "role": request[0],
                "wall_seconds": time.monotonic() - started,
                **fields,
            }
            with self.record_path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class ScriptedClient:
    """Offline model: JSON for single-call tasks; for agent messages, text or {"text", "tool_calls", "truncated"}."""

    def __init__(self, responder, model: str = "scripted", role_models: dict[str, str] | None = None):
        self.responder = responder
        self.model = model
        self.role_models = dict(role_models or {})
        self.usage = empty_usage()
        self.requests = []

    def model_name(self, role: str) -> str:
        return self.role_models.get(role, self.model)

    def complete(self, role: str, system: str, prompt: str) -> dict:
        return self._respond(role, system, prompt, dict)

    def generate(self, role: str, messages: list[dict], stop_sequences=None, *,
                 tools: list[dict] | None = None, tool_choice: str | None = None) -> Generation:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        prompt = json.dumps(messages, ensure_ascii=False)
        result = self._respond(role, system, prompt, (str, dict),
                               stop=stop_sequences, tools=tools, tool_choice=tool_choice)
        if isinstance(result, str):
            return Generation(result)
        return Generation(result.get("text", ""), result.get("tool_calls", []), result.get("truncated", False))

    def _respond(self, role, system, prompt, expected, **request):
        self.requests.append({"role": role, "system": system, "prompt": prompt, **request})
        result = self.responder(role, system, prompt)
        if not isinstance(result, expected):
            raise LLMError(f"Scripted response has the wrong type: {type(result).__name__}")
        input_tokens = max(1, (len(system) + len(prompt)) // 4)
        output_tokens = max(1, len(json.dumps(result)) // 4)
        add_usage(
            self.usage,
            {
                "calls": 1,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )
        return result
