"""Apply YAML judge settings to the native KramaBench SDK transport."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from openai import OpenAI

from .llm import _SECRET_KEYS, merge_kwargs


def _public(value, credentials):
    if isinstance(value, dict):
        return {key: "[REDACTED]" if key.lower().replace("-", "_") in _SECRET_KEYS
                or key.lower().replace("-", "_").endswith(("_api_key", "_secret"))
                else _public(item, credentials) for key, item in value.items()}
    if isinstance(value, list):
        return [_public(item, credentials) for item in value]
    if isinstance(value, str):
        for credential in credentials:
            if credential:
                value = value.replace(credential, "[REDACTED]")
    return value


def _append(journal: Path | None, record: dict, credentials) -> None:
    if journal is not None:
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_public(record, credentials), ensure_ascii=False) + "\n")


class _ConfiguredCompletions:
    def __init__(self, completions, model, options, journal, credentials):
        self.completions = completions
        self.model = model
        self.options = options
        self.journal = journal
        self.credentials = credentials

    def create(self, **kwargs):
        request = merge_kwargs(kwargs, self.options)
        if self.model:
            request["model"] = self.model
        started = time.monotonic()
        _append(self.journal, {"state": "started", "time": time.time(), "request": request}, self.credentials)
        try:
            response = self.completions.create(**request)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            _append(self.journal, {"state": "failed", "wall_seconds": time.monotonic() - started,
                                   "error": message}, self.credentials)
            raise
        if self.journal is not None:
            saved = response.model_dump()
            choices = [{"finish_reason": choice.get("finish_reason"),
                        "message": {key: value for key, value in choice.get("message", {}).items()
                                    if key in {"role", "content", "tool_calls", "function_call"}}}
                       for choice in saved.get("choices", [])]
            _append(self.journal, {"state": "completed", "wall_seconds": time.monotonic() - started,
                                   "response": {"id": saved.get("id"), "model": saved.get("model"),
                                                "choices": choices, "usage": saved.get("usage")}}, self.credentials)
        return response


def configure_judge(journal: Path | None = None) -> None:
    """Configure only the SDK constructor used by the original GPTInterface.

    Call after adding the assembled benchmark root to sys.path. The original
    class, prompts, parsing, metrics and evaluator are left intact.
    """
    from benchmark.llm_tools import gpt_interface

    model = os.environ.get("EVAL_MODEL")
    options = json.loads(os.environ.get("LLM_JUDGE_KWARGS", "{}"))
    api_key = os.environ.get("EVAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("EVAL_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    headers = [str(value) for value in (options.get("extra_headers") or {}).values()]
    credentials = (api_key, os.environ.get("OPENAI_API_KEY"), *headers,
                   *(value.removeprefix("Bearer ") for value in headers))
    journal = Path(journal) if journal is not None else None

    def configured_openai(*args, **kwargs):
        kwargs["api_key"] = api_key or kwargs.get("api_key")
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(*args, **kwargs)
        completions = _ConfiguredCompletions(client.chat.completions, model, options, journal, credentials)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    gpt_interface.OpenAI = configured_openai
