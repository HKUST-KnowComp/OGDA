"""Apply YAML request settings to DiscoveryBench's original judge SDK clients."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

from openai import OpenAI

from .llm import merge_kwargs


class _ConfiguredCompletions:
    def __init__(self, completions, model, options):
        self.completions, self.model, self.options = completions, model, options

    def create(self, **kwargs):
        request = merge_kwargs(kwargs, self.options)
        if self.model:
            request["model"] = self.model
        return self.completions.create(**request)


def configure_judge() -> None:
    """Configure module-local SDK constructors; leave native response handling intact.

    Call after adding the assembled benchmark root to sys.path and applying the
    evaluator YAML environment. Original prompts, parsers, retries and scoring
    functions execute unchanged, receiving the raw SDK response.
    """
    from eval import lm_utils, new_eval

    model = os.environ.get("EVAL_MODEL")
    options = json.loads(os.environ.get("LLM_JUDGE_KWARGS", "{}"))
    api_key = os.environ.get("EVAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("EVAL_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

    def configured_openai(*args, **kwargs):
        kwargs["api_key"] = api_key or kwargs.get("api_key")
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(*args, **kwargs)
        completions = _ConfiguredCompletions(client.chat.completions, model, options)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    lm_utils.OpenAI = new_eval.OpenAI = configured_openai
