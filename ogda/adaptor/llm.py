#!/usr/bin/env python3
"""Load one LLM YAML file and launch legacy commands with its settings.

Usage: python -m ogda.adaptor.llm --config llm_config/Kramabench/evaluator.yaml run -- command ...
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, MutableMapping
from urllib.parse import urlsplit

from .artifacts import WORKSPACE


DEFAULT_MAX_COMPLETION_TOKENS = 32768
_COMPLETION_TOKEN_KEYS = {"max_tokens", "max_completion_tokens", "max_output_tokens", "max_new_tokens"}


class ConfigError(ValueError):
    """Invalid or unavailable model configuration."""


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str | None = None
    reasoning_effort: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def litellm_model(self) -> str:
        return self.model if self.model.startswith(self.provider + "/") else f"{self.provider}/{self.model}"

    @property
    def provider_model(self) -> str:
        return self.model.removeprefix(self.provider + "/")

    def resolve_api_key(self, environ: MutableMapping[str, str]) -> str | None:
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env:
            value = environ.get(self.api_key_env)
            if not value:
                raise ConfigError(f"Credential variable {self.api_key_env!r} is not set")
            return value
        return None


@dataclass(frozen=True)
class LLMConfig(ModelConfig):
    judge: ModelConfig | None = None
    roles: dict[str, ModelConfig] = field(default_factory=dict, repr=False)
    graph_context: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = field(default=None, repr=False)


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROVIDER_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
_EFFORT = {"no", "none", "minimal", "low", "medium", "high", "xhigh", "max", "default"}
_PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY", "google": "GOOGLE_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY", "azure_openai": "AZURE_OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY", "openrouter": "OPENROUTER_API_KEY",
}
_DEFAULT_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "openrouter": "https://openrouter.ai/api/v1",
}
_OPENAI_COMPATIBLE = {"openai", "deepseek", "openrouter", "azure", "azure_openai"}
_SECRET_KEYS = {"api_key", "password", "secret", "authorization", "credential", "access_token", "refresh_token", "bearer_token"}


def _parse_yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError:
        # JSON is a YAML subset. This keeps the CLI usable in a fresh Python
        # environment; conventional YAML requires PyYAML.
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise ConfigError("Install PyYAML to read block-style YAML (python -m pip install pyyaml)") from error
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError("Invalid YAML configuration") from error


def _nonempty_string(data: dict[str, Any], key: str, location: str, required: bool = False) -> str | None:
    value = data.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location}.{key} must be a nonempty string")
    return value.strip()


def _model_config(data: Any, location: str, fallback_provider: str | None = None) -> ModelConfig:
    if not isinstance(data, dict):
        raise ConfigError(f"{location} must be a mapping")
    data = dict(data)
    if "api_base" in data:
        if "base_url" in data:
            raise ConfigError(f"{location}: use only api_base, not base_url as well")
        data["base_url"] = data.pop("api_base")
    allowed = {"provider", "model", "base_url", "api_key", "api_key_env", "reasoning_effort", "model_kwargs"}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"Unknown {location} field(s): {', '.join(sorted(map(str, unknown)))}")
    model = _nonempty_string(data, "model", location, required=True)
    provider = _nonempty_string(data, "provider", location) or (model.split("/", 1)[0] if "/" in model else fallback_provider or "openai")
    if not provider or not _PROVIDER_NAME.fullmatch(provider):
        raise ConfigError(f"{location}.provider must be a lowercase provider name")
    base_url = _nonempty_string(data, "base_url", location)
    if base_url:
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ConfigError(f"{location}.base_url must be an HTTP(S) URL without embedded credentials or query")
    api_key = _nonempty_string(data, "api_key", location)
    api_key_env = _nonempty_string(data, "api_key_env", location)
    if api_key and api_key_env:
        raise ConfigError(f"{location} may specify only one of api_key and api_key_env")
    if api_key_env and not _ENV_NAME.fullmatch(api_key_env):
        raise ConfigError(f"{location}.api_key_env must name an environment variable")
    raw_effort = data.get("reasoning_effort")
    # PyYAML follows YAML 1.1 and parses an unquoted `no` as False.
    reasoning_effort = "no" if raw_effort is False else _nonempty_string(data, "reasoning_effort", location)
    if reasoning_effort == "none":
        reasoning_effort = "no"
    if reasoning_effort and reasoning_effort not in _EFFORT:
        raise ConfigError(f"{location}.reasoning_effort must be one of {', '.join(sorted(_EFFORT))}")
    kwargs = data.get("model_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ConfigError(f"{location}.model_kwargs must be a mapping")
    if set(kwargs) & {"model", "messages", "api_key", "api_base", "base_url"}:
        raise ConfigError(f"{location}.model_kwargs must not override model, messages, or credentials")
    if len(set(kwargs) & _COMPLETION_TOKEN_KEYS) > 1:
        raise ConfigError(f"{location}.model_kwargs must specify only one completion token limit")
    return ModelConfig(provider, model, base_url, api_key, api_key_env, reasoning_effort, deepcopy(kwargs))


def merge_kwargs(defaults: dict, overrides: dict) -> dict:
    """Merge request options; an explicit token limit replaces any inherited alias."""
    if not isinstance(overrides, dict):
        raise ConfigError("model_kwargs must be a mapping")
    result = deepcopy(defaults)
    if set(overrides) & _COMPLETION_TOKEN_KEYS:
        for key in _COMPLETION_TOKEN_KEYS:
            result.pop(key, None)
    for key, value in overrides.items():
        result[key] = (merge_kwargs(result[key], value)
                       if isinstance(result.get(key), dict) and isinstance(value, dict)
                       else deepcopy(value))
    return result


def _role_config(values: Any, primary: ModelConfig, location: str) -> ModelConfig:
    if not isinstance(values, dict):
        raise ConfigError(f"{location} must be a mapping")
    values = dict(values)
    if "api_base" in values:
        if "base_url" in values:
            raise ConfigError(f"{location}: use only api_base, not base_url as well")
        values["base_url"] = values.pop("api_base")
    model = values.get("model")
    provider = values.get("provider", model.split("/", 1)[0] if isinstance(model, str) and "/" in model else primary.provider)
    if provider != primary.provider and "model" not in values:
        raise ConfigError(f"{location}.model is required when provider changes")
    inherited = {"provider": provider, "model": primary.model}
    if provider == primary.provider:
        inherited.update(base_url=primary.base_url, api_key=primary.api_key,
                         api_key_env=primary.api_key_env, reasoning_effort=primary.reasoning_effort,
                         model_kwargs=primary.model_kwargs)
    kwargs = merge_kwargs(inherited.get("model_kwargs", {}), values.get("model_kwargs", {}))
    inherited.update(values)
    inherited["model_kwargs"] = kwargs
    # An explicit credential replaces the inherited credential of the other kind.
    if "api_key" in values:
        inherited.pop("api_key_env", None)
    if "api_key_env" in values:
        inherited.pop("api_key", None)
    return _model_config(inherited, location)


def select_role(config: LLMConfig, role: str) -> LLMConfig:
    """Select a YAML role, falling back to the top-level model for older files."""
    selected = config.roles.get(role.lower())
    if selected is None:
        return config
    return replace(config, judge=config.judge or ModelConfig(**{name: getattr(config, name) for name in ModelConfig.__dataclass_fields__}), **vars(selected))


def load_config(path: str | Path, *, _reference_stack: frozenset[Path] = frozenset()) -> LLMConfig:
    """Validate a YAML configuration without putting credentials in logs."""
    source = Path(path)
    if not source.is_absolute():
        source = WORKSPACE / source
    source = source.resolve()
    if source in _reference_stack:
        raise ConfigError(f"Circular config_file reference: {source}")
    reference_stack = _reference_stack | {source}
    try:
        raw = _parse_yaml(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"Cannot read configuration: {source}") from error
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping")
    allowed = {"provider", "model", "base_url", "api_key", "api_key_env", "reasoning_effort", "judge", "roles", "graph_context", "api_base", "model_kwargs"}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"Unknown configuration field(s): {', '.join(sorted(map(str, unknown)))}")
    graph_context = raw.get("graph_context", {})
    if not isinstance(graph_context, dict):
        raise ConfigError("graph_context must be a mapping")
    # Graph settings are persisted in run metadata. Keep all secret-bearing
    # fields out of that mapping, including nested provider kwargs.
    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ConfigError("graph_context keys must be strings")
                if key.lower() in _SECRET_KEYS or key.lower().endswith(("_api_key", "_secret")):
                    raise ConfigError("graph_context must not contain credentials")
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
    inspect(graph_context)
    primary = _model_config({k: v for k, v in raw.items() if k in allowed - {"judge", "roles", "graph_context"}}, "llm")
    roles_raw = raw.get("roles", {})
    if not isinstance(roles_raw, dict):
        raise ConfigError("roles must be a mapping")
    roles = {}
    for name, values in roles_raw.items():
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name.lower()):
            raise ConfigError("Role names must use letters, digits, and underscores")
        role_primary = primary
        if isinstance(values, dict) and "config_file" in values:
            reference = _nonempty_string(values, "config_file", f"roles.{name}", required=True)
            role_primary = load_config(source.parent / reference, _reference_stack=reference_stack)
            values = {key: value for key, value in values.items() if key != "config_file"}
        roles[name.lower()] = _role_config(values, role_primary, f"roles.{name}")
    judge = _role_config(raw["judge"], primary, "judge") if raw.get("judge") is not None else None
    if "evaluator" in roles:
        judge = roles["evaluator"]
    return LLMConfig(**vars(primary), judge=judge, roles=roles, graph_context=graph_context,
                     source_path=str(source.resolve()))


def validate_openai_compatible_judge(config: LLMConfig) -> None:
    """Reject judges unsupported by native OpenAI-SDK benchmark evaluators."""
    judge = config.judge or config
    if judge.provider not in {"openai", "deepseek", "openrouter"}:
        raise ConfigError(
            f"Native benchmark judging requires an OpenAI-compatible provider; got {judge.provider!r}"
        )


def model_request_kwargs(config: ModelConfig) -> dict[str, Any]:
    """Preserve explicit budgets and supply a provider-compatible completion limit."""
    options = merge_kwargs(_legacy_request_kwargs(config), config.model_kwargs)
    limits = set(options) & _COMPLETION_TOKEN_KEYS
    if len(limits) > 1:
        raise ConfigError("model_kwargs must specify only one completion token limit")
    # Native output/new-token options already describe their transport's limit.
    if limits & {"max_output_tokens", "max_new_tokens"}:
        return options
    name = config.provider_model.lower().split("/")[-1]
    modern_openai = (config.provider in {"openai", "azure", "azure_openai"}
                     and (re.match(r"gpt-(?:[5-9]|\d{2,})(?:[.-]|$)", name)
                          or re.match(r"o[134](?:-|$)", name)))
    key = "max_completion_tokens" if modern_openai else "max_tokens"
    value = options.pop(next(iter(limits))) if limits else DEFAULT_MAX_COMPLETION_TOKENS
    options[key] = value
    return options


def _legacy_request_kwargs(config: ModelConfig) -> dict[str, Any]:
    """Translate a role's effort into provider request options where supported."""
    effort = "no" if config.reasoning_effort == "none" else config.reasoning_effort
    if not effort or effort == "default":
        return {}
    name = config.provider_model.lower()
    if effort == "no":
        if (name.startswith("deepseek-r1") or name.startswith(("o1", "o3", "o4", "gemini-3"))
                or (name.startswith("gpt-5") and (not name.startswith("gpt-5.") or "-pro" in name))):
            raise ConfigError(f"{config.model} cannot disable reasoning")
        if name.startswith("deepseek"):
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        if name.startswith("claude") and not name.startswith(
                ("claude-3-5", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku")):
            return {"thinking": {"type": "disabled"}}
        if name.startswith("gpt-5"):
            return {"reasoning_effort": "none"}
        if name.startswith("gemini-2.5"):
            return {"thinking": {"type": "disabled", "budget_tokens": 0}}
        return {}
    if name.startswith("deepseek") and not name.startswith("deepseek-r1"):
        return {"extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": effort}
    if name.startswith("claude") and not name.startswith(
            ("claude-3-5", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku")):
        budgets = {"minimal": 1024, "low": 1024, "medium": 2048, "high": 4096, "xhigh": 6144, "max": 6144}
        return {"thinking": {"type": "enabled", "budget_tokens": budgets[effort]}}
    if name.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
        return {"reasoning_effort": effort}
    if name.startswith("gemini-2.5"):
        budgets = {"minimal": 1024, "low": 1024, "medium": 2048, "high": 4096, "xhigh": 6144, "max": 6144}
        return {"thinking": {"type": "enabled", "budget_tokens": budgets[effort]}}
    if name.startswith("gemini-3") and effort in {"minimal", "low", "medium", "high"}:
        return {"reasoning_effort": effort}
    raise ConfigError(f"No reasoning effort mapping for {config.model}")


def graph_context_settings(config: LLMConfig) -> dict[str, Any]:
    """Build graph settings that are safe to persist in run artifacts."""
    settings = dict(config.graph_context)
    settings["model"] = config.litellm_model
    settings["builder_model"] = config.litellm_model
    if config.base_url:
        settings["api_base"] = config.base_url
    else:
        settings.pop("api_base", None)
    kwargs = dict(settings.get("model_kwargs", {}))
    if config.reasoning_effort == "no":
        for key in ("reasoning_effort", "thinking", "think"):
            kwargs.pop(key, None)
        if "extra_body" in kwargs:
            body = dict(kwargs["extra_body"])
            body.pop("thinking", None)
            kwargs["extra_body"] = body
    settings["model_kwargs"] = model_request_kwargs(
        replace(config, model_kwargs=merge_kwargs(kwargs, config.model_kwargs))
    )
    return settings


def apply_environment(config: LLMConfig, environ: MutableMapping[str, str] | None = None) -> MutableMapping[str, str]:
    """Set variables used by current clients; return the mutated environment."""
    environ = os.environ if environ is None else environ
    # Resolve references before clearing inherited transport settings. YAML is
    # authoritative even when a previous shell exported another endpoint/key.
    key = config.resolve_api_key(environ)
    judge = config.judge or config
    judge_key = judge.resolve_api_key(environ) if config.judge else key
    if config.judge and judge_key is None and judge.provider == config.provider:
        judge_key = key
    provider_key = _PROVIDER_KEYS.get(config.provider, config.provider.upper().replace("-", "_") + "_API_KEY")
    judge_provider_key = _PROVIDER_KEYS.get(judge.provider, judge.provider.upper().replace("-", "_") + "_API_KEY")
    for name in {*_PROVIDER_KEYS.values(), provider_key, judge_provider_key,
                 "OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_KEY_PATH",
                 "ANTHROPIC_BASE_URL", "AZURE_API_BASE", "AZURE_API_KEY",
                 "API_CONFIG", "MODEL_CONFIG", "EVO_BUILD_MODEL", "EVAL_API_KEY", "EVAL_BASE_URL"}:
        environ.pop(name, None)
    environ["LLM_MODEL_KWARGS"] = json.dumps(model_request_kwargs(config))
    environ["LLM_JUDGE_KWARGS"] = json.dumps(model_request_kwargs(judge))
    environ["LLM_PROVIDER"] = config.provider
    if key:
        environ[provider_key] = key
        if config.provider in _OPENAI_COMPATIBLE:
            environ["OPENAI_API_KEY"] = key
    endpoint = config.base_url or (_DEFAULT_ENDPOINTS.get(config.provider) if config.provider != "openai" else None)
    if endpoint:
        environ["OPENAI_BASE_URL"] = endpoint
        environ["OPENAI_API_BASE"] = endpoint
    if config.provider in {"gemini", "google"}:
        environ["DSSTAR_PROVIDER"] = "gemini"
    elif config.provider in _OPENAI_COMPATIBLE:
        environ["DSSTAR_PROVIDER"] = "openai"
    else:
        environ["DSSTAR_PROVIDER"] = config.provider
    if config.source_path:
        environ["LLM_CONFIG_FILE"] = config.source_path
    model = config.provider_model
    for name in ("DSGURU_MODEL", "DSSTAR_MODEL", "SMOLAGENTS_MODEL", "EVO_BUILD_MODEL"):
        environ[name] = model
    environ["LLM_JUDGE_PROVIDER"] = judge.provider
    environ["EVAL_MODEL"] = judge.provider_model
    if judge_key:
        environ["EVAL_API_KEY"] = judge_key
        if judge.provider != config.provider and not (
                judge_provider_key == "OPENAI_API_KEY" and config.provider in _OPENAI_COMPATIBLE):
            environ[judge_provider_key] = judge_key
    judge_endpoint = judge.base_url or _DEFAULT_ENDPOINTS.get(judge.provider)
    if judge_endpoint:
        environ["EVAL_BASE_URL"] = judge_endpoint
    if config.reasoning_effort == "no":
        environ["KRAMABENCH_THINKING"] = "disabled"
        environ["KRAMABENCH_REASONING_EFFORT"] = "default"
        environ["REASONING_EFFORT"] = "none"
    elif config.reasoning_effort and config.reasoning_effort != "default":
        environ["KRAMABENCH_THINKING"] = "enabled"
        environ["KRAMABENCH_REASONING_EFFORT"] = config.reasoning_effort
        environ["REASONING_EFFORT"] = config.reasoning_effort
    else:
        environ["KRAMABENCH_THINKING"] = "disabled"
        environ["KRAMABENCH_REASONING_EFFORT"] = "default"
        environ.pop("REASONING_EFFORT", None)
    environ["EVAL_REASONING_EFFORT"] = judge.reasoning_effort or "default"
    environ["LLM_CONFIG_APPLIED"] = "1"
    return environ


def resolve_credentials(config: LLMConfig) -> LLMConfig:
    """Bind explicit secret references before switching between agent and judge."""
    judge = config.judge or config
    return replace(config, api_key=config.resolve_api_key(os.environ), api_key_env=None,
                   judge=replace(judge, api_key=judge.resolve_api_key(os.environ), api_key_env=None))


def apply_judge_environment(config: LLMConfig) -> None:
    """Switch to the YAML judge, clearing the agent's endpoint and credentials."""
    judge = config.judge or config
    apply_environment(replace(config, **{name: getattr(judge, name)
                                        for name in ModelConfig.__dataclass_fields__}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--role", default="agent", help="Named model role under roles in YAML")
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run", help="Run a command with YAML-derived LLM environment")
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        config = select_role(load_config(args.config), args.role)
        apply_environment(config)
    except ConfigError as error:
        parser.error(str(error))
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("run requires a command after --")
    os.execvpe(command[0], command, os.environ)
    return 0


if __name__ == "__main__":
    sys.exit(main())
