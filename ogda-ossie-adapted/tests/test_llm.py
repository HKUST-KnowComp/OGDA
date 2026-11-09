"""The LiteLLM transport: requests from parallel clients share one event loop."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import yaml

from ogda.llm import OpenAICompatibleClient


def test_parallel_clients_send_every_request_from_one_event_loop(tmp_path, monkeypatch):
    # LiteLLM keeps one global logging queue bound to one loop; a loop per worker makes its tasks conflict.
    import litellm

    loops = set()
    both_waiting = threading.Barrier(2, timeout=5)

    async def acompletion(**kwargs):
        loops.add(asyncio.get_running_loop())
        await asyncio.to_thread(both_waiting.wait)  # Both requests are in flight at once.
        message = SimpleNamespace(content="ok", tool_calls=None)
        usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"llm": {"model": "m", "base_url": "https://example.invalid/v1",
                                              "api_key": "offline-test-key"}}))

    def request(_):
        client = OpenAICompatibleClient(config)
        try:
            return client.generate("proposer", [{"role": "user", "content": "go"}]).text
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(request, range(2))) == ["ok", "ok"]
    assert len(loops) == 1


def test_model_kwargs_extra_body_reaches_litellm_as_extra_body(tmp_path, monkeypatch):
    # DeepSeek turns thinking off with extra_body {"thinking": {"type": "disabled"}}; LiteLLM rejects `thinking`
    # as a top-level argument for an OpenAI-compatible model.
    import litellm

    sent = {}

    async def acompletion(**kwargs):
        sent.update(kwargs)
        message = SimpleNamespace(content="ok", tool_calls=None)
        usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    config = tmp_path / "config.yaml"
    thinking = {"thinking": {"type": "disabled"}}
    config.write_text(yaml.safe_dump({"llm": {
        "model": "m", "base_url": "https://example.invalid/v1", "api_key": "offline-test-key",
        "roles": {"data_agent": {"model_kwargs": {"extra_body": thinking}}}}}))
    client = OpenAICompatibleClient(config)
    client.generate("data_agent", [{"role": "user", "content": "go"}])
    assert sent["extra_body"] == thinking and "thinking" not in sent
    sent.clear()
    client.generate("evaluator", [{"role": "user", "content": "go"}])
    assert "extra_body" not in sent  # Only the role that sets it, so the judge keeps thinking on.


def test_a_busy_provider_is_retried_for_about_ten_minutes_before_the_request_fails(tmp_path, monkeypatch):
    # Providers answer 503 in bursts; a few seconds of retries let one burst end a construction run.
    import litellm
    import pytest
    import ogda.llm as llm

    busy, waits = [5], []

    async def acompletion(**kwargs):
        if busy[0]:
            busy[0] -= 1
            raise litellm.ServiceUnavailableError(message="busy", llm_provider="openai", model="m")
        message = SimpleNamespace(content="ok", tool_calls=None)
        usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)

    async def pause(seconds):
        waits.append(seconds)

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(llm, "pause", pause)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"llm": {"model": "m", "base_url": "https://example.invalid/v1",
                                              "api_key": "offline-test-key"}}))
    client = llm.OpenAICompatibleClient(config)
    assert client.generate("proposer", [{"role": "user", "content": "go"}]).text == "ok"
    assert waits == [1, 2, 4, 8, 16] and client.usage["calls"] == 6
    busy[0], waits[:] = 100, []
    with pytest.raises(llm.LLMError, match="503"):
        client.generate("proposer", [{"role": "user", "content": "go"}])
    assert 500 <= sum(waits) <= 660 and max(waits) == 60  # Provider outages of several minutes were seen.
