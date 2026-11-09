"""Model profiles choose how CodeAgent receives code: a native `code` call or a text code block."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import yaml
from smolagents import CodeAgent
from smolagents.models import MessageRole

from helpers import call, code
from ogda.agent import (CODE_TOOL, CODE_TOOL_NOTE, NO_CALL_NOTE, TRUNCATED_NOTE, AgentConfig, LLMModel, code_text,
                                run_agent)
from ogda.llm import Generation, OpenAICompatibleClient, ScriptedClient
from ogda.ontology import Harness
from ogda.retrieval import ontology_tools
from ogda.sandbox import IMPORTS


def visit(tmp_path, client, *, role="proposer", validate=lambda answer: None, tools=()):
    lake = tmp_path / "lake"
    lake.mkdir(exist_ok=True)
    (lake / "values.csv").write_text("value\n42\n")
    return run_agent(client, role=role, instructions="Inspect the File and submit the result.", task="values.csv",
                     validate=validate, lake_root=lake, scratch_root=tmp_path / "scratch",
                     artifact_root=tmp_path / "logs", config=AgentConfig(max_steps=3), tools=tools)


@pytest.mark.parametrize("model", ["scripted", "openai/other-model", "deepseek-v4.1-flash-extra"])
def test_unknown_models_offer_the_code_call_and_accept_text_code_blocks(tmp_path, model):
    client = ScriptedClient(lambda role, system, prompt: code("final_answer({'value': 42})"), model=model)
    result = visit(tmp_path, client)
    assert result.answer == {"value": 42} and result.steps == 1
    request = client.requests[0]
    assert request["tools"] == [CODE_TOOL] and request["tool_choice"] == "auto" and request["stop"]
    instructions = f"{CODE_TOOL_NOTE}\n\nInspect the File and submit the result."
    stock = CodeAgent(tools=[], model=LLMModel(client, "proposer"), additional_authorized_imports=IMPORTS,
                      instructions=instructions, verbosity_level=-1)
    assert request["system"] == stock.initialize_system_prompt()


def test_gpt_profile_requires_the_code_call(tmp_path):
    client = ScriptedClient(lambda role, system, prompt: call("final_answer({'value': 42})"), model="openai/gpt-5")
    result = visit(tmp_path, client)
    assert result.answer == {"value": 42} and result.steps == 1
    request = client.requests[0]
    assert request["tools"] == [CODE_TOOL] and request["tool_choice"] == "required" and request["stop"] is None
    assert CODE_TOOL_NOTE in request["system"]


@pytest.mark.parametrize("model", [
    "deepseek-flash", "deepseek-v4.1-flash", "deepseek-v4-flash", "openai/deepseek-flash", "deepseek/DeepSeek-V4.1-Flash",
])
def test_deepseek_profile_asks_only_for_code_calls(tmp_path, model):
    replies = iter([call("value = int((LAKE_ROOT / 'values.csv').read_text().split()[1])", thought="Read the File."),
                    call("final_answer({'value': value})", thought="Submit.")])
    client = ScriptedClient(lambda role, system, prompt: next(replies), model=model)
    result = visit(tmp_path, client)
    assert result.answer == {"value": 42} and result.steps == 2
    for request in client.requests:
        assert request["tools"] == [CODE_TOOL] and request["tool_choice"] == "auto"  # Thinking mode rejects required.
        system = request["system"]
        assert CODE_TOOL_NOTE in system and "<code>" not in system and "Thought:" not in system
        assert '{"thought": "The result is built and supported, so submit it now.", "code": "final_answer(result)"}' in system
        assert "{{" not in system and "{%" not in system
    step, _ = history(client.requests[1])
    assert code_call(step)["thought"] == "Read the File."


def history(request) -> list[dict]:
    return json.loads(request["prompt"])[2:]  # After the system prompt and the task.


def code_call(message) -> dict:
    [tool_call] = message["tool_calls"]
    assert message["role"] == "assistant" and tool_call["function"]["name"] == "code"
    return json.loads(tool_call["function"]["arguments"])


def test_code_calls_keep_python_state_tools_and_native_history(tmp_path):
    source = ("knowledge = search_ontology('values')\nrows = (LAKE_ROOT / 'values.csv').read_text().splitlines()\n"
              "value = int(rows[1])\nprint(value, len(knowledge['datasets']))")
    replies = iter([call(source, thought="Read the File first."),
                    call("final_answer({'value': value, 'datasets': len(knowledge['datasets'])})")])
    client = ScriptedClient(lambda role, system, prompt: next(replies), model="openai/other-model")
    result = visit(tmp_path, client, tools=ontology_tools(Harness.empty("toy")))
    assert result.answer == {"value": 42, "datasets": 0} and result.steps == 2
    step, observation = history(client.requests[1])
    assert code_call(step) == {"thought": "Read the File first.", "code": source} and step["content"] is None
    assert "reasoning_content" not in step  # Only profiles with return_reasoning send it.
    assert observation["role"] == "tool" and observation["tool_call_id"] == step["tool_calls"][0]["id"]
    assert "42 0" in observation["content"] and "<code>" not in json.dumps(history(client.requests[1]))


def test_deepseek_gets_its_reasoning_back_with_each_code_call(tmp_path):
    # DeepSeek's thinking mode rejects an earlier tool call sent back without reasoning_content, even an empty one.
    replies = iter([{**call("print(1)"), "reasoning": "Count first."}, call("print(2)"), call("final_answer({'value': 42})")])
    client = ScriptedClient(lambda role, system, prompt: next(replies), model="deepseek-flash")
    assert visit(tmp_path, client).answer == {"value": 42}
    first, _, second, _ = history(client.requests[2])
    assert first["reasoning_content"] == "Count first." and second["reasoning_content"] == ""


def test_every_step_reaches_the_model_as_a_code_call(tmp_path):
    # A text code block, a malformed call, a cut-off response and prose all come back without text-code hints.
    replies = iter([f"Thought: Count.\n{code('print(1)')}",
                    {"text": "", "tool_calls": [{"name": "code", "arguments": '{"thought": "x"}'}]},
                    {"text": "", "truncated": True},
                    "I will now count the rows.",
                    call("print(1)", thought="Count."),
                    call("final_answer({'value': 42})")])
    client = ScriptedClient(lambda role, system, prompt: next(replies), model="openai/other-model")
    lake = tmp_path / "lake"
    lake.mkdir()
    result = run_agent(client, role="proposer", instructions="Submit the value.", task="values",
                       validate=lambda answer: None, lake_root=lake, scratch_root=tmp_path / "scratch",
                       artifact_root=tmp_path / "logs", config=AgentConfig(max_steps=6))
    assert result.answer == {"value": 42} and result.steps == 6
    messages = history(client.requests[5])
    assert [message["role"] for message in messages] == ["assistant", "tool", "assistant", "tool", "user",
                                                         "assistant", "user", "assistant", "tool"]
    assert code_call(messages[0]) == {"thought": "Count.", "code": "print(1)"} and "\n1\n" in messages[1]["content"]
    assert code_call(messages[2]) == {"thought": "x"}
    assert messages[3]["content"].startswith("The `code` call had invalid arguments (KeyError: 'code')")
    assert messages[4]["content"].endswith(TRUNCATED_NOTE)  # After the "steps remain" note.
    assert messages[5] == {"role": "assistant", "content": "I will now count the rows."}
    assert messages[6]["content"].endswith(NO_CALL_NOTE)
    assert code_call(messages[7]) == code_call(messages[0])  # The same step twice keeps its own call id.
    ids = [m["tool_calls"][0]["id"] for m in messages if "tool_calls" in m]
    assert len(set(ids)) == len(ids) == 3
    assert "<code>" not in client.requests[5]["prompt"].split('"role": "assistant"', 1)[1]


def test_invalid_call_arguments_and_answers_receive_corrections(tmp_path):
    broken = {"text": "", "tool_calls": [{"name": "code", "arguments": '{"thought": "x", "code": "final_answer('}]}
    replies = iter([broken, call("final_answer('wrong type')"), call("final_answer({'value': 0})"),
                    call("final_answer({'value': 42})")])
    client = ScriptedClient(lambda role, system, prompt: next(replies), model="openai/other-model")

    def validate(answer):  # run_agent hands any final_answer value to validate, a bare string included.
        if not isinstance(answer, dict):
            raise ValueError("Submit a dict")
        if answer.get("value") != 42:
            raise ValueError("value must equal 42")

    lake = tmp_path / "lake"
    lake.mkdir()
    result = run_agent(client, role="proposer", instructions="Submit the value.", task="values", validate=validate,
                       lake_root=lake, scratch_root=tmp_path / "scratch", artifact_root=tmp_path / "logs",
                       config=AgentConfig(max_steps=4))
    assert result.answer == {"value": 42} and result.steps == 4
    assert "The `code` call had invalid arguments" in client.requests[1]["prompt"]
    assert "Submit a dict" in client.requests[2]["prompt"]
    assert "value must equal 42" in client.requests[3]["prompt"]


def test_code_text_runs_only_the_call_code():
    tags = ("<code>", "</code>")
    with_text = Generation("I check the File.", call("print(1)", thought="Count rows.")["tool_calls"])
    assert code_text(with_text, tags) == "I check the File.\nThought: Count rows.\n<code>\nprint(1)\n</code>"
    duplicate = Generation(code("print(2)"), call("print(1)")["tool_calls"])
    assert code_text(duplicate, tags).count("<code>") == 1 and "print(2)" not in code_text(duplicate, tags)
    assert code_text(Generation(code("print(3)")), tags) == code("print(3)")


def test_truncated_responses_run_no_code_and_receive_a_correction(tmp_path):
    # A reasoning model can spend the whole output limit before any text, or stop inside a code block.
    partial = call("open(SCRATCH_ROOT / 'ran.txt', 'w').write('x')\nprint(", thought="Write.")
    replies = iter([{"text": "", "truncated": True},
                    {"text": partial["text"], "tool_calls": partial["tool_calls"], "truncated": True},
                    {"text": "<code>\nopen(SCRATCH_ROOT / 'ran.txt', 'w').write('x')\n", "truncated": True},
                    call("final_answer({'value': 42})")])
    client = ScriptedClient(lambda role, system, prompt: next(replies), model="openai/other-model")
    lake = tmp_path / "lake"
    lake.mkdir()
    result = run_agent(client, role="proposer", instructions="Submit the value.", task="values",
                       validate=lambda answer: None, lake_root=lake, scratch_root=tmp_path / "scratch",
                       artifact_root=tmp_path / "logs", config=AgentConfig(max_steps=4))
    assert result.answer == {"value": 42} and result.steps == 4
    assert all(TRUNCATED_NOTE in request["prompt"] for request in client.requests[1:])
    assert not list(tmp_path.rglob("ran.txt"))
    assert code_text(Generation("<code>\nprint(1)\n", truncated=True), ("<code>", "</code>")) == TRUNCATED_NOTE


def test_openai_client_sends_the_code_tool_and_reads_tool_calls(tmp_path, monkeypatch):
    import litellm

    sent = {}

    async def acompletion(**kwargs):
        sent.update(kwargs)
        function = SimpleNamespace(name="code", arguments='{"thought": "t", "code": "print(1)"}')
        message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(function=function)])
        usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")], usage=usage)

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"llm": {"model": "deepseek-flash", "base_url": "https://example.invalid/v1",
                                            "api_key": "offline-test-key"}}))
    client = OpenAICompatibleClient(path, record_path=tmp_path / "llm.jsonl")
    try:
        generation = client.generate("proposer", [{"role": "user", "content": "go"}], None,
                                     tools=[CODE_TOOL], tool_choice="required")
    finally:
        client.close()
    assert sent["tools"] == [CODE_TOOL] and sent["tool_choice"] == "required" and "stop" not in sent
    assert sent["max_tokens"] == 8192  # The default output limit leaves room for reasoning.
    assert generation == Generation("", [{"name": "code", "arguments": '{"thought": "t", "code": "print(1)"}'}])
    assert not generation.truncated
    record = json.loads((tmp_path / "llm.jsonl").read_text())
    assert record["tool_calls"] == generation.tool_calls and record["finish_reason"] == "tool_calls"


def test_role_model_names_reuse_yaml_inheritance(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"llm": {
        "model": "openai/base-model", "base_url": "https://example.invalid/v1", "api_key": "offline-test-key",
        "roles": {"proposer": {"model": "deepseek/DeepSeek-V4.1-Flash"}, "verifier": {"model": "other-model"},
                  "experience": {}},
    }}))
    client = OpenAICompatibleClient(path)
    try:
        assert client.model_name("proposer") == "deepseek/DeepSeek-V4.1-Flash"
        assert client.model_name("verifier") == "other-model"
        assert client.model_name("experience") == "deepseek/DeepSeek-V4.1-Flash"
        assert client.model_name("other-role") == "openai/base-model"
        assert LLMModel(client, "proposer").model_id == "deepseek/DeepSeek-V4.1-Flash"
        client.config["roles"]["experience"]["model"] = "experience-model"
        assert client.model_name("experience") == "experience-model"
    finally:
        client.close()


def test_new_and_edited_yaml_profiles_are_read_on_each_visit(tmp_path, monkeypatch):
    from ogda import agent

    package = tmp_path / "package"
    models = package / "models"
    models.mkdir(parents=True)
    default = yaml.safe_load(agent.importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text())
    profile = models / "custom.yaml"
    original_files = agent.importlib.resources.files
    monkeypatch.setattr(agent.importlib.resources, "files",
                        lambda name: package if name == "ogda" else original_files(name))
    monkeypatch.chdir(tmp_path)

    for opening, closing in [("<custom>", "</custom>"), ("<edited>", "</edited>")]:
        settings = {"code_block_tags": [opening, closing]}
        if opening == "<edited>":
            settings["system_prompt"] = default["system_prompt"] + "\nEdited profile."
        profile.write_text(yaml.safe_dump({"model_names": ["custom-model"], "code_agent": settings}))
        client = ScriptedClient(lambda role, system, prompt: f"{opening}\nfinal_answer({{'value': 42}})\n{closing}",
                                model="provider/custom-model")
        result = visit(tmp_path, client)
        assert result.answer == {"value": 42} and result.steps == 1
        assert opening in client.requests[0]["system"] and closing in client.requests[0]["system"]
        assert ("Edited profile." in client.requests[0]["system"]) == (opening == "<edited>")


def test_profile_tool_choice_none_sends_no_tools_and_bad_values_fail(tmp_path, monkeypatch):
    from ogda import agent

    models = tmp_path / "package" / "models"
    models.mkdir(parents=True)
    original_files = agent.importlib.resources.files
    monkeypatch.setattr(agent.importlib.resources, "files",
                        lambda name: tmp_path / "package" if name == "ogda" else original_files(name))
    profile = models / "text.yaml"
    profile.write_text(yaml.safe_dump({"model_names": ["text-model"], "code_agent": {"tool_choice": "none"}}))
    client = ScriptedClient(lambda role, system, prompt: code("final_answer({'value': 42})"), model="text-model")
    assert visit(tmp_path, client).answer == {"value": 42}
    assert client.requests[0]["tools"] is None and CODE_TOOL_NOTE not in client.requests[0]["system"]
    profile.write_text(yaml.safe_dump({"model_names": ["text-model"], "code_agent": {"tool_choice": "always"}}))
    with pytest.raises(ValueError, match="tool_choice"):
        agent.load_model_profile("text-model")


def test_model_profiles_name_only_tools_and_shapes_that_exist():
    from pathlib import Path
    for profile in (Path(__file__).parent.parent / "ogda" / "models").glob("*.yaml"):
        text = profile.read_text()
        assert "read_ontology" not in text and "['schemas']" not in text, profile.name


SEARCH_EXAMPLE = (r'''- {"thought": "Search the Ontology for what already exists, and keep the results for later '''
                  r'''steps.", "code": "found = search_ontology('household income')\nconcepts = {c['concept']: c '''
                  r'''for c in found['concepts']}\nprint(sorted(concepts) or 'Nothing matches yet.')"}''')
FINAL_EXAMPLE = '- {"thought": "The result is built and supported, so submit it now.", "code": "final_answer(result)"}'


def deepseek_prompt(*tools: str) -> str:
    """The DeepSeek profile's system prompt as CodeAgent renders it, for tools with these names."""
    from smolagents.agents import populate_template

    from ogda.agent import load_model_profile

    stubs = {name: SimpleNamespace(to_code_prompt=lambda name=name: f"def {name}(): ...") for name in tools}
    return populate_template(load_model_profile("deepseek-flash")["system_prompt"],
                             {"tools": stubs, "authorized_imports": "['pandas']", "custom_instructions": "Answer."})


def test_deepseek_profile_shows_a_ground_example_only_when_ground_is_a_tool():
    for tools in [("final_answer",), ("search_ontology", "get_concept", "get_dataset", "final_answer")]:
        prompt = deepseek_prompt(*tools)  # The baseline, then construction: the search example, as before.
        assert f"single steps:\n{SEARCH_EXAMPLE}\n{FINAL_EXAMPLE}" in prompt and "ground(" not in prompt
    ogda = deepseek_prompt("ground", "search", "final_answer")
    assert "fires, acres = ground('fires', 'burned.acres')" in ogda and "search_ontology" not in ogda
