"""OGDA, the Data agent: the task message and Lake map, ground and search, the Task graph, answer repair and worker
access."""

import copy
import json

import pytest

from helpers import code, income_harness, make_lake
from ogda.agent import AgentConfig
from ogda.data_agent import DataAgentConfig, answer as answer_task
from ogda.grounding import Grounding
from ogda.lake import copy_lake
from ogda.llm import ScriptedClient
from ogda.ontology import ExperienceEntry, OntologyStore
from ogda.ontology_search import OntologySearch

TASK = {"id": "task-001", "question": "How many households earn more than 150 in annual income?",
        "answer_format": "One integer."}
GROUND = code("df = ground('income')")
COUNT = code("high = df[df['annual_income'] > 150]\nfinal_answer(len(high))")


class Roles:
    """Answer each role from its own list of responses; the last response repeats."""

    def __init__(self, **responses):
        self.responses = {role: list(queue) for role, queue in responses.items()}
        self.prompts = {role: [] for role in responses}

    def __call__(self, role, system, prompt):
        self.prompts[role].append(prompt)
        queue = self.responses[role]
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        return response(prompt) if callable(response) else response


@pytest.fixture
def store(tmp_path):
    lake = make_lake(tmp_path)
    (lake / "notes.txt").write_text("Household H2 moved in 2020.\n")
    store = OntologyStore(tmp_path / "run" / "ontology")
    copy_lake(lake, store.lake_root)
    harness = income_harness(store.lake_root)
    harness.experience.append(ExperienceEntry(id="experience-1", kind="data_quirk", text="Income is top coded at 150",
                                              anchors=["income.annual_income"]))
    harness.metadata["progress"] = {"files": {"notes.txt": {"status": "unresolved", "reason": "Free text"}}}
    store.save(harness)
    return store


def ask(tmp_path, store, responses, task=TASK, check=None, max_steps=3, **settings):
    roles = Roles(data_agent=responses)
    config = DataAgentConfig(agent=AgentConfig(max_steps=max_steps), **settings)
    client = ScriptedClient(roles)
    result = answer_task(store, client, task, config, tmp_path / "scratch", tmp_path / "out", check=check)
    return result, roles.prompts["data_agent"], client


def events(tmp_path, name):
    lines = (tmp_path / "out" / "logs" / "data_agent.jsonl").read_text().splitlines()
    return [event for event in map(json.loads, lines) if event["event"] == name]


def initial_message(prompts):
    return next(m["content"] for m in json.loads(prompts[0]) if m["role"] == "user").removeprefix("New task:\n")


def test_the_task_message_holds_the_task_and_then_the_lake_map(tmp_path, store):
    _, prompts, _ = ask(tmp_path, store, [code("final_answer(1)")])
    task, lake = initial_message(prompts).split("\n\nLAKE MAP\n", 1)
    assert json.loads(task) == {"task": {"question": TASK["question"], "answer_format": TASK["answer_format"]},
                                "max_steps": 3, "workspace": str(tmp_path / "scratch" / "workspace")}
    assert "- income: income.csv; one Household. Annual income of each household." in lake
    assert "- notes.txt: 28 bytes" in lake.splitlines()
    assert any(line.startswith("- README.md: ") and line.endswith(", Documentation") for line in lake.splitlines())


def test_agent_grounds_a_table_sees_its_meaning_and_answers_from_it(tmp_path, store):
    result, prompts, _ = ask(tmp_path, store, [GROUND, COUNT])
    assert "income: one Household, 2 rows" in prompts[1] and "Income is top coded at 150" in prompts[1]
    assert result["status"] == "answered" and result["answer"] == 1 and "searches" not in result
    assert [node["id"] for node in result["graph"]["nodes"]] == ["income", "high@2"]
    assert result["graph"]["answer"] == {"typed": False, "from": ["income"]}
    assert [event["names"] for event in events(tmp_path, "ground")] == [["income"]]


def test_agent_grounds_a_table_and_a_column_in_one_call(tmp_path, store):
    # The prompt's first step; the sandbox forbids dunder access, so the agent's code checks the types itself.
    both = code("a, b = ground('income', 'income.annual_income')\n"
                "print(isinstance(a, pd.DataFrame), isinstance(b, pd.Series))")
    result, prompts, _ = ask(tmp_path, store, [both, code("final_answer(len(a[b > 150]))")])
    harness = store.load()
    grounding = Grounding(harness, store.lake_root, tmp_path / "cache", OntologySearch(harness))
    views = [grounded.view for grounded in grounding.ground(["income", "income.annual_income"])]
    assert "\n\n".join(views) + "\nTrue True\n" in json.loads(prompts[1])[-1]["content"]
    [event] = events(tmp_path, "ground")
    assert event["names"] == ["income", "income.annual_income"] and event["rows"] == [2, 2]
    assert [(node["id"], node["kind"]) for node in result["graph"]["nodes"]] == [
        ("income", "dataset"), ("income.annual_income", "field")]
    assert result["answer"] == 1
    assert result["graph"]["answer"] == {"typed": False, "from": ["income", "income.annual_income"]}


def test_agent_has_no_lake_root_and_reads_a_grounded_file(tmp_path, store):
    read = code("p = ground('notes.txt')\nfinal_answer(open(p).read().strip())")
    result, prompts, _ = ask(tmp_path, store, [code("print(LAKE_ROOT)"), read])
    assert "The variable `LAKE_ROOT` is not defined" in prompts[1]  # The worker's own error, not the step's code.
    assert result["answer"] == "Household H2 moved in 2020."
    assert result["graph"]["answer"] == {"typed": False, "from": ["notes.txt"]}


def test_agent_cannot_write_a_grounded_file_and_copies_it_to_the_workspace(tmp_path, store):
    write = code("p = ground('income.csv')\nopen(p, 'w').write('x')")
    copy = code("import shutil\nshutil.copyfile(p, WORKSPACE / 'income.csv')\nfinal_answer(1)")
    result, prompts, _ = ask(tmp_path, store, [write, copy])
    assert "PermissionError" in prompts[1] and result["answer"] == 1
    assert (store.lake_root / "income.csv").read_text().startswith("household_id")
    assert (tmp_path / "scratch" / "workspace" / "income.csv").is_file()


def test_search_prints_matching_columns(tmp_path, store):
    _, prompts, _ = ask(tmp_path, store, [code("search('annual income')"), code("final_answer(1)")])
    # The prompt is JSON text, whose strings escape the header's quotation marks, so the line is found in a message.
    line = 'income.annual_income  "annual_income"  Decimal  Annual household income in US dollars'
    assert any(line in (message.get("content") or "") for message in json.loads(prompts[1]))
    assert [(event["words"], event["fields"]) for event in events(tmp_path, "search")] == [
        ("annual income", ["income.annual_income"])]


def test_a_bad_search_limit_reaches_the_agent_with_its_message(tmp_path, store):
    _, prompts, _ = ask(tmp_path, store, [code("search('income', limit=2.5)"), code("final_answer(1)")])
    assert "limit must be a whole number of columns, such as 15, not 2.5" in prompts[1]


def test_an_unknown_name_reaches_the_agent_with_close_names(tmp_path, store):
    _, prompts, _ = ask(tmp_path, store, [code("ground('incomes')"), code("final_answer(1)")])
    assert "No table, column or File named 'incomes'. Close names: income" in prompts[1]


@pytest.mark.parametrize("call", ["ground(3)", "ground([['income']])"])
def test_a_name_that_is_not_a_string_reaches_the_agent_as_a_correction(tmp_path, store, call):
    _, prompts, _ = ask(tmp_path, store, [code(call), code("final_answer(1)")])
    assert "ground takes each name as a string" in prompts[1]


def test_views_too_long_for_the_output_limit_reach_the_agent_as_a_correction(tmp_path, store):
    roles = Roles(data_agent=[code("a, b = ground('income', 'income.annual_income')"), code("final_answer(1)")])
    config = DataAgentConfig(agent=AgentConfig(max_steps=3, max_output_chars=400))
    result = answer_task(store, ScriptedClient(roles), TASK, config, tmp_path / "scratch", tmp_path / "out")
    assert "more than one call prints (300)" in roles.prompts["data_agent"][1]
    assert result["graph"]["nodes"] == [] and events(tmp_path, "ground") == []


def test_agent_grounds_two_tables_joins_them_and_answers(tmp_path):
    from county_lake import county_harness

    store = OntologyStore(tmp_path / "run" / "ontology")
    store.lake_root.mkdir(parents=True)
    store.save(county_harness(store.lake_root))
    join = code("c, b = ground('counties', 'burned')\nm = b.merge(c, on='County')")
    total = code("nevada = m[m.State == 'Nevada']['2020'].sum()\nfinal_answer(float(nevada))")
    task = {"id": "t", "question": "How many acres burned in Nevada in 2020?"}
    result, prompts, _ = ask(tmp_path, store, [join, total], task=task)
    assert '  counties on County: "County" (County.name) ↔ "County" (County.name)' in json.loads(
        prompts[1])[-1]["content"]
    assert result["answer"] == 20.0 and result["graph"]["answer"] == {"typed": False, "from": ["burned", "counties"]}
    saved = json.loads((tmp_path / "out" / "result.json").read_text())
    assert saved["graph"]["edges"][0]["derived"] == {"m@1": ["burned", "counties"]}
    assert [node["id"] for node in saved["graph"]["nodes"]] == ["counties", "burned", "m@1", "nevada@2"]


def test_ground_takes_one_name_given_as_the_names_keyword(tmp_path, store):
    # The tool list shows ground(names: array), so an agent may pass one name by keyword.
    result, _, _ = ask(tmp_path, store, [code("df = ground(names='income')\nfinal_answer(len(df))")])
    assert result["answer"] == 2 and [event["names"] for event in events(tmp_path, "ground")] == [["income"]]
    assert result["graph"]["answer"] == {"typed": False, "from": ["income"]}


def test_a_typed_answer_is_recorded_as_typed(tmp_path, store):
    result, _, _ = ask(tmp_path, store, [code("final_answer(2)")])
    assert result["graph"]["answer"] == {"typed": True, "from": []}


def test_agent_sees_ground_and_search_and_no_lake_root(tmp_path, store):
    _, prompts, _ = ask(tmp_path, store, [code("final_answer(1)")])
    assert "ground(" in prompts[0] and "search(" in prompts[0]
    assert "search_ontology" not in prompts[0] and "LAKE_ROOT" not in prompts[0]


def test_the_prompt_says_what_code_checked_and_to_trust_the_data():
    from ogda.prompts import DATA_AGENT

    prompt = " ".join(DATA_AGENT.split())  # The prompt wraps its lines as the spec does, so words meet across them.
    assert "Code checked each table's File, region and header" in prompt
    assert "when the data and the description disagree, trust the data" in prompt
    assert "Every File comes through ground; there is no other way into the Lake." in prompt


def test_the_prompt_says_columns_are_named_by_their_file_header():
    from ogda.prompts import DATA_AGENT

    prompt = " ".join(DATA_AGENT.split())
    assert ("It returns a DataFrame with one column per column it printed, named by its File header (by its column "
            "name when it has none); the index is the row number in the File.") in prompt
    assert ("For a File of a table that stands for several Files, it returns that File's rows of the table "
            "instead.") in prompt


@pytest.mark.parametrize("invalid, error", [
    ("None", "with the answer itself"),
    ("{'answer': 1}", 'not final_answer({"answer": value})'),
    ("{'answer': 1, 'explanation': 'Counted'}", 'not final_answer({"answer": value})'),
])
def test_agent_repairs_invalid_final_answer(tmp_path, store, invalid, error):
    result, prompts, _ = ask(tmp_path, store, [code(f"final_answer({invalid})"), code("final_answer(1)")])
    assert any(error in (message.get("content") or "") for message in json.loads(prompts[1]))
    assert result["status"] == "answered" and result["answer"] == 1 and "explanation" not in result


@pytest.mark.parametrize("value", ["'No'", "0", "[]", "{'hypothesis': 'h', 'workflow': 'w'}"])
def test_agent_submits_the_answer_value_itself(tmp_path, store, value):
    # Zero and an empty list are answers, and an object answer such as a DiscoveryBench one passes through.
    result, _, _ = ask(tmp_path, store, [code(f"final_answer({value})")])
    assert result["status"] == "answered" and result["answer"] == eval(value)


@pytest.mark.parametrize("quoted, unwrapped", [('"Singapore"', "Singapore"), (['"Yes"', "No"], ["Yes", "No"])])
def test_a_json_serialized_string_answer_is_unwrapped(tmp_path, store, quoted, unwrapped):
    # A string scorer compares '"Singapore"' with its quotation marks, so the saved answer drops them.
    result, prompts, _ = ask(tmp_path, store, [code(f"final_answer({quoted!r})")], check=lambda value: None)
    assert result["status"] == "answered" and result["answer"] == unwrapped and result["steps"] == 1
    [event] = events(tmp_path, "answer_unwrapped")
    assert event["submitted"] == quoted and event["answer"] == unwrapped


def test_a_doubly_serialized_string_answer_is_still_rejected(tmp_path, store):
    result, prompts, _ = ask(tmp_path, store, [code("import json\nfinal_answer(json.dumps(json.dumps('Singapore')))"),
                                               code("final_answer('Singapore')")])
    assert "not json.dumps of it" in prompts[1]
    assert result["answer"] == "Singapore" and result["steps"] == 2


def test_agent_hears_how_many_steps_remain_near_the_limit(tmp_path, store):
    result, prompts, _ = ask(tmp_path, store, [code("print('a')"), code("print('b')"), code("final_answer(1)")],
                             max_steps=5)
    assert "steps remain" not in prompts[1] and "3 of 5 steps remain" in prompts[2]
    assert result["answer"] == 1


def test_agent_repairs_an_answer_that_fails_the_adapter_check(tmp_path, store):
    def integer(value):
        if not isinstance(value, int):
            raise ValueError("answer must be a JSON integer")

    result, prompts, _ = ask(tmp_path, store, [code("final_answer('1')"),
                                               code("final_answer(1)")],
                             check=integer)
    assert "required shape: answer must be a JSON integer" in prompts[1]
    assert result["status"] == "answered" and result["answer"] == 1
    assert events(tmp_path, "answer_rejected")[0]["answer"] == "1"


def test_agent_that_runs_out_of_steps_leaves_an_incomplete_result(tmp_path, store):
    result, prompts, _ = ask(tmp_path, store, [code("df = ground('income')\nprint('still working')")])
    assert result["status"] == "incomplete" and result["stop_reason"] == "incomplete"
    assert result["answer"] is None and result["steps"] == 3 and len(prompts) == 3
    assert [node["id"] for node in result["graph"]["nodes"]] == ["income"] and result["graph"]["answer"] is None
    assert [event["names"] for event in events(tmp_path, "ground")] == [["income"]] * 3


@pytest.mark.parametrize("config", [DataAgentConfig(), DataAgentConfig.from_dict({}),
                                   DataAgentConfig.from_dict({"max_tokens": None})], ids=["default", "omitted", "null"])
def test_agent_without_a_token_budget_can_continue_after_a_million_tokens(tmp_path, store, config):
    roles = Roles(data_agent=[GROUND, COUNT])

    class Client(ScriptedClient):
        def generate(self, *args, **kwargs):
            response = super().generate(*args, **kwargs)
            if self.usage["calls"] == 1:
                self.usage["input_tokens"] += 1_000_001
                self.usage["total_tokens"] += 1_000_001
            return response

    client = Client(roles)
    result = answer_task(store, client, TASK, config, tmp_path / "scratch", tmp_path / "out")

    assert result["status"] == "answered" and result["answer"] == 1 and result["steps"] == 2
    assert result["usage"]["total_tokens"] > 1_000_000 and client.usage["calls"] == 2
    assert json.loads((tmp_path / "out" / "result.json").read_text()) == result


def test_spent_budget_saves_an_incomplete_result(tmp_path, store):
    result, _, client = ask(tmp_path, store, [GROUND, COUNT], max_tokens=1)
    assert result["status"] == "incomplete" and result["stop_reason"] == "budget" and result["answer"] is None
    assert [node["id"] for node in result["graph"]["nodes"]] == ["income"]
    assert json.loads((tmp_path / "out" / "result.json").read_text())["status"] == "incomplete"
    with pytest.raises(FileExistsError):
        answer_task(store, client, TASK, DataAgentConfig(), tmp_path / "scratch", tmp_path / "out")


def test_config_and_task_checks():
    config = DataAgentConfig.from_dict({"max_tokens": 5, "agent": {"cell_timeout_seconds": 60}})
    assert config.agent.max_steps == 20 and config.agent.cell_timeout_seconds == 60 and config.max_tokens == 5
    default = DataAgentConfig.from_dict(None)
    assert default.agent.max_steps == 20 and default.max_tokens is None
    assert DataAgentConfig.from_dict({}).max_tokens is None
    assert DataAgentConfig.from_dict({"max_tokens": None}).max_tokens is None
    with pytest.raises(ValueError, match="Unknown data_agent settings"):
        DataAgentConfig.from_dict({"planner": {"max_steps": 3}})
    from ogda.data_agent import check_task
    with pytest.raises(ValueError):
        check_task({"id": "t", "question": ""})
    assert check_task(copy.deepcopy(TASK)) == TASK


def test_data_agent_role_falls_back_to_analyst_settings(tmp_path):
    import yaml

    from ogda.llm import OpenAICompatibleClient

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"llm": {"model": "openai/base-model", "base_url": "https://example.invalid/v1",
                                            "api_key": "offline-test-key", "roles": {"analyst": {"model": "analyst-model"}}}}))
    client = OpenAICompatibleClient(path)
    try:
        assert client.model_name("data_agent") == "analyst-model"
    finally:
        client.close()


def test_cli_ask_answers_a_task_file(tmp_path, store, monkeypatch, capsys):
    import yaml

    from ogda import cli

    roles = Roles(data_agent=[GROUND, COUNT])

    class Client(ScriptedClient):
        def __init__(self, config, record_path=None):
            super().__init__(roles)

        def close(self):
            pass

    monkeypatch.setattr(cli, "OpenAICompatibleClient", Client)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"data_agent": {"agent": {"max_steps": 3}}}))
    (tmp_path / "task.json").write_text(json.dumps(TASK))
    cli.main(["ask", "--config", str(tmp_path / "config.yaml"), "--ontology", str(store.root),
              "--task", str(tmp_path / "task.json"), "--output", str(tmp_path / "out"), "--scratch", str(tmp_path / "scratch")])
    summary = json.loads(capsys.readouterr().out)
    assert summary["answer"] == 1 and summary["steps"] == 2
    assert summary["graph"]["grounded"] == ["income"]
    assert (tmp_path / "out" / "result.json").is_file()


def test_the_prompt_says_to_keep_the_plain_reading_and_round_only_the_answer():
    from ogda.prompts import DATA_AGENT

    prompt = " ".join(DATA_AGENT.split())
    assert ("- Answer the question as written. Change a table, filter or definition only when the data shows the plain "
            "reading is wrong, not to refine an answer you already have.") in prompt
    assert "- Compare and rank unrounded values; round only the final answer." in prompt
    assert ("Call final_answer as soon as the question is answered, in a step after the one that printed the "
            "value.") in prompt
