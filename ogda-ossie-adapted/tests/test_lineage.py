"""The Task graph: grounded nodes, derived nodes and edges read from each cell's code, and the answer's lineage."""

from ogda.lineage import TaskGraph


def cell(graph, step, code, summaries=None, error=None, grounds=()):
    """Run one cell through the recorder as the executor does, grounding `grounds` while it runs."""
    graph.current_step = lambda: step
    graph.before(code)
    for name in grounds:
        graph.grounded(name, "dataset", [f"{name}.csv"], 2)
    graph.after(code, error, summaries or {}, False)


def grounded(*names):
    graph = TaskGraph()
    targets = ", ".join(f"v{i}" for i in range(len(names)))
    cell(graph, 1, f"{targets} = ground({', '.join(map(repr, names))})", grounds=names)
    return graph


def test_before_names_every_variable_a_cell_may_assign():
    assert TaskGraph().before("x = 1\nfor i in range(3):\n    y = i\ndf['a'] = 2\ndef f():\n    z = 1") == [
        "df", "i", "x", "y"]


def test_ground_assignment_makes_grounded_nodes_only():
    data = grounded("burned", "states").to_json()
    assert data["nodes"] == [{"id": "burned", "kind": "dataset", "step": 1, "files": ["burned.csv"], "rows": 2},
                             {"id": "states", "kind": "dataset", "step": 1, "files": ["states.csv"], "rows": 2}]
    assert data["edges"] == [] and data["answer"] is None


def test_an_output_takes_only_the_inputs_of_its_own_statement():
    graph = grounded("burned", "states")
    code = "x = v0.acres_2020.sum()\nprint(v1.head())\ny = len(v1)"
    cell(graph, 2, code, {"x": {"type": "float", "value": 30.0}, "y": {"type": "int", "value": 2}})
    nodes = {node["id"]: node for node in graph.to_json()["nodes"]}
    assert nodes["x@2"] == {"id": "x@2", "kind": "derived", "variable": "x", "step": 2, "from": ["burned"],
                            "summary": {"type": "float", "value": 30.0}}
    assert graph.to_json()["edges"] == [{"step": 2, "code": code, "derived": {"x@2": ["burned"], "y@2": ["states"]}}]


def test_item_attribute_and_inplace_writes_derive_the_variable():
    graph = grounded("burned")
    cell(graph, 2, 'v0["z"] = 1\nv0.dropna(inplace=True)')
    assert graph.to_json()["edges"][0]["derived"] == {"v0@2": ["burned"]}


def test_a_function_defined_earlier_carries_what_its_body_reads():
    graph = grounded("burned")
    cell(graph, 2, "def total():\n    return v0.acres_2020.sum()")
    cell(graph, 3, "t = total()")
    assert graph.to_json()["edges"] == [{"step": 3, "code": "t = total()", "derived": {"t@3": ["burned"]}}]


def test_a_variable_assigned_without_inputs_stops_holding_node_data():
    graph = grounded("burned")
    cell(graph, 2, "v0 = 5\nb = v0 + 1")
    assert graph.to_json()["edges"] == []


def test_names_built_in_code_take_every_name_grounded_during_the_cell():
    graph = TaskGraph()
    cell(graph, 1, 'tables = [ground(n) for n in ["burned", "states"]]', grounds=["burned", "states"])
    assert graph.to_json()["edges"][0]["derived"] == {"tables@1": ["burned", "states"]}


def test_a_failed_cell_keeps_the_statements_before_the_failing_one():
    graph = grounded("burned")
    cell(graph, 2, "p = len(v0)\nq = 1 / 0\nr = len(v0)",
         error="InterpreterError: Code execution failed at line 'q = 1 / 0' due to: ZeroDivisionError: division by zero")
    assert graph.to_json()["edges"][0]["derived"] == {"p@2": ["burned"]}


def test_a_restarted_worker_forgets_every_variable():
    graph = grounded("burned")
    graph.current_step = lambda: 2
    graph.before("import time")
    graph.after("import time", "TimeoutError: Code execution exceeded 30 seconds", {}, True)
    cell(graph, 3, "b = v0")
    assert graph.to_json()["edges"] == []


def test_the_answer_records_its_lineage_or_that_it_was_typed():
    graph = grounded("burned")
    cell(graph, 2, "x = v0.acres_2020.sum()")
    cell(graph, 3, "if x > 0:\n    final_answer(round(x, 2))")
    assert graph.to_json()["answer"] == {"typed": False, "from": ["burned"]}
    cell(graph, 4, "final_answer(0.632)")
    assert graph.to_json()["answer"] == {"typed": True, "from": []}


def test_a_built_in_name_is_not_a_variable_the_answer_reads():
    graph = grounded("burned")
    cell(graph, 2, "final_answer(str(round(1990.4)))")
    assert graph.to_json()["answer"] == {"typed": True, "from": []}
    cell(graph, 3, "len = len(v0)\nfinal_answer(len)")
    assert graph.to_json()["answer"] == {"typed": False, "from": ["burned"]}


def test_several_answers_in_one_statement_are_typed_only_if_all_are():
    graph = grounded("burned", "states")
    cell(graph, 2, "x = len(v0)\ny = len(v1)")
    cell(graph, 3, "if x > y:\n    final_answer(x)\nelse:\n    final_answer(y)")
    assert graph.to_json()["answer"] == {"typed": False, "from": ["burned", "states"]}
    cell(graph, 4, "if x > y:\n    final_answer(x)\nelse:\n    final_answer('none')")
    assert graph.to_json()["answer"] == {"typed": False, "from": ["burned"]}
    cell(graph, 5, "if x > y:\n    final_answer(1)\nelse:\n    final_answer('none')")
    assert graph.to_json()["answer"] == {"typed": True, "from": []}


def test_an_augmented_assignment_keeps_the_lineage():
    graph = grounded("burned")
    cell(graph, 2, "x = v0.acres_2020.sum()")
    cell(graph, 3, "x += 1")
    cell(graph, 4, "final_answer(x)")
    assert graph.to_json()["edges"][1] == {"step": 3, "code": "x += 1", "derived": {"x@3": ["x@2"]}}
    assert graph.to_json()["answer"] == {"typed": False, "from": ["burned"]}


def test_variables_written_in_turn_within_one_step_never_form_a_cycle():
    # astronomy-easy-6: d feeds n, then n feeds d again, in one cell; the answer walk used to loop forever.
    graph = grounded("omm")
    cell(graph, 2, "d = v0[v0.id == 1].copy()\nn = d['mean_motion'] * 2\nd['a'] = n ** 2")
    assert graph.to_json()["edges"][0]["derived"] == {"d@2": ["omm"], "n@2": ["omm"]}
    cell(graph, 3, "rate = d['a'].mean()")
    cell(graph, 4, "final_answer(rate)")
    assert graph.to_json()["answer"] == {"typed": False, "from": ["omm"]}


def test_the_answer_walk_visits_each_node_once():
    # Each step derives both variables from both earlier ones, so a walk that revisits nodes takes 2**60 paths.
    graph = grounded("burned", "states")
    cell(graph, 2, "a = len(v0)\nb = len(v1)")
    for step in range(3, 63):
        cell(graph, step, "a, b = a + b, a - b")
    cell(graph, 63, "final_answer(a)")
    assert graph.to_json()["answer"] == {"typed": False, "from": ["burned", "states"]}


def test_a_method_call_whose_result_is_dropped_changes_its_receiver():
    # environment-hard-8: rows.append(...) in a loop, then pd.concat(rows), used to lose every source.
    graph = grounded("burned")
    code = ("import pandas as pd\nrows = []\nfor kind in ['a', 'b']:\n    rows.append(v0[v0.kind == kind])\n"
            "out = pd.concat(rows)\npd.set_option('display.width', 120)\ntotal = out.acres.sum()")
    assert graph.before(code) == ["kind", "out", "rows", "total"]
    cell(graph, 2, code)
    assert graph.to_json()["edges"][0]["derived"] == {
        "kind@2": ["burned"], "rows@2": ["burned"], "out@2": ["burned"], "total@2": ["burned"]}
    cell(graph, 3, "final_answer(total)")
    assert graph.to_json()["answer"] == {"typed": False, "from": ["burned"]}


def test_an_imported_name_is_not_a_variable_the_answer_reads():
    graph = grounded("burned")
    cell(graph, 2, "import json\nfrom math import sqrt")
    cell(graph, 3, 'final_answer(json.dumps("Ashburnham"))')
    assert graph.to_json()["answer"] == {"typed": True, "from": []}
    cell(graph, 4, "final_answer(sqrt(2))")
    assert graph.to_json()["answer"] == {"typed": True, "from": []}


def test_a_failing_compound_statement_keeps_the_statements_before_it():
    graph = grounded("burned")
    code = "total = sum(v0.acres_2020)\nfor value in v0.acres_2020:\n    print(value / 0)"
    failing = "for value in v0.acres_2020:\n    print(value / 0)"
    cell(graph, 2, code, error=f"InterpreterError: Code execution failed at line '{failing}' due to: "
                               "ZeroDivisionError: float division by zero")
    cell(graph, 3, "final_answer(total)")
    assert graph.to_json()["answer"] == {"typed": False, "from": ["burned"]}
