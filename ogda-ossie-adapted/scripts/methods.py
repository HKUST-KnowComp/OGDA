"""Answer one Task with OGDA or with a baseline CodeAgent, and rebuild the pipeline it ran.

OGDA grounds a finished Harness through `ground` and `search` and records the Task graph as `graph`. The baseline has
no Ontology and no tools: it runs Python over the whole read-only Lake, in the same isolated sandbox. Both save
`result.json` with the same keys apart from OGDA's `graph`, so later code (scoring, pipeline extraction) treats them
alike.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from ogda.agent import Budget, run_agent
from ogda.data_agent import DataAgentConfig, answer, check_answer, check_task
from ogda.llm import BudgetExceeded, LLMError
from ogda.ontology import OntologyStore

ROLE = "data_agent"
WORKSPACE = "workspace"
WORKERS = "workers"

BASELINE = ('Answer the question by running Python on the Files under LAKE_ROOT, which you can read but not change. '
           'Write any files you make to WORKSPACE. Finish by calling final_answer(answer), where answer is the '
           'answer itself in the requested format.')


def run_ogda(task: dict, ontology: Path, llm, config: DataAgentConfig, scratch: Path, output: Path,
             check: Callable[[Any], None] | None = None) -> dict:
    """Answer one Task with OGDA against the Harness at `ontology`."""
    return answer(OntologyStore(ontology), llm, task, config, scratch, output, check)


def run_baseline(task: dict, lake: Path, llm, config: DataAgentConfig, scratch: Path, output: Path,
                 check: Callable[[Any], None] | None = None) -> dict:
    """Answer one Task with a plain CodeAgent: no Ontology, no tools, a read-only Lake, the same sandbox.

    Saves `output/result.json` with the same keys as the Data agent's result and returns it.
    """
    task = check_task(task)
    scratch, output = Path(scratch).resolve(), Path(output).resolve()
    workspace = scratch / WORKSPACE
    agent_task = {"task": {key: task[key] for key in ("question", "answer_format") if key in task},
                  "max_steps": config.agent.max_steps, "workspace": str(workspace)}
    budget = Budget(config.max_tokens, start=llm.usage["total_tokens"]) if config.max_tokens else None
    started, usage_before = time.monotonic(), dict(llm.usage)
    final, steps, stop_reason = None, None, "answered"
    try:
        visit = run_agent(
            llm, role=ROLE, instructions=BASELINE, task=json.dumps(agent_task, ensure_ascii=False),
            validate=lambda value: check_answer(value, check), tools=[], lake_root=lake, scratch_root=scratch / WORKERS,
            artifact_root=output / "logs" / "baseline", config=config.agent, budget=budget, lake="read",
            workspace=workspace)
        steps, final = visit.steps, visit.answer
        if final is None:
            stop_reason = "incomplete"
    except LLMError as error:  # Save what the Task has so far.
        stop_reason = "budget" if isinstance(error, BudgetExceeded) else "llm_error"
    result = {
        "task": task, "status": "incomplete" if final is None else "answered", "stop_reason": stop_reason,
        "answer": final, "steps": steps,
        "usage": {key: llm.usage[key] - usage_before.get(key, 0) for key in llm.usage},
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    temporary.replace(output / "result.json")
    return result


def pipeline_code(logs: Path) -> str:
    """Rebuild the Python the agent ran from every `steps.json` under `logs`, in file and step order."""
    actions = []
    for path in sorted(Path(logs).rglob("steps.json")):
        for step in json.loads(path.read_text()):
            if isinstance(step, dict) and step.get("code_action"):
                actions.append(step["code_action"])
    return "\n\n".join(actions)
