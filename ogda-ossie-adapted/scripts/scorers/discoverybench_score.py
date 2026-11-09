"""Score one DiscoveryBench hypothesis with the official eval and your judge model.

Run by `benchmarks.run_scorer` with cwd and PYTHONPATH set to `../benchmark/discoverybench`, so `eval` and `utils`
are its own packages. Reads one job `{"query", "gold_hypo", "gold_workflow", "gen_hypo", "gen_workflow",
"dataset_meta", "dataset_type"}` from stdin and prints the eval record, with `final_score`, as one JSON line; the
eval's own progress output goes to stderr. Every judge call uses `JUDGE_MODEL`; the eval's `OpenAI()` clients read
`OPENAI_API_KEY` and `OPENAI_BASE_URL`.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import types


def main() -> None:
    # The eval imports IPython.embed but never calls it; IPython is not installed.
    ipython = types.ModuleType("IPython")
    ipython.embed = lambda *args, **kwargs: None
    sys.modules["IPython"] = ipython
    job = json.loads(sys.stdin.read())
    with contextlib.redirect_stdout(sys.stderr):
        from eval.new_eval import run_eval_gold_vs_gen_NL_hypo_workflow

        record = run_eval_gold_vs_gen_NL_hypo_workflow(
            query=job["query"], gold_hypo=job["gold_hypo"], gold_workflow=job["gold_workflow"],
            gen_hypo=job["gen_hypo"], gen_workflow=job["gen_workflow"], dataset_meta=job["dataset_meta"],
            llm_used=os.environ["JUDGE_MODEL"], dataset_type=job["dataset_type"], use_column_metadata=True)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
