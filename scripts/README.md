# Ontology construction and benchmark evaluation

The launchers reuse OGDA's existing benchmark suite and native KramaBench and DiscoveryBench evaluators. Run the following setup from the repository root, with [uv](https://docs.astral.sh/uv/) and the operating-system prerequisites in the [main README](../README.md#installation).

## Setup

First obtain the external benchmark checkouts and datasets using [benchmark setup](../benchmark/README.md).

```bash
uv sync --locked --extra benchmarks
uv run python scripts/prepare_datalakes.py --benchmark all
```

The preparer copies KramaBench's six input trees and DiscoveryBench's raw files and sanitized documentation to `datalake/`. It verifies content hashes and refuses existing benchmark directories. Use `--benchmark Kramabench` or `--benchmark discoverybench` to prepare only one benchmark. `--output-root /path/to/new/datalake` selects another destination for preparation; the suite reads `datalake/` under its workspace root.

Copy and fill in the [model configuration templates](../llm_config/README.md). Generation and evaluator settings use separate YAML files. Generation reasoning must be explicitly disabled using your provider's supported setting.

## Run a complete pipeline

First check the models, configuration, data hashes, sanitized documentation, and task inventory without making model calls:

```bash
RUN_ID=ogda-krama-001 uv run bash scripts/run/kramabench_ogda.sh --dry-run
RUN_ID=ogda-discovery-001 uv run bash scripts/run/discoverybench_ogda.sh --dry-run
```

Run either benchmark's construction, task answering, and scoring:

```bash
RUN_ID=ogda-krama-001 uv run bash scripts/run/kramabench_ogda.sh
RUN_ID=ogda-discovery-001 uv run bash scripts/run/discoverybench_ogda.sh
```

These commands make model calls. KramaBench runs 104 main tasks across six domain lakes; DiscoveryBench runs the 239 real test questions. Each preset defaults to 16 task/evaluation workers, 16 construction workers, and 1,800-second task and evaluation deadlines. Construction has no overall wall-clock deadline unless `--construction-timeout` is supplied.

## Run stages separately

Set a fixed run ID and keep both YAML files unchanged across stages:

```bash
export RUN_ID=ogda-krama-002
uv run bash scripts/run/kramabench_ogda.sh --stage construction
uv run bash scripts/run/kramabench_ogda.sh --stage tasks --resume
uv run bash scripts/run/kramabench_ogda.sh --stage evaluation --resume
```

Use `discoverybench_ogda.sh` for DiscoveryBench. Construction must finish successfully before the suite freezes the ontologies and allows task execution. Evaluation reads the saved predictions and uses the original upstream scorer. `--resume` skips compatible completed work and continues unfinished jobs; reuse the original run ID. A new run ID starts a new experiment.

## Direct Python launcher

The shared launcher exposes the same suite options:

```bash
uv run python scripts/run_benchmarks.py \
  --benchmark Kramabench \
  --config llm_config/Kramabench/config.yaml \
  --evaluator-config llm_config/Kramabench/evaluator.yaml \
  --run-id ogda-krama-003 \
  --stage pipeline \
  --workers 16 --lake-workers 16 \
  --task-timeout 1800 --evaluation-timeout 1800
```

`--benchmark` accepts `Kramabench`, `discoverybench`, or `all`. Selecting `all` runs the benchmarks sequentially with the supplied configuration. The module entry point `uv run python -m ogda.adaptor.suite` is equivalent. Run `uv run python scripts/run_benchmarks.py --help` for the full options.

Paths resolve from this repository by default, including when invoking a launcher from another working directory. `OGDA_WORKSPACE` can explicitly select another root containing `benchmark/`, `datalake/`, and output directories. `OGDA_SOURCE` optionally selects another core checkout for suite workers.

## Outputs

| Location | Contents |
| --- | --- |
| `results/<benchmark>/<run-id>/run_config.json` | Saved configuration and inventory digests |
| `results/<benchmark>/<run-id>/lakes/` | Ontologies, copied inputs, construction summaries, and logs |
| `results/<benchmark>/<run-id>/tasks/` | Answers, task graphs, and generation evidence |
| `results/<benchmark>/<run-id>/evaluation/` | Native scores, per-task results, and summary |
| `results/<benchmark>/<run-id>/logs/` | Worker console, model, and task logs |
| `exp_env/<benchmark>/<run-id>/` | Isolated agent workspaces |
| `exp_env/adaptor/` | Exported upstream runtime code |
| `exp_env/setup/<run-id>/dry-run-manifest.json` | Validated inputs and public model settings from a dry run |

See the [editable preset reference](run/README.md) for environment overrides.
