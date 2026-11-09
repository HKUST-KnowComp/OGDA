# Editable OGDA presets

Each preset runs one benchmark through [`../run_benchmarks.py`](../run_benchmarks.py). Follow the [setup instructions](../README.md) first.

| Preset | Tasks |
| --- | --- |
| `kramabench_ogda.sh` | All 104 main KramaBench tasks across six domains |
| `discoverybench_ogda.sh` | All 239 DiscoveryBench real test questions |

Launch with Bash; executable permissions and the caller's current directory do not matter:

```bash
RUN_ID=ogda-krama-001 uv run bash scripts/run/kramabench_ogda.sh --dry-run
RUN_ID=ogda-discovery-001 uv run bash scripts/run/discoverybench_ogda.sh --stage construction
```

Settings at the top of each script can also be supplied as environment variables:

| Variable | Default |
| --- | --- |
| `PYTHON` | `python` |
| `LLM_CONFIG` | `llm_config/<benchmark>/config.yaml` |
| `EVALUATOR_CONFIG` | `llm_config/<benchmark>/evaluator.yaml` |
| `RUN_ID` | A new timestamped OGDA run ID |
| `NUM_WORKERS` | `16` (also accepts `WORKERS` as a fallback) |
| `LAKE_WORKERS` | `16` |
| `TASK_TIMEOUT` | `1800` seconds |
| `EVALUATION_TIMEOUT` | `1800` seconds |

Model, credential, endpoint, reasoning, and method settings belong in YAML. Additional arguments are forwarded to the shared launcher, including `--stage construction|tasks|evaluation|pipeline`, `--dry-run`, and `--resume`.

Use a fixed `RUN_ID` when running stages separately or resuming. Keep its configuration files unchanged. Use a new run ID for a new experiment or different model settings.
