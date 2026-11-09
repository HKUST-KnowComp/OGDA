# Usage

Run commands from the repository root after creating the environment with `uv sync --locked`; run each command through `uv run`. The [quick start](../README.md#quick-start) shows the three commands.

## Configuration

Copy [`config.example.yaml`](../config.example.yaml) to `config.yaml`. A single YAML file configures the model, construction, and data agent.

| Setting | Purpose |
| --- | --- |
| `llm.model` | Provider/model identifier |
| `llm.base_url` | Provider API root; `api_base` is also accepted |
| `llm.api_key_env` | Name of the environment variable containing the credential |
| `llm.api_key` | Alternative explicit credential; prefer `llm.api_key_env` to keep credentials out of configuration files |
| `llm.api` | `chat_completions` (default, through LiteLLM) or `responses` |
| `llm.max_output_tokens` | Per-response output limit; default 3,000 |
| `llm.timeout_seconds` | Per-request-attempt timeout; default 120 seconds |
| `llm.model_kwargs` | Provider-specific request options |
| `llm.reasoning`, `llm.reasoning_effort` | Provider-supported reasoning settings |
| `llm.max_total_tokens`, `llm.max_calls`, `llm.max_cost_usd` | Optional client-wide usage caps |

`llm.roles.proposer`, `llm.roles.verifier`, `llm.roles.experience`, and `llm.roles.data_agent` override shared settings. The experience role inherits proposer overrides first. For older configurations, the data-agent role inherits `llm.roles.analyst` before applying its own overrides. Environment credentials are read only when `api_key_env` names them.

The Responses transport sends requests to `<base_url>/responses`; keep the configured URL at the API root. It uses `max_output_tokens`, including reasoning tokens, and honors HTTP proxy environment variables. Choose reasoning settings supported by your provider explicitly when reproducing an experiment.

### Construction and execution limits

| Setting | Default | Meaning |
| --- | --- | --- |
| `construction.max_passes` | `3` | Schema-enrichment passes after file mapping |
| `construction.max_tokens` | `2000000` | Construction token budget |
| `construction.files_per_visit` | `8` | File-mapping batch size |
| `construction.max_claims_per_visit` | `20` | Claims in one proposer visit |
| `construction.agent.max_steps` | `20` | Agent steps per construction visit |
| `data_agent.agent.max_steps` | `40` | Steps for answering one task |
| `data_agent.max_tokens` | `null` | Optional total token cap for one task |

Both `construction.agent` and `data_agent.agent` accept `cell_timeout_seconds` (default `30`), `memory_mb` (`4096`), and `max_output_chars` (`20000`). These settings control code execution and observations. A per-cell timeout is not a total task deadline.

### Model profiles

[`ogda/models/`](../ogda/models/) contains model-specific CodeAgent prompts and options. A profile matches `model_names` without provider prefixes or case sensitivity. Its `code_agent` section can set `system_prompt`, two `code_block_tags`, and `tool_choice`:

- `required`: require the native `code(thought, code)` function.
- `auto`: allow a native code call or a text code block; this is the fallback for unmatched models.
- `none`: omit API tools and accept text code blocks.

An optional `data_agent.instructions` string adds instructions only for the matching data-agent model. These YAML files are packaged with the module.

## Prepare a lake

Place only data and supporting documentation in the lake. Keep task questions, reference answers, and evaluation metadata outside it. For mixed source directories, `build` can exclude paths and strip JSON/YAML keys:

```bash
uv run ogda build \
  --config config.yaml --lake /path/to/lake \
  --exclude 'answers/*' --strip-key answer \
  --output results/my-lake/run-001 \
  --scratch exp_env/my-lake/run-001
```

Choose exclusion patterns and keys for your own data; the example is not a complete filter for every benchmark. Construction copies the lake to `<output>/ontology/lake`, applies the exclusions, and leaves the original unchanged. Agents can edit files in the copy during construction.

Use a fresh output directory for a new run. To continue interrupted or budget-stopped construction, repeat its command with `--resume`. Submitted claims and completed verifier judgments persist. Increase `--max-passes` or `--max-tokens` when continuing a run stopped at the corresponding limit.

Inspect `construction-summary.json` and the saved progress before using an ontology. A stopped run can still have unresolved mappings or pending verification; the CLI does not make every stopped run complete.

## Search and answer

Search uses the saved ontology and makes no model calls:

```bash
uv run ogda search \
  --ontology results/my-lake/run-001/ontology \
  --limit 10 --hops 2 'household income'
```

A task is a JSON object containing `id`, `question`, and optionally `answer_format`:

```json
{
  "id": "income-001",
  "question": "What is the median household income by region?",
  "answer_format": "A JSON object mapping each region to its median income."
}
```

Save it outside the lake as `task.json`, then run:

```bash
uv run ogda ask \
  --config config.yaml \
  --ontology results/my-lake/run-001/ontology \
  --task task.json \
  --output results/my-lake/run-001/tasks/income-001 \
  --scratch exp_env/my-lake/run-001/tasks/income-001
```

The data agent uses `search_ontology`, `get_schema`, and `ground` to find and ground evidence. It reads the lake copy without changing it and writes artifacts under the task scratch directory's `workspace/`. An unmapped file can be grounded by path and is recorded as outside the ontology.

One code step can search, ground files, and compute over them. New `ground` context is automatically included in the next model observation even if Python ignores the return value, prints too much output, or fails later in the step. Shared schema definitions and their field descriptions are shown once; each grounded table identifies its `schema_id`. The agent must review new context before finalizing: a `final_answer` in the same step is rejected, while its completed computations remain available for the next step. Repeated grounding that adds no context does not delay finalization. If the step budget ends before context can be reviewed and an answer accepted, the task remains incomplete.

The agent submits `final_answer({"answer": ...})`; an optional `explanation` is saved alongside it. In Python integrations, `answer(..., check=...)` accepts an answer-shape callback that raises `ValueError` for an invalid shape, allowing the agent to repair it within its remaining steps. Shape checks do not establish answer correctness.

## Saved outputs

| Path relative to a construction output directory | Contents |
| --- | --- |
| `ontology/ontology.json` | Schemas, tables, concepts, links, claims, experience, and progress; format version 5 |
| `ontology/lake/` | The copied lake |
| `construction-summary.json` | Stop reason, counts, claim outcomes, and usage |
| `logs/construction.jsonl` | Construction stages, file dispositions, and claim outcomes |
| `logs/proposer/`, `logs/verifier/` | Agent visit transcripts |
| `logs/llm.jsonl` | Model calls and usage |

Each task output directory contains `result.json`, `logs/data_agent.jsonl`, agent transcripts under `logs/data_agent/`, and `logs/llm.jsonl`. The result records the answer, explanation, status, usage, time, and task graph, including grounded objects, reads, and files outside the ontology.

Keep generated results in `results/` and scratch files in `exp_env/`. To obtain and prepare the external benchmarks and run ontology construction, task generation, and native scoring, follow the [benchmark scripts](../scripts/README.md).
