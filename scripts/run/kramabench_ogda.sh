#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG="${LLM_CONFIG:-llm_config/Kramabench/config.yaml}"
EVALUATOR_CONFIG="${EVALUATOR_CONFIG:-llm_config/Kramabench/evaluator.yaml}"
RUN_ID="${RUN_ID:-ogda-$(date -u +%Y%m%dT%H%M%S%NZ)}"
NUM_WORKERS="${NUM_WORKERS:-${WORKERS:-16}}"
LAKE_WORKERS="${LAKE_WORKERS:-16}"
TASK_TIMEOUT="${TASK_TIMEOUT:-1800}"
EVALUATION_TIMEOUT="${EVALUATION_TIMEOUT:-1800}"
# All 104 main tasks across six domains; model and method budgets stay in YAML.

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKSPACE"
exec "$PYTHON" -B "$WORKSPACE/scripts/run_benchmarks.py" \
  --benchmark Kramabench --config "$CONFIG" --evaluator-config "$EVALUATOR_CONFIG" --run-id "$RUN_ID" \
  --workers "$NUM_WORKERS" --lake-workers "$LAKE_WORKERS" \
  --task-timeout "$TASK_TIMEOUT" --evaluation-timeout "$EVALUATION_TIMEOUT" "$@"
