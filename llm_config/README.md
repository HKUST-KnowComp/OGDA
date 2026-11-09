# Benchmark model configuration

Each benchmark has a generation template (`config.example.yaml`) and a separate native evaluator template (`evaluator.example.yaml`). From the repository root:

```bash
cp llm_config/Kramabench/config.example.yaml llm_config/Kramabench/config.yaml
cp llm_config/Kramabench/evaluator.example.yaml llm_config/Kramabench/evaluator.yaml
cp llm_config/discoverybench/config.example.yaml llm_config/discoverybench/config.yaml
cp llm_config/discoverybench/evaluator.example.yaml llm_config/discoverybench/evaluator.yaml
```

Replace the placeholder models and endpoints, and set the environment variables named by `api_key_env`. The examples contain no credentials and do not specify the paper's model choices.

Generation uses the core [`config.example.yaml`](../config.example.yaml) format: `llm`, `construction`, and `data_agent`. The suite requires default data-agent method settings and explicitly disabled reasoning for the proposer, verifier, experience, and data-agent roles. Adjust the off setting to match your provider; the examples use `reasoning_effort: "none"`. Provider-specific request options belong in `model_kwargs`.

The evaluator format has top-level `provider`, `model`, `base_url`, and `api_key_env`, with overrides under `roles.evaluator`. It requires an OpenAI-compatible endpoint. Evaluator reasoning and token limits are configured independently of generation, using `reasoning_effort` and `model_kwargs` as needed. The native scorer's prompts, response parsing, and metrics remain unchanged.

Both configuration paths are required for every suite stage so saved runs can check configuration consistency when resumed. Use the same files throughout a staged run; use a new run ID after changing model settings. See the [script instructions](../scripts/README.md).
