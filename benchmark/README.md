# Benchmarks

Benchmark sources and datasets are not bundled in this copy. The adaptors expect separate upstream checkouts in the directories below. The recorded reference revisions are listed for setup; this copy contains no benchmark checkout to verify against them.

| Directory | Upstream | Reference revision |
| --- | --- | --- |
| `Kramabench/` | [mitdbg/Kramabench](https://github.com/mitdbg/Kramabench) | `b2e0d77540263f8b6119f977fe79a8c4386b5a02` |
| `discoverybench/` | [allenai/discoverybench](https://github.com/allenai/discoverybench) | `c31fcf011e070f021a5f5b906896d0821f6880e8` |

From the OGDA repository root:

```bash
git clone https://github.com/mitdbg/Kramabench.git benchmark/Kramabench
git -C benchmark/Kramabench checkout b2e0d77540263f8b6119f977fe79a8c4386b5a02
git clone https://github.com/allenai/discoverybench.git benchmark/discoverybench
git -C benchmark/discoverybench checkout c31fcf011e070f021a5f5b906896d0821f6880e8
```

Obtain any datasets required by each upstream project, then prepare the lakes:

```bash
uv sync --locked --extra benchmarks
uv run python scripts/prepare_datalakes.py --benchmark all
```

Preparation creates independent copies under `datalake/`. KramaBench copies only each domain's `data/<domain>/input/` tree. DiscoveryBench copies raw data and builds `documentation/schema.json` from dataset and column names and descriptions. Questions, answers, hypotheses, workflows, and evaluation metadata remain outside these construction inputs. Preparation verifies the copied files against the sources and refuses to overwrite an existing destination.

The DiscoveryBench preparer includes all upstream splits; the evaluation launcher selects only `real/test`. The KramaBench launcher selects all 104 main tasks across six domains.

The runtime exporter requires Git metadata in each upstream checkout; deleting it prevents benchmark execution. Committed upstream code is exported into `exp_env/adaptor/` before execution. Data directories are linked there, and DiscoveryBench receives two package marker files to keep its imports isolated. Native evaluator code and prompts are unchanged. Integration code lives in [`ogda/adaptor/`](../ogda/adaptor/); the runtime exporter leaves the upstream checkouts unchanged.

See the [run instructions](../scripts/README.md) for construction and evaluation. Each upstream repository retains its own licensing and data terms.
