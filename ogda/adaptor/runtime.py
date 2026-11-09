"""Export committed benchmark code into an isolated runtime outside the upstream checkouts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from .artifacts import WORKSPACE

DATA_DIRECTORIES = {
    "Kramabench": ("data", "dr-input"),
    "discoverybench": ("discoverybench",),
}
NATIVE_KRAMABENCH_EVALUATION_FILES = (
    "evaluate.py", "benchmark/evaluator.py", "benchmark/metrics.py",
    "benchmark/llm_tools/__init__.py", "benchmark/llm_tools/gpt_interface.py",
    "benchmark/llm_tools/llm_interface.py", "benchmark/llm_tools/prompts.py",
)
NATIVE_DISCOVERYBENCH_EVALUATION_FILES = (
    "eval/new_eval.py", "eval/lm_utils.py", "utils/openai_helpers.py",
    "utils/openai_semantic_gen_prompts.py", "utils/helpers.py", "utils/arguments.py",
)


def git(repository: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repository), *args])


def repository_commit(source: Path) -> str:
    if not (source / ".git").exists():
        raise FileNotFoundError(
            f"Benchmark Git checkout is missing: {source}. "
            "Obtain the upstream checkout and datasets as described in benchmark/README.md."
        )
    return git(source, "rev-parse", "HEAD").decode().strip()


def export_repository(source: Path, target: Path, commit: str, assets: tuple[str, ...]) -> None:
    """Export committed code, linking only upstream dataset directories."""
    target.mkdir(parents=True)
    entries = git(source, "ls-tree", "--name-only", "-z", commit).decode().split("\0")
    entries = [name for name in entries if name and name not in assets]
    if entries:
        with tempfile.TemporaryFile() as archive:
            exclusions = [f":(exclude){name}" for name in assets]
            subprocess.run(["git", "-C", str(source), "archive", commit, "--", *entries, *exclusions],
                           stdout=archive, check=True)
            archive.seek(0)
            with tarfile.open(fileobj=archive) as stream:
                stream.extractall(target, filter="data")
    for name in assets:
        dataset = source / name
        if dataset.is_dir():
            output = target / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.symlink_to(dataset, target_is_directory=True)


def prepare(workspace: Path = WORKSPACE) -> Path:
    """Cache native sources by repository location, commits and runtime implementation."""
    workspace = Path(workspace).resolve()
    commits = {name: repository_commit(workspace / "benchmark" / name) for name in DATA_DIRECTORIES}
    digest = hashlib.sha256(str(workspace).encode() + Path(__file__).read_bytes())
    digest.update(json.dumps(commits, sort_keys=True).encode())
    cache = workspace / "exp_env/adaptor"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / digest.hexdigest()[:20]
    with (cache / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (target / ".ready.json").is_file():
            return target
        temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=cache))
        try:
            for name, commit in commits.items():
                export_repository(workspace / "benchmark" / name, temporary / "benchmark" / name,
                                  commit, DATA_DIRECTORIES[name])
            # Package markers prevent collisions with unrelated installed packages.
            # All native evaluator modules and prompts remain byte-identical to upstream.
            for name in ("eval", "utils"):
                (temporary / "benchmark/discoverybench" / name / "__init__.py").write_text(
                    f'"""DiscoveryBench {name}."""\n'
                )
            (temporary / ".ready.json").write_text(json.dumps({
                "upstream_commits": commits, "adapter_digest": digest.hexdigest(),
            }, indent=2) + "\n")
            os.rename(temporary, target)
        except BaseException:
            shutil.rmtree(temporary)
            raise
    return target


def benchmark_path(name: str) -> Path:
    if name not in DATA_DIRECTORIES:
        raise ValueError(f"No adapter runtime for {name}")
    return prepare() / "benchmark" / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("workspace", *DATA_DIRECTORIES), nargs="?", default="workspace")
    args = parser.parse_args()
    print(prepare() if args.target == "workspace" else benchmark_path(args.target))


if __name__ == "__main__":
    main()
