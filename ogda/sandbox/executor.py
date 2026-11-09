"""A smolagents executor that runs agent code in an isolated worker process.

The worker can read the visit artifacts and write a private scratch directory. By default it
can also write the Lake; `lake="read"` makes the Lake read-only and `lake="none"` hides it.
A `workspace` is a writable directory that outlives the worker, for example the files that the
Steps of one Task build toward its answer; agent code sees it as WORKSPACE. It has no network. Tools run in this process, and their time does not
count toward the cell timeout.

A `gate` returns the Lake Files that agent code may open; the worker refuses other Lake Files and asks for them
again after each tool call. `on_files` receives the worker's log of Lake reads and Workspace reads and writes after
each cell. Both see Python file opens only; C libraries that open files themselves bypass them.
"""

from __future__ import annotations

import json
import mimetypes
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from smolagents.local_python_executor import CodeOutput, PythonExecutor

IMPORTS = ["pandas", "numpy", "numpy.*", "scipy", "scipy.*", "statsmodels", "statsmodels.*", "pathlib", "os", "posixpath", "glob", "shutil", "io", "json", "csv", "math", "statistics", "datetime",
           "re", "collections", "itertools", "sqlite3", "duckdb", "time", "openpyxl", "yaml",
           "struct", "zlib", "gzip", "bz2", "lzma", "zipfile", "tarfile", "tempfile"]
# Every worker runs this first, so code that forgets an import of these modules still runs.
PRELUDE = "import json, math, re, csv, datetime, collections, itertools, statistics\nimport numpy as np\nimport pandas as pd"


class SandboxUnavailable(RuntimeError):
    pass


class IsolatedExecutor(PythonExecutor):
    def __init__(self, lake_root, scratch_root, artifact_root, *, timeout=30, memory_mb=4096, max_output_chars=20000,
                 lake="write", workspace=None, long_tools=False, gate=None, on_files=None):
        """`long_tools`: a tool call can outlast the cell timeout, so only the host deadline, which
        excludes tool time, bounds a cell; the worker's own thread timeout would count tool waits."""
        if lake not in ("write", "read", "none"):
            raise ValueError(f"Unknown Lake access mode {lake!r}")
        self.lake_root, self.scratch_root, self.artifact_root = (
            Path(p).resolve() for p in (lake_root, scratch_root, artifact_root))
        self.workspace = None if workspace is None else Path(workspace).resolve()
        for output in (self.scratch_root, self.artifact_root, *([self.workspace] if self.workspace else [])):
            if output == self.lake_root or output.is_relative_to(self.lake_root) or self.lake_root.is_relative_to(output):
                raise ValueError("The Lake, scratch, artifact and workspace directories must be separate trees")
        self.lake, self.long_tools, self.gate, self.on_files = lake, long_tools, gate, on_files
        self.timeout, self.memory_mb, self.max_output_chars = timeout, memory_mb, max_output_chars
        self.tools, self.variables = {}, {}
        self.process = None
        self._directory = None
        self._buffer = b""

    def send_tools(self, tools):
        self.tools = {name: tool for name, tool in tools.items() if name != "final_answer"}

    def send_variables(self, variables):
        self.variables.update(variables)

    def _start(self):
        for directory in (self.artifact_root, self.scratch_root, *([self.workspace] if self.workspace else [])):
            directory.mkdir(parents=True, exist_ok=True)
        self._directory = tempfile.TemporaryDirectory(prefix="worker-", dir=self.scratch_root)
        work = Path(self._directory.name)
        interpreter = Path(sys.executable).resolve()
        # mimetypes reads its known files on first use (openpyxl does), and fails if they exist but are denied.
        runtime = ["/usr", "/lib", "/lib64", "/bin", "/etc/ld.so.cache", "/etc/localtime", *mimetypes.knownfiles,
                   "/dev/null", "/dev/urandom", str(interpreter.parent.parent), sys.prefix, sys.base_prefix]
        lake = [str(self.lake_root)]
        config = {
            "read_roots": list(dict.fromkeys(runtime + [str(self.artifact_root)] + (lake if self.lake == "read" else []))),
            "write_roots": ([str(work)] + ([str(self.workspace)] if self.workspace else [])
                            + (lake if self.lake == "write" else [])),
            "lake_root": None if self.lake == "none" else str(self.lake_root),
            "workspace": None if self.workspace is None else str(self.workspace),
            "scratch_root": str(work), "artifact_root": str(self.artifact_root),
            "memory_mb": self.memory_mb, "cpu_seconds": self.timeout * 64,
            "timeout": None if self.long_tools else self.timeout,
            "max_output_chars": self.max_output_chars, "imports": IMPORTS, "prelude": PRELUDE,
            "tools": {name: list(tool.inputs) for name, tool in self.tools.items()},
            "gate": None if self.gate is None else sorted(self.gate()),
            "log_files": self.on_files is not None,
        }
        (work / "worker.json").write_text(json.dumps(config))
        environment = {"PATH": f"{interpreter.parent}:/usr/bin:/bin", "HOME": str(work), "TMPDIR": str(work),
                       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LANG": "C.UTF-8"}
        log = (self.artifact_root / "worker.log").open("ab")
        self.process = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).with_name("_worker.py")), str(work / "worker.json")],
            cwd=work, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
            start_new_session=True, close_fds=True)
        log.close()
        self._selector = selectors.DefaultSelector()
        self._selector.register(self.process.stdout, selectors.EVENT_READ)
        if self._receive(time.monotonic() + 30).get("event") != "ready":
            raise SandboxUnavailable("Worker isolation setup failed; see worker.log")
        if self.variables:
            self._send({"event": "variables", "variables": self.variables})

    def _send(self, value):
        self.process.stdin.write((json.dumps(value, default=str) + "\n").encode())
        self.process.stdin.flush()

    def _receive(self, deadline):
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._selector.select(remaining):
                raise TimeoutError(f"Code execution exceeded {self.timeout} seconds")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                raise RuntimeError("The code worker exited; see worker.log")
            self._buffer += block
            if len(self._buffer) > 16 * 1024 * 1024:
                raise RuntimeError("Worker output exceeded its size limit")
        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line)

    def __call__(self, code_action):
        if self.process is None:
            self._start()
        self._send({"event": "execute", "code": code_action})
        deadline = time.monotonic() + self.timeout
        try:
            while (message := self._receive(deadline))["event"] == "tool":
                started = time.monotonic()
                self._send(self._run_tool(message["name"], message["arguments"]))
                deadline += time.monotonic() - started  # Tool time does not count toward the cell limit.
        except Exception:
            # A stuck or dead worker is unusable. The next cell starts a new worker.
            self.cleanup()
            raise
        if self.on_files is not None and message.get("files"):
            self.on_files(message["files"])
        if not message["success"]:
            raise RuntimeError("\n".join(part for part in (message["stdout"], message["stderr"]) if part))
        return CodeOutput(output=message["output"], logs=message["stdout"], is_final_answer=message["is_final_answer"])

    def _run_tool(self, name, arguments):
        try:
            response = {"output": json.loads(json.dumps(self.tools[name](**arguments), default=str))}
        except Exception as error:
            # Host exception messages can contain private paths, so send the type only.
            response = {"error": type(error).__name__}
        if self.gate is not None:
            response["gate"] = sorted(self.gate())
        return response

    def cleanup(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self._selector.close()
            self.process, self._buffer = None, b""
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
