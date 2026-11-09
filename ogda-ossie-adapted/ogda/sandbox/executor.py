"""A smolagents executor that runs agent code in an isolated worker process.

The worker can read the visit artifacts and write a private scratch directory. By default it
can also write the Lake; `lake="read"` makes the Lake read-only, `lake="ground"` makes it read-only without
defining LAKE_ROOT, so code opens the paths that tools return, and `lake="none"` hides it. `readable` paths outside
the Lake are readable too.
A `workspace` is a writable directory that outlives the worker, for example the files that the
Steps of one Task build toward its answer; agent code sees it as WORKSPACE. It has no network.
On Linux the worker confines itself with Landlock and seccomp. On macOS it runs inside sandbox-runtime
(`srt`), which applies the same read and write roots and blocks the network; macOS cannot cap memory. Tools run in
this process, and a cell's time limit excludes their time: the host keeps the deadline and replaces a worker that
passes it. The worker has no timer of its own, since smolagents' timer cannot stop code and would count tool time.
A tool can print text among the cell's own prints and return tables and paths (see `_worker.tool_result`). A
`recorder` sees each cell and summaries of the variables it names.
"""

from __future__ import annotations

import json
import mimetypes
import os
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Protocol

from smolagents.local_python_executor import CodeOutput, PythonExecutor

IMPORTS = ["pandas", "numpy", "numpy.*", "scipy", "scipy.*", "statsmodels", "statsmodels.*", "pathlib", "os", "posixpath", "glob", "shutil", "io", "json", "csv", "math", "statistics", "datetime",
           "re", "collections", "itertools", "sqlite3", "duckdb", "time", "openpyxl", "yaml",
           "struct", "zlib", "gzip", "bz2", "lzma", "zipfile", "tarfile", "tempfile",
           # Geospatial Files (GeoPackage, shapefile) and their coordinate systems, CDF, TLE orbits, NRLMSISE-00.
           "geopandas", "geopandas.*", "shapely", "shapely.*", "pyogrio", "pyogrio.*", "pyproj", "pyproj.*",
           "cdflib", "cdflib.*", "sgp4", "sgp4.*", "skyfield", "skyfield.*", "pymsis", "pymsis.*"]
# Every worker runs this first, so code that forgets an import of these modules still runs.
PRELUDE = "import json, math, re, csv, datetime, collections, itertools, statistics\nimport numpy as np\nimport pandas as pd"


class SandboxUnavailable(RuntimeError):
    pass


class ToolCorrection(ValueError):
    """A tool error that tells the agent what to correct. The worker raises it as a ValueError with this message;
    other host exceptions reach the worker as their type only."""


class CellRecorder(Protocol):
    """Sees each cell that the executor runs."""

    def before(self, code: str) -> list[str]:
        """Called before the cell is sent: the variables for the worker to summarize once the cell ends."""

    def after(self, code: str, error: str | None, summaries: dict[str, dict], restarted: bool) -> None:
        """Called once the cell ends. `error` is None on success and, on a failed cell, the interpreter's whole error
        without the cell's other stderr (which the agent sees cut to 4,000 characters); when the worker died or timed
        out it is the exception text, `restarted` is True and the variables are gone.
        `summaries` describe the named variables that the cell left, with nothing after a restart."""


def _python_roots() -> list[str]:
    """The interpreter's install trees, including each symlink hop from the venv to the real binary."""
    roots, path = [sys.prefix, sys.base_prefix], Path(sys.executable)
    for _ in range(8):
        roots.append(str(path.parent.parent))
        if not path.is_symlink():
            break
        path = path.parent / os.readlink(path)
    return roots + [str(Path(sys.executable).resolve().parent.parent)]


def _srt_command(work: Path, read_roots: list[str], write_roots: list[str], worker: list[str]) -> tuple[list[str], str]:
    """Wrap the worker in sandbox-runtime: deny all reads but the roots, writes but the write roots, all network."""
    srt = shutil.which("srt")
    if srt is None:
        raise SandboxUnavailable("The macOS sandbox needs srt: npm install -g @anthropic-ai/sandbox-runtime")
    settings = {
        "filesystem": {"denyRead": ["/"], "allowRead": read_roots + write_roots,
                       "allowWrite": write_roots, "denyWrite": []},
        "network": {"allowedDomains": [], "deniedDomains": []},
    }
    (work / "srt-settings.json").write_text(json.dumps(settings))
    # srt is a Node script and finds ripgrep on PATH; the worker cannot read either.
    tools = [str(Path(found).parent) for name in ("node", "rg") if (found := shutil.which(name))]
    # srt keeps its proxy socket in TMPDIR, so srt gets a short one; the worker's TMPDIR is set inside.
    worker = ["/usr/bin/env", f"TMPDIR={work}", *worker]
    command = [srt, "--settings", str(work / "srt-settings.json"), "-c", shlex.join(worker)]
    return command, ":".join(dict.fromkeys([str(Path(srt).parent), *tools]))


class IsolatedExecutor(PythonExecutor):
    def __init__(self, lake_root, scratch_root, artifact_root, *, timeout=30, memory_mb=4096, max_output_chars=20000,
                 lake="write", workspace=None, readable: Iterable[str | Path] = (),
                 recorder: CellRecorder | None = None):
        if lake not in ("write", "read", "ground", "none"):
            raise ValueError(f"Unknown Lake access mode {lake!r}")
        self.lake_root, self.scratch_root, self.artifact_root = (
            Path(p).resolve() for p in (lake_root, scratch_root, artifact_root))
        self.workspace = None if workspace is None else Path(workspace).resolve()
        self.readable = [Path(p).resolve() for p in readable]
        for root in (self.scratch_root, self.artifact_root, *([self.workspace] if self.workspace else []),
                     *self.readable):
            if root == self.lake_root or root.is_relative_to(self.lake_root) or self.lake_root.is_relative_to(root):
                raise ValueError("The Lake, scratch, artifact, workspace and readable paths must be separate trees")
        self.lake, self.recorder = lake, recorder
        self.timeout, self.memory_mb, self.max_output_chars = timeout, memory_mb, max_output_chars
        self.tools, self.variables = {}, {}
        self.process = None
        self._directory = None
        self._buffer = b""

    def send_tools(self, tools):
        self.tools = {name: tool for name, tool in tools.items() if name != "final_answer"}

    def send_variables(self, variables):
        """Set variables for agent code: at the worker's start, or at once in a running worker."""
        self.variables.update(variables)
        if self.process is not None:
            self._send({"event": "variables", "variables": variables})

    def _start(self):
        for path in self.readable:
            if not path.exists():
                raise FileNotFoundError(f"The readable path {path} does not exist")
        for directory in (self.artifact_root, self.scratch_root, *([self.workspace] if self.workspace else [])):
            directory.mkdir(parents=True, exist_ok=True)
        self._directory = tempfile.TemporaryDirectory(prefix="worker-", dir=self.scratch_root)
        work = Path(self._directory.name)
        interpreter = Path(sys.executable).resolve()
        # mimetypes reads its known files on first use (openpyxl does), and fails if they exist but are denied.
        if sys.platform == "darwin":
            # srt applies before Python starts, so the worker's own sources must be readable too.
            runtime = ["/usr", "/bin", "/System", "/private/etc", "/etc", "/private/var/db/timezone",
                       *mimetypes.knownfiles, "/dev", *_python_roots(), str(Path(__file__).parent)]
        else:
            runtime = ["/usr", "/lib", "/lib64", "/bin", "/etc/ld.so.cache", "/etc/localtime", *mimetypes.knownfiles,
                       "/dev/null", "/dev/urandom", str(interpreter.parent.parent), sys.prefix, sys.base_prefix]
        lake = [str(self.lake_root)]
        config = {
            "read_roots": list(dict.fromkeys(runtime + [str(self.artifact_root)]
                                             + (lake if self.lake in ("read", "ground") else [])
                                             + [str(path) for path in self.readable])),
            "write_roots": ([str(work)] + ([str(self.workspace)] if self.workspace else [])
                            + (lake if self.lake == "write" else [])),
            "lake_root": None if self.lake in ("ground", "none") else str(self.lake_root),
            "workspace": None if self.workspace is None else str(self.workspace),
            "scratch_root": str(work), "artifact_root": str(self.artifact_root),
            "memory_mb": self.memory_mb, "cpu_seconds": self.timeout * 64,
            "max_output_chars": self.max_output_chars, "imports": IMPORTS, "prelude": PRELUDE,
            "tools": {name: list(tool.inputs) for name, tool in self.tools.items()},
            "backend": "srt" if sys.platform == "darwin" else "landlock",
        }
        (work / "worker.json").write_text(json.dumps(config))
        command = [sys.executable, "-I", str(Path(__file__).with_name("_worker.py")), str(work / "worker.json")]
        path = f"{interpreter.parent}:/usr/bin:/bin"
        if config["backend"] == "srt":
            command, tools = _srt_command(work, config["read_roots"], config["write_roots"], command)
            path = f"{tools}:{path}"
        elif sys.platform != "linux":
            raise SandboxUnavailable(f"No code sandbox for {sys.platform}; use Linux or macOS")
        environment = {"PATH": path, "HOME": str(work), "TMPDIR": "/tmp" if config["backend"] == "srt" else str(work),
                       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LANG": "C.UTF-8"}
        log = (self.artifact_root / "worker.log").open("ab")
        self.process = subprocess.Popen(
            command, cwd=work, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
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
        names = [] if self.recorder is None else self.recorder.before(code_action)
        self._send({"event": "execute", "code": code_action, "summarize": names})
        deadline = time.monotonic() + self.timeout
        try:
            while (message := self._receive(deadline))["event"] == "tool":
                started = time.monotonic()
                self._send(self._run_tool(message["name"], message["arguments"]))
                deadline += time.monotonic() - started  # Tool time does not count toward the cell limit.
        except Exception as error:
            # A stuck or dead worker is unusable. The next cell starts a new worker without its variables.
            self.cleanup()
            if self.recorder is not None:
                self.recorder.after(code_action, str(error), {}, True)
            note = f"{error}. The interpreter restarted, so variables and imports from earlier steps are gone."
            raise (TimeoutError if isinstance(error, TimeoutError) else RuntimeError)(note) from error
        if self.recorder is not None:
            error = None if message["success"] else message["error"]
            self.recorder.after(code_action, error, message["summaries"], False)
        if not message["success"]:
            raise RuntimeError("\n".join(part for part in (message["stdout"], message["stderr"]) if part))
        return CodeOutput(output=message["output"], logs=message["stdout"], is_final_answer=message["is_final_answer"])

    def _run_tool(self, name, arguments):
        try:
            response = {"output": json.loads(json.dumps(self.tools[name](**arguments), default=str))}
        except ToolCorrection as error:
            response = {"error": "ValueError", "message": str(error)}
        except Exception as error:
            # Host exception messages can contain private paths, so send the type only.
            response = {"error": type(error).__name__}
        return response

    def cleanup(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass  # The group is gone; macOS reports EPERM when only an exited process remains.
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self._selector.close()
            self.process, self._buffer = None, b""
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
