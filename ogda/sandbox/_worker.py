"""Private CodeAgent interpreter process. No model credentials cross this boundary."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path


class FileGate:
    """An audit hook that logs Lake reads and Workspace reads and writes and, with `allowed`, refuses other Lake reads."""

    def __init__(self, lake_root, workspace, allowed, log):
        self.lake = lake_root and os.path.realpath(lake_root)
        self.workspace = workspace and os.path.realpath(workspace)
        self.allowed = None if allowed is None else set(allowed)
        self.log, self.files = log, []

    def __call__(self, event, args):
        if event == "open":
            path, mode, flags = args
        elif event == "sqlite3.connect":
            path, mode, flags = args[0], "r", 0
        else:
            return
        if path is None or isinstance(path, int):
            return
        try:
            full = os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError):
            return
        if isinstance(mode, str):
            write = any(letter in mode for letter in "wax+")
        else:
            write = bool((flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
        for root, area in ((self.lake, "lake"), (self.workspace, "workspace")):
            if root and full.startswith(root + os.sep):
                relative = full[len(root) + 1:]
                if area == "lake" and write:
                    return  # Landlock refuses Lake writes in the Data agent.
                if area == "lake" and self.allowed is not None and full not in self.allowed and not os.path.isdir(full):
                    raise PermissionError(f"{relative} is not in the task graph. Call ground({relative!r}) or ground "
                                          "the Schema or Table that maps it, then read it again.")
                if self.log:
                    self.files.append({"area": area, "path": relative, "mode": "write" if write else "read"})
                return

    def take(self):
        files, self.files = self.files, []
        return files


class BoundedOutput:
    def __init__(self, limit):
        self.limit, self.text = limit, ""

    def write(self, value):
        self.text += value[:max(0, self.limit - len(self.text))]
        return len(value)

    def flush(self):
        pass


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    spec = importlib.util.spec_from_file_location("_ogda_isolate", Path(__file__).with_name("_isolate.py"))
    isolation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolation)
    channel = os.fdopen(os.dup(1), "w", buffering=1)

    def send(value):
        channel.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def receive():
        line = sys.stdin.buffer.readline(16 * 1024 * 1024 + 1)
        if not line or len(line) > 16 * 1024 * 1024:
            raise EOFError("Closed or oversized worker message")
        return json.loads(line)

    try:
        isolation.restrict_filesystem(config["read_roots"], config["write_roots"])
        isolation.restrict_resources(config["memory_mb"], config["cpu_seconds"])
        isolation.restrict_syscalls()
        from smolagents.local_python_executor import LocalPythonExecutor

        executor = LocalPythonExecutor(
            additional_authorized_imports=config["imports"],
            max_print_outputs_length=config["max_output_chars"],
            timeout_seconds=config["timeout"],
            # Landlock limits which paths open() can reach.
            additional_functions={"open": open, "repr": repr},
        )
        variables = {"SCRATCH_ROOT": Path(config["scratch_root"]), "ARTIFACT_ROOT": Path(config["artifact_root"])}
        if config["lake_root"] is not None:
            variables["LAKE_ROOT"] = Path(config["lake_root"])
        if config["workspace"] is not None:
            variables["WORKSPACE"] = Path(config["workspace"])
        executor.send_variables(variables)

        def bind_tool(name, parameters):
            def invoke(*args, **kwargs):
                if len(args) > len(parameters):
                    raise TypeError("Too many positional tool arguments")
                arguments = dict(zip(parameters, args))
                if arguments.keys() & kwargs.keys():
                    raise TypeError("Duplicate tool argument")
                arguments.update(kwargs)
                send({"event": "tool", "name": name, "arguments": arguments})
                response = receive()
                if "gate" in response:
                    gate.allowed = set(response["gate"])
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response["output"]
            return invoke

        tools = {name: bind_tool(name, parameters) for name, parameters in config["tools"].items()}
        tools["final_answer"] = lambda answer: answer
        executor.send_tools(tools)
        executor(config["prelude"])
        gate = FileGate(config["lake_root"], config["workspace"], config["gate"], config["log_files"])
        if config["gate"] is not None or config["log_files"]:
            sys.addaudithook(gate)
        send({"event": "ready"})
    except BaseException as error:
        send({"event": "setup_error", "error": type(error).__name__})
        return

    while True:
        try:
            request = receive()
        except EOFError:
            return
        if request["event"] == "variables":
            executor.send_variables(request["variables"])
            continue
        stdout = BoundedOutput(config["max_output_chars"])
        stderr = BoundedOutput(4000)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = executor(request["code"])
            output = result.output
            # Final results must be JSON values. Exploratory expressions can
            # yield DataFrames and other objects, whose display is sufficient.
            try:
                json.dumps(output, allow_nan=False)
            except (TypeError, ValueError):
                if result.is_final_answer:
                    raise ValueError("final_answer must contain finite JSON values")
                output = str(output)[:config["max_output_chars"]]
            response = {
                "event": "result", "success": True, "output": output,
                "is_final_answer": result.is_final_answer,
                "stdout": (stdout.text + result.logs)[:config["max_output_chars"]],
                "stderr": stderr.text, "files": gate.take(),
            }
        except BaseException as error:
            logs = str(executor.state.get("_print_outputs", ""))
            response = {
                "event": "result", "success": False, "is_final_answer": False,
                "output": None, "stdout": (stdout.text + logs)[:config["max_output_chars"]],
                "stderr": (stderr.text + f"{type(error).__name__}: {error}")[:4000], "files": gate.take(),
            }
        send(response)


if __name__ == "__main__":
    main()
