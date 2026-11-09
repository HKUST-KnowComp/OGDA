"""Private CodeAgent interpreter process. No model credentials cross this boundary."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path


class BoundedOutput:
    def __init__(self, limit):
        self.limit, self.text = limit, ""

    def write(self, value):
        self.text += value[:max(0, self.limit - len(self.text))]
        return len(value)

    def flush(self):
        pass


def patch_interpreter(interpreter):
    """Fix two smolagents 1.26.0 interpreter bugs that silently change what agent code computes."""
    evaluate_ast = interpreter.evaluate_ast
    signals = (interpreter.BreakException, interpreter.ContinueException, interpreter.ReturnException)

    def evaluate_try(node, state, *tools):
        # The original let `except:` and `except Exception:` catch break, continue and return.
        try:
            for statement in node.body:
                evaluate_ast(statement, state, *tools)
        except signals:
            raise
        except Exception as error:
            for handler in node.handlers:
                if handler.type is None or isinstance(error, evaluate_ast(handler.type, state, *tools)):
                    if handler.name:
                        state[handler.name] = error
                    for statement in handler.body:
                        evaluate_ast(statement, state, *tools)
                    break
            else:
                raise
        else:
            for statement in node.orelse:
                evaluate_ast(statement, state, *tools)
        finally:
            for statement in node.finalbody:
                evaluate_ast(statement, state, *tools)

    def evaluate_generatorexp(node, state, *tools):
        # The original ran each `for` clause after the last instead of nesting them, as comprehensions do.
        return interpreter._evaluate_comprehensions(
            node.generators, lambda inner: evaluate_ast(node.elt, inner, *tools), state, *tools)

    interpreter.evaluate_try = evaluate_try
    interpreter.evaluate_generatorexp = evaluate_generatorexp


def tool_result(output, state):
    """What agent code gets from a tool. An output with "__print__" adds that text to the cell's printed output, in
    order with the code's own prints, and returns its "__values__" decoded: the only value (None when there is none)
    when "__single__" is set, and otherwise a tuple."""
    if not (isinstance(output, dict) and "__print__" in output):
        return output
    state["_print_outputs"] += output["__print__"] + "\n"
    values = [decode(value) for value in output["__values__"]]
    if output["__single__"]:
        return values[0] if values else None
    return tuple(values)


def decode(value):
    """A cached table as a DataFrame, with "__file__" only that File's rows, or with "__field__" one column as a
    Series; a path or paths as `Path`s."""
    if not isinstance(value, dict):
        return value
    if "__table__" in value:
        import pandas  # Already loaded: the prelude imports it.

        if "__field__" in value:
            return pandas.read_parquet(value["__table__"], columns=[value["__field__"]])[value["__field__"]]
        table = pandas.read_parquet(value["__table__"])
        if "__file__" in value:
            return table[table[value["__file_column__"]] == value["__file__"]]
        return table
    if "__path__" in value:
        return Path(value["__path__"])
    if "__paths__" in value:
        return [Path(path) for path in value["__paths__"]]
    return value


def summarize(state, names):
    """For each named variable in the interpreter state: its type, its shape or length and, when small, its value."""
    summaries = {}
    for name in names:
        if name in state:
            try:
                summary = describe(state[name])
                json.dumps(summary, ensure_ascii=False, allow_nan=False).encode()  # send() must be able to write it.
            except Exception:
                continue  # A value that cannot be summarized is left out.
            summaries[name] = summary
    return summaries


def describe(value):
    """One value's type, its shape or length, and its value when it is a scalar or a list of at most 20 scalars."""
    summary = {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and shape:
        summary["shape"] = list(shape)
    if isinstance(value, (list, tuple, dict, set)):
        summary["length"] = len(value)
    if isinstance(value, (list, tuple)) and len(value) <= 20:
        items = [scalar(item) for item in value]
        if None not in items:
            summary["value"] = items
    elif (item := scalar(value)) is not None:
        summary["value"] = item
    return summary


def scalar(value):
    """A finite number, a bool or a string of at most 200 characters, as JSON; None for any other value."""
    import numpy  # Already loaded: the prelude imports it.

    if isinstance(value, numpy.generic) and value.dtype.kind in "biuf":
        value = value.item()  # Booleans and numbers only: item() turns a datetime into an integer.
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value if isinstance(value, int) or (isinstance(value, str) and len(value) <= 200) else None


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    spec = importlib.util.spec_from_file_location("_ogda_isolate", Path(__file__).with_name("_isolate.py"))
    isolation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolation)
    channel = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # Direct writes to fd 1 (os.write, child processes) go to worker.log, not into the channel.

    def send(value):
        channel.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def receive():
        line = sys.stdin.buffer.readline(16 * 1024 * 1024 + 1)
        if not line or len(line) > 16 * 1024 * 1024:
            raise EOFError("Closed or oversized worker message")
        return json.loads(line)

    try:
        if config["backend"] == "landlock":
            isolation.restrict_filesystem(config["read_roots"], config["write_roots"])
        elif config["backend"] != "srt" or sys.platform != "darwin":
            raise RuntimeError(f"Unknown sandbox backend {config['backend']!r} on {sys.platform}")
        # Under srt the sandbox already holds: srt started this process with the same roots and no network.
        isolation.restrict_resources(config["memory_mb"], config["cpu_seconds"])
        if config["backend"] == "landlock":
            isolation.restrict_syscalls()
        import smolagents.local_python_executor as interpreter

        patch_interpreter(interpreter)
        executor = interpreter.LocalPythonExecutor(
            additional_authorized_imports=config["imports"],
            max_print_outputs_length=config["max_output_chars"],
            timeout_seconds=None,  # The host keeps each cell's deadline, which excludes tool time.
            # The OS sandbox limits which paths open() can reach.
            additional_functions={"open": open, "repr": repr, "object": object, "bytes": bytes},
        )
        variables = {"SCRATCH_ROOT": Path(config["scratch_root"]), "ARTIFACT_ROOT": Path(config["artifact_root"])}
        if config["lake_root"] is not None:
            variables["LAKE_ROOT"] = Path(config["lake_root"])
        if config["workspace"] is not None:
            variables["WORKSPACE"] = Path(config["workspace"])
        executor.send_variables(variables)

        def bind_tool(name, parameters):
            def invoke(*args, **kwargs):
                if parameters == ["names"] and "names" not in kwargs:
                    # A tool that takes names gets them as separate arguments or as one list.
                    args = (list(args[0]) if len(args) == 1 and isinstance(args[0], (list, tuple)) else list(args),)
                if len(args) > len(parameters):
                    raise TypeError("Too many positional tool arguments")
                arguments = dict(zip(parameters, args))
                if arguments.keys() & kwargs.keys():
                    raise TypeError("Duplicate tool argument")
                arguments.update(kwargs)
                send({"event": "tool", "name": name, "arguments": arguments})
                response = receive()
                if "message" in response:
                    raise ValueError(response["message"])  # A correction the tool addressed to the agent.
                if "error" in response:
                    raise RuntimeError(response["error"])
                return tool_result(response["output"], executor.state)
            return invoke

        tools = {name: bind_tool(name, parameters) for name, parameters in config["tools"].items()}
        tools["final_answer"] = lambda answer: answer
        executor.send_tools(tools)
        executor(config["prelude"])
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
                "stderr": stderr.text,
            }
        except BaseException as error:
            logs = str(executor.state.get("_print_outputs", ""))
            response = {
                "event": "result", "success": False, "is_final_answer": False,
                "output": None, "stdout": (stdout.text + logs)[:config["max_output_chars"]],
                "stderr": (stderr.text + f"{type(error).__name__}: {error}")[:4000],
                "error": f"{type(error).__name__}: {error}",  # Whole, for the recorder to find the failing statement.
            }
        response["summaries"] = summarize(executor.state, request["summarize"])
        send(response)


if __name__ == "__main__":
    main()
