"""The Task graph: grounded nodes, derived nodes and edges, read from each cell's code with `ast`."""

import ast
import builtins
import re
from collections.abc import Callable

# smolagents quotes the whole failing top-level statement, which spans lines when it is compound.
_FAILED_AT = re.compile(r"Code execution failed at line '(.*?)' due to", re.DOTALL)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
           ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_SKIPPED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)


def _loads(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def _base(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _base(item)]
    if isinstance(target, ast.Starred):
        return _base(target.value)
    if isinstance(target, (ast.Subscript, ast.Attribute)):
        return _base(target.value)
    return []


def _imported(statement: ast.stmt) -> set[str]:
    """The names an import statement binds."""
    if isinstance(statement, ast.Import):
        return {(alias.asname or alias.name).split(".")[0] for alias in statement.names}
    if isinstance(statement, ast.ImportFrom):
        return {alias.asname or alias.name for alias in statement.names}
    return set()


def _writes(statement: ast.stmt) -> set[str]:
    """Variables a statement may assign or change in place, never inside a nested def, lambda, class or
    comprehension. A method call whose result is dropped, such as `rows.append(x)`, changes its receiver."""
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, _SCOPES):
            return
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_base(target))
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr, ast.For, ast.AsyncFor)):
            names.update(_base(node.target))
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            names.update(_base(node.optional_vars))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func,
                                                                                           ast.Attribute):
            names.update(_base(node.value.func.value))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and any(
                k.arg == "inplace" and isinstance(k.value, ast.Constant) and k.value.value is True
                for k in node.keywords):
            names.update(_base(node.func.value))
        for child in ast.iter_child_nodes(node):
            visit(child)

    if not isinstance(statement, _SKIPPED):
        visit(statement)
    return names


def _ground_names(call: ast.Call) -> tuple[list[str], bool]:
    """The string constants a ground(...) call names, and whether it has other arguments."""
    names = [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    return names, len(names) < len(call.args) + len(call.keywords)


class TaskGraph:
    """Records, cell by cell, which grounded data each variable and the answer come from."""

    def __init__(self) -> None:
        self.current_step: Callable[[], int] = lambda: 0
        self._nodes: dict[str, dict] = {}
        self._edges: list[dict] = []
        self._answer: dict | None = None
        self._variables: dict[str, str] = {}
        self._functions: dict[str, set[str]] = {}
        self._modules: set[str] = set()
        self._cell_grounded: list[str] = []
        self._taken: set[str] = set()

    def grounded(self, name: str, kind: str, files: list[str], rows: int | None) -> None:
        if name not in self._nodes:
            self._nodes[name] = {"id": name, "kind": kind, "step": self.current_step(),
                                 "files": list(files), "rows": rows}
        if name not in self._cell_grounded:
            self._cell_grounded.append(name)

    def before(self, code: str) -> list[str]:
        self._cell_grounded, self._taken = [], set()
        tree = self._parse(code)
        if tree is None:
            return []
        modules = self._modules.union(*[_imported(s) for s in tree.body])
        return sorted(set().union(*[_writes(s) for s in tree.body]) - modules)

    def after(self, code: str, error: str | None, summaries: dict[str, dict], restarted: bool) -> None:
        if restarted:
            self._variables, self._functions, self._modules = {}, {}, set()
            return
        tree = self._parse(code)
        if tree is None:
            return
        statements = tree.body
        if error is not None:
            match = _FAILED_AT.search(error)
            failing = [i for i, s in enumerate(statements)
                       if match and ast.get_source_segment(code, s) == match.group(1)]
            statements = statements[:failing[0]] if failing else []
        step = self.current_step()
        made: list[str] = []
        for statement in statements:
            self._statement(statement, step, made)
        derived = {}
        for node_id in made:
            node = self._nodes[node_id]
            node["summary"] = summaries.get(node["variable"], {})
            derived[node_id] = node["from"]
        if derived:
            self._edges.append({"step": step, "code": code, "derived": derived})

    def to_json(self) -> dict:
        return {"nodes": list(self._nodes.values()), "edges": list(self._edges),
                "answer": dict(self._answer) if self._answer else None}

    @staticmethod
    def _parse(code: str) -> ast.Module | None:
        try:
            return ast.parse(code)
        except SyntaxError:
            return None

    def _reads(self, node: ast.AST) -> set[str]:
        names = _loads(node)
        for target in ast.walk(node):
            if isinstance(target, ast.AugAssign):
                names.update(_base(target.target))
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in self._functions:
                names |= self._functions[call.func.id]
        return names

    def _statement(self, statement: ast.stmt, step: int, made: list[str]) -> None:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = {a.arg for a in ast.walk(statement.args) if isinstance(a, ast.arg)}
            assigned = set().union(*[_writes(s) for s in statement.body])
            body = {n for s in statement.body for n in _loads(s)}
            self._functions[statement.name] = body - params - assigned
            return
        if imported := _imported(statement):
            self._modules |= imported
            for name in imported:
                self._variables.pop(name, None)
            return
        if isinstance(statement, _SKIPPED):
            return
        inputs = {self._variables[n] for n in self._reads(statement) if n in self._variables}
        extra = False
        named: list[str] = []
        for call in _calls(statement, "ground"):
            names, other = _ground_names(call)
            named += [n for n in names if n in self._nodes]
            extra = extra or other
        if extra:
            named += [n for n in self._cell_grounded if n not in self._taken]
        inputs |= set(named)
        self._taken |= set(named)
        alias = self._alias(statement)
        if alias:
            self._variables.update(alias)
        else:
            for variable in sorted(_writes(statement) - self._modules):
                self._write(variable, inputs, step, made)
        answers = [call.args[0] for call in _calls(statement, "final_answer") if call.args]
        if answers:
            self._answer_of(answers)

    def _alias(self, statement: ast.stmt) -> dict[str, str]:
        """`a, b = ground("x", "y")`: each target names the grounded node itself."""
        if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "ground"):
            return {}
        names, other = _ground_names(statement.value)
        target = statement.targets[0]
        targets = target.elts if isinstance(target, ast.Tuple) else [target]
        if other or len(targets) != len(names) or not all(isinstance(t, ast.Name) for t in targets):
            return {}
        if not all(n in self._nodes for n in names):
            return {}
        return {t.id: n for t, n in zip(targets, names)}

    def _write(self, variable: str, inputs: set[str], step: int, made: list[str]) -> None:
        node_id = f"{variable}@{step}"
        sources: set[str] = set()
        for source in inputs:
            # A node this step derived already passes on its own sources, so every node points only at grounded
            # nodes or nodes of earlier steps, and the graph has no cycle.
            node = self._nodes[source]
            sources |= set(node["from"]) if node["kind"] == "derived" and node["step"] == step else {source}
        if not sources:
            self._variables.pop(variable, None)
            return
        node = self._nodes.get(node_id)
        if node is None:
            node = self._nodes[node_id] = {"id": node_id, "kind": "derived", "variable": variable,
                                           "step": step, "from": [], "summary": {}}
        node["from"] = sorted(set(node["from"]) | sources)
        self._variables[variable] = node_id
        if node_id not in made:
            made.append(node_id)

    def _answer_of(self, arguments: list[ast.AST]) -> None:
        """The answer of a statement's `final_answer` calls: typed when no argument reads a variable (a built-in or
        imported name is none unless code assigned it), from the grounded nodes behind every argument."""
        roots: set[str] = set()
        seen: set[str] = set()
        pending = [self._variables[n] for argument in arguments for n in self._reads(argument) if n in self._variables]
        while pending:
            node_id = pending.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = self._nodes[node_id]
            if node["kind"] == "derived":
                pending += node["from"]
            else:
                roots.add(node_id)
        typed = all(all((hasattr(builtins, n) or n in self._modules) and n not in self._variables
                        for n in _loads(argument)) for argument in arguments)
        self._answer = {"typed": typed, "from": sorted(roots)}
