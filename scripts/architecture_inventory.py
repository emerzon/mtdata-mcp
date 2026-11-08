"""Print an AST-only dependency and special-case inventory without importing MT5."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def inventory(root: Path) -> dict:
    sources = {}
    for path in sorted((root / "src" / "mtdata").rglob("*.py")):
        module = ".".join(path.relative_to(root / "src").with_suffix("").parts)
        module = module.removesuffix(".__init__")
        sources[module] = (path, ast.parse(path.read_text(encoding="utf-8-sig")))

    imports = set()
    special_cases = []
    definitions = {}
    for module, (path, tree) in sources.items():
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        definitions[module] = [
            {"name": node.name, "line": node.lineno}
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                parts = package.split(".")
                base = (
                    ".".join(parts[: len(parts) - node.level + 1])
                    if node.level else ""
                )
                target = ".".join(filter(None, (base, node.module)))
                targets = [target, *(f"{target}.{alias.name}" for alias in node.names)]
            for target in targets:
                if target in sources:
                    imports.add((module, target))

            kind = None
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    kind = "bare_except"
                elif any(isinstance(child, ast.Pass) for child in node.body):
                    kind = "pass_exception"
            elif isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name.endswith(".sleep") or name == "sleep":
                    kind = "sleep"
            elif isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                if any(isinstance(value, ast.Constant) for value in node.test.comparators):
                    kind = "literal_branch"
            if kind:
                special_cases.append({
                    "path": path.relative_to(root).as_posix(),
                    "line": node.lineno,
                    "kind": kind,
                })

    disabled_tests = []
    for path in sorted((root / "tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    name = ast.unparse(decorator)
                    function = decorator.func if isinstance(decorator, ast.Call) else decorator
                    marker = ast.unparse(function).rsplit(".", 1)[-1]
                    if marker in {"skip", "skipif", "skipIf", "skipUnless", "xfail"}:
                        disabled_tests.append({
                            "path": path.relative_to(root).as_posix(),
                            "line": node.lineno,
                            "decorator": name,
                        })
            elif isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name.rsplit(".", 1)[-1] in {"skip", "skipTest", "xfail", "importorskip"}:
                    disabled_tests.append({
                        "path": path.relative_to(root).as_posix(),
                        "line": node.lineno,
                        "call": ast.unparse(node),
                    })
    return {
        "modules": len(sources),
        "import_edges": len(imports),
        "imports": sorted(imports),
        "definitions": definitions,
        "special_case_candidates": special_cases,
        "disabled_test_candidates": disabled_tests,
    }


if __name__ == "__main__":
    print(json.dumps(inventory(Path(__file__).resolve().parents[1]), indent=2))
