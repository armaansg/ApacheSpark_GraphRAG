import ast
import json
from pathlib import Path

from typing import Optional, Union

ROOT = Path("data/raw/git/apache-spark")
MANIFEST = Path("data/manifests/file_paths.txt")
OUT_DIR = Path("data/normalized/graph")
NODES_OUT = OUT_DIR / "nodes.jsonl"
EDGES_OUT = OUT_DIR / "edges.jsonl"

PREFIX = "python/pyspark/"

def iter_target_files():
    for line in MANIFEST.read_text().splitlines():
        path = line.strip()
        if not path.startswith(PREFIX):
            continue
        if not path.endswith(".py"):
            continue
        if "/test/" in path or "/tests/" in path:
            continue
        yield path

def call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
    return None

def qualified_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = qualified_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return None

class FileGraphBuilder(ast.NodeVisitor):
    def __init__(self, rel_path: str, source: str):
        self.rel_path = rel_path
        self.source = source
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self._class_stack: list[str] = []
        self._func_stack: list[str] = []

    @property
    def file_id(self) -> str:
        return f"file:{self.rel_path}"
    
    def _add_node(self, node_type: str, node_id: str, **props):
        self.nodes.append({"id": node_id, "type": node_type, "file": self.rel_path, **props})
    
    def _add_edge(self, edge_type: str, src: str, dst: str, **props):
        self.edges.append({"type": edge_type, "src": src, "dst": dst, **props})
    
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            module = alias.name
            import_id = f"import:{module}"
            self._add_node("import", import_id, name=module)
            self._add_edge("IMPORTS", self.file_id, import_id, line=node.lineno)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        if node.level:
            module = "." * node.level + module
        import_id = f"import:{module}"
        self._add_node("import", import_id, name=module)
        self._add_edge("IMPORTS", self.file_id, import_id, line=node.lineno)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        class_id = f"class:{self.rel_path}:{node.name}"
        self._add_node("class", class_id, name=node.name, line=node.lineno, end_line=getattr(node, "end_lineno", None),)
        self._add_edge("CONTAINS", self.file_id, class_id)

        for base in node.bases:
            base_name = qualified_name(base)
            if base_name:
                self._add_edge("INHERITS", class_id, f"symbol:{base_name}", line=node.lineno)
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._register_function(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._register_function(node)

    def _register_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        if self._class_stack:
            owner = self._class_stack[-1]
            func_id = f"method:{self.rel_path}:{owner}.{node.name}"
            node_type = "method"
            parent_id = f"class:{self.rel_path}:{owner}"
        else:
            func_id = f"function:{self.rel_path}:{node.name}"
            node_type = "function"
            parent_id = self.file_id

        self._add_node( node_type, func_id, name=node.name, line=node.lineno, end_line=getattr(node, "end_lineno", None),
                       class_name = self._class_stack[-1] if self._class_stack else None,)
        self._add_edge("CONTAINS", parent_id, func_id)

        self._func_stack.append(func_id)
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = call_name(child.func)
                if callee:
                    self._add_edge("CALLS", func_id, f"call:{callee}", line=child.lineno)
        self._func_stack.pop()

    def build(self):
        tree = ast.parse(self.source, filename=self.rel_path)
        self._add_node("file", self.file_id, name=Path(self.rel_path).name)
        self.visit(tree)
        return self.nodes, self.edges
    
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_nodes: dict[str, dict] = {}
    all_edges: list[dict] = []
    parsed_files = 0
    skipped_files = 0
    
    with NODES_OUT.open("w") as nodes_f, EDGES_OUT.open("w") as edges_f:
        for rel_path in sorted(iter_target_files()):
            full_path = ROOT / rel_path
            if not full_path.is_file():
                skipped_files += 1
                continue
            try:
                source = full_path.read_text(encoding="utf-8", errors="replace")
                builder = FileGraphBuilder(rel_path, source)
                nodes, edges = builder.build()
            except SyntaxError as exc:
                skipped_files += 1
                print(f"skip syntax error: {rel_path} ({exc})")
                continue

            parsed_files += 1
            for node in nodes:
                all_nodes[node["id"]] = node
                nodes_f.write(json.dumps(node) + "\n")
            for edge in edges:
                all_edges.append(edge)
                edges_f.write(json.dumps(edge) + "\n")

    print(f"Parsed files: {parsed_files}")
    print(f"Skipped files: {skipped_files}")
    print(f"Nodes: {len(all_nodes)}")
    print(f"Edges: {len(all_edges)}")
    print(f"Wrote {NODES_OUT}")
    print(f"Wrote {EDGES_OUT}")

if __name__ == "__main__":
    main()