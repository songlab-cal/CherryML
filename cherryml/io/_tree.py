import tempfile
from typing import List, Optional, Tuple

import ete3


class Tree:
    def __init__(self) -> None:
        self._num_nodes = 0
        self._num_edges = 0
        self._adj_list = {}
        self._edges = []
        self._m = 0
        self._out_deg = {}
        self._in_deg = {}
        self._parent = {}

    def add_node(self, v: str) -> None:
        self._adj_list[v] = []
        self._out_deg[v] = 0
        self._in_deg[v] = 0
        self._num_nodes += 1

    def add_nodes(self, nodes: List[str]) -> None:
        for node in nodes:
            self.add_node(node)

    def add_edge(self, u: str, v: str, length: float) -> None:
        if v in self._parent:
            raise Exception(
                f"Node {v} already has a parent ({self._parent[v][0]}), cannot "
                f"also have parent {u} - graph is not a tree."
            )
        self._adj_list[u].append((v, length))
        self._edges.append((u, v, length))
        self._m += 1
        self._out_deg[u] += 1
        self._in_deg[v] += 1
        self._parent[v] = (u, length)
        self._num_edges += 1

    def add_edges(self, edges: List[Tuple[str, str, float]]) -> None:
        for u, v, length in edges:
            self.add_edge(u, v, length)

    def edges(self) -> List[Tuple[str, str, float]]:
        return self._edges[:]

    def is_node(self, v: str) -> bool:
        return v in self._adj_list

    def nodes(self) -> List[str]:
        return list(self._adj_list.keys())[:]

    def root(self) -> str:
        roots = [u for u in self._adj_list.keys() if self._in_deg[u] == 0]
        if len(roots) != 1:
            raise Exception(f"Tree should have one root, but found: {roots}")
        return roots[0]

    def __str__(self) -> str:
        res = ""
        res += f"Tree with {self._num_nodes} nodes, and {self._m} edges:\n"
        for u in self._adj_list.keys():
            for v, length in self._adj_list[u]:
                res += f"{u} -> {v}: {length}\n"
        return res

    def children(self, u: str) -> List[str]:
        return list(self._adj_list[u])[:]

    def is_leaf(self, u: str) -> bool:
        return self._out_deg[u] == 0

    def is_root(self, u: str) -> bool:
        return self._in_deg[u] == 0

    def num_nodes(self) -> int:
        return self._num_nodes

    def num_edges(self) -> int:
        return self._num_edges

    def preorder_traversal(self) -> List[str]:
        res = []

        def dfs(v: str):
            res.append(v)
            for u, _ in self.children(v):
                dfs(u)

        dfs(self.root())
        return res

    def postorder_traversal(self) -> List[str]:
        res = []

        def dfs(v: str):
            for u, _ in self.children(v):
                dfs(u)
            res.append(v)

        dfs(self.root())
        return res

    def parent(self, u: str) -> Tuple[str, float]:
        return self._parent[u]

    def leaves(self) -> List[str]:
        return [u for u in self.nodes() if self.is_leaf(u)]

    def internal_nodes(self) -> List[str]:
        return [u for u in self.nodes() if not self.is_leaf(u)]

    def scaled(self, scaling_factor: float, node_name_prefix: str = ""):
        """
        Tree scaled by the given scaling_factor.

        Node names are additionally prefixed with node_name_prefix.
        """
        with tempfile.NamedTemporaryFile("w") as scaled_tree_file:
            scaled_tree_path = scaled_tree_file.name
            write_tree(
                tree=self,
                tree_path=scaled_tree_path,
                scaling_factor=scaling_factor,
                node_name_prefix=node_name_prefix,
            )
            return read_tree(scaled_tree_path)

    def to_ete3(self) -> ete3.Tree:
        """
        Return the ete3 version of this tree
        """
        tree_ete = ete3.Tree(name=self.root())
        ete3_node_dict = {}
        ete3_node_dict[self.root()] = tree_ete
        for node in self.preorder_traversal():
            for child, dist in self.children(node):
                ete3_node_dict[child] = ete3_node_dict[node].add_child(
                    name=child, dist=dist
                )
        return tree_ete

    def to_ete3_resolve_root_trifurcation(self) -> ete3.Tree:
        """
        Return the ete3 version of this tree, resolving the
        trifurcation at the root if it exists
        """
        if len(self.children(self.root())) == 2:
            # No trifurcation
            return self.to_ete3()
        assert len(self.children(self.root())) == 3

        tree_ete = ete3.Tree(name=self.root() + "_fakeroot")
        ete3_node_dict = {}
        ete3_node_dict[self.root() + "_fakeroot"] = tree_ete

        # Create binary root manually
        first_root_child, dist = self.children(self.root())[0]
        ete3_node_dict[first_root_child] = ete3_node_dict[
            self.root() + "_fakeroot"
        ].add_child(name=first_root_child, dist=dist / 2)
        ete3_node_dict[self.root()] = ete3_node_dict[
            self.root() + "_fakeroot"
        ].add_child(name=self.root(), dist=dist / 2)

        for node in self.preorder_traversal():
            for i, (child, dist) in enumerate(self.children(node)):
                if i == 0 and self.is_root(node):
                    # This edge was split into two manually above
                    continue
                ete3_node_dict[child] = ete3_node_dict[node].add_child(
                    name=child, dist=dist
                )
        return tree_ete

    def to_newick(self, format: str) -> str:
        """
        Return the newick representation of this tree.
        """
        tree_ete = self.to_ete3()
        return tree_ete.write(format=format)

    def to_newick_resolve_root_trifurcation(self, format: str) -> str:
        """
        Return the newick representation of this tree.
        """
        tree_ete = self.to_ete3_resolve_root_trifurcation()
        return tree_ete.write(format=format)


def write_tree(
    tree: Tree,
    tree_path: str,
    scaling_factor: float = 1.0,
    node_name_prefix: str = "",
) -> None:
    res_list = []
    res_list.append(f"{tree.num_nodes()} nodes\n")
    for node in tree.nodes():
        res_list.append(f"{node_name_prefix + node}\n")
    res_list.append(f"{tree.num_edges()} edges\n")
    for u, v, d in tree.edges():
        res_list.append(
            f"{node_name_prefix + u} {node_name_prefix + v} "
            f"{d * scaling_factor}\n"
        )
    res = "".join(res_list)
    with open(tree_path, "w") as tree_file:
        tree_file.write(res)


def read_tree(
    tree_path: str,
) -> Tree:
    with open(tree_path, "r") as tree_file:
        lines = tree_file.read().strip().split("\n")
    try:
        n, s = lines[0].split(" ")
        if s != "nodes":
            raise Exception
        n = int(n)
    except Exception:
        raise Exception(
            f"Tree file: {tree_path} should start with '[num_nodes] nodes'. "
            f"It started with: '{lines[0]}'"
        )
    tree = Tree()
    for i in range(1, n + 1, 1):
        v = lines[i]
        tree.add_node(v)
    try:
        m, s = lines[n + 1].split(" ")
        if s != "edges":
            raise Exception
        m = int(m)
    except Exception:
        raise Exception(
            f"Tree file: {tree_path} should have line '[num_edges] edges' at "
            f"position {n + 1}, but it had line: '{lines[n + 1]}'"
        )
    if len(lines) != n + 1 + m + 1:
        raise Exception(
            f"Tree file: {tree_path} should have {m} edges, but it has "
            f"{len(lines) - n - 2} edges instead."
        )
    for i in range(n + 2, n + 2 + m, 1):
        try:
            u, v, length = lines[i].split(" ")
            length = float(length)
        except Exception:
            raise Exception(
                f"Tree file: {tree_path} should have line '[u] [v] [length]' at"
                f" position {i}, but it had line: '{lines[i]}'"
            )
        if not tree.is_node(u) or not tree.is_node(v):
            raise Exception(
                f"In Tree file {tree_path}: {u} and {v} should be nodes in the"
                f" tree, but the nodes are: {tree.nodes()}"
            )
        tree.add_edge(u, v, length)
    assert tree.num_nodes() == n
    assert tree.num_edges() == m
    return tree


def _name_internal_nodes(t: ete3.Tree) -> None:
    """
    Assigns names to the internal nodes of tree t if they don't already have a
    name.
    """

    def node_name_generator():
        """Generates unique node names for the tree."""
        internal_node_id = 1
        while True:
            yield f"internal-{internal_node_id}"
            internal_node_id += 1

    names = node_name_generator()

    def dfs_name_internal_nodes(p: Optional[ete3.Tree], v: ete3.Tree) -> None:
        global internal_node_id
        if v.name == "":
            v.name = next(names)
        if p:
            # print(f"{p.name} -> {v.name}")
            pass
        for u in v.get_children():
            dfs_name_internal_nodes(v, u)

    dfs_name_internal_nodes(None, t)


def _translate_tree(tree_ete: ete3.Tree) -> Tree:
    tree = Tree()

    def dfs_translate_tree(p, v) -> None:
        tree.add_node(v.name)
        if p is not None:
            try:
                tree.add_edge(p.name, v.name, v.dist)
            except Exception:
                raise Exception("Could not translate tree")
        for u in v.get_children():
            dfs_translate_tree(v, u)

    dfs_translate_tree(None, tree_ete)
    return tree


def convert_newick_to_CherryML_Tree(
    tree_newick: str,
) -> Tree:
    tree_ete = ete3.Tree(tree_newick)
    _name_internal_nodes(tree_ete)
    tree = _translate_tree(tree_ete)
    return tree


def test_convert_newick_to_CherryML_Tree():
    tree_newick = "((Homo_sapiens:0.00655,Pan_troglodytes:0.00684):0.00422);"
    tree = convert_newick_to_CherryML_Tree(
        tree_newick=tree_newick,
    )
    assert(
        tree.to_newick(format=1) == "((Homo_sapiens:0.00655,Pan_troglodytes:0.00684)internal-2:0.00422);"
    )
