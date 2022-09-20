from typing import Optional

from ete3 import Tree as TreeETE

from cherryml.io import Tree


def name_internal_nodes(t: TreeETE) -> None:
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

    def dfs_name_internal_nodes(p: Optional[Tree], v: Tree) -> None:
        global internal_node_id
        if v.name == "":
            v.name = next(names)
        if p:
            # print(f"{p.name} -> {v.name}")
            pass
        for u in v.get_children():
            dfs_name_internal_nodes(v, u)

    dfs_name_internal_nodes(None, t)


def translate_tree(tree_ete: TreeETE) -> Tree:
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
