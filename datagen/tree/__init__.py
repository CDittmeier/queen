"""datagen/tree — search tree + narrative for stage-5 proof-tree datagen.

    from datagen.tree import get_tree
    t = get_tree(fen)          # depth-3 SF-NNUE alpha-beta + maia-1100 branches
    print(t.string("human"))   # flatten to the search narrative
"""
from datagen.tree.tree import Tree
from datagen.tree.search import get_tree, close_engines

__all__ = ["Tree", "get_tree", "close_engines"]
