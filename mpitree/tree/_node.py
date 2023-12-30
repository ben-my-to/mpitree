""""""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from numpy.typing import ArrayLike


@dataclass(kw_only=True)
class Node:
    """A tree node.

    Parameters
    ----------
    value: float
        The descriptive or target feature value.

    threshold : float, optional
        Add Description Here.

    depth : int, default=0
        The number of levels from the root to a node. The root node `depth`
        is initialized to 0 and successor nodes are one depth lower from
        its parent.

    parent : Node, optional
        The precedent node.

    left : Node, optional
        The node whose `value` is less than or equal to its parent.

    right : Node, optional
        The node whose `value` is greater than its parent.
    """

    value: float
    threshold: Optional[float] = None
    depth: int = field(default_factory=int)
    parent: Optional[Node] = field(default=None, repr=False)
    left: Optional[Node] = field(default=None, repr=False)
    right: Optional[Node] = field(default=None, repr=False)

    def __post_init__(self):
        if self.parent is not None:
            self.depth = self.parent.depth + 1

    def __str__(self):
        """Export a string-formatted node.

        Returns
        -------
        str
            The string-formatted node.
        """

        spacing = self.depth * "│  " + (
            "└──" if self.is_leaf else "├──" if self.depth else "┌──"
        )

        root_node_style = "{spacing} {value}"
        interior_node_style = "{spacing} {value} [{sign} {threshold}]"

        feature_name = (
            f"class: {self.value}" if self.is_leaf else f"feature_{self.value}"
        )

        if not self.depth:
            return root_node_style.format(spacing=spacing, value=feature_name)

        return interior_node_style.format(
            spacing=spacing,
            value=feature_name,
            sign=("<=" if self is self.parent.left else ">"),
            threshold=round(self.parent.threshold, 2),
        )

    @property
    def is_leaf(self):
        """Return whether a node is terminal.

        A node is a leaf proprty if it contains no children.

        Returns
        -------
        bool
        """
        return self.left is None and self.right is None
