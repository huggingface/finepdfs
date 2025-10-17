#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
from collections import deque

import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import html


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value"""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1"""
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag,
                self.colspan,
                self.rowspan,
                self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class TEDScorer:
    r"""
    Compute Tree-Edit-Distance Score on HTML tables with support for cell content
    """

    def __init__(self):
        self._tokens = []

    def __call__(self, gt_table: html, pred_table: html, structure_only: bool) -> float:
        r"""
        Compute the tree-edit-distance score TEDS
        TEDS is a float between [0, 1] where 0 is the worst and 1 is the best
        """
        n_nodes_pred = len(pred_table.xpath(".//*"))
        n_nodes_gt = len(gt_table.xpath(".//*"))

        # Convert the html objects into APTED trees
        tree_pred = self._tree_convert_html(pred_table, convert_cell=not structure_only)
        tree_gt = self._tree_convert_html(gt_table, convert_cell=not structure_only)

        n_nodes = max(n_nodes_pred, n_nodes_gt)
        distance = APTED(tree_pred, tree_gt, CustomConfig()).compute_edit_distance()
        teds = 1.0 - (float(distance) / n_nodes)
        return teds

    def _tokenize(self, node: html):
        r"""
        Tokenizes table cells
        """
        self._tokens.append(f"<{node.tag}")
        if node.text is not None:
            self._tokens += list(node.text)
        for n in node.getchildren():
            self._tokenize(n)
        if node.tag != "unk":
            self._tokens.append(f"</{node.tag}>")
        if node.tag != "td" and node.tail is not None:
            self._tokens += list(node.tail)

    def _tree_convert_html(
        self, node: html, convert_cell: bool = False, parent: html = None
    ) -> TableTree:
        r"""
        Converts HTML tree to the format required by apted
        """
        if node.tag == "td":
            if convert_cell:
                self._tokens = []
                self._tokenize(node)
                cell = self._tokens[1:-1].copy()
            else:
                cell = []
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self._tree_convert_html(n, convert_cell, new_node)
        # if parent is None:
        #     return new_node
        return new_node
