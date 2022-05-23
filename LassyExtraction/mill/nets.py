from __future__ import annotations

import pdb

from .types import Type, Atom, Functor, Box, Diamond
from .terms import (Term, Variable, Constant, ArrowElimination, ArrowIntroduction,
                    DiamondIntroduction, DiamondElimination, BoxIntroduction, BoxElimination)
from typing import Literal, Iterable
from abc import ABC, abstractmethod
from itertools import product


########################################################################################################################
# Formula Trees
########################################################################################################################


class FormulaTree(ABC):
    @property
    @abstractmethod
    def polarity(self) -> bool: ...
    def to_type(self) -> Type: return tree_to_type(self)

    @staticmethod
    def from_type(_type: Type, polarity: bool, index: int) -> tuple[FormulaTree, int]:
        return type_to_tree(_type, polarity, index)

    @staticmethod
    def from_types(types: Iterable[Type]) -> tuple[list[FormulaTree], int]:
        ...

    @abstractmethod
    def __repr__(self) -> str: ...


class LeafFT(FormulaTree):
    __match_args__ = ('atom', 'index', 'polarity')

    atom:       str
    index:      int
    _polarity:   bool

    def __init__(self, atom: str, polarity: bool, index: int):
        self.atom = atom
        self.index = index
        self._polarity = polarity

    @property
    def polarity(self) -> bool: return self._polarity
    def __repr__(self) -> str: return f'{self.atom}({"+" if self.polarity else "-"},{self.index})'


class UnaryFT(FormulaTree):
    __match_args__ = ('modality', 'content', 'decoration', 'polarity')
    modality:   Literal['box'] | Literal['diamond']
    decoration: str
    content:    FormulaTree

    def __init__(self, modality: Literal['box'] | Literal['diamond'], content: FormulaTree, decoration: str):
        self.modality = modality
        self.decoration = decoration
        self.content = content

    @property
    def polarity(self) -> bool: return self.content.polarity
    def __repr__(self) -> str: return f'U({"□" if self.modality == "box" else "◇"}{self.decoration}, {self.content})'


class BinaryFT(FormulaTree):
    __match_args__ = ('left', 'right')

    left:   FormulaTree
    right:  FormulaTree

    def __init__(self, left: FormulaTree, right: FormulaTree):
        assert left.polarity == (not right.polarity)
        self.left = left
        self.right = right

    @property
    def polarity(self) -> bool: return self.right.polarity
    def __repr__(self) -> str: return f'B({self.left},{self.right})'


def tree_to_type(tree: FormulaTree) -> Type:
    match tree:
        case LeafFT(a, _, _): return Atom(a)
        case UnaryFT('box', c, decoration, _): return Box(decoration, tree_to_type(c))
        case UnaryFT('diamond', c, decoration, _): return Diamond(decoration, tree_to_type(c))
        case BinaryFT(l, r, _): return Functor(tree_to_type(l), tree_to_type(r))


def type_to_tree(_type: Type, polarity: bool = True,  index: int = 0, step: int = 1) -> tuple[FormulaTree, int]:
    match _type:
        case Atom(a): return LeafFT(a, polarity, index), index + step
        case Functor(left, right):
            ltree, lindex = type_to_tree(left, not polarity, index)
            rtree, rindex = type_to_tree(right, polarity, lindex)
            return BinaryFT(ltree, rtree), rindex
        case Box(decoration, content):
            ctree, index = type_to_tree(content, polarity, index)
            return UnaryFT('box', ctree, decoration), index
        case Diamond(decoration, content):
            ctree, index = type_to_tree(content, polarity, index)
            return UnaryFT('diamond', ctree, decoration), index


def match(left: FormulaTree, right: FormulaTree) -> dict[LeafFT, LeafFT]:
    match left, right:
        case LeafFT(l_atom, _, l_polarity), LeafFT(r_atom, _, r_polarity):
            assert l_atom == r_atom and l_polarity == (not r_polarity)
            return {right: left} if l_polarity else {left: right}  # type: ignore
        case UnaryFT(l_mod, l_content, l_deco, _), UnaryFT(r_mod, r_content, r_deco, _):
            assert l_mod == r_mod and l_deco == r_deco
            return match(l_content, r_content)
        case BinaryFT(l_left, l_right), BinaryFT(r_left, r_right):
            return match(l_left, r_left) | match(l_right, r_right)


def flip(tree: FormulaTree) -> FormulaTree:
    match tree:
        case LeafFT(atom, index, polarity): return LeafFT(atom, not polarity, index)
        case UnaryFT(mod, content, decoration, _): return UnaryFT(mod, flip(content), decoration)
        case BinaryFT(left, right): return BinaryFT(flip(left), flip(right))


def term_to_links(term: Term) -> tuple[dict[LeafFT, LeafFT], dict[int, FormulaTree]]:
    constants, (conclusion, index), lex_trees = term.constants(), type_to_tree(term.type, False), {}

    for constant in constants:
        formula_tree, index = type_to_tree(constant.type, True, index)
        lex_trees[constant.index] = formula_tree

    def go(_term: Term, _index: int, _vars: dict[int, FormulaTree]) -> \
            tuple[dict[LeafFT, LeafFT], FormulaTree, int, dict[int, FormulaTree]]:
        match _term:
            case Variable(_type, _index):
                return {}, _vars.pop(_index), _index, _vars
            case Constant(_type, _index):
                return {}, lex_trees[_index], _index, _vars
            case ArrowElimination(function, argument):
                left_links, (_, left_match, rem), _index, _vars = go(function, _index, _vars)
                right_links, right_match, _index, _vars = go(argument, _index, _vars)
                return left_links | right_links | match(left_match, right_match), rem, _index, _vars
            case ArrowIntroduction(abstraction, body):
                var_tree, _index = type_to_tree(abstraction.type, True, _index, step=-1)
                _vars[abstraction.index] = var_tree
                _links, tree, _index, _vars = go(body, _index, _vars)
                return _links, BinaryFT(flip(var_tree), tree), _index, _vars
            case BoxElimination(_, body) | DiamondElimination(_, body):
                _links, tree, _index, _vars = go(body, _index, _vars)
                return _links, tree.content, _index, _vars
            case BoxIntroduction(decoration, body):
                _links, tree, _index, _vars = go(body, _index, _vars)
                return _links, UnaryFT('box', tree, decoration), _index, _vars
            case DiamondIntroduction(decoration, body):
                _links, tree, _index, _vars = go(body, _index, _vars)
                return _links, UnaryFT('diamond', tree, decoration), _index, _vars

    links, output_tree, _, _ = go(term, -1, {})
    links |= match(output_tree, conclusion)

    def beta_norm(_links: dict[LeafFT, LeafFT]) -> dict[LeafFT, LeafFT]:
        detours = {(x, y) for x, y in product(_links.items(), _links.items())
                   if x[0].index == y[1].index and x[0].index < 0}
        if not detours:
            return _links
        long_links = {x for x, _ in detours} | {y for _, y in detours}
        norm_links = {y[0]: x[1] for x, y in detours}
        next_round = {x: y for x, y in _links.items() if (x, y) not in long_links} | norm_links
        return beta_norm(next_round)

    return beta_norm(links), lex_trees

#
# def reachable_positives(tree: Tree) -> set[int]:
#     match tree:
#         case Leaf(_, _, index): return {index}
#         case Unary(_, _, _, content): return reachable_positives(content)
#         case Binary(True, _, right): return reachable_positives(right)
#         case Binary(False, _, _): return set()
#         case _: raise ValueError(f'{tree} must be a formula Tree')
#
#
# def par_trees(tree: Tree, par: bool = False) -> list[Tree]:
#     match tree:
#         case Leaf(_, _, _): return [tree] if par else []
#         case Unary(_, _, _, content): return ([tree] if par else []) + par_trees(content, False)
#         case Binary(polarity, left, right):
#             return ([tree] if par else []) + par_trees(left, not polarity) + par_trees(right, False)
#         case _: raise ValueError(f'{tree} must be a formula Tree')
#
#
# def tree_to_type(tree: Tree) -> T:
#     match tree:
#         case Leaf(atom, _, _): return Atom(atom)
#         case Unary(_, '□', decoration, content): return Box(decoration, tree_to_type(content))
#         case Unary(_, '◇', decoration, content): return Diamond(decoration, tree_to_type(content))
#         case Binary(_, left, right): return Functor(tree_to_type(left), tree_to_type(right))
#         case _: raise ValueError(f'{tree} must be a formula Tree')
#
#
# def rooting_branch(container: Tree, subtree: Tree) -> Tree | None:
#     match container:
#         case Unary(_, _, _, content):
#             return container if content == subtree else rooting_branch(content, subtree)
#         case Binary(_, left, right):
#             return (container if left == subtree or right == subtree
#                     else rooting_branch(left, subtree) or rooting_branch(right, subtree))
#
#
# def links_to_term(
#         links: dict[Leaf, Leaf], formula_assignments: dict[int, Tree]) -> T:
#     i = -1
#     hypotheses = {(i := i-1): par for key in sorted(formula_assignments)
#                   for par in par_trees(formula_assignments[key])}
#     atom_to_word = {atom_idx: w_idx for w_idx, tree in formula_assignments.items()
#                     for atom_idx in reachable_positives(tree)}
#     atom_to_var = {atom_idx: var_idx for var_idx, tree in hypotheses.items()
#                    for atom_idx in reachable_positives(tree)}
#
#     def negative_traversal(negative_tree: Tree) -> T:
#         assert not negative_tree.polarity
#         match negative_tree:
#             case Leaf(_, _, _): return positive_traversal(links[negative_tree])
#             case Unary(_, '□', decoration, content): return Proof.box(decoration, negative_traversal(content))
#             case Unary(_, '◇', decoration, content): return Proof.diamond(decoration, negative_traversal(content))
#             case Binary(_, left, right):
#                 abstraction = tree_to_type(left).var(abs(next(k for k in hypotheses if hypotheses[k] == left)))
#                 return Proof.abstract(abstraction, negative_traversal(right))
#
#     def positive_traversal(positive_tree: Tree, grounding: tuple[int, Tree] | None = None) -> T:
#         assert positive_tree.polarity
#
#         if grounding is None:
#             atom_idx = positive_tree.index
#             index = atom_to_word[atom_idx] if atom_idx in atom_to_word else atom_to_var[atom_idx]
#             container = formula_assignments[index] if index > 0 else hypotheses[index]
#             grounding = (index, container)
#         else:
#             index, container = grounding
#
#         if positive_tree == container:
#             proof_type = tree_to_type(positive_tree)
#             return proof_type.con(index) if index > 0 else proof_type.var(abs(index))
#         rooted_in = rooting_branch(container, positive_tree)
#         match rooted_in:
#             case Unary(_, '□', _, _):
#                 return Proof.unbox(positive_traversal(rooted_in, grounding))
#             case Unary(_, '◇', _, _):
#                 return Proof.undiamond(positive_traversal(rooted_in, grounding))
#             case Binary(True, left, _):
#                 return Proof.apply(positive_traversal(rooted_in, grounding), negative_traversal(left))
#             case Binary(False, _, right):
#                 return Proof.apply(positive_traversal(rooted_in, grounding), negative_traversal(right))
#         pdb.set_trace()
#         raise NotImplementedError
#     return negative_traversal(next(iter(leaf for leaf in links if leaf.index == 0)))
