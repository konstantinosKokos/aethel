from __future__ import annotations

import pdb

from .types import Type, Atom, Functor, Box, Diamond
from .proofs import Proof, Rule, Structural, Logical, constant, variable, deep_extract
from .terms import Variable
from ..utils.tex import proof_to_tex, compile_tex
from typing import Literal
from abc import ABC, abstractmethod
from LassyExtraction.utils.tex import *


def panic(x):compile_tex(proof_to_tex(x))


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

    @abstractmethod
    def __repr__(self) -> str: ...
    @abstractmethod
    def __eq__(self, other: object) -> bool: ...
    @abstractmethod
    def __hash__(self) -> int: ...


class LeafFT(FormulaTree):
    __match_args__ = ('atom', 'index', 'polarity')

    atom:       str
    index:      int
    _polarity:   bool

    def __init__(self, atom: str, index: int, polarity: bool):
        self.atom = atom
        self.index = index
        self._polarity = polarity

    @property
    def polarity(self) -> bool: return self._polarity
    def __repr__(self) -> str: return f'{self.atom}({"+" if self.polarity else "-"},{self.index})'

    def __eq__(self, other) -> bool:
        match other:
            case LeafFT(a, i, p): return self.atom == a and self.index == i and self.polarity == p
            case _: return False

    def __hash__(self) -> int: return hash((self.atom, self.index, self.polarity))


Modality = Literal['box', 'diamond']
AxiomLinks = dict[LeafFT, LeafFT]


def modality_to_str(modality: Modality) -> str:
    match modality:
        case 'box': return '□'
        case 'diamond': return '◇'
        case _: raise ValueError(f'{modality} is not a modality')


class UnaryFT(FormulaTree):
    __match_args__ = ('modality', 'content', 'decoration', 'polarity')
    modality:   Modality
    decoration: str
    content:    FormulaTree

    def __init__(self, modality: Modality, content: FormulaTree, decoration: str):
        self.modality = modality
        self.decoration = decoration
        self.content = content

    @property
    def polarity(self) -> bool: return self.content.polarity
    def __repr__(self) -> str: return f'U({modality_to_str(self.modality)}{self.decoration}, {self.content})'

    def __eq__(self, other: object) -> bool:
        match other:
            case UnaryFT(m, c, d, p):
                return self.modality == m and self.content == c and self.decoration == d and self.polarity == p
            case _:
                return False

    def __hash__(self) -> int: return hash((self.modality, self.content, self.decoration, self.polarity))


class BinaryFT(FormulaTree):
    __match_args__ = ('left', 'right', 'polarity')

    left:   FormulaTree
    right:  FormulaTree

    def __init__(self, left: FormulaTree, right: FormulaTree):
        assert left.polarity == (not right.polarity)
        self.left = left
        self.right = right

    @property
    def polarity(self) -> bool: return self.right.polarity
    def __repr__(self) -> str: return f'B({self.left},{self.right})'

    def __eq__(self, other: object) -> bool:
        match other:
            case BinaryFT(l, r, p): return self.left == l and self.right == r and self.polarity == p
            case _: return False

    def __hash__(self) -> int: return hash((self.left, self.right, self.polarity))


def leaves(tree: FormulaTree) -> tuple[list[LeafFT], list[LeafFT]]:
    match tree:
        case LeafFT(_, _, polarity):
            return ([tree], []) if polarity else ([], [tree])
        case UnaryFT(_, content, _, _):
            return leaves(content)
        case BinaryFT(left, right, _):
            lp, ln = leaves(left)
            rp, rn = leaves(right)
            return lp + rp, ln + rn


def contains(tree: FormulaTree, content: FormulaTree) -> bool:
    if tree == content:
        return True
    match tree:
        case BinaryFT(left, right, _): return contains(left, content) or contains(right, content)
        case UnaryFT(_, nested, _, _): return contains(nested, content)
        case _: return False


def tree_to_type(tree: FormulaTree) -> Type:
    match tree:
        case LeafFT(a, _, _): return Atom(a)
        case UnaryFT('box', c, decoration, _): return Box(decoration, tree_to_type(c))
        case UnaryFT('diamond', c, decoration, _): return Diamond(decoration, tree_to_type(c))
        case BinaryFT(l, r, _): return Functor(tree_to_type(l), tree_to_type(r))
        case _: raise ValueError(f'{tree} is not a formula tree')


def type_to_tree(_type: Type, polarity: bool = True,  index: int = 0, step: int = 1) -> tuple[FormulaTree, int]:
    match _type:
        case Atom(a): return LeafFT(a, index, polarity), index + step
        case Functor(left, right):
            ltree, lindex = type_to_tree(left, not polarity, index, step)
            rtree, rindex = type_to_tree(right, polarity, lindex, step)
            return BinaryFT(ltree, rtree), rindex
        case Box(decoration, content):
            ctree, index = type_to_tree(content, polarity, index, step)
            return UnaryFT('box', ctree, decoration), index
        case Diamond(decoration, content):
            ctree, index = type_to_tree(content, polarity, index, step)
            return UnaryFT('diamond', ctree, decoration), index
        case _: raise ValueError(f'{_type} is not a type')


def match(left: FormulaTree, right: FormulaTree) -> AxiomLinks:
    match left, right:
        case LeafFT(l_atom, _, l_polarity), LeafFT(r_atom, _, r_polarity):
            assert l_atom == r_atom and l_polarity == (not r_polarity)
            return {right: left} if l_polarity else {left: right}  # type: ignore
        case UnaryFT(l_mod, l_content, l_deco, _), UnaryFT(r_mod, r_content, r_deco, _):
            assert l_mod == r_mod and l_deco == r_deco
            return match(l_content, r_content)
        case BinaryFT(l_left, l_right, _), BinaryFT(r_left, r_right, _):
            return match(l_left, r_left) | match(l_right, r_right)
        case _:
            raise ValueError


def mirrors(negative: FormulaTree, positive: FormulaTree, links: AxiomLinks) -> bool:
    assert not negative.polarity
    match negative, positive:
        case LeafFT(_, _, _), LeafFT(_, _, _):
            return links[negative] == positive  # type: ignore
        case UnaryFT(l_mod, l_content, l_deco, _), UnaryFT(r_mod, r_content, r_deco, _):
            return l_mod == r_mod and l_deco == r_deco and mirrors(l_content, r_content, links)
        case BinaryFT(l_left, l_right, False), BinaryFT(r_left, r_right, True):
            return mirrors(r_left, l_left, links) and mirrors(l_right, r_right, links)
        case _:
            return False


def flip(tree: FormulaTree) -> FormulaTree:
    match tree:
        case LeafFT(atom, index, polarity): return LeafFT(atom, index, not polarity)
        case UnaryFT(mod, content, decoration, _): return UnaryFT(mod, flip(content), decoration)
        case BinaryFT(left, right, _): return BinaryFT(flip(left), flip(right))


def beta_norm_links(links: AxiomLinks) -> AxiomLinks:
    def follow_link(positive: LeafFT):
        if positive.index >= 0:
            return positive
        return follow_link(links[LeafFT(positive.atom, positive.index, False)])
    return {k: follow_link(v) for k, v in links.items() if k.index >= 0}


def proof_to_links(proof: Proof) -> tuple[AxiomLinks, dict[int, FormulaTree], FormulaTree]:
    lex_trees, index = {}, 0
    for _constant in sorted(proof.term.constants(), key=lambda t: t.index):
        formula_tree, index = type_to_tree(_constant.type, True, index)
        lex_trees[_constant.index] = formula_tree
    conclusion, index = type_to_tree(proof.term.type, False, index)

    def go(_proof: Proof,
           var_trees: dict[int, FormulaTree],
           _index: int) -> tuple[AxiomLinks, FormulaTree, int, dict[int, FormulaTree]]:
        match _proof.rule:
            case Logical.Variable:
                var_tree = var_trees[_proof.term.index]
                return {}, var_tree, _index, {k: v for k, v in var_trees.items() if k != _proof.term.index}
            case Logical.Constant:
                return {}, lex_trees[_proof.term.index], _index, var_trees
            case Logical.ArrowElimination:
                (function, argument) = _proof.premises
                left_links, left_tree, _index, left_vars = go(function, var_trees, _index)
                right_links, right_tree, _index, right_vars = go(argument, var_trees, _index)
                assert isinstance(left_tree, BinaryFT)
                return (left_links | right_links | match(left_tree.left, right_tree),
                        left_tree.right,
                        _index,
                        left_vars | right_vars)
            case Logical.ArrowIntroduction:
                (body,) = _proof.premises
                var_tree, _index = type_to_tree(_proof.focus.type, True, _index, step=-1)
                var_trees = {**var_trees, _proof.focus.index: var_tree}
                _links, _tree, _index, var_trees = go(body, var_trees, _index)
                return _links, BinaryFT(flip(var_tree), _tree), _index, var_trees
            case Logical.BoxElimination:
                (body,) = _proof.premises
                _links, _tree, _index, var_trees = go(body, var_trees, _index)
                assert isinstance(_tree, UnaryFT) and _tree.modality == 'box'
                return _links, _tree.content, _index, var_trees
            case Logical.BoxIntroduction:
                (body,) = _proof.premises
                (struct,) = body.structure
                _links, _tree, _index, var_trees = go(body, var_trees, _index)
                return _links, UnaryFT('box', _tree, struct.brackets), _index, var_trees
            case Logical.DiamondIntroduction:
                (body,) = _proof.premises
                (struct,) = _proof.structure
                _links, _tree, _index, var_trees = go(body, var_trees, _index)
                return _links, UnaryFT('diamond', _tree, struct.brackets), _index, var_trees
            case Logical.DiamondElimination:
                (original, becomes) = _proof.premises
                _becomes_links, _becomes_tree, _index, var_trees = go(becomes, var_trees, _index)
                assert isinstance(_becomes_tree, UnaryFT) and _becomes_tree.modality == 'diamond'
                var_trees = {**var_trees, _proof.focus.index: _becomes_tree.content}
                _original_links, _original_tree, _index, var_trees = go(original, var_trees, _index,)
                return _original_links | _becomes_links, _original_tree, _index, var_trees
            case Structural.Extract:
                (body,) = _proof.premises
                return go(body, var_trees, _index)
            case _:
                raise ValueError

    links, tree, _, _ = go(proof, {}, -1)
    links |= match(tree, conclusion)
    return beta_norm_links(links), lex_trees, conclusion


def reachable_positives(tree: FormulaTree) -> set[int]:
    match tree:
        case LeafFT(_, index, _): return {index}
        case UnaryFT(_, content, _, _): return reachable_positives(content)
        case BinaryFT(_, right, True): return reachable_positives(right)
        case BinaryFT(_, _, False): return set()
        case _: raise ValueError(f'{tree} must be a formula Tree')


def par_trees(tree: FormulaTree, par: bool) -> list[FormulaTree]:
    match tree:
        case LeafFT(_, _, _):
            return [tree] * par
        case UnaryFT(_, content, _, _):
            return [tree] * par + par_trees(content, False)
        case BinaryFT(left, right, polarity):
            return [tree] * par + par_trees(left, not polarity) + par_trees(right, False)
        case _:
            raise ValueError


def rooting_branch(container: FormulaTree, subtree: FormulaTree) -> FormulaTree | None:
    match container:
        case UnaryFT(_, content, _, _):
            return container if content == subtree else rooting_branch(content, subtree)
        case BinaryFT(left, right, _):
            return (container if left == subtree or right == subtree
                    else
                    rooting_branch(left, subtree) or rooting_branch(right, subtree))
        case _:
            return None


def links_to_proof(links: AxiomLinks, lex_trees: dict[int, FormulaTree], conclusion: FormulaTree) -> Proof:
    i = 0
    var_trees = {(i := i - 1): par
                 for _, tree in [*sorted(lex_trees.items(), key=lambda x: x[0]), (None, conclusion)]
                 for par in par_trees(tree, False)}
    atom_to_word = {atom_idx: w_idx for w_idx, tree in lex_trees.items()
                    for atom_idx in reachable_positives(tree)}
    atom_to_var = {atom_idx: var_idx for var_idx, tree in var_trees.items()
                   for atom_idx in reachable_positives(tree)}

    def go_neg(tree: FormulaTree) -> Proof:
        assert not tree.polarity

        absolute_match = next(((k, v) for k, v in (lex_trees | var_trees).items() if mirrors(tree, v, links)), None)
        if absolute_match is not None:
            index, pos_tree = absolute_match
            return (constant if index >= 0 else variable)(tree_to_type(pos_tree), abs(index))
        return step_neg(tree)

    def step_neg(tree: FormulaTree) -> Proof:
        match tree:
            case LeafFT(_, _):
                assert isinstance(tree, LeafFT)
                return go_pos(links[tree])
            case UnaryFT('box', content, decoration, _):
                return step_neg(content).box(decoration)
            case UnaryFT('diamond', content, decoration, _):
                ret = step_neg(content)
                # todo: tidy this up
                if ret.rule == Logical.Variable or (ret.rule == Logical.BoxElimination and ret.premises[0].rule == Logical.Variable):
                    return ret
                return ret.diamond(decoration)
            case BinaryFT(left, right, _):
                body = step_neg(right)
                var_type = tree_to_type(left)
                if isinstance(var_type, Diamond) and var_type.decoration == 'x':
                    # deferred diamond elimination
                    var_type = var_type.content
                    abstraction = Variable(var_type, abs(next(k for k in var_trees if var_trees[k] == left)))
                    body, abstraction = deep_extract(body, abstraction)
                else:
                    abstraction = Variable(var_type, abs(next(k for k in var_trees if var_trees[k] == left)))
                return body.abstract(abstraction)

    def go_pos(leaf: FormulaTree) -> Proof:
        assert isinstance(leaf, LeafFT)
        atom_idx = leaf.index
        is_var = atom_idx in atom_to_var
        index = (atom_to_var if is_var else atom_to_word)[atom_idx]
        tree = (var_trees if is_var else lex_trees)[index]
        if isinstance(tree, UnaryFT) and tree.modality == 'diamond' and tree.decoration == 'x':
            tree = tree.content
        ground = (variable if is_var else constant)(tree_to_type(tree), abs(index))
        return step_pos(tree, ground, abs(index))

    def step_pos(tree: FormulaTree,
                 context: Proof,
                 index: int,
                 cut: tuple[Variable, Proof] | None = None) -> Proof:
        match tree:
            case LeafFT(_, _, _):
                return context
            case UnaryFT('box', content, decoration, _):
                return step_pos(content, context.unbox(decoration), index, cut)
            case UnaryFT('diamond', content, outer, _):
                match content:
                    case LeafFT(_, _, _):
                        return context
                    case UnaryFT('box', _, inner, _):
                        assert inner == outer
                        focus = Variable(tree_to_type(content), index)
                        becomes = context
                        return step_pos(content, variable(tree_to_type(content), index), index, (focus, becomes))
                    case BinaryFT(_, _, _):
                        # this should never happen
                        raise NotImplementedError
            case BinaryFT(left, right, _):
                if cut is not None:
                    focus, becomes = cut
                    context = context.undiamond(focus, becomes)
                return step_pos(right, context @ go_neg(left), index)
    return go_neg(conclusion).beta_norm().eta_norm().de_bruijn()
