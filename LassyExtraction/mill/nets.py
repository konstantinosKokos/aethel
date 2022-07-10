from __future__ import annotations

import pdb

from .types import Type, Atom, Functor, Box, Diamond
from .proofs import Proof, Rule, Structural, Logical, constant, variable, deep_extract
from .terms import Variable
# from .terms import (Term, Variable, Constant, ArrowElimination, ArrowIntroduction,
#                     DiamondIntroduction, DiamondElimination, BoxIntroduction, BoxElimination)
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
        trees, index = [], 0
        for _type in types:
            tree, index = FormulaTree.from_type(_type, True, index)
            trees.append(tree)
        return trees, index

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

    def __init__(self, atom: str, polarity: bool, index: int):
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


def tree_to_type(tree: FormulaTree) -> Type:
    match tree:
        case LeafFT(a, _, _): return Atom(a)
        case UnaryFT('box', c, decoration, _): return Box(decoration, tree_to_type(c))
        case UnaryFT('diamond', c, decoration, _): return Diamond(decoration, tree_to_type(c))
        case BinaryFT(l, r, _): return Functor(tree_to_type(l), tree_to_type(r))
        case _: raise ValueError(f'{tree} is not a formula tree')


def type_to_tree(_type: Type, polarity: bool = True,  index: int = 0, step: int = 1) -> tuple[FormulaTree, int]:
    match _type:
        case Atom(a): return LeafFT(a, polarity, index), index + step
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


def match(left: FormulaTree, right: FormulaTree) -> dict[LeafFT, LeafFT]:
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


def flip(tree: FormulaTree) -> FormulaTree:
    match tree:
        case LeafFT(atom, index, polarity): return LeafFT(atom, not polarity, index)
        case UnaryFT(mod, content, decoration, _): return UnaryFT(mod, flip(content), decoration)
        case BinaryFT(left, right, _): return BinaryFT(flip(left), flip(right))


def beta_norm_links(_links: dict[LeafFT, LeafFT]) -> dict[LeafFT, LeafFT]:
    detours = {(x, y) for x, y in product(_links.items(), _links.items())
               if x[0].index == y[1].index and x[0].index < 0}
    if not detours:
        return _links
    long_links = {x for x, _ in detours} | {y for _, y in detours}
    norm_links = {y[0]: x[1] for x, y in detours}
    next_round = {x: y for x, y in _links.items() if (x, y) not in long_links} | norm_links
    return beta_norm_links(next_round)


def proof_to_links(proof: Proof) -> tuple[dict[LeafFT, LeafFT], dict[int, FormulaTree], FormulaTree]:
    lex_trees, index = {}, 0
    for _constant in sorted(proof.term.constants(), key=lambda t: t.index):
        formula_tree, index = type_to_tree(_constant.type, True, index)
        lex_trees[_constant.index] = formula_tree
    conclusion, index = type_to_tree(proof.term.type, False, index)

    def go(_proof: Proof,
           var_trees: dict[int, FormulaTree],
           _index: int) -> tuple[dict[LeafFT, LeafFT], FormulaTree, int]:
        match _proof.rule:
            case Logical.Variable:
                var_tree = var_trees.pop(_proof.term.index)
                return {}, var_tree, _index
            case Logical.Constant:
                return {}, lex_trees[_proof.term.index], _index
            case Logical.ArrowElimination:
                (function, argument) = _proof.premises
                left_links, left_tree, _index = go(function, var_trees, _index)
                right_links, right_tree, _index = go(argument, var_trees, _index)
                assert isinstance(left_tree, BinaryFT)
                return left_links | right_links | match(left_tree.left, right_tree), left_tree.right, _index
            case Logical.ArrowIntroduction:
                (body,) = _proof.premises
                var_tree, _index = type_to_tree(_proof.focus.type, True, _index, step=-1)
                var_trees[_proof.focus.index] = var_tree
                _links, _tree, _index = go(body, var_trees, _index)
                return _links, BinaryFT(flip(var_tree), _tree), _index
            case Logical.BoxElimination:
                (body,) = _proof.premises
                _links, _tree, _index = go(body, var_trees, _index)
                assert isinstance(_tree, UnaryFT) and _tree.modality == 'box'
                return _links, _tree.content, _index
            case Logical.BoxIntroduction:
                (body,) = _proof.premises
                (struct,) = body.structure
                _links, _tree, _index = go(body, var_trees, _index)
                return _links, UnaryFT('box', _tree, struct.brackets), _index
            case Logical.DiamondIntroduction:
                (body,) = _proof.premises
                (struct,) = _proof.structure
                _links, _tree, _index = go(body, var_trees, _index)
                return _links, UnaryFT('diamond', _tree, struct.brackets), _index
            case Logical.DiamondElimination:
                (original, becomes) = _proof.premises
                _becomes_links, _becomes_tree, _index = go(becomes, var_trees, _index)
                assert isinstance(_becomes_tree, UnaryFT) and _becomes_tree.modality == 'diamond'
                var_trees[_proof.focus.index] = _becomes_tree.content
                _original_links, _original_tree, _index = go(original, var_trees, _index)
                return _original_links | _becomes_links, _original_tree, _index
            case Structural.Extract:
                (body,) = _proof.premises
                return go(body, var_trees, _index)
            case _:
                raise ValueError

    links, tree, _ = go(proof, {}, -1)
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


def links_to_proof(links: dict[LeafFT, LeafFT], lex_trees: dict[int, FormulaTree], conclusion: FormulaTree) -> Proof:
    i = -1
    var_trees = {(i := i - 1): par
                 for _, tree in [*sorted(lex_trees.items(), key=lambda x: x[0]), (None, conclusion)]
                 for par in par_trees(tree, False)}
    atom_to_word = {atom_idx: w_idx for w_idx, tree in lex_trees.items()
                    for atom_idx in reachable_positives(tree)}
    atom_to_var = {atom_idx: var_idx for var_idx, tree in var_trees.items()
                   for atom_idx in reachable_positives(tree)}

    # def printy(f):
    #     def g(*args):
    #         print('=' * 80)
    #         print(f'Called with {args}')
    #         ret = f(*args)
    #         print(f'Returned {ret} for {args}\n')
    #         return ret
    #     return g

    def go_neg(tree: FormulaTree) -> Proof:
        assert not tree.polarity
        match tree:
            case LeafFT(_, _): return go_pos(links[tree])  # type: ignore
            case UnaryFT('box', content, decoration, _): return go_neg(content).box(decoration)
            case UnaryFT('diamond', content, decoration, _): return go_neg(content).diamond(decoration)
            case BinaryFT(left, right, _):
                body = go_neg(right)
                var_type = tree_to_type(left)
                if isinstance(var_type, Diamond) and var_type.decoration == 'x':
                    # deferred diamond elimination
                    var_type = var_type.content
                    abstraction = Variable(var_type, abs(next(k for k in var_trees if var_trees[k] == left)))
                    body, abstraction = deep_extract(body, abstraction)
                else:
                    abstraction = Variable(var_type, abs(next(k for k in var_trees if var_trees[k] == left)))
                return body.abstract(abstraction)

    # @printy
    def go_pos(tree: FormulaTree, grounding: tuple[int, FormulaTree] | None = None) -> Proof:
        assert tree.polarity

        if grounding is None:
            assert isinstance(tree, LeafFT)
            atom_idx = tree.index
            index = atom_to_word[atom_idx] if atom_idx in atom_to_word else atom_to_var[atom_idx]
            container = (lex_trees if index >= 0 else var_trees)[index]
            grounding = (index, container)
        else:
            index, container = grounding

        if tree == container:
            out_type = tree_to_type(tree)
            return (constant if index >= 0 else variable)(out_type, abs(index))
        rooted_in = rooting_branch(container, tree)
        match rooted_in:
            case UnaryFT('box', _, decoration, _):
                future = go_pos(rooted_in, grounding)
                match rooting_branch(container, rooted_in):
                    case UnaryFT('diamond', _, inner, _):
                        if inner == decoration:
                            return future
                return future.unbox(decoration)
            case UnaryFT('diamond', UnaryFT('box', nested, inner, _), outer, _):
                assert inner == outer
                body = go_pos(nested, (index, UnaryFT('box', nested, inner)))
                if outer == 'x':
                    assert rooted_in == container   # assert this is the bottom
                    return body                     # defer the elimination until X-rule
                becomes = go_pos(rooted_in, (index, container))
                _where = Variable(Box(inner, tree_to_type(nested)), abs(index))
                return body.undiamond(_where, becomes)
            case BinaryFT(left, _, True): return go_pos(rooted_in, grounding) @ go_neg(left)
            case BinaryFT(_, right, False): return go_pos(rooted_in, grounding) @ go_neg(right)
            case _: raise ValueError(f'{rooted_in} must be a formula Tree')
    return go_neg(conclusion).eta_norm()
