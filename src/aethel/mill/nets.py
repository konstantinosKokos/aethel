from __future__ import annotations

from .types import Type, Atom, Functor, Box, Diamond
from .proofs import Proof, Structural, Logical, constant, variable, deep_extract
from .terms import Variable
from typing import Literal
from abc import ABC, abstractmethod


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


def flip(tree: FormulaTree) -> FormulaTree:
    match tree:
        case LeafFT(atom, index, polarity): return LeafFT(atom, index, not polarity)
        case UnaryFT(mod, content, decoration, _): return UnaryFT(mod, flip(content), decoration)
        case BinaryFT(left, right, _): return BinaryFT(flip(left), flip(right))


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


########################################################################################################################
# N.D. <---> Proof Net
########################################################################################################################


def beta_norm_links(links: AxiomLinks) -> AxiomLinks:
    def follow_link(positive: LeafFT):
        if positive.index >= 0:
            return positive
        return follow_link(links[LeafFT(positive.atom, positive.index, False)])
    return {k: follow_link(v) for k, v in links.items() if k.index >= 0}


def proof_to_links(proof: Proof) -> tuple[AxiomLinks, dict[int, FormulaTree], FormulaTree]:
    lex_trees, _index = {}, 0
    for _constant in sorted(proof.term.constants(), key=lambda t: t.index):
        formula_tree, _index = type_to_tree(_constant.type, True, _index)
        lex_trees[_constant.index] = formula_tree
    conclusion, _index = type_to_tree(proof.term.type, False, _index)

    def go(_proof: Proof,
           index: int,
           var_trees: dict[int, FormulaTree]) -> tuple[AxiomLinks, int, FormulaTree]:

        match _proof.rule:
            case Logical.Variable:
                return {}, index, var_trees[_proof.term.index]
            case Logical.Constant:
                return {}, index, lex_trees[_proof.term.index]
            case Logical.ArrowIntroduction:
                (body,) = _proof.premises
                focus = _proof.focus
                var_tree, index = type_to_tree(focus.type, True, index, -1)
                var_trees = {**var_trees, focus.index: var_tree}
                links, index, result = go(body, index, var_trees)
                return links, index, BinaryFT(flip(var_tree), result)
            case Logical.ArrowElimination:
                (fn, arg) = _proof.premises
                fn_links, index, fn_tree = go(fn, index, var_trees)
                arg_links, index, arg_tree = go(arg, index, var_trees)
                assert isinstance(fn_tree, BinaryFT)
                return fn_links | arg_links | match(fn_tree.left, arg_tree), index, fn_tree.right
            case Logical.DiamondElimination:
                (original, becomes) = _proof.premises
                focus = _proof.focus
                becomes_links, index, becomes_tree = go(becomes, index, var_trees)
                assert isinstance(becomes_tree, UnaryFT) and becomes_tree.modality == 'diamond'
                var_trees = {**var_trees, focus.index: becomes_tree.content}
                original_links, index, original_tree = go(original, index, var_trees)
                return original_links | becomes_links, index, original_tree
            case Logical.BoxElimination:
                (body,) = _proof.premises
                links, index, result = go(body, index, var_trees)
                assert isinstance(result, UnaryFT) and result.modality == 'box'
                return links, index, result.content
            case Logical.DiamondIntroduction:
                (body,) = _proof.premises
                (struct,) = _proof.structure
                links, index, result = go(body, index, var_trees)
                return links, index, UnaryFT('diamond', result, struct.brackets)
            case Logical.BoxIntroduction:
                (body,) = _proof.premises
                (struct,) = body.structure
                links, index, result = go(body, index, var_trees)
                return links, index, UnaryFT('box', result, struct.brackets)
            case Structural.Extract:
                (body,) = _proof.premises
                return go(body, index, var_trees)
            case _:
                raise ValueError

    axiom_links, _, output = go(proof, -1, {})
    axiom_links |= match(output, conclusion)
    return beta_norm_links(axiom_links), lex_trees, conclusion


def single_var_proof(proof: Proof) -> bool:
    return proof.rule == Logical.Variable or (len(proof.premises) == 1 and single_var_proof(proof.premises[0]))


def links_to_proof(links: AxiomLinks, lex_to_tree: dict[int, FormulaTree], conclusion: FormulaTree) -> Proof:
    tree_to_lex, i = {v: k for k, v in lex_to_tree.items()}, 0
    var_to_tree: dict[int, FormulaTree] = {(i := i-1): par
                                           for _, tree in [*sorted(lex_to_tree.items(), key=lambda x: x[0]),
                                                           (None, conclusion)]
                                           for par in par_trees(tree, False)}
    tree_to_var: dict[FormulaTree, int] = {v: k for k, v in var_to_tree.items()}
    atom_to_var = {k: index for index in var_to_tree for k in reachable_positives(var_to_tree[index])}
    atom_to_lex: dict[int, int] = {k: index for index in lex_to_tree for k in reachable_positives(lex_to_tree[index])}
    inv_links = {k: v for v, k in links.items()}

    def mirror(tree: FormulaTree) -> FormulaTree:
        if isinstance(tree, LeafFT):
            return links[tree] if not tree.polarity else inv_links[tree]
        match tree:
            case UnaryFT(modality, content, decoration, _):
                return UnaryFT(modality, mirror(content), decoration)
            case BinaryFT(left, right, _):
                return BinaryFT(mirror(left), mirror(right))
            case _:
                raise ValueError

    def try_direct_match(tree: FormulaTree) -> Proof | None:
        if tree in tree_to_lex:
            return constant(tree_to_type(tree), tree_to_lex[tree])
        if tree in tree_to_var:
            return variable(tree_to_type(tree), abs(tree_to_var[tree]))

    def go_neg(tree: FormulaTree) -> Proof:
        assert not tree.polarity
        mirror_image = mirror(tree)
        return step_neg(tree, mirror_image)

    def go_pos(tree: FormulaTree) -> Proof:
        assert tree.polarity and isinstance(tree, LeafFT)
        atom_index = tree.index
        is_var = atom_index in atom_to_var
        index = (atom_to_var if is_var else atom_to_lex)[atom_index]
        tree = (var_to_tree if is_var else lex_to_tree)[index]
        if isinstance(tree, UnaryFT) and tree.modality == 'diamond' and tree.decoration == 'x':
            tree = tree.content
        ground = (variable if is_var else constant)(tree_to_type(tree), abs(index))
        return step_pos(tree, ground)

    def step_neg(tree: FormulaTree, mirror_image: FormulaTree) -> Proof:
        nonlocal i
        if (direct_match := try_direct_match(mirror_image)) is not None:
            return direct_match

        if isinstance(tree, LeafFT):
            return go_pos(links[tree])
        match tree, mirror_image:
            case BinaryFT(left, right, _), BinaryFT(_, mirror_right, _):
                body = step_neg(right, mirror_right)
                var_type = tree_to_type(left)
                if isinstance(var_type, Diamond) and var_type.decoration == 'x':
                    var_type = var_type.content
                    abstraction = Variable(var_type, abs(tree_to_var[left]))
                    body, abstraction = deep_extract(body.eta_norm(), abstraction, renaming=abs(i := i - 1))
                else:
                    abstraction = Variable(var_type, abs(tree_to_var[left]))
                return body.abstract(abstraction)
            case UnaryFT('diamond', content, decoration, _), UnaryFT(_, mirror_content, _, _):
                ret = step_neg(content, mirror_content)
                return ret if single_var_proof(ret) else ret.diamond(decoration)
            case UnaryFT('box', content, decoration, _), UnaryFT(_, mirror_content, _, _):
                return step_neg(content, mirror_content).box(decoration)
            case _:
                raise ValueError

    def step_pos(tree: FormulaTree,
                 context: Proof,
                 cut: tuple[Variable, Proof] | None = None) -> Proof:
        nonlocal i
        match tree:
            case LeafFT(_, _, _):
                return context
            case UnaryFT('diamond', content, decoration, _):
                match content:
                    case UnaryFT('box', _, inner, _):
                        assert inner == decoration
                        nested_var = Variable(tree_to_type(content), abs(i := i - 1))
                        nested_proof = Logical.Variable(nested_var)
                        return step_pos(content, nested_proof, (nested_var, context))
                    case LeafFT(_, _, _):
                        return context
                    case _:
                        ret = try_direct_match(tree)
                        assert ret is not None
                        return ret
            case UnaryFT('box', content, decoration, _):
                return step_pos(content, context.unbox(decoration), cut)
            case BinaryFT(left, right, _):
                if cut is not None:
                    focus, becomes = cut
                    context = context.undiamond(focus, becomes)
                return step_pos(right, context @ go_neg(left))
            case _:
                raise ValueError

    return go_neg(conclusion).beta_norm().eta_norm().standardize_vars()
