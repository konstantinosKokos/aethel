import pdb

from .types import Type, Atom, Functor, Box, Diamond, Proof, Modal, T
from typing import NamedTuple


class Leaf(NamedTuple):
    atom: Atom
    polarity: bool
    index: int

    def __repr__(self):
        return f'{self.atom}({"+" if self.polarity else "-"},{self.index})'


class Unary(NamedTuple):
    polarity: bool
    modality: str
    decoration: str
    content: 'Tree'


class Binary(NamedTuple):
    polarity: bool
    left: 'Tree'
    right: 'Tree'


Tree = Leaf | Unary | Binary


def type_to_tree(_type: Type, polarity: bool = True, index: int = 0, step: int = 1) -> tuple[Tree, int]:
    match _type:
        case Atom(_):
            return Leaf(_type, polarity, index), index + step                                             # type: ignore
        case Functor(argument, result):
            left_tree, index = type_to_tree(argument, not polarity, index, step)
            right_tree, index = type_to_tree(result, polarity, index, step)
            return Binary(polarity, left_tree, right_tree), index
        case Box(decoration, content):
            content_tree, index = type_to_tree(content, polarity, index, step)
            return Unary(polarity, '□', decoration, content_tree), index
        case Diamond(decoration, content):
            content_tree, index = type_to_tree(content, polarity, index, step)
            return Unary(polarity, '◇', decoration, content_tree), index


def match_trees(left: Tree, right: Tree) -> dict[Leaf, Leaf]:
    match left, right:
        case Leaf(latom, lpolarity, _), Leaf(ratom, rpolarity, _):
            assert latom == ratom and lpolarity != rpolarity
            return {right: left} if lpolarity else {left: right}
        case Unary(lpolarity, lmodality, ldecoration, lcontent), Unary(rpolarity, rmodality, rdecoration, rcontent):
            assert lmodality == rmodality and ldecoration == rdecoration and lpolarity != rpolarity
            return match_trees(lcontent, rcontent)
        case Binary(lpolarity, lleft, lright), Binary(rpolarity, rleft, rright):
            assert lpolarity != rpolarity
            left_mapping = match_trees(lleft, rleft)
            right_mapping = match_trees(lright, rright)
            return left_mapping | right_mapping
        case _:
            raise ValueError(f'Cannot match trees: {left} and {right}')


def flip_polarity(tree: Tree) -> Tree:
    match tree:
        case Leaf(atom, polarity, index):
            return Leaf(atom, not polarity, index)
        case Unary(polarity, modality, decoration, content):
            return Unary(not polarity, modality, decoration, flip_polarity(content))
        case Binary(polarity, left, right):
            return Binary(not polarity, flip_polarity(left), flip_polarity(right))


def term_to_links(proof: T) -> tuple[dict[Leaf, Leaf], dict[int, Tree], dict[int, Tree]]:
    hypotheses, constants = proof.vars(), proof.constants()
    conclusion, index = type_to_tree(type(proof), False)
    lex_trees, ax_trees = {}, {}
    for term in constants:
        formula_tree, index = type_to_tree(type(term), True, index, 1)
        lex_trees[term.constant] = formula_tree
    index = -1
    for term in hypotheses:
        formula_tree, index = type_to_tree(type(term), True, index, -1)
        ax_trees[term.variable] = formula_tree

    def f(_proof: Proof) -> tuple[dict[Leaf, Leaf], Tree]:
        match _proof.rule:
            case Proof.Rule.Lexicon: return {}, lex_trees[_proof.constant]
            case Proof.Rule.Axiom: return {}, ax_trees[_proof.variable]
            case Proof.Rule.ArrowElimination:
                left_links, (_, left_match, rem) = f(_proof.function)
                right_links, right_match = f(_proof.argument)
                return left_links | right_links | match_trees(left_match, right_match), rem
            case Proof.Rule.ArrowIntroduction:
                (body_links, tree), variable = f(_proof.body), _proof.abstraction.variable
                return body_links, Binary(tree.polarity, flip_polarity(ax_trees[variable]), tree)
            case Proof.Rule.BoxElimination | Proof.Rule.DiamondElimination:
                internal_links, tree = f(_proof.body)
                return internal_links, tree.content
            case Proof.Rule.BoxIntroduction:
                (internal_links, tree), decoration = f(_proof.body), _proof.decoration
                return internal_links, Unary(tree.polarity, '□', decoration, tree)
            case Proof.Rule.DiamondIntroduction:
                (internal_links, tree), decoration = f(_proof.body), _proof.decoration
                return internal_links, Unary(tree.polarity, '◇', decoration, tree)

    links, output_tree = f(proof)
    links |= match_trees(output_tree, conclusion)

    def beta_norm(_links: dict[Leaf, Leaf]) -> dict[Leaf, Leaf]:
        detours = {(x, y) for x in _links.items() for y in _links.items()
                   if x[0].index == y[1].index and x[1].index > 0}
        beta_long_links = {x for x, _ in detours} | {y for _, y in detours}
        beta_norm_links = {y[0]: x[1] for x, y in detours}
        return (beta_norm({x: y for x, y in _links.items() if (x, y) not in beta_long_links} | beta_norm_links)
                if beta_long_links else _links)
    return beta_norm(links), lex_trees, ax_trees


def reachable_positives(tree: Tree) -> set[int]:
    match tree:
        case Leaf(_, _, index): return {index}
        case Unary(_, _, _, content): return reachable_positives(content)
        case Binary(True, _, right): return reachable_positives(right)
        case Binary(False, _, _): return set()
        case _: raise ValueError(f'{tree} must be a formula Tree')


def par_trees(tree: Tree, par: bool = False) -> list[Tree]:
    match tree:
        case Leaf(_, _, _): return [tree] if par else []
        case Unary(_, _, _, content): return ([tree] if par else []) + par_trees(content, False)
        case Binary(polarity, left, right):
            return ([tree] if par else []) + par_trees(left, not polarity) + par_trees(right, False)
        case _: raise ValueError(f'{tree} must be a formula Tree')


def tree_to_type(tree: Tree) -> T:
    match tree:
        case Leaf(atom, _, _): return atom
        case Unary(_, '□', decoration, content): return Box(decoration, tree_to_type(content))
        case Unary(_, '◇', decoration, content): return Diamond(decoration, tree_to_type(content))
        case Binary(_, left, right): return Functor(tree_to_type(left), tree_to_type(right))
        case _: raise ValueError(f'{tree} must be a formula Tree')


def rooting_branch(container: Tree, subtree: Tree) -> Tree | None:
    match container:
        case Unary(_, _, _, content):
            return container if content == subtree else rooting_branch(content, subtree)
        case Binary(_, left, right):
            return (container if left == subtree or right == subtree
                    else rooting_branch(left, subtree) or rooting_branch(right, subtree))


def links_to_term(
        links: dict[Leaf, Leaf], formula_assignments: dict[int, Tree]) -> T:
    i = -1
    hypotheses = {(i := i-1): par for key in sorted(formula_assignments)
                  for par in par_trees(formula_assignments[key])}
    atom_to_word = {atom_idx: w_idx for w_idx, tree in formula_assignments.items()
                    for atom_idx in reachable_positives(tree)}
    atom_to_var = {atom_idx: var_idx for var_idx, tree in hypotheses.items()
                   for atom_idx in reachable_positives(tree)}

    def negative_traversal(negative_tree: Tree) -> T:
        assert not negative_tree.polarity
        match negative_tree:
            case Leaf(_, _, _): return positive_traversal(links[negative_tree])
            case Unary(_, '□', decoration, content): return Proof.box(decoration, negative_traversal(content))
            case Unary(_, '◇', decoration, content): return Proof.diamond(decoration, negative_traversal(content))
            case Binary(_, left, right):
                abstraction = tree_to_type(left).var(abs(next(k for k in hypotheses if hypotheses[k] == left)))
                return Proof.abstract(abstraction, negative_traversal(right))

    def positive_traversal(positive_tree: Tree, grounding: tuple[int, Tree] | None = None) -> T:
        assert positive_tree.polarity

        if grounding is None:
            atom_idx = positive_tree.index
            index = atom_to_word[atom_idx] if atom_idx in atom_to_word else atom_to_var[atom_idx]
            container = formula_assignments[index] if index > 0 else hypotheses[index]
            grounding = (index, container)
        else:
            index, container = grounding

        if positive_tree == container:
            proof_type = tree_to_type(positive_tree)
            return proof_type.con(index) if index > 0 else proof_type.var(abs(index))
        rooted_in = rooting_branch(container, positive_tree)
        match rooted_in:
            case Unary(_, '□', _, _):
                return Proof.unbox(positive_traversal(rooted_in, grounding))
            case Unary(_, '◇', _, _):
                return Proof.undiamond(positive_traversal(rooted_in, grounding))
            case Binary(True, left, _):
                return Proof.apply(positive_traversal(rooted_in, grounding), negative_traversal(left))
            case Binary(False, _, right):
                return Proof.apply(positive_traversal(rooted_in, grounding), negative_traversal(right))
        pdb.set_trace()
        raise NotImplementedError
    return negative_traversal(next(iter(leaf for leaf in links if leaf.index == 0)))
