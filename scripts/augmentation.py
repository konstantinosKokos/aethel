from LassyExtraction.transformations import DAG, get_material
from LassyExtraction.mill.types import *


def extract_contiguous_subgraphs(dag: DAG[str]):
    """
        Extracts all contiguous phrases from a transformed graph.
    """
    for node in dag.nodes - dag.get_roots() - dag.get_leaves():
        subgraph = dag.get_rooted_subgraph(node)
        if not any(get_material(subgraph, leaf) is None for leaf in subgraph.get_leaves()):
            subgraph.meta['name'] = subgraph.meta['name'] + f'({node})'
            yield subgraph


def mod_combinations(proof: T) -> list[T]:
    def is_mod(_proof: T) -> bool:
        if _proof.rule == Proof.Rule.BoxElimination and (deco := _proof.decoration) in {'mod', 'app'}:
            return not (_proof.body.rule == Proof.Rule.DiamondElimination and _proof.body.decoration == deco)
        return False

    match proof.rule:
        case Proof.Rule.Axiom: return [proof]
        case Proof.Rule.Lexicon: return [proof]
        case Proof.Rule.ArrowElimination:
            ret = [Proof.apply(f, a)
                   for f in mod_combinations(proof.function) for a in mod_combinations(proof.argument)]
            if is_mod(proof.function):
                ret.append(proof.argument)
            return ret
        case Proof.Rule.ArrowIntroduction:
            return [Proof.abstract(proof.variable, body) for body in mod_combinations(proof.body)]
        case Proof.Rule.BoxElimination:
            return [Proof.unbox(body) for body in mod_combinations(proof.body)]
        case Proof.Rule.BoxIntroduction:
            return [Proof.box(proof.box, body) for body in mod_combinations(proof.body)]
