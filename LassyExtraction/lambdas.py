from LassyExtraction.milltypes import *
from typing import *
from dataclasses import dataclass

IntMapping = Dict[int, int]
StrMapping = Dict[str, str]

SUB = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
SUP = str.maketrans('abcdefghijklmnoprstuvwxyz1', 'ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ¹')
TYPES = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ1→', 'ᴀʙᴄᴅᴇғɢʜɪᴊᴋʟᴍɴᴏᴘǫʀsᴛᴜᴠᴡxʏᴢ1→')


def translate_id(idx: int) -> str:
    return str(idx).translate(SUB)


def translate_decoration(decoration: str) -> str:
    return decoration.lower().translate(SUP)


def nodecorate(*args) -> str:
    return ''


def translate_type(wordtype: WordType) -> str:
    return str(wordtype).upper().translate(TYPES)


@dataclass
class Node:
    idx: str
    name: str
    polarity: bool
    terminal: bool
    decoration: str = ''

    def __init__(self, idx: str, name: str, polarity: bool, terminal: bool, decoration: str = ''):
        self.idx = idx
        self.name = name
        self.polarity = polarity
        self.terminal = terminal
        self.decoration = decoration

    def __hash__(self):
        return (self.idx, self.name, self.polarity).__hash__()


class Graph(object):
    def __init__(self):
        self.nodes = set()
        self.intra_edges = set()
        self.inter_edges = set()

    def add_intra_graphs(self, words: strings, types: WordTypes):
        nodes, edges = list(zip(*list(map(make_intra_graphs, words, types))))
        self.nodes.update(set.union(*nodes))
        self.intra_edges.update(set.union(*edges))

    def add_inter_graphs(self, proofnet: IntMapping):
        self.inter_edges.update({(str(k), str(v)) for k, v in proofnet.items()})

    def get_node(self, idx: str) -> Node:
        nodes = list(filter(lambda node: node.idx == idx, self.nodes))
        assert len(nodes) == 1
        return nodes[0]

    def get_root(self, node: Node) -> Optional[Node]:
        node_ids = set(map(lambda edge: edge[0], filter(lambda edge: edge[1] == node.idx, self.intra_edges)))
        nodes = list(filter(lambda other: other.idx in node_ids, self.nodes))
        if len(nodes) == 1:
            return nodes[0]
        elif len(nodes) == 0:
            return None
        else:
            raise AssertionError('More than one root.')

    def get_daughters(self, node: Node) -> Set[Node]:
        node_ids = set(map(lambda edge: edge[1], filter(lambda edge: edge[0] == node.idx, self.intra_edges)))
        return set(filter(lambda other: other.idx in node_ids, self.nodes))

    def get_pos_root(self, node: Node) -> Optional[Node]:
        root = self.get_root(node)
        if root is None or not root.polarity:
            return None
        return root

    def get_neg_daughter(self, node: Node) -> Optional[Node]:
        daughters = self.get_daughters(node)
        neg_daughters = list(filter(lambda daughter: not daughter.polarity, daughters))
        if len(neg_daughters) == 1:
            return neg_daughters[0]
        return None


def make_graph(words: strings, premises: WordTypes, conclusion: WordType, add_types: bool) -> Graph:
    graph = Graph()
    words = list(map(lambda w, t:
                     f'{w}::{translate_type(t.depolarize().decolor())}' if add_types else f'{w}',
                     words,
                     premises + [conclusion]))
    graph.add_intra_graphs(words, premises + [conclusion])
    return graph


def get_type_indices(wordtype: WordType) -> List[int]:
    pos, neg = get_polarities_and_indices(wordtype)
    return sorted(reduce(add, map(lambda x: [x[1]], pos), []) + reduce(add, map(lambda x: [x[1]], neg), []))


def make_atomic_nodes(types: WordTypes) -> Set[AtomicType]:
    return set.union(*list(map(get_atomic, types)))


def make_intra_graphs(word: str, wordtype: WordType, polarity: bool = True, parent: Optional[str] = None) \
        -> Tuple[Set[Node], Set[Tuple[str, str]]]:
    if isinstance(wordtype, AtomicType):
        return ({Node(idx=str(wordtype.index), polarity=wordtype.polarity, name=word, terminal=True)},
                {(parent, str(wordtype.index))} if parent is not None else set())
    else:
        # get identifiers for each subtree
        left_id = '.'.join(list(map(str, get_type_indices(wordtype.argument))))
        right_id = '.'.join(list(map(str, get_type_indices(wordtype.result))))
        # init the implication node
        node = Node(idx=f'{left_id} | {right_id}', polarity=polarity, name=word, terminal=False,
                    decoration=get_decoration(wordtype))
        edge = {(parent, node.idx)} if parent is not None else set()
        left_nodes, left_edges = make_intra_graphs(word, wordtype.argument, not polarity, node.idx)
        right_nodes, right_edges = make_intra_graphs(word, wordtype.result, polarity, node.idx)
        return {node}.union(left_nodes).union(right_nodes), edge.union(left_edges).union(right_edges)


def get_decoration(functor: FunctorType):
    return functor.diamond if isinstance(functor, DiamondType) else functor.box if isinstance(functor, BoxType) else ''


def traverse(graph: Graph, idx: str, forward_dict: StrMapping, backward_dict: StrMapping, upward: bool,
             varcount: int, add_dependencies: bool) -> Tuple[str, int]:
    if add_dependencies:
        decorate = translate_decoration
    else:
        decorate = nodecorate
    node = graph.get_node(idx)
    if upward:
        if node.terminal:
            return traverse(graph, backward_dict[idx], forward_dict, backward_dict, False, varcount, add_dependencies)
        else:
            ret = f'λx{translate_id(varcount)}.'
            varcount += 1
            ret2, varcount = traverse(graph, graph.get_neg_daughter(node).idx, forward_dict,
                                      backward_dict, True, varcount, add_dependencies)
            return ret+ret2, varcount
    else:
        proot = graph.get_pos_root(node)
        root = graph.get_root(node)
        if proot is None:
            if node.terminal:
                if root:
                    return f'x{translate_id(varcount - 1)}', varcount - 1
                return f' {node.name}', varcount
            else:
                assert node.polarity
                neg_daughter = graph.get_neg_daughter(node)
                if neg_daughter is None:
                    raise AssertionError
                ret, varcount = traverse(graph, neg_daughter.idx, forward_dict, backward_dict,
                                         True, varcount, add_dependencies)
                if node.decoration in {'mod', 'app', 'predm', 'det'}:
                    return (f'({node.name}{decorate(node.decoration)} {ret})', varcount) if not root else \
                        (f'(x{translate_id(varcount - 1)}{decorate(node.decoration)} {ret})', varcount - 1)
                return (f'({node.name} {ret}{decorate(node.decoration)})', varcount) if not root \
                    else (f'(x{translate_id(varcount - 1)} {ret}{decorate(node.decoration)})', varcount)
        else:
            if node.terminal:
                return traverse(graph, proot.idx, forward_dict, backward_dict, False, varcount, add_dependencies)
            else:
                daughter = graph.get_neg_daughter(node)
                ret_left, varcount = traverse(graph, proot.idx, forward_dict, backward_dict,
                                              False, varcount, add_dependencies)
                ret_right, varcount = traverse(graph, daughter.idx, forward_dict, backward_dict,
                                               True, varcount, add_dependencies)
                if node.decoration in {'mod', 'app', 'predm', 'det'}:
                    ret_left += decorate(node.decoration)
                else:
                    ret_right += decorate(node.decoration)
                return f'({ret_left} {ret_right})', varcount
