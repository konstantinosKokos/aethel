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
        self.edges = set()

    def add_intra_graphs(self, words: strings, types: WordTypes):
        nodes, edges = list(zip(*list(map(make_intra_graphs, words, types))))
        self.nodes.update(set.union(*nodes))
        self.edges.update(set.union(*edges))

    def add_inter_graphs(self, proofnet: IntMapping):
        self.edges.update({(str(k), str(v)) for k, v in proofnet.items()})

    def get_node(self, idx: str) -> Node:
        nodes = list(filter(lambda node: node.idx == idx, self.nodes))
        assert len(nodes) == 1
        return nodes[0]

    def downward(self, node: Node) -> Set[Node]:
        node_ids = set(map(lambda edge: edge[0], filter(lambda edge: edge[1] == node.idx, self.edges)))
        nodes = set(filter(lambda other: other.idx in node_ids, self.nodes))
        return nodes

    def upward(self, node: Node) -> Set[Node]:
        node_ids = set(map(lambda edge: edge[1], filter(lambda edge: edge[0] == node.idx, self.edges)))
        nodes = set(filter(lambda other: other.idx in node_ids, self.nodes))
        return nodes

    def to_input(self, node: Node) -> Node:
        downward = self.downward(node)
        downward_input = list(filter(lambda other: other.polarity, downward))
        assert len(downward_input) == 1
        return downward_input[0]

    def to_output(self, node: Node) -> Node:
        upward = self.upward(node)
        upward_output = list(filter(lambda other: not other.polarity, upward))
        assert len(upward_output) == 1
        return upward_output[0]


def make_graph(words: strings, premises: WordTypes, conclusion: WordType) -> Graph:
    graph = Graph()
    words = list(map(lambda w, t:
                     f'{w}::{translate_type(t.depolarize().decolor())}',
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
    return functor.diamond if isinstance(functor, DiamondType) else functor.box if isinstance(functor, BoxType) else '→'


def traverse(graph: Graph, idx: str, forward_dict: StrMapping, backward_dict: StrMapping, upward: bool,
             varcount: int) -> Tuple[str, int]:
    node = graph.get_node(idx)
    if upward:
        # leaf case, cross to input
        if node.terminal:
            ret, varcount = traverse(graph, backward_dict[idx], forward_dict, backward_dict, not upward, varcount)
            return ret, varcount
        else:
            ret = f'λx{translate_id(varcount)}.'
            varcount += 1
            ret2, varcount = traverse(graph, graph.to_output(node).idx, forward_dict,
                                      backward_dict, upward, varcount)
            return ret+ret2, varcount
    else:
        # root case
        if not graph.downward(node):
            # terminal root case
            if node.terminal:
                ret = f' {node.name}'
                return ret, varcount
            else:
                # root impl case, move to other branch, switch mode
                if node.polarity:
                    ret = f'({node.name}'
                    ret2, varcount = traverse(graph, graph.to_output(node).idx, forward_dict,
                                              backward_dict, not upward, varcount)
                    if node.decoration in {'mod', 'app', 'predm', 'det'}:
                        return f'{ret}{translate_decoration(node.decoration)} {ret2})', varcount
                    else:
                        return f'{ret} {ret2}{translate_decoration(node.decoration)})', varcount
                else:
                    raise NotImplementedError
        else:
            if node.terminal:
                try:
                    ret, varcount = traverse(graph, graph.to_input(node).idx, forward_dict,
                                             backward_dict, upward, varcount)
                    return ret, varcount
                except AssertionError:
                    ret = f'x{translate_id(varcount - 1)}'
                    return ret, varcount
            else:
                ret2, varcount = traverse(graph, graph.to_input(node).idx, forward_dict,
                                          backward_dict, upward, varcount)
                ret3, varcount = traverse(graph, graph.to_output(node).idx, forward_dict,
                                          backward_dict, not upward, varcount)
                if node.decoration in {'mod', 'app', 'predm', 'det'}:
                    ret2 += translate_decoration(node.decoration)
                else:
                    ret3 += translate_decoration(node.decoration)
                return f'({ret2} {ret3})', varcount

