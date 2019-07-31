from src.WordType import *
from src.WordType import binarizer as ColoredType

import os
import xml.etree.cElementTree as ET
from glob import glob
from copy import deepcopy
import graphviz

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from collections import Counter
from itertools import groupby, chain

from typing import Optional, Iterable, Tuple, Union, Any, Generator, Dict, List, Callable, NamedTuple

from warnings import warn

Rel = NamedTuple('Rel', [('label', str), ('rank', str)])
ChildRel = Tuple[ET.Element, Union[Rel, str]]
ChildRels = List[ChildRel]
Grouped = Dict[ET.Element, ChildRels]

fst = lambda x: x[0]
snd = lambda x: x[1]


class Lassy(Dataset):
    """
        Lassy dataset. A wrapper that feeds samples into the extraction algorithm.
    """

    def __init__(self, root_dir: str = '/home/kokos/Documents/Projects/Lassy/LassySmall 4.0',
                 treebank_dir: str = '/Treebank', transform: Optional[Compose] = None,
                 ignore: Optional[str] = 'src/utils/ignored.txt') -> None:
        """
            Initialize a Lassy dataset.

        :param root_dir: The root directory containing all data and metadata files.
        :type root_dir: str
        :param treebank_dir: The subdirectory containing the parse trees in xml format.
        :type treebank_dir: test
        :param transform: A torchvision Compose object, specifying the transformations to be applied on the data.
        :type transform: Compose
        :param ignore: (Optional) Filename containing the xml files to be ignored.
        :type ignore: Optional[str]
        """

        if os.path.isdir(root_dir) and os.path.isdir(root_dir+treebank_dir):
            self.root_dir = root_dir # might be useful to store meta information
            self.treebank_dir = root_dir + treebank_dir
        else:
            raise ValueError('%s and %s must be existing directories' % (root_dir, treebank_dir))

        self.ignored = []
        if ignore is not None:
            try:
                with open(ignore, 'r') as f:
                    self.ignored = list(map(lambda x: x[0:-1], f.readlines()))
                    print('Ignoring {} samples..'.format(len(self.ignored)))
            except FileNotFoundError:
                warn('Could not open the ignore file.')

        self.filelist = [y for x in os.walk(self.treebank_dir) for y in glob(os.path.join(x[0], '*.[xX][mM][lL]'))
                         if y not in self.ignored]
        self.transform = transform

        print('Dataset constructed with {} samples.'.format(len(self.filelist)))

    def __len__(self) -> int:
        """
        :return: The number of xml files used by this dataset.
        :rtype: int
        """
        return len(self.filelist)

    def __getitem__(self, index: Union[int, str]) -> Any:
        """
            Itemgetter function, used to retrieve items from this dataset.

        :param index: The index or filename of the sample to be retrieved.
        :type index: Union[int, str]
        :return: If no transformation is provided, a tuple containing the index, the filename and the ElementTree parse
            of the sample. If a transformation is provided, returns the output of that transformation.
        :rtype: Any
        """

        if type(index) == int:
            file = self.filelist[index]
        elif type(index) == str:
            file = index
        else:
            raise TypeError('file argument has to be int or str')

        parse = ET.parse(file)
        parse.getroot().set('type', 'Tree')

        sample = (index, file, parse)

        if self.transform:
            return self.transform(sample)

        return sample

    @staticmethod
    def extract_nodes(xtree: ET.ElementTree) -> Generator[Tuple[ET.Element, Optional[ET.Element]], None, None]:
        """
            A simple iterator over an xml parse that returns the parse tree's nodes. This is necessary as the default \
        ET iterator does not provide parent or depth info.

        :param xtree: The ElementTree to extract the nodes from.
        :type xtree: ET.ElementTree
        :return: A generator yielding tuples of (child, parent) nodes.
        :rtype: Generator[Tuple[ET.Element, Optional[int]], None, None]
        """

        root = xtree.getroot().find('node')
        parents = [root]

        yield (root, None)
        while parents:
            children = []
            for parent in parents:
                for child in parent.findall('node'):
                    children.append(child)
                    yield (child, parent)
                parents = children

    @staticmethod
    def find_main_coindex(xtree: ET.ElementTree) -> Tuple[Dict[int, List[ET.Element]],
                                                          Dict[int, List[ET.Element]]]:
        """
            Takes an ElementTree representing a parse tree, finds out nodes corresponding to the same lexical unit
        and selects a single node to act as the "main" node (the one to collapse dependencies to, when removing
        its cross-references).

        :param xtree: The ElementTree to operate on.
        :type xtree: ET.ElementTree
        :return: Returns a tuple of dictionaries, the first mapping each co-indexing identifier to a list of nodes
            sharing it, and the second mapping each co-indexing identifier to the main node of the group.
        :rtype: Tuple[Dict[int, List[ET.Element, ...]], Dict[int, List[ET.Element, ...]]]
        """

        coindexed = list(filter(lambda x: 'index' in x.attrib.keys(), xtree.iter('node')))
        if not coindexed:
            return dict(), dict()
        all_coind = {i: [node for node in group] for i, group in
                     groupby(sorted(coindexed, key=lambda x: x.attrib['index']), key=lambda x: x.attrib['index'])}
        # find the 'main' child for each set of siblings
        main_coind = {i: list(filter(lambda x: 'cat' in x.attrib or 'word' in x.attrib, nodes))[0]
                      for i, nodes in all_coind.items()}
        return all_coind, main_coind

    @staticmethod
    def tree_to_dag(xtree: ET.ElementTree, inplace: bool=False) -> ET.ElementTree:
        """
            Takes an ElementTree representing a parse tree, possibly containing duplicate nodes (that is, nodes that
        correspond to the same lexical unit but with a different identifier to preserve the tree format). Removes
        duplicate nodes by constructing new dependency links between cross-references (moving to a DAG format), and
        returns the resulting ElementTree.

        :param xtree: The ElementTree to transform.
        :type xtree: ET.ElementTree
        :param inplace: Whether to do the transformation inplace or on a deep copy of the input.
        :type inplace: bool
        :return: The ElementTree with the ghost nodes replaced by secondary dependencies.
        :rtype: ET.ElementTree
        """

        if not inplace:
            xtree = deepcopy(xtree)

        nodes = list(Lassy.extract_nodes(xtree))

        _, main_coind = Lassy.find_main_coindex(xtree)

        for node, parent in nodes[1:]:
            if node in main_coind.values():
                node.attrib['rel'] = {parent.attrib['id']: Rel(label=node.attrib['rel'], rank='primary')}

        for node, parent in nodes[1:]:
            if isinstance(node.attrib['rel'], str):
                if 'index' in node.attrib.keys():
                    main_coind[node.attrib['index']].attrib['rel'] = {
                        parent.attrib['id']: Rel(label=node.attrib['rel'], rank='secondary'),
                        **main_coind[node.attrib['index']].attrib['rel']}
                    parent.remove(node)
                else:
                    node.attrib['rel'] = {parent.attrib['id']: node.attrib['rel']}
        return xtree


class Decompose:
    """
        Class wrapper implementing the functionalities of the extraction algorithm.
    """
    def __init__(self, unify: bool=False, separation_symbol: str='↔', visualize: bool=False,
                 text_pipeline: Callable[[str], str]=lambda x: x.lower()) -> None:
        """
            Initialize an extraction class.

        :param unify: If True, will return a single lexicon from a sentence. If False, will return a lexicon for each
            subunit.
        :type unify: bool
        :param return_lists: If False, will return a lexicon Dict[str, WordType]. If True, will return an iterable of
            words and iterable of WordTypes instead.
        :type return_lists: bool
        :param text_pipeline: A function applied on the text before adding a word to the lexicon.
        :type text_pipeline: Callable[[str], str]
        :param separation_symbol: The separation symbol between a node's text content and its id.
        :type separation_symbol: str
        :param visualize: If True, will visualize the input DAG.
        :type visualize: bool
        """
        # type_dict: POS → Type
        # removed: conj, top
        cat_dict = {'advp': 'ADV', 'ahi': 'AHI', 'ap': 'AP', 'cp': 'CP', 'detp': 'DETP', 'du': 'DU',
                    'inf': 'INF', 'np': 'NP', 'oti': 'OTI', 'pp': 'PP', 'ppart': 'PPART', 'ppres': 'PPRES',
                    'rel': 'REL', 'smain': 'SMAIN', 'ssub': 'SSUB', 'sv1': 'SV1', 'svan': 'SVAN', 'ti': 'TI',
                    'whq': 'WHQ', 'whrel': 'WHREL', 'whsub': 'WHSUB'}
        pos_dict = {'adj': 'ADJ', 'adv': 'ADV', 'comp': 'COMP', 'comparative': 'COMPARATIVE', 'det': 'DET',
                    'fixed': 'FIXED', 'name': 'NAME', 'noun': 'N', 'num': 'NUM', 'part': 'PART',
                    'prefix': 'PREFIX', 'prep': 'PREP', 'pron': 'PRON', 'punct': 'PUNCT', 'tag': 'TAG',
                    'verb': 'VERB', 'vg': 'VG'}
        pt_dict = {'adj': 'ADJ', 'bw': 'BW', 'let': 'LET', 'lid': 'LID', 'n': 'N', 'spec': 'SPEC', 'tsw': 'TSW',
                   'tw': 'TW', 'vg': 'VG', 'vnw': 'VNW', 'vz': 'VZ', 'ww': 'WW'}
        self.type_dict = {**cat_dict, **pt_dict}
        self.pos_set = 'pt'

        # convert to Types
        self.type_dict = {k: AtomicType(v) for k, v in self.type_dict.items()}
        self.text_pipeline = text_pipeline
        self.separation_symbol = separation_symbol
        # the function applied to convert the processed text of a node into a dictionary key
        self.get_key = lambda node: \
            self.text_pipeline(node.attrib['word']) + self.separation_symbol + node.attrib['id'] \
                if 'word' in node.attrib.keys() else node.attrib['id']
        self.head_candidates = ('hd', 'rhd', 'whd', 'cmp', 'crd', 'dlink')
        self.mod_candidates = ('mod', 'predm', 'app')
        self.visualize = visualize

    @staticmethod
    def is_leaf(node: ET.Element) -> bool:
        """

        :param node: The node to check.
        :type node: ET.Element
        :return: True if the node has a 'word' field, False otherwise.
        :rtype: bool
        """
        return True if 'word' in node.attrib.keys() else False

    def majority_vote(self, node: ET.Element, grouped: Grouped) -> str:
        """

        :param node:
        :type node:
        :param grouped:
        :type grouped:
        :return:
        :rtype:
        """
        sibling_types = [self.get_type_key(n[0], grouped) for n in grouped[node]]
        votes = {c: len([x for x in sibling_types if x == c]) for c in set(sibling_types)}
        votes = sorted(votes, key=lambda x: votes[x], reverse=True)

        return 'smain' if 'smain' in votes \
            else 'np' if 'np' in votes or 'n' in votes \
            else 'ap' if 'ap' in votes or 'adj' in votes \
            else votes[0]

    def get_type_key(self, node: ET.Element, grouped: Dict[ET.Element, List]):
        """
            This will return the pos/cat of a node, performing majority vote on conjunctions

        :param node:
        :param grouped:
        :return:
        """
        if 'cat' in node.attrib.keys():
            if node.attrib['cat'] == 'conj':
                return self.majority_vote(node, grouped)
            return node.attrib['cat']
        return node.attrib[self.pos_set]

    def get_type_plain(self, node: ET.Element, grouped: Grouped) -> WordType:
        # plain type assignment
        if 'cat' in node.attrib.keys():
            # non-terminal node
            if node.attrib['cat'] == 'conj':
                # conjunction
                return self.type_dict[self.majority_vote(node, grouped)]
            else:
                # non-conjunction
                return self.type_dict[node.attrib['cat']]
        elif self.pos_set in node.attrib.keys():
            # terminal node
            return self.type_dict[node.attrib[self.pos_set]]
        else:
            raise KeyError('No pos or cat in node {}.'.format(node.attrib['id']))

    def get_type_mod(self, rel: str, parent_type: WordType) -> WordType:
        return ColoredType(arguments=(parent_type,), colors=(rel,), result=parent_type)

    def get_type_gap(self, node: ET.Element, rel: str, prior_type: ColoredType, grouped: Grouped) -> WordType:
        try:
            new_arg = ColoredType(arguments=(self.get_type_plain(node, grouped),),
                                  colors=(rel,),
                                  result=prior_type.argument)
        except AttributeError:
            # todo
            raise NotImplementedError('Modified gap.')
        return ColoredType(arguments=(new_arg,), colors=(prior_type.color,), result=prior_type.result)

    def get_type_gap_mod(self, rel: str, prior_type: ColoredType, parent_type: WordType) -> WordType:
        mod_type = ColoredType(arguments=(parent_type,), colors=(rel,), result=parent_type)
        new_arg = ColoredType(arguments=(mod_type,),
                              colors=('embedded',),
                              result=prior_type.argument)
        return ColoredType(arguments=(new_arg,), colors=(prior_type.color,), result=prior_type.result)

    @staticmethod
    def group_by_parent(xtree: ET.ElementTree) -> Grouped:
        """
            Given a DAG, will organize the nodes into a dictionary with non-terminal nodes (parents) as keys, and their
        children as values.

        :param xtree: The DAG to convert.
        :type xtree: ET.ElementTree
        :return: The dictionary mapping parents to their children.
        :rtype: Grouped
        """
        nodes = list(xtree.iter('node'))
        grouped = []
        for node in nodes:
            if type(node.attrib['rel']) == str:
                grouped.append([node, -1, 'top'])
            else:
                for parent, rel in node.attrib['rel'].items():
                    grouped.append([node, parent, rel])
        grouped = sorted(grouped, key=lambda x: int(x[1]))
        grouped = groupby(grouped, key=lambda x: int(x[1]))
        grouped = {k: [(v[0], v[2]) for v in V] for k, V in grouped}
        grouped = dict(map(lambda x: (x[0], x[1]), grouped.items()))

        newdict = dict()

        for key in grouped.keys():
            if key == -1:
                newdict[None] = grouped[key]
                continue
            newkey = list(filter(lambda x: x.attrib['id'] == str(key), nodes))[0]
            newdict[newkey] = grouped[key]
        return newdict

    @staticmethod
    def split_dag(grouped: Grouped, cats_to_remove: Iterable[str] = ('du',),
                  rels_to_remove: Iterable[str] = ('dp', 'sat', 'nucl', 'tag', '--', 'top')) -> Grouped:
        """
            Takes a dictionary describing a DAG and breaks it into a possibly disconnected one by removing links between
        the category and dependency labels specified. Useful for breaking headless structures apart.

        :param grouped: The dictionary to operate on.
        :type grouped: Grouped
        :param cats_to_remove: The category labels that are subject to removal.
        :type cats_to_remove: Iterable[str]
        :param rels_to_remove: The dependency labels that are subject to removal.
        :type rels_to_remove: Iterable[str]
        :return: The disconnected DAG dictionary.
        :rtype: Grouped
        """
        keys_to_remove = list()

        for key in grouped.keys():
            if key is not None:
                if 'cat' in key.attrib.keys():
                    if key.attrib['cat'] in cats_to_remove:
                        keys_to_remove.append(key)

        for key in keys_to_remove:
            for c, r in grouped[key]:
                del c.attrib['rel'][key.attrib['id']]
            del grouped[key]  # here we delete the node from being a parent (outgoing edges)

        # here we remove children
        for key in grouped.keys():
            children_to_remove = list()
            for c, r in grouped[key]:
                if c in keys_to_remove:
                    # the child was a cut-off parent; remove the incoming edge
                    children_to_remove.append((c, r))
                    if key is not None:
                        del c.attrib['rel'][key.attrib['id']]
                elif r in rels_to_remove:
                    # the child has a 'bad' incoming edge; remove it
                    children_to_remove.append((c, r))
                    if key is not None:
                        del c.attrib['rel'][key.attrib['id']]
            for c in children_to_remove:
                grouped[key].remove(c)

        # check the parse tree for parent nodes with zero children
        empty_keys = [key for key in grouped.keys() if not grouped[key]]
        while len(empty_keys):
            for key in empty_keys:
                del grouped[key]
            for key in grouped.keys():
                for c, r in grouped[key]:
                    if c in empty_keys:
                        grouped[key].remove((c, r))
                        if key is not None:
                            del c.attrib['rel'][key.attrib['id']]
            empty_keys = [key for key in grouped.keys() if not grouped[key]]

        return grouped

    @staticmethod
    def get_disconnected(grouped: Grouped) -> Set[ET.Element]:
        """
            Takes a dictionary, possibly containing headless structures. Returns all nodes that are parents without
        being children of any other node (i.e. the starting points for the type-assignment recursion).

        :param grouped: The DAG to operate on.
        :type grouped: Grouped
        :return: The set of nodes that are starting points.
        :rtype: Set[ET.Element]
        """
        all_keys = set(grouped.keys())  # all parents
        all_children = set([x[0] for k in all_keys for x in grouped[k]])  # all children
        # sanity check: make sure that {all_children} - {all_parents} == all_leaves
        assert(all(map(lambda x: Decompose.is_leaf(x), all_children.difference(all_keys))))
        return all_keys.difference(all_children)

    @staticmethod
    def abstract_object_to_subject(grouped: Grouped) -> Grouped:
        """
            Takes a dictionary containing abstract objects/subjects and applies a series of conventions to re-assign the
        main objects/subjects.

        :param grouped: The DAG to operate on.
        :type grouped: Grouped
        :return: The transformed DAG.
        :rtype: Grouped
        """
        def is_abstract(node_rel, real_so):
            node, rel = node_rel
            if (node.attrib['index'] in real_so.attrib['index']) and \
                    rel in (('obj1', 'primary'), ('su', 'primary'), ('sup', 'primary')):
                return True
            return False

        for main_parent in grouped.keys():

            parent = main_parent

            if main_parent.attrib['cat'] not in ('ssub', 'smain', 'sv1'):
                continue

            real_so = list(filter(lambda x: x[1] in (('su', 'secondary'),
                                                     ('obj', 'secondary'),
                                                     ('obj1', 'secondary'),
                                                     ('sup', 'secondary'),
                                                     ('obj2', 'secondary')),
                                  [x for x in grouped[main_parent]]))
            if not real_so:
                continue

            assert isinstance(real_so[0][1], Rel)
            parent_dep = real_so[0][1].label
            real_so = real_so[0][0]

            # om te construction --  go one level lower
            ti = list(filter(lambda x: x[0].attrib['cat'] == 'ti',
                             [x for x in grouped[main_parent] if 'cat' in x[0].attrib]))
            if ti:
                parent = ti[0][0]

            # all children of the main parent that are pparts or infinites
            ppart = list(filter(lambda x: x[0].attrib['cat'] in ('ppart', 'inf'),
                                [x for x in grouped[parent] if 'cat' in x[0].attrib]))
            if not ppart:
                continue
            ppart = ppart[0][0]
            abstract_so = list(filter(lambda x: is_abstract(x, real_so),
                                      [x for x in grouped[ppart] if 'index' in x[0].attrib]))

            # chained inf / ppart construction
            if not abstract_so:
                ppart = list(filter(lambda x: (x[0].attrib['cat'] == 'ppart' or x[0].attrib['cat'] == 'inf'),
                             [x for x in grouped[ppart] if 'cat' in x[0].attrib]))
                if ppart:
                    ppart = ppart[0][0]
                    abstract_so = list(filter(lambda x: is_abstract(x, real_so),
                                              [x for x in grouped[ppart] if 'index' in x[0].attrib]))
            if not abstract_so:
                continue

            rel = abstract_so[0][1].label
            abstract_so = abstract_so[0][0]
            # # # Dictionary changes
            # remove the abstract real_so / object from being a child of the ppart/inf
            grouped[ppart].remove((abstract_so, Rel(label=rel, rank='primary')))
            # remove the abstract so from being a child of the ssub with a secondary label
            grouped[main_parent].remove((abstract_so, (parent_dep, 'secondary')))
            # add it again with a primary label
            grouped[main_parent].append((abstract_so, Rel(label=parent_dep, rank='primary')))
            # # # Internal node changes (for consistency)
            # remove the primary edge property from abstract object
            del abstract_so.attrib['rel'][ppart.attrib['id']]
            # convert the secondary edge to primary internally
            abstract_so.attrib['rel'][main_parent.attrib['id']] = Rel(label='su', rank='primary')

        return grouped

    @staticmethod
    def remove_abstract_so(grouped: Grouped, candidates: Iterable[str]=('su', 'obj', 'obj1', 'obj2', 'sup')) -> Grouped:
        """
            Takes a dictionary containing abstract subjects/objects and removes them. The main subject/object must have
        been properly assigned before.

        :param grouped: The DAG to operate on.
        :type grouped: Grouped
        :param candidates: The potential abstract subject/object dependency labels.
        :type candidates: Iterable[str]
        :return: The transformed DAG.
        :rtype: Grouped
        """

        removable = []
        for parent in grouped.keys():
            if parent.attrib['cat'] not in ('sv1', 'smain', 'ssub'):
                continue
            else:
                for child, rel in grouped[parent]:
                    removable.append(child)

        for parent in grouped.keys():
            if parent.attrib['cat'] not in ('ppart', 'inf'):
                # this is a proper secondary edge (non-abstract) and should not be removed
                continue
            for child, rel in grouped[parent]:
                if isinstance(rel, str):
                    # ignore non-coindexed
                    continue
                red_rel = Decompose.get_rel(rel)
                if red_rel not in candidates:
                    continue
                if rel.rank == 'secondary' and child in removable:
                    # # # Dictionary changes
                    # remove the abstract s/o from being a child of the ppart/inf
                    grouped[parent].remove((child, rel))
                    # # # Internal node changes (for consistency)
                    del child.attrib['rel'][parent.attrib['id']]
                else:
                    # Trying to remove main coindex
                    if 'index' in child.keys():
                        continue

        return grouped

    @staticmethod
    def order_siblings(siblings: ChildRels) -> ChildRels:
        """
            Arranges a list of sibling nodes by their order of appearance. Optionally excludes one of them (useful for
        head-daughters).

        :param siblings: The siblings to arrange.
        :type siblings: List[Tuple[ET.Element, Union[Rel, str]]]
        :param exclude: (Optional) The node to exclude. Defaults to None.
        :type exclude: Optional[ET.Element]
        :return: The arranged siblings.
        :rtype: List[Tuple[ET.Element, Union[Rel, str]]]
        """
        return sorted(siblings,
                      key=lambda x: tuple(map(int, (x[0].attrib['begin'], x[0].attrib['end'], x[0].attrib['id']))))

    def collapse_mwu(self, grouped: Grouped) -> Grouped:
        """
            Collapses multi-word expressions into a single node. Updates the node's word and category fields.

        :param grouped: The DAG to operate on.
        :type grouped: Grouped
        :return: The transformed DAG.
        :rtype: Grouped
        """
        to_remove = []

        # find all mwu parents
        for key in grouped.keys():
            if key is not None:
                if 'cat' in key.attrib.keys():
                    if key.attrib['cat'] == 'mwu':
                        nodes = Decompose.order_siblings(grouped[key])
                        collapsed_text = ' '.join([x[0].attrib['word'] for x in nodes])
                        key.attrib['word'] = collapsed_text  # update the parent text
                        del key.attrib['cat']
                        key.attrib['pt'] = self.majority_vote(key, grouped)  # todo
                        to_remove.append(key)

        # parent is not a parent anymore (since no children are inherited)
        for key in to_remove:
            del (grouped[key])
        return grouped

    def swap_determiner_head(self, grouped: Grouped) -> Grouped:
        """
            Turns determiner dependency labels into heads, replacing original heads with an 'invdet' dependency.
            Maintains the rank of the original dependency.

        :param grouped: The DAG to operate on
        :type grouped: Grouped
        :return: The transformed DAG
        :rtype: Grouped
        """
        for parent in grouped:
            children_rels = grouped[parent]
            children_rels = self.order_siblings(children_rels)
            rels = list(map(self.get_rel, [x[1] for x in children_rels]))
            if 'det' in rels:
                det_idx = rels.index('det')  # the first determiner in the phrase
                hd_idx = rels.index('hd')  # originally the noun

                if isinstance(children_rels[det_idx][1], Rel):
                    # det becomes a hd, inheriting the rank of previous det
                    new_hd = Rel(label='hd', rank=children_rels[det_idx][1].rank)
                else:
                    new_hd = 'hd'

                if isinstance(children_rels[hd_idx][1], Rel):
                    # head becomes an invdet, inheriting the rank of previous head
                    new_det = Rel(label='invdet', rank=children_rels[hd_idx][1].rank)
                else:
                    new_det = 'invdet'

                # change internal node attributes
                del children_rels[det_idx][0].attrib['rel'][parent.attrib['id']]
                children_rels[det_idx][0].attrib['rel'][parent.attrib['id']] = new_hd
                del children_rels[hd_idx][0].attrib['rel'][parent.attrib['id']]
                children_rels[hd_idx][0].attrib['rel'][parent.attrib['id']] = new_det

                # change grouped structure
                new_children_rels = [children_rels[i] for i in range(len(children_rels)) if i not in [det_idx, hd_idx]]
                new_children_rels += [(children_rels[det_idx][0], new_hd), (children_rels[hd_idx][0], new_det)]
                grouped[parent] = new_children_rels

                # # assertions
                # rels = list(map(self.get_rel, [x[1] for x in new_children_rels]))
                # if 'det' in rels and 'invdet' in rels:
                #     ToGraphViz()(grouped)
                #     raise NotImplementedError

        return grouped

    def refine_body(self, grouped: Grouped) -> Grouped:
        """
            Turns the uninformative 'body' dependency label into a richer variant that indicates what it is the body to.

        :param grouped: The DAG to operate on.
        :type grouped: Grouped
        :return: The transformed DAG.
        :rtype: Grouped
        """
        for parent in grouped:
            children_rels = grouped[parent]
            rels = list(map(self.get_rel, [x[1] for x in children_rels]))
            if 'body' in rels:
                body_idx = rels.index('body')
                if 'cmp' in rels:
                    continue
                elif 'rhd' in rels:
                    if isinstance(children_rels[body_idx][1], Rel):
                        new_body = Rel(label='rhd_body', rank=children_rels[body_idx][1].rank)
                    else:
                        new_body = 'rhd_body'
                elif 'whd' in rels:
                    if isinstance(children_rels[body_idx][1], Rel):
                        new_body = Rel(label='whd_body', rank=children_rels[body_idx][1].rank)
                    else:
                        new_body = 'whd_body'
                else:
                    ToGraphViz()(grouped)
                    raise NotImplementedError('Unknown head?')

                children_rels[body_idx][0].attrib['rel'][parent.attrib['id']] = new_body
                new_children_rels = [children_rels[i] for i in range(len(children_rels)) if i != body_idx]
                new_children_rels += [(children_rels[body_idx][0], new_body)]
                grouped[parent] = new_children_rels
        return grouped

    def tw_to_mod(self, grouped: Grouped) -> Grouped:
        """

        :param grouped:
        :type grouped:
        :return:
        :rtype:
        """
        ## de anderhalve
        ## een enkele // de beide : lid vnw
        ## geen enekele: vnw vnw

        for parent in grouped:
            children_rels = grouped[parent]
            dets = [(i, c, r) for i, (c, r) in enumerate(children_rels) if self.get_rel(r) == 'det']
            if len(dets) > 1:
                if all(list(map(lambda x: 'pt' in x[1].attrib.keys(), dets))):
                    tw = [x for x in dets if x[1].attrib['pt'] == 'tw' or x[1].attrib['word'] == 'beide']
                    # de tw construction -- convert tw to mod
                    if tw:
                        tw = tw[0]
                        if isinstance(tw[2], Rel):
                            new_tw = Rel(label='mod', rank=children_rels[tw[0]][1].rank)
                        else:
                            new_tw = 'mod'
                        tw[1].attrib['rel'][parent.attrib['id']] = new_tw
                        new_children_rels = [children_rels[i] for i in range(len(children_rels)) if i != tw[0]]
                        new_children_rels += [(tw[1], new_tw)]
                        grouped[parent] = new_children_rels
                    # note: no catch-all rule for other cases; type assigner defaults to a blank type (mwu det)
                else:
                    # chained detp with internal tw construction -- convert detp to mod
                    detp = [x for x in list(filter(lambda x: 'cat' in x[1].attrib.keys(), dets))][0]

                    # assert any(list(map(lambda x: x[0].attrib['pt'] == 'tw', grouped[detp[1]])))

                    if isinstance(detp[2], Rel):
                        new_detp = Rel(label='mod', rank=detp[2].rank)
                    else:
                        new_detp = 'mod'

                    children_rels[detp[0]][0].attrib['rel'][parent.attrib['id']] = new_detp
                    new_children_rels = [children_rels[i] for i in range(len(children_rels)) if i != detp[0]]
                    new_children_rels += [(detp[1], new_detp)]
                    grouped[parent] = new_children_rels

        return grouped

    def choose_head(self, children_rels: ChildRels) -> Optional[ChildRel]:
        """
            Takes a list of siblings and returns their head and its dependency label. Returns Nones if no head is found.

        :param children_rels: The siblings to look for a head in.
        :type children_rels: List[Tuple[ET.Element, Union[Rel, str]]]
        :return: A tuple consisting of the head node and its dependency label, or Nones if no head is found.
        :rtype: Union[Tuple[ET.Element, Union[Rel, str]], Tuple[None, None]]
        """
        for i, (candidate, rel) in enumerate(children_rels):
            if Decompose.get_rel(rel) in self.head_candidates:
                return candidate, rel
        return None

    @staticmethod
    def is_copy(node: ET.Element) -> bool:
        all_incoming_edges = list(map(Decompose.get_rel, node.attrib['rel'].values()))
        all_incoming_edges = Counter(all_incoming_edges)
        if any(list(map(lambda x: x > 1, all_incoming_edges.values()))):
            return True
        return False

    @staticmethod
    def get_rel(rel: Union[Rel, str]) -> str:
        if isinstance(rel, str):
            return rel
        else:
            return rel.label

    @staticmethod
    def collapse_single_non_terminals(grouped: Grouped, depth: int=0) -> Grouped:
        """
            Takes a dictionary containing intermediate non-terminals with only a single child (by-product of previous
        transformations) and collapses those with their respective children. The dependency label is inherited from
        upwards, but the dependency rank is inherited from downwards.

        :param grouped: The DAG to operate on.
        :type grouped: Grouped.
        :param depth: The recursion depth. Defaults to 0, useful for debugging.
        :type depth: int
        :return: The transformed DAG.
        :rtype: Grouped
        """
        # list of nodes containing a single child
        intermediate_nodes = [k for k in grouped.keys() if len(grouped[k]) == 1]
        if not intermediate_nodes:
            return grouped  # tying the knot

        # for each intermediate node
        for k in intermediate_nodes:
            # find its only child
            points_to, old_rel = grouped[k][0]
            # for each other node
            for kk in grouped.keys():
                # find its dependencies pointing to the intermediate node
                rels = [r for n, r in grouped[kk] if n == k]
                if not rels:
                    continue
                if len(rels) > 1:
                    ToGraphViz()(grouped)
                    raise ValueError('Many rels @ non-terminal node {}.'.format(k.attrib['id']))
                new_rel = rels[0]
                grouped[kk].remove((k, new_rel))
                new_rel = Decompose.get_rel(new_rel)
                if isinstance(old_rel, Rel):
                    grouped[kk].append((points_to, Rel(label=rels[0], rank=old_rel[1])))
                    points_to.attrib['rel'][kk.attrib['id']] = Rel(label=rels[0], rank=old_rel[1])
                else:
                    grouped[kk].append((points_to, new_rel))
                    points_to.attrib['rel'][kk.attrib['id']] = new_rel
            del points_to.attrib['rel'][k.attrib['id']]
        for k in intermediate_nodes:
            del(grouped[k])
        return Decompose.collapse_single_non_terminals(grouped, depth=depth+1)

    def reattach_conj_items(self, grouped: Grouped) -> Grouped:
        """
            Detaches modifiers applied to conjunction daughters and attaches them to conjunction parent.

        :param grouped:
        :return:
        """
        conj_nodes = [node for node in grouped.keys() if node.attrib['cat'] == 'conj']

        to_add = []
        to_remove = []

        removable = self.mod_candidates

        for node in conj_nodes:
            for daughter, _ in grouped[node]:
                if daughter in grouped.keys():
                    for granddaughter, rel in grouped[daughter]:
                        if self.get_rel(rel) in removable and Decompose.is_copy(granddaughter):
                            to_remove.append((daughter, granddaughter, rel))
                            to_add.append((node, granddaughter, self.get_rel(rel)))

        to_remove = set(to_remove)
        to_add = set(to_add)

        if len(to_remove):
            for parent, child, rel in to_remove:
                grouped[parent].remove((child, rel))
                del child.attrib['rel'][parent.attrib['id']]
            for parent, child, rel in to_add:
                grouped[parent].append((child, rel))
                child.attrib['rel'][parent.attrib['id']] = rel
            grouped = Decompose.collapse_single_non_terminals(grouped)
        return self.detatch_low_conj_mods(grouped)

    def detatch_low_conj_mods(self, grouped: Grouped) -> Grouped:
        """
            Detaches modifiers attached both high and low in a conjunction.

        :param grouped:
        :type grouped:
        :return:
        :rtype:
        """
        non_cnj_mod_fathers = [(node, [x for x in grouped[node] if self.get_rel(x[1]) in self.mod_candidates])
                               for node in grouped.keys() if node.attrib['cat'] != 'conj'
                               and any(list(map(lambda x: self.get_rel(x[1]) in self.mod_candidates, grouped[node])))]
        cnj_mod_fathers = [(node, [x for x in grouped[node] if self.get_rel(x[1]) in self.mod_candidates])
                           for node in grouped.keys() if node.attrib['cat'] == 'conj'
                           and any(list(map(lambda x: self.get_rel(x[1]) in self.mod_candidates, grouped[node])))]

        to_remove = []

        for non_cnj_father, non_cnj_mods in non_cnj_mod_fathers:
            for cnj_father, cnj_mods in cnj_mod_fathers:
                for non_cnj_mod in non_cnj_mods:
                    for cnj_mod in cnj_mods:
                        if non_cnj_mod[0] == cnj_mod[0]:
                            to_remove.append((non_cnj_father, non_cnj_mod[0], non_cnj_mod[1]))

        to_remove = set(to_remove)

        if len(to_remove):
            for parent, child, rel in to_remove:
                grouped[parent].remove((child, rel))
                del child.attrib['rel'][parent.attrib['id']]
            grouped = Decompose.collapse_single_non_terminals(grouped)

        return grouped

    def split_grouped(self, grouped: Grouped) -> Sequence[Grouped]:
        def reconstruct_grouped(new_grouped, fringe):
            new_grouped[fringe] = grouped[fringe]
            for k in new_grouped[fringe]:
                if k[0] in grouped.keys():
                    reconstruct_grouped(new_grouped, k[0])

        top_nodes = Decompose.get_disconnected(grouped)
        if len(top_nodes) == 1:
            return [grouped]
        else:
            out = []
            for tn in top_nodes:
                o = dict()
                reconstruct_grouped(o, tn)
                out.append(o)
            return out

    def update_lexicon(self, lexicon: Dict[str, WordType], wt: Sequence[Tuple[ET.Element, WordType]]) -> None:
        for k, t in wt:
            try:
                lexicon[k.attrib['id']] = t
            except AttributeError as e:
                raise e

    def type_assign(self, grouped: Grouped, lexicon: Dict[str, WordType], node_dict: Dict[str, ET.Element]) -> Any:
        """

        :param grouped:
        :type grouped:
        :param lexicon:
        :type lexicon:
        :param node_dict:
        :type node_dict:
        :return:
        :rtype:
        """
        root_type = self.type_assign_top(grouped, lexicon)
        # type assign constants
        self.type_assign_bot(grouped, lexicon)
        self.type_assign_mods(grouped, lexicon)
        self.type_assign_heads(grouped, lexicon)
        self.type_assign_gaps(grouped, lexicon)
        self.type_assign_copies(grouped, lexicon)
        self.type_assign_pairs(grouped, lexicon)
        self.annotate_nodes(lexicon, node_dict)
        return root_type

    def type_assign_top(self, grouped: Grouped, lexicon: Dict[str, WordType]) -> Any:
        top_nodes = Decompose.get_disconnected(grouped)
        assert len(top_nodes) == 1
        top_node_types = tuple(map(lambda x: (x, self.get_type_plain(x, grouped)), top_nodes))
        self.update_lexicon(lexicon, top_node_types)
        return top_node_types[0][1]

    def type_assign_bot(self, grouped: Grouped, lexicon: Dict[str, WordType]) -> None:

        def is_fringe(node: ET.Element) -> bool:
            if node.attrib['id'] in lexicon.keys():
                return False
            if all(map(lambda r: self.get_rel(r) in self.mod_candidates,
                       node.attrib['rel'].values())):
                return False
            if 'word' in node.attrib.keys():
                return True
            else:
                if all(map(lambda daughter: daughter[0].attrib['id'] in lexicon.keys(),
                           filter(lambda child: self.get_rel(snd(child))
                                                not in self.mod_candidates,
                                  grouped[node]))):
                    return True
                else:
                    return False

        fringe = filter(lambda x: is_fringe(fst(x)), chain.from_iterable(grouped.values()))
        fringe_types = tuple(map(lambda x: (fst(x), self.get_type_plain(fst(x), grouped)), fringe))
        self.update_lexicon(lexicon, fringe_types)
        if fringe_types:
            self.type_assign_bot(grouped, lexicon)

    def type_assign_mods(self, grouped: Grouped, lexicon: Dict[str, WordType]) -> None:
        fringe = Decompose.get_disconnected(grouped)
        for f in fringe:
            self.type_assign_mod_recursive(f, grouped, lexicon)

    def type_assign_mod_recursive(self, parent: ET.Element, grouped: Grouped, lexicon: Dict[str, WordType]) -> None:
        branch = grouped[parent]
        mods = filter(lambda child: self.get_rel(snd(child)) in self.mod_candidates, branch)
        mod_types = list(map(lambda x: (fst(x), self.get_type_mod(self.get_rel(snd(x)),
                                                                  lexicon[parent.attrib['id']])), mods))
        cnjs = filter(lambda child: self.get_rel(snd(child)) == 'cnj', branch)
        cnj_types = list(map(lambda x: (fst(x), lexicon[parent.attrib['id']]), cnjs))
        self.update_lexicon(lexicon, mod_types)
        self.update_lexicon(lexicon, cnj_types)
        for f in filter(lambda x: x[0] in grouped.keys(), branch):
            self.type_assign_mod_recursive(f[0], grouped, lexicon)

    def is_gap(self, node: ET.Element) -> bool:
        all_incoming_edges = list(map(self.get_rel, node.attrib['rel'].values()))
        # for something to be a gap it needs to have at least two different rels, of which at least one is hd
        head_edges = [x for x in all_incoming_edges if x in self.head_candidates]
        if len(head_edges) >= 1 and len(set(all_incoming_edges)) > 1:
            return True
        else:
            return False

    @staticmethod
    def fringe_heads_top_down(grouped: Grouped, left: List[ET.Element], done: List[ET.Element]) -> List[ET.Element]:
        return [k for k in left if all(map(lambda x: k not in map(fst, grouped[x]) or x in done,
                                           grouped.keys()))]

    @staticmethod
    def fringe_heads_bottom_up(grouped: Grouped, left: List[ET.Element], done: List[ET.Element]) -> List[ET.Element]:
        return [k for k in left if all(list(map(lambda x: x not in grouped.keys() or x in done,
                                                list(map(fst, grouped[k])))))]

    def type_assign_heads(self, grouped: Grouped, lexicon: Dict[str, WordType]) -> None:
        fringe_heads = lambda l, d: Decompose.fringe_heads_top_down(grouped, l, d)

        done = []
        left = list(grouped.keys())

        while left:
            fringe = fringe_heads(left, done)
            for f in fringe:
                self.type_assign_head_single_branch(f, grouped, lexicon)
                left.remove(f)
                done += [f]

    def type_assign_head_single_branch(self, parent: ET.Element, grouped: Grouped, lexicon: Dict[str, WordType]) -> None:
        head = self.choose_head(grouped[parent])
        if head is None:
            return head
        arglist = [x for x in grouped[parent] if self.get_rel(snd(x)) not in self.mod_candidates + ('crd', 'det')
                   and x != head]
        # exclude secondary gaps from arglist
        gaplist = list(filter(lambda x: self.is_gap(fst(x)) and snd(x).rank == 'secondary',
                              arglist))

        arglist = list(filter(lambda x: x not in gaplist, arglist))
        arglist = list(map(lambda x: (lexicon[fst(x).attrib['id']], self.get_rel(snd(x))),
                           arglist))
        arglist += list(map(lambda x: (self.get_type_plain(fst(x), grouped), self.get_rel(snd(x))),
                            gaplist))
        if arglist:
            args, colors = list(zip(*arglist))
            head_type = ColoredType(arguments=args, colors=colors, result=lexicon[parent.attrib['id']])
        else:
            head_type = lexicon[parent.attrib['id']]
        self.update_lexicon(lexicon, [(fst(head), head_type)])

        if fst(head) in grouped.keys():
            # todo: handle this properly with a function
            modlist = list(filter(lambda x: self.get_rel(snd(x)) in self.mod_candidates, grouped[fst(head)]))
            if any(list(map(lambda x: fst(x) in grouped.keys(), modlist))):
                raise NotImplementedError('Complex head modifier.')
            modlist = list(map(lambda x: (fst(x), ColoredType(arguments=(head_type,), colors=(self.get_rel(snd(x)),),
                                                              result=head_type)),
                               modlist))
            self.update_lexicon(lexicon, modlist)

    def type_assign_gaps(self, grouped: Grouped, lexicon: Dict[str, WordType]) -> None:
        # keep track of assigned gaps, dont assign twice
        gaps_assigned = set()
        for k in grouped.keys():
            gaps = list(filter(lambda x: self.is_gap(fst(x)) and fst(x) not in gaps_assigned, grouped[k]))
            gaps = list(filter(lambda x: isinstance(snd(x), Rel) and snd(x).rank == 'secondary', gaps))
            if any(map(lambda x: fst(x) in grouped.keys(), gaps)):
                raise NotImplementedError('Non-terminal gap.')
            gaps_assigned = gaps_assigned.union(set(list(map(fst, gaps))))
            gap_mods = list(filter(lambda x: self.get_rel(snd(x)) in self.mod_candidates, gaps))
            gap_non_mods = list(filter(lambda x: x not in gap_mods, gaps))
            gaptypes = list(map(lambda x: (fst(x), self.get_type_gap(fst(x), self.get_rel(snd(x)),
                                                                     lexicon[fst(x).attrib['id']], grouped)),
                                gap_non_mods))
            self.update_lexicon(lexicon, gaptypes)
            gaptypes = list(map(lambda x: (fst(x), self.get_type_gap_mod(self.get_rel(snd(x)),
                                                                         lexicon[fst(x).attrib['id']],
                                                                         lexicon[k.attrib['id']])),
                                gap_mods))
            self.update_lexicon(lexicon, gaptypes)

    def type_assign_copies(self, grouped: Grouped, lexicon: Dict[str, WordType]) -> None:
        cnjs = [k for k in grouped.keys() if k.attrib['cat'] == 'conj']
        for cnj in cnjs:
            head = self.choose_head(grouped[cnj])
            if head is None:
                raise NotImplementedError('Headless conjunction.')

            headtype = self.iterate_conj(cnj, grouped, lexicon)
            if headtype:
                self.update_lexicon(lexicon, [(fst(head), headtype)])

    def type_assign_pairs(self, grouped: Grouped, lexicon: Dict[str, WordType]) -> None:
        for k in grouped.keys():
            secondary_coordinators = [(d, r) for (d, r) in grouped[k] if self.get_rel(r) == 'crd'
                                      and (d, r) != self.choose_head(grouped[k])]
            crd_placeholders = list(map(lambda x: (fst(x), AtomicType('_CRD')), secondary_coordinators))
            self.update_lexicon(lexicon, crd_placeholders)
            secondary_determiners = [(d, r) for (d, r) in grouped[k] if self.get_rel(r) == 'det'
                                     and (d, r) != self.choose_head(grouped[k])]
            det_placeholders = list(map(lambda x: (fst(x), AtomicType('_DET')), secondary_determiners))
            self.update_lexicon(lexicon, det_placeholders)

    def iterate_conj(self, conj: ET.Element, grouped: Grouped, lexicon: Dict[str, WordType]) -> Optional[ColoredType]:
        def retrieve_ctype(x: ET.Element) -> WordType:
            if self.is_gap(x):
                return lexicon[x.attrib['id']].argument.argument
            else:
                return lexicon[x.attrib['id']]

        daughters = [(d, r) for (d, r) in grouped[conj] if self.get_rel(r) not in self.mod_candidates + ('crd', 'det')]

        copied = []
        non_copied = []

        for daughter, rel in daughters:
            if daughter in grouped.keys():
                granddaughters = [(n, r) for (n, r) in grouped[daughter]
                                  if self.get_rel(r) not in self.mod_candidates + ('det',)]

                shared = list(filter(lambda nr: self.is_copy(fst(nr)), granddaughters))
                non_shared = list(filter(lambda nr: fst(nr) not in list(map(fst, shared)), granddaughters))

                # multiset of types shared between conj daughters
                shared_types = Counter(list(map(lambda x: (retrieve_ctype(fst(x)), self.get_rel(snd(x))),
                                                shared)))
                # multiset of types unique to each conj daughter
                non_shared_types = Counter(list(map(lambda x: (retrieve_ctype(fst(x)), self.get_rel(snd(x))),
                                                    non_shared)))

                copied.append(shared_types)
                non_copied.append(non_shared_types)

        if not any(list(map(len, copied))):
            polymorphic_x = lexicon[conj.attrib['id']]
        else:
            try:
                assert all(list(map(lambda x: x == copied[0], copied[1::])))
            except AssertionError:
                raise NotImplementedError('Incompatible copies.')

            copies = list(copied[0].keys())

            copied_types, copied_deps = list(zip(*copies))
            copied_heads = [x in self.head_candidates for x in copied_deps]

            # case management:
            if all(copied_heads):
                # Case of head copying
                if all(map(lambda x: not len(x), non_copied)):
                    raise NotImplementedError('Copied head with no arguments.')
                if not all(list(map(lambda x: x == non_copied[0], non_copied[1::]))):
                    raise NotImplementedError('Copied head with different arguments.')
                polymorphic_x = ColoredType(arguments=(copied_types[0],), colors=('embedded',),
                                            result=lexicon[conj.attrib['id']])
            elif not any(copied_heads):
                # Case of argument copying
                polymorphic_x = ColoredType(arguments=tuple(copied_types), result=lexicon[conj.attrib['id']],
                                            colors=tuple(copied_deps))
            else:
                # Case of mixed
                if not all(list(map(lambda x: x == non_copied[0], non_copied[1::]))):
                    raise NotImplementedError('Copied head with different arguments.')
                if all(map(lambda x: not len(x), non_copied)):
                    raise NotImplementedError('Copied head and arguments with no real arguments.')
                non_copies = list(non_copied[0].keys())
                non_copied_argtypes, non_copied_argdeps = list(zip(*non_copies))
                # copied_heads = list(filter(lambda c: snd(c) in self.head_candidates, copies))
                # copied_headtypes, copied_headargs = list(zip(*copied_heads))

                hot = ColoredType(arguments=non_copied_argtypes, colors=non_copied_argdeps,
                                  result=lexicon[conj.attrib['id']])
                polymorphic_x = ColoredType(arguments=(hot,), result=lexicon[conj.attrib['id']], colors=('cnj',))

        self.update_lexicon(lexicon, list(map(lambda x: (fst(x), polymorphic_x), daughters)))
        return ColoredType(arguments=tuple(polymorphic_x for _ in range(len(daughters))),
                           colors=tuple('cnj' for _ in range(len(daughters))),
                           result=polymorphic_x)

    def lexicon_to_list(self, sublex: Dict[str, WordType], grouped: Grouped) \
            -> Tuple[Iterable[str], Iterable[WordType]]:
        """
            Takes a dictionary and a lexicon partially mapping dictionary leaves to types and converts it to either an
        iterable of (word, WordType) tuples, if to_sequences=True, or two iterables of words and WordTypes otherwise.

        :param sublex: The partially filled lexicon.
        :type sublex: Dict[str, WordType]
        :param grouped: The DAG that is being assigned.
        :type grouped: Grouped
        :return: The partial lexicon, converted into iterable(s)
        :rtype: Union[List[Tuple[str, WordType]], List[Iterable[str], Iterable[WordType]]]
        """
        # find all items that need to be labeled
        all_leaves = set(list(filter(lambda x: 'word' in x.attrib.keys(),
                              map(lambda x: x[0], chain.from_iterable(grouped.values())))))
        # sort them by occurrence
        all_leaves = sorted(all_leaves,
                            key=lambda x: tuple(map(int, (x.attrib['begin'], x.attrib['end'], x.attrib['id']))))

        # mapping from linear order to dictionary keys
        enum = {i: l.attrib['id'] for i, l in enumerate(all_leaves)}

        ret = [(enum[i], sublex[enum[i]])
               for i in range(len(all_leaves)) if enum[i] in sublex.keys()]

        return tuple(zip(*ret))

    def update_with_polarities(self, type_dict: Dict[str, WordType], node_dict: Dict[str, ET.Element],
                               lexlist: Tuple[Iterable[str], Iterable[WordType]],
                               top_type: WordType, g: Grouped) -> Tuple[Sequence[str], WordTypes, PolarizedIndexedType]:
        index_sequence = fst(lexlist)
        type_sequence = snd(lexlist)
        # add polarity and index information
        l, type_sequence = polarize_and_index_many(type_sequence, 0)

        top_type = PolarizedIndexedType(top_type.result, False, l)

        # update the type dict
        for (i, t) in zip(index_sequence, type_sequence):
            type_dict[i] = t

        root = list(self.get_disconnected(g))
        assert len(root) == 1

        type_dict[root[0].attrib['id']] = top_type

        # update the lexicon
        self.annotate_nodes(type_dict, node_dict)

        return index_sequence, type_sequence, top_type

    def annotate_nodes(self, lexicon: Dict[str, WordType], node_dict: Dict[str, ET.Element]) -> None:
        """
            Writes the extracted type of a node into its attributes.

        :param lexicon: The mapping from words and their identifiers into their types.
        :type lexicon: Dict[str, WordType]
        :param node_dict: The mapping from node identifiers into nodes.
        :type node_dict: Dict[str, ET.Element]
        :return: None
        """
        for lex_key in lexicon:
            if self.separation_symbol in lex_key:
                key_id = lex_key.split(self.separation_symbol)[1]
            else:
                key_id = lex_key
            node_dict[key_id].attrib['type'] = str(lexicon[lex_key])

    def __call__(self, grouped: Grouped) -> Any:
        node_dict = {node.attrib['id']: node for node in
                     set(grouped.keys()).union(set([v[0] for v in chain.from_iterable(grouped.values())]))}

        new_grouped = self.split_grouped(grouped)
        dicts = [dict() for _ in range(len(new_grouped))]
        top_types = []

        for i, g in enumerate(new_grouped):
            top_types.append(self.type_assign(g, dicts[i], node_dict))

        # convert to sequences
        lexicons = list(map(lambda ix: self.lexicon_to_list(ix[1], new_grouped[ix[0]]), enumerate(dicts)))

        # update with extra info
        lexicons = list(map(lambda d, l, t, g:
                            self.update_with_polarities(d, node_dict, l, t, g),
                            dicts, lexicons, top_types, new_grouped))

        # rearrange and split
        lexicons, top_types = list(zip(*[(l[0:2], l[2]) for l in lexicons]))

        for i, l in enumerate(lexicons):
            if not typecheck(list(l[1]), top_types[i]):
                ToGraphViz()(new_grouped[i])
                raise NotImplementedError('Generic type-checking error')

        return list(zip(*(new_grouped, dicts, lexicons, top_types)))


def main(viz: bool=False, remove_mods: bool=False) -> Any:
    if remove_mods:
        rels_to_remove = ('dp', 'sat', 'nucl', 'tag', '--', 'mod', 'predm')
    else:
        rels_to_remove = ('dp', 'sat', 'nucl', 'tag', '--')

    # a non-processed dataset for comparisons
    L0 = Lassy()
    # a processed dataset that yields a lexicon
    decomposer = Decompose(visualize=viz)
    lexicalizer = Compose([lambda x: [x[0], x[2]],  # keep only index and parse tree
                           lambda x: [x[0], Lassy.tree_to_dag(x[1])],  # convert to DAG
                           lambda x: [x[0], Decompose.group_by_parent(x[1])],  # convert to dict format
                           lambda x: [x[0], Decompose.split_dag(x[1],  # split into disjoint trees if needed
                                                                cats_to_remove=('du',),
                                                                rels_to_remove=rels_to_remove)],
                           lambda x: [x[0], decomposer.collapse_mwu(x[1])],  # remove mwus
                           lambda x: [x[0], Decompose.abstract_object_to_subject(x[1])],  # relabel abstract so's
                           lambda x: [x[0], Decompose.remove_abstract_so(x[1])],  # remove abstract so's
                           lambda x: [x[0], Decompose.collapse_single_non_terminals(x[1])],
                           lambda x: [x[0], decomposer.refine_body(x[1])],
                           lambda x: [x[0], decomposer.tw_to_mod(x[1])],
                           lambda x: [x[0], decomposer.swap_determiner_head(x[1])],
                           lambda x: [x[0], decomposer.reattach_conj_items(x[1])],
                           lambda x: [x[0], decomposer(x[1])],  # decompose into a lexicon
                           ])
    L = Lassy(transform=lexicalizer)
    return L0, L, ToGraphViz(), decomposer


def iterate(lassy: Lassy, **kwargs: int) -> \
        Tuple[List[Tuple[int, Sequence[str]]], List[Tuple[int, WordTypes]]]:
    """
        Iterates over a Lassy dataset as returned by main(), either serially or in parallel. To enable parallel mode
        provide num_workers and/or batch_size as keyword arguments.

    :param lassy: The dataset to iterate. Note that its itemgetter function must be composed with a lexicalization
                  transformation.
    :type lassy: Lassy
    :param kwargs: If any of num_workers, batch_size is provided as a kwarg, the iteration will be done in parallel.
                   Set num_workers to the number of your CPU threads and batch_size to a power of 2 between 1 and 128.
    :type kwargs: int
    :return: X: A list of tuples (i, S) where S a list of words forming a sentence and i the index indicating which
             data sample this sentence came from, and a corresponding list of tuples (i, T) where T is the list of
             types that this sentence maps to.
    :rtype: Tuple[List[Tuple[int, Sequence[str]]], List[Tuple[int, Sequence[WordType]]], List[Tuple[int], WordType]]
    """

    DAGS, X, Y, TD = [], [], [], []

    def merge_dicts(dict_args):
        """
        Given any number of dicts, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts.
        """
        result = {}
        for dictionary in dict_args:
            result = {**result, **dictionary}
        return result

    # parallel case
    if kwargs:
        dl = DataLoader(lassy, **kwargs, collate_fn=lambda x: list(x))
        for b in dl:
            for sample in b:
                idx = sample[0]
                DAGS.extend([(idx, sample[1][0]) for _ in sample[1][1]])
                X.extend([(idx, x[0]) for x in sample[1][1]])
                Y.extend([(idx, x[1]) for x in sample[1][1]])
                TD.extend([(idx, merge_dicts(sample[1][2])) for _ in sample[1][2]])
        return DAGS, X, Y, TD

    # sequential case
    for i in range(len(lassy)):
        l = lassy[i][1]
        X.extend([(i, x[0]) for x in l[1]])
        Y.extend([(i, x[1]) for x in l[1]])
    return X, Y

# # # # # # # # Visualization Utility # # # # # # # #


class ToGraphViz:
    def __init__(self, to_show: Iterable[str]=('id', 'word', 'pos', 'cat', 'index', 'type', 'pt')) -> None:
        self.to_show = to_show

    def construct_node_label(self, child: ET.Element):
        """
        :param child:
        :return:
        """
        label = ''
        for key in self.to_show:
            if key != 'span':
                try:
                    label += str(child[key]) + '\n'
                except KeyError:
                    pass
            else:
                label += child['begin'] + '-' + child['end'] + '\n'
        return label

    def construct_edge_label(self, rel: Union[Rel, str]) -> str:
        if isinstance(rel, str):
            return rel
        else:
            return ' '.join(rel)

    def get_edge_style(self, rel: Union[Rel, str]) -> str:
        return 'dashed' if isinstance(rel, Rel) and rel.rank == 'secondary' else ''

    def xml_to_gv(self, xtree: ET.ElementTree) -> graphviz.Digraph:
        nodes = list(Lassy.extract_nodes(xtree))  # a list of triples
        graph = graphviz.Digraph()

        graph.node('title', label=xtree.findtext('sentence'), shape='none')
        graph.node(nodes[0][0].attrib['id'], label='ROOT')
        graph.edge('title', nodes[0][0].attrib['id'], style='invis')

        for child, parent in nodes[1:]:
            node_label = self.construct_node_label(child.attrib)
            graph.node(child.attrib['id'], label=node_label)

            if isinstance(child.attrib['rel'], str):
                graph.edge(parent.attrib['id'], child.attrib['id'], label=child.attrib['rel'])
            else:
                for parent_id, dependency in child.attrib['rel'].items():
                    graph.edge(parent_id, child.attrib['id'], label=self.construct_edge_label(dependency))

        return graph

    def grouped_to_gv(self, grouped: Grouped, show_sentence: bool) -> graphviz.Digraph:
        graph = graphviz.Digraph()
        if show_sentence:
            reduced_sentence = ' '.join([x.attrib['word'] for x in sorted(
                set(grouped.keys()).union(set([y[0] for x in grouped.values() for y in x])),
                key=lambda x: (int(x.attrib['begin']), int(x.attrib['end']), int(x.attrib['id']))) if 'word' in x.attrib])

            graph.node('title', label=reduced_sentence, shape='none')

        for parent in grouped.keys():
            node_label = self.construct_node_label(parent.attrib)
            graph.node(parent.attrib['id'], label=node_label)
            for child, rel in grouped[parent]:
                node_label = self.construct_node_label(child.attrib)
                graph.node(child.attrib['id'], label=node_label)
                graph.edge(parent.attrib['id'], child.attrib['id'], style=self.get_edge_style(rel),
                           label=self.construct_edge_label(rel))
        return graph

    def __call__(self, parse: Union[Grouped, ET.ElementTree], output: str='gv_output', view: bool=True,
                 show_sentence: bool = True) -> None:
        graph = self.xml_to_gv(parse) if isinstance(parse, ET.ElementTree) \
            else self.grouped_to_gv(parse, show_sentence)
        if output:
            graph.render(output, view=view)
