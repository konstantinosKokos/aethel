import os
import xml.etree.cElementTree as ET
from glob import glob
from copy import deepcopy
import graphviz
from pprint import pprint as print

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from itertools import groupby, chain

from WordType import *

from typing import Optional, Iterable, Tuple, Union, Any, Generator, Dict, List, Callable, NamedTuple

from warnings import warn

Rel = NamedTuple('Rel', [('label', str), ('rank', str)])
Grouped = Dict[ET.Element, List[Tuple[ET.Element, Union['Rel', str]]]]


class Lassy(Dataset):
    """
        Lassy dataset. A wrapper that feeds samples into the extraction algorithm.
    """

    def __init__(self, root_dir: str='/home/kokos/Documents/Projects/LassySmall 4.0', treebank_dir: str='/Treebank',
                 transform: Optional[Compose]=None, ignore: Optional[str] = 'ignored.txt') -> None:
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

        ignored = []
        if ignore is not None:
            try:
                with open(ignore, 'r') as f:
                    ignored = f.readlines()
                    ignored = list(map(lambda x: x[0:-1], ignored))
                    print('Ignoring {} samples..'.format(len(ignored)))
            except FileNotFoundError:
                warn('Could not open the ignore file.')

        self.filelist = [y for x in os.walk(self.treebank_dir) for y in glob(os.path.join(x[0], '*.[xX][mM][lL]'))
                         if y not in ignored]
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
            Takes an ElementTree representing a parse tree, possibly containing duplicate nodes (that is, nodes that \
        correspond to the same lexical unit but with a different identifier to preserve the tree format). Removes \
        duplicate nodes by constructing new dependency links between cross-references (moving to a DAG format), and \
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

        # todo
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
    def __init__(self, type_dict: Optional[Dict[str, str]]=None, unify: bool=False, return_lists: bool=True,
                 text_pipeline: Callable[[str], str]=lambda x: x.lower(), separation_symbol: str='↔',
                 visualize: bool=False) -> None:
        """
            Initialize an extraction class.

        :param type_dict: (Optional) A dictionary mapping part of speech and category labels to their corresponding
            types.
        :type type_dict: Optional[Dict[str, str]]
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
        if not type_dict:
            self.type_dict = {'adj': 'ADJ', 'adv': 'ADV', 'advp': 'ADV', 'ahi': 'AHI', 'ap': 'AP', 'comp': 'COMP',
                              'comparative': 'COMPARATIVE', 'conj': 'CONJ', 'cp': 'CP', 'det': 'DET', 'detp': 'DET',
                              'du': 'DU', 'fixed': 'FIXED', 'inf': 'INF', 'mwu': 'MWU', 'name': 'NAME', 'noun': 'N',
                              'np': 'NP', 'num': 'NUM', 'oti': 'OTI', 'part': 'PART', 'pp': 'PP', 'ppart': 'PPART',
                              'ppres': 'PPRES', 'prefix': 'PREFIX', 'prep': 'PREP', 'pron': 'NP', 'punct': 'PUNCT',
                              'rel': 'REL', 'smain': 'S', 'ssub': 'S', 'sv1': 'S', 'svan': 'SVAN',
                              'tag': 'TAG', 'ti': 'TI', 'top': 'TOP', 'verb': 'VERB', 'vg': 'VG', 'whq': 'WHQ',
                              'whrel': 'WHREL', 'whsub': 'WHSUB'}
            # convert to Types
            self.type_dict = {k: AtomicType(v) for k, v in self.type_dict.items()}

        self.unify = unify
        self.return_lists = return_lists
        self.text_pipeline = text_pipeline
        self.separation_symbol = separation_symbol
        # the function applied to convert the processed text of a node into a dictionary key
        self.get_key = lambda node: self.text_pipeline(node.attrib['word']) + self.separation_symbol + node.attrib['id']
        self.head_candidates = ('hd', 'rhd', 'whd', 'cmp', 'crd', 'dlink')
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
            todo

        :param node:
        :type node:
        :param grouped:
        :type grouped:
        :return:
        :rtype:
        """
        sibling_types = [self.get_type_key(n[0], grouped) for n in grouped[node]]
        votes = {c: len([x for x in sibling_types if x == c]) for c in set(sibling_types)}
        votes = sorted(votes, key=lambda x: -votes[x])
        return votes[0]

    def get_type_key(self, node: ET.Element, grouped: Dict[ET.Element, List]):
        """
        This will return the pos/cat of a node, performing majority vote on conjunctions
        :param node:
        :param grouped:
        :return:
        """
        # todo: this will not work after the noun-ification change
        if 'cat' in node.attrib.keys():
            if node.attrib['cat'] == 'conj':
                return self.majority_vote(node, grouped)
            return node.attrib['cat']
        return node.attrib['pos']

    def get_type(self, node: ET.Element, grouped: Grouped, rel: Optional[Union[Rel, str]]=None,
                 parent: Optional[ET.Element]=None) -> WordType:
        """
            Returns the type of a node within a given context.

            If no optional arguments are passed, context is ignored and the type returned is simply the type dictionary
            \ mapping of the node's syntactic category or part of speech. In the case of conjunction, a majority vote \
            is performed on the node's daughters to infer its type translation, i.e. conjunction is treated as \
            polymorphic, with the production type being the same as that of the (majority of the) daughters.

            If the dependency label is also passed, an additional check will be made; if the dependency label is that \
            of a modifier, the type assigned will be X -> X with a 'mod' color on the arrow, where X is the parent type.
            \ If no parent type is provided (e.g. in the case of gaps, where we don't know the non-local source of an \
            incoming edge, a placeholder type MOD is assigned instead.

            Alternatively, if the parent but no dependency is passed, and the node has a part of speech label of either
            \ noun or adjective, the assigned type will depend on the parent's syntactic category, according to the \
            rule: if parent is a phrase (either NP or AP), then the daughter is non-phrasal (N or ADJ, respectively), \
            otherwise the daughter is acting as a phrase on her own.

        :param node: The node to assign a type to.
        :type node: ET.Element
        :param grouped: The dictionary containing the parenthood relations between nodes and their corresponding \
            dependency labels.
        :type grouped: dict
        :param rel: (Optional) The currently inspected dependency label.
        :type rel: Optional[Union[list, str]]
        :param parent: (Optional) The parent node of the node inspected.
        :type parent: Optional[ET.Element]
        :return: The WordType that the context imposes on the input node.
        :rtype: WordType
        """
        # case management
        # dependency label is given and is mod
        if rel is not None and self.get_rel(rel) == 'mod':
            if parent is None:
                # we do not know the parent of this, needs post-processing
                warn('Assigning placeholder type.')
                return AtomicType('MOD')
            return ColoredType(arguments=(self.get_type(parent, grouped),), result=self.get_type(parent, grouped),
                               colors=('mod',))
        # node is terminal (i.e. has a POS attribute) and a parent is provided
        elif 'pos' in node.attrib.keys() and parent is not None:
            # subcase management
            # three indicators; pos of parent, dependency, postag of self
            # pos of parent useful in case of cnj but not really
            # dependency most reliable (obj1, su, body)
            # postag of self only covers some cases

            # issues:
            # if I dont take into account all three information sources, I get partial information
            # i.e. a predc might be a name or a noun-- do I assign it AP then?
            #
            if rel == 'cnj':
                return self.get_type(parent, grouped)
            if self.get_rel(rel) in ('obj1', 'su', 'body', 'vc', 'obj2'):
                print('Transforming {} ({}) to NP.'.format(node.attrib['pos'], node.attrib['id']))
                return AtomicType('NP')
            elif self.get_rel(rel) in ('predc', 'predm'):
                if self.get_type(node, grouped) == AtomicType('ADJ'):
                    return AtomicType('AP')
                else:
                    print('should be ap but is not adj')
                    ToGraphViz()(grouped)
                    print(rel)
                    print(node.attrib['id'])
                    print(node.attrib['pos'])
                    print(node.attrib['postag'])
                    print(self.get_type(parent, grouped))
                    raise AssertionError('NP under NP.')
            elif any(pt in node.attrib['postag'] for pt in (',nom', 'soort,', 'eigen', 'vrij', 'pers,pron')):
                if self.get_rel(rel) not in ('app', 'predc', 'svp'):
                    print(rel)
                    print('POSTag looks like NP but no direct dependency')
                    ToGraphViz()(grouped)
                    print(node.attrib['id'])
                    print(node.attrib['pos'])
                    print(node.attrib['postag'])
                    print(self.get_type(parent, grouped))
                    raise AssertionError('NP under NP.')
                return AtomicType('NP')
            elif self.get_type(node, grouped) == AtomicType('ADJ') and \
                    self.get_type(parent, grouped) != AtomicType('AP'):
                return AtomicType('AP')

        # plain type assignment
        if 'cat' in node.attrib.keys():
            # non-terminal node
            if node.attrib['cat'] == 'conj':
                # conjunction
                return self.type_dict[self.majority_vote(node, grouped)]
            else:
                # non-conjunction
                return self.type_dict[node.attrib['cat']]
        elif 'pos' in node.attrib.keys():
            # terminal node
            return self.type_dict[node.attrib['pos']]
        else:
            raise KeyError('No pos or cat in node {}.'.format(node.attrib['id']))

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
    def split_dag(grouped: Grouped, cats_to_remove: Iterable[str]=('du',),
                  rels_to_remove: Iterable[str]=('dp', 'sat', 'nucl', 'tag', '--', 'top')) -> Grouped:
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
                # elif (len(c.attrib['rel'].values()) and
                #       all(map(lambda x: Decompose.get_rel(x) == 'mod', c.attrib['rel'].values()))):
                #     # explicit treatment of modifiers
                #     children_to_remove.append([c, r])
                #     if key is not None:
                #         del c.attrib['rel'][key.attrib['id']]

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
                        grouped[key].remove((c,r))
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
            real_so = list(filter(lambda x: x[1] in (('su', 'secondary'), ('obj1', 'secondary')),
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
                if rel.rank == 'secondary':
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
    def order_siblings(siblings: List[Tuple[ET.Element, Union[Rel, str]]], exclude: Optional[ET.Element]=None) \
            -> List[Tuple[ET.Element, Union[Rel, str]]]:
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
        if exclude is not None:
            siblings = list(filter(lambda x: x[0] != exclude, siblings))
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
                        key.attrib['cat'] = self.majority_vote(key, grouped)  # todo
                        to_remove.append(key)

        # parent is not a parent anymore (since no children are inherited)
        for key in to_remove:
            del (grouped[key])
        return grouped

    def choose_head(self, children_rels: List[Tuple[ET.Element, Union[Rel, str]]]) \
            -> Union[Tuple[ET.Element, Union[Rel, str]], Tuple[None, None]]:
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
        return None, None

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
                    points_to.attrib['rel'][kk.attrib['id']] = (rels[0], old_rel[1])
                else:
                    grouped[kk].append((points_to, new_rel))
                    points_to.attrib['rel'][kk.attrib['id']] = new_rel
            del points_to.attrib['rel'][k.attrib['id']]
        for k in intermediate_nodes:
            del(grouped[k])
        return Decompose.collapse_single_non_terminals(grouped, depth=depth+1)

    def recursive_assignment(self, current: ET.Element, grouped: Grouped, top_type: Optional[WordType],
                             lexicon: Dict[str, WordType], node_dict: Dict[str, ET.Element]) -> None:
        """
            Blah.
        :param current:
        :type current:
        :param grouped:
        :type grouped:
        :param top_type:
        :type top_type:
        :param lexicon:
        :type lexicon:
        :param node_dict:
        :type node_dict:
        :return:
        :rtype:
        """
        # ToGraphViz()(grouped)

        def is_gap(node: ET.Element) -> bool:
            # this node is involved in some sort of magic trickery if it has more than one incoming edges
            all_incoming_edges = list(map(self.get_rel, node.attrib['rel'].values()))
            if len(all_incoming_edges) > 1:
                head_edges = ([x for x in all_incoming_edges if x in self.head_candidates])
                if head_edges:
                    if len(set(head_edges)) != 1 or len(head_edges) < len(all_incoming_edges):
                        return True
                return False
            else:
                return False

        # find all of the node's siblings
        siblings = grouped[current]
        # pick a head
        headchild, headrel = self.choose_head(siblings)

        # if no type given from above, assign one now (no nested types)
        if top_type is None:
            top_type = self.get_type(current, grouped)

        # pose some linear order and exclude the picked head
        siblings = Decompose.order_siblings(siblings, exclude=headchild)

        if headchild is not None:
            # classify the headchild
            gap = is_gap(headchild)

            # pick all the arguments
            arglist = [[self.get_type(sib, grouped, parent=current, rel=self.get_rel(rel)), self.get_rel(rel)]
                       for sib, rel in siblings if self.get_rel(rel) != 'mod']

            # whether to type assign on this head -- True by default
            assign = True

            # case management
            if arglist and gap:
                argtypes, argdeps = list(zip(*arglist))

                # hard case first -- gap with siblings
                # assert that there is just one argument to project into
                if len(argtypes) != 1:
                    print(argtypes)
                    ToGraphViz()(grouped)
                    raise NotImplementedError('Case of non-terminal gap with many arguments {} {}.'.
                                              format(headchild.attrib['id'], current.attrib['id']))
                # find the dependencies not projected by current head
                internal_edges = list(filter(lambda x: x[0] != current.attrib['id'], headchild.attrib['rel'].items()))
                internal_edges = list(map(lambda x: (node_dict[x[0]], self.get_rel(x[1])), internal_edges))

                # assert that there is just one (class) of those
                if len(set(internal_edges)) == 1:
                    # construct the internal type (which includes a hypothesis for the gap)
                    internal_type = ColoredType(arguments=(self.get_type(headchild, grouped, rel=internal_edges[0][1],
                                                                         parent=internal_edges[0][0]),),
                                                result=argtypes[0], colors=(internal_edges[0][1],))
                    # (X -> Y)
                    # construct the external type (which takes the internal type back to the top type)
                    headtype = ColoredType(arguments=(internal_type,), result=top_type, colors=(argdeps[0],))
                    # (X -> Y) -> Z
                else:
                    assert len(argdeps) == 1  # not even sure why but this is necessary
                    types = []
                    for internal_head, internal_edge in set(internal_edges):
                        # if internal_edge == 'mod':
                        #     raise NotImplementedError('[Case of unknown parent] Modifier gap {} {}.'.
                        #                               format(headchild.attrib['id'], current.attrib['id']))
                        internal_type = ColoredType(arguments=(self.get_type(headchild, grouped, rel=internal_edge,
                                                                             parent=internal_head),),
                                                    result=argtypes[0], colors=(internal_edge,))
                        types.append(internal_type)
                    headtype = ColoredType(arguments=(CombinatorType(tuple(types), combinator='&'),),
                                           result=top_type, colors=(argdeps[0],))
            elif arglist:
                argtypes, argdeps = list(zip(*arglist))
                #  easy case -- standard type assignment
                headtype = ColoredType(arguments=argtypes, result=top_type, colors=argdeps)  # /W EXCHANGE
            elif gap:
                # weird case -- gap with no non-modifier siblings (most likely simply an intermediate non-terminal)
                headtype = self.get_type(headchild, grouped)
                # we avoid type assigning here
                assign = False
                # raise NotImplementedError('[What is this?] Case of head with only modifier siblings {}.'
                #                           .format(headchild.attrib['id']))
            else:
                # neither gap nor has siblings -- must be the end
                headtype = top_type
                if not self.is_leaf(headchild):
                    raise NotImplementedError('[Dead End] Case of head non-terminal with no siblings {}.'
                                              .format(headchild.attrib['id']))

            # finish the head assignment
            if assign:
                if self.is_leaf(headchild):
                    # tying the knot
                    if self.get_key(headchild) not in lexicon.keys():
                        lexicon[self.get_key(headchild)] = headtype
                    else:
                        old_value = lexicon[self.get_key(headchild)]
                        if old_value != headtype:
                            headtype = CombinatorType((headtype, old_value), combinator='&')
                            lexicon[self.get_key(headchild)] = headtype
                else:
                    self.recursive_assignment(headchild, grouped, headtype, lexicon, node_dict)

        # now deal with the siblings
        for sib, rel in siblings:
            if not is_gap(sib):
                sib_type = self.get_type(sib, grouped, rel, parent=current)
                if Decompose.is_leaf(sib):
                    # assign to lexicon
                    lexicon[self.get_key(sib)] = sib_type
                else:
                    # .. or iterate down
                    self.recursive_assignment(sib, grouped, None, lexicon, node_dict)
            else:
                pass
                # raise ValueError('??')

    def lexicon_to_list(self, sublex: Dict[str, WordType], grouped: Grouped, to_sequences: bool=True) \
            -> Union[List[Tuple[str, WordType]], List[Iterable[str], Iterable[WordType]]]:
        """
            Takes a dictionary and a lexicon partially mapping dictionary leaves to types and converts it to either an
        iterable of (word, WordType) tuples, if to_sequences=True, or two iterables of words and WordTypes otherwise.

        :param sublex: The partially filled lexicon.
        :type sublex: Dict[str, WordType]
        :param grouped: The DAG that is being assigned.
        :type grouped: Grouped
        :param to_sequences: If True, will return an iterable of words and an iterable of WordTypes. If False, will
            return an iterable of (word, WordType) tuples.
        :type to_sequences: bool
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
        enum = {i: self.get_key(l) for i, l in enumerate(all_leaves)}
        # convert to a list [(word, WordType), ...]
        ret = [(enum[i].split(self.separation_symbol)[0], sublex[enum[i]])
               for i in range(len(all_leaves)) if enum[i] in sublex.keys()]

        if to_sequences:
            # convert to two tuples (word1, word2, ..), (type1, type2, ..)
            return list(zip(*ret))
        return ret

    def __call__(self, grouped: Grouped):
        if self.visualize:
            ToGraphViz()(grouped)
        top_nodes = Decompose.get_disconnected(grouped)

        top_node_types = map(lambda x: self.get_type(x, grouped), top_nodes)

        node_dict = {node.attrib['id']: node for node in
                     set(grouped.keys()).union(set([v[0] for v in chain.from_iterable(grouped.values())]))}

        if self.unify:
            # init lexicon here
            lexicon = dict()

            # recursively iterate from each top node
            for top_node in top_nodes:
                self.recursive_assignment(top_node, grouped, None, lexicon, node_dict)
            if self.return_lists:
                return Decompose.lexicon_to_list(lexicon, grouped)  # return the dict transformation
            return lexicon  # or return the dict

        else:
            # one dict per disjoint sequence
            dicts = [dict() for _ in top_nodes]
            for i, top_node in enumerate(top_nodes):
                # recursively iterate each
                self.recursive_assignment(top_node, grouped, None, dicts[i], node_dict)

            if self.return_lists:
                return list(map(lambda x: self.lexicon_to_list(x, grouped), dicts))  # return the dict transformation
            return dicts  # or return the dicts


def main(return_lists: bool=False, viz: bool=False, remove_mods: bool=False, ignore: str='ignored.txt'):
    if remove_mods:
        rels_to_remove = ('dp', 'sat', 'nucl', 'tag', '--', 'mod')
    else:
        rels_to_remove = ('dp', 'sat', 'nucl', 'tag', '--')

    # a non-processed dataset for comparisons
    L0 = Lassy(ignore=ignore)

    # a processed dataset that yields a lexicon
    decomposer = Decompose(return_lists=return_lists, visualize=viz)
    lexicalizer = Compose([lambda x: [x[0], x[2]],  # keep only index and parse tree
                           lambda x: [x[0], Lassy.tree_to_dag(x[1])],  # convert to DAG
                           lambda x: [x[0], Decompose.group_by_parent(x[1])],  # convert to dict format
                           lambda x: [x[0], decomposer.collapse_mwu(x[1])],  # remove mwus
                           lambda x: [x[0], Decompose.split_dag(x[1],  # split into disjoint trees if needed
                                                                cats_to_remove=('du',),
                                                                rels_to_remove=rels_to_remove)],
                           lambda x: [x[0], Decompose.abstract_object_to_subject(x[1])],  # relabel abstract so's
                           lambda x: [x[0], Decompose.remove_abstract_so(x[1])],  # remove abstract so's
                           lambda x: [x[0], Decompose.collapse_single_non_terminals(x[1])],
                           lambda x: [x[1], decomposer(x[1])],  # decompose into a lexicon
                           ])
    L = Lassy(transform=lexicalizer, ignore=ignore)
    # #
    # X, Y = [], []
    # for i in range(len(L)):
    #     if i == 29897: continue
    #     l = L[i][1]
    #     X.extend([x[0] for x in l])
    #     Y.extend([x[1] for x in l])
    # return X, Y

    return L0, L, ToGraphViz()


class ToGraphViz:
    def __init__(self, to_show: Iterable[str]=('id', 'word', 'pos', 'cat', 'index')) -> None:
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
                    label += child[key] + '\n'
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

    def grouped_to_gv(self, grouped: Grouped) -> graphviz.Digraph:
        graph = graphviz.Digraph()
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
                graph.edge(parent.attrib['id'], child.attrib['id'], style=self.get_edge_style(rel), label=self.construct_edge_label(rel))
        return graph

    def __call__(self, parse: Union[Grouped, ET.ElementTree], output: str='gv_output', view: bool=True) -> None:
        graph = self.xml_to_gv(parse) if isinstance(parse, ET.ElementTree) else self.grouped_to_gv(parse)
        if output:
            graph.render(output, view=view)

# def reduce_lexicon(main_lex, new_lex, key_reducer=lambda x: x.split(' ')[0]):
#     """
#
#     :param main_lex:
#     :param new_lex:
#     :param key_reducer:
#     :return:
#     """
#     # for each word of the new lexicon
#     for key in new_lex:
#         # remove the internal id
#         reduced_key = key_reducer(key)
#         # if the word exists in the original lexicon
#         if reduced_key in main_lex.keys():
#             # if the assigned type has already been assigned to the word in the original lexicon
#             if new_lex[key] in main_lex[reduced_key].keys():
#                 # increase each occurrence count
#                 main_lex[reduced_key][new_lex[key]] += 1
#             else:
#                 # otherwise add it to the lexicon
#                 main_lex[reduced_key][new_lex[key]] = 1
#         # if the word does not exist in the original lexicon
#         else:
#             # init a new dictionary in the original lexicon, with this type as its only key and a single occurrence
#             main_lex[reduced_key] = {new_lex[key]: 1}