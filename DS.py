import os
import xml.etree.cElementTree as ET
from glob import glob
from copy import deepcopy
import graphviz
from pprint import pprint as print

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from itertools import groupby, chain
from collections import Counter

from WordType import *

from warnings import warn


class Lassy(Dataset):
    """
    Lassy dataset
    """

    def __init__(self, root_dir='/home/kokos/Documents/Projects/LassySmall 4.0', treebank_dir='/Treebank',
                 transform=None, ignore=True):
        """

        :param root_dir:
        :param treebank_dir:
        :param transform:
        """

        if os.path.isdir(root_dir) and os.path.isdir(root_dir+treebank_dir):
            self.root_dir = root_dir # might be useful to store meta information
            self.treebank_dir = root_dir + treebank_dir
        else:
            raise ValueError('%s and %s must be existing directories' % (root_dir, treebank_dir))

        ignored = []
        if ignore:
            try:
                with open('ignored.txt', 'r') as f:
                    ignored = f.readlines()
                    ignored = list(map(lambda x: x[0:-1], ignored))
            except FileNotFoundError:
                pass
            print('Ignoring {} samples..'.format(len(ignored)))

        self.filelist = [y for x in os.walk(self.treebank_dir) for y in glob(os.path.join(x[0], '*.[xX][mM][lL]'))
                         if y not in ignored]
        self.transform = transform

        print('Dataset constructed with {} samples.'.format(len(self.filelist)))

    def __len__(self):
        """
        :return:
        """
        return len(self.filelist)

    def __getitem__(self, id):
        """
        :param file:
        :return: id (INT), FILENAME (STR), PARSE (XMLTREE)
        """

        if type(id) == int:
            file = self.filelist[id]
        elif type(id) == str:
            file = id
        else:
            raise TypeError('file argument has to be int or str')

        parse = ET.parse(file)
        parse.getroot().set('type', 'Tree')

        sample = (id, file, parse)

        if self.transform:
            return self.transform(sample)

        return sample

    @staticmethod
    def extract_nodes(xtree):
        """
        A simple iterator over an xml parse that returns the parse tree's nodes. This is necessary as the default ET
        iterator does not provide parent or depth info.
        :param xtree:
        :return (child_node, parent_node, depth)
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
    def find_main_coindex(xtree):
        """
        Takes an ElementTree representing a parse tree, finds out nodes corresponding to the same lexical unit, and
        selects a single node to act as the "main" node (the one to collapse dependencies to, when removing its
        cross-references).
        :param cElementTree xtree: the ElementTree to operate on
        :return dict all_coind a dictionary mapping each co-indexing identifier to a list of nodes sharing it,
                dict main_coind a dictionary mapping each co-indexing identifier to the main node of the group:

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
    def tree_to_dag(xtree, inline=False):
        """
        Takes an ElementTree representing a parse tree, possibly containing duplicate nodes (that is, nodes that
        correspond to the same lexical unit but with a different identifier to preserve the tree format). Removes
        duplicate nodes by constructing new dependency links between cross-references (moving to a DAG format), and
        returns the resulting ElementTree.
        :param xtree: the ElementTree to process
        :param inline: whether to apply the changes to the original tree or a copy of it
        :return xtree: the modified ElementTree
        """
        if not inline:
            xtree = deepcopy(xtree)

        nodes = list(Lassy.extract_nodes(xtree))

        _, main_coind = Lassy.find_main_coindex(xtree)

        for node, parent in nodes[1:]:
            if node in main_coind.values():
                node.attrib['rel'] = {parent.attrib['id']: [node.attrib['rel'], 'primary']}

        for node, parent in nodes[1:]:
            if type(node.attrib['rel']) == str:
                if 'index' in node.attrib.keys():

                    main_coind[node.attrib['index']].attrib['rel'] = {parent.attrib['id']: [node.attrib['rel'],
                                                                                            'secondary'],
                                                                      **main_coind[node.attrib['index']].attrib['rel']}
                    parent.remove(node)
                else:
                    node.attrib['rel'] = {parent.attrib['id']: node.attrib['rel']}
        return xtree


class Decompose:
    def __init__(self, type_dict=None, unify=False, return_lists=True, text_pipeline=lambda x: x.lower(),
                 separation_symbol='↔', visualize=False):
        # type_dict: POS → Type
        if not type_dict:
            # todo: instantiate the type dict with Types and refactor all downwards call to allow for complex types
            self.type_dict = {'adj': 'ADJ', 'adv': 'ADV', 'advp': 'ADVP', 'ahi': 'AHI', 'ap': 'AP', 'comp': 'COMP',
                              'comparative': 'COMPARATIVE', 'conj': 'CONJ', 'cp': 'CP', 'det': 'DET', 'detp': 'DET',
                              'du': 'DU', 'fixed': 'FIXED', 'inf': 'INF', 'mwu': 'MWU', 'name': 'NP', 'noun': 'NP',
                              'np': 'NP', 'num': 'NP', 'oti': 'OTI', 'part': 'PART', 'pp': 'PP', 'ppart': 'PPART',
                              'ppres': 'PPRES', 'prefix': 'PREFIX', 'prep': 'PREP', 'pron': 'NP', 'punct': 'PUNCT',
                              'rel': 'REL', 'smain': 'S', 'ssub': 'S', 'sv1': 'S', 'svan': 'SVAN',
                              'tag': 'TAG', 'ti': 'TI', 'top': 'TOP', 'verb': 'VERB', 'vg': 'VG', 'whq': 'WHQ',
                              'whrel': 'WHREL', 'whsub': 'WHSUB'}
            # convert to Types
            self.type_dict = {k: AtomicType(v) for k, v in self.type_dict.items()}

            # Type conventions (richard)
            # # todo: need to change the direction of headedness before applying
            # self.type_dict['det'] = WordType([('NP', 'det')], 'NP')
            # self.type_dict['adj'] = WordType([('NP', 'mod')], 'NP')
            # self.type_dict['ap'] = WordType([('NP', 'mod')], 'NP')
            # self.type_dict['adv'] = WordType([('S', 'mod')], 'S')
            # self.type_dict['advp'] = WordType([('S', 'mod')], 'S')
        self.unify = unify  # whether to return a single lexicon from a sentence, or a lexicon for each subunit
        self.return_lists = return_lists  # whether to convert lexicons to word and type sequences
        self.text_pipeline = text_pipeline  # the function applied on the text before adding a word to the lexicon
        self.separation_symbol = separation_symbol  # the separation symbol between a node's text content and its id
        # the function applied to convert the processed text of a node into a dictionary key
        self.get_key = lambda node: self.text_pipeline(node.attrib['word']) + self.separation_symbol + node.attrib['id']
        self.head_candidates = ('hd', 'rhd', 'whd', 'cmp', 'crd', 'dlink')
        self.visualize = visualize

    @staticmethod
    def is_leaf(node):
        """
        Returns True if a node is a leaf in the parse tree, False otherwise.
        :param node: the node to decide
        :return True | False:
        """
        return True if 'word' in node.attrib.keys() else False

    def majority_vote(self, node, grouped):
        """
        Returns the majority voting of a node's children base types.
        :param node:
        :param grouped:
        :return:
        """
        sibling_types = [self.get_type_key(n[0], grouped) for n in grouped[node]]
        votes = {c: len([x for x in sibling_types if x == c]) for c in set(sibling_types)}
        votes = sorted(votes, key=lambda x: -votes[x])
        return votes[0]

    def get_type_key(self, node, grouped):
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
        return node.attrib['pos']

    def get_plain_type(self, node, grouped, rel=None):
        """
        This will return the plain (i.e. ignoring context) of a node, based on the type_dict of the class
        :param node:
        :param grouped:
        :return:
        """
        if rel is not None and self.get_rel(rel) == 'mod':
            return AtomicType('MOD')
        if 'cat' in node.attrib.keys():
            if node.attrib['cat'] == 'conj':
                return self.type_dict[self.majority_vote(node, grouped)]
            return self.type_dict[node.attrib['cat']]
        return self.type_dict[node.attrib['pos']]

    @staticmethod
    def group_by_parent(xtree):
        """
        Converts the representation from ETree to a dictionary mapping parents to their children
        :param ElementTree xtree:
        :return dict grouped: a dictionary mapping each node to its children
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
        grouped = {k: [[v[0], v[2]] for v in V] for k, V in grouped}
        grouped = dict(map(lambda x: [x[0], x[1]], grouped.items()))

        newdict = dict()

        for key in grouped.keys():
            if key == -1:
                newdict[None] = grouped[key]
                continue
            newkey = list(filter(lambda x: x.attrib['id'] == str(key), nodes))[0]
            newdict[newkey] = grouped[key]
        return newdict

    @staticmethod
    def split_dag(grouped, cats_to_remove=('du',), rels_to_remove=('dp', 'sat', 'nucl', 'tag', '--', 'top')):
        """
        Takes a dictionary possibly containing headless structures, and returns multiple dictionaries that don't.
        Essentially splits a parse DAG into a set of disjoint DAGs.
        :param grouped:
        :param cats_to_remove:
        :param rels_to_remove:
        :return:
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
                    children_to_remove.append([c, r])
                    if key is not None:
                        del c.attrib['rel'][key.attrib['id']]
                elif r in rels_to_remove:
                    # the child has a 'bad' incoming edge; remove it
                    children_to_remove.append([c, r])
                    if key is not None:
                        del c.attrib['rel'][key.attrib['id']]
                elif (len(c.attrib['rel'].values()) and
                      all(map(lambda x: Decompose.get_rel(x) == 'mod', c.attrib['rel'].values()))):
                    # explicit treatment of modifiers
                    children_to_remove.append([c, r])
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
                        grouped[key].remove([c, r])
                        if key is not None:
                            del c.attrib['rel'][key.attrib['id']]
            empty_keys = [key for key in grouped.keys() if not grouped[key]]

        return grouped

    @staticmethod
    def get_disconnected(grouped):
        """
        Takes a dictionary, possibly containing headless structures. Returns all nodes that are parents without being
        children of any other node.
        :param grouped:
        :return:
        """
        all_keys = set(grouped.keys())  # all parents
        all_children = set([x[0] for k in all_keys for x in grouped[k]])  # all children
        # sanity check: make sure that {all_children} - {all_parents} == all_leaves
        assert(all(map(lambda x: Decompose.is_leaf(x), all_children.difference(all_keys))))
        return all_keys.difference(all_children)

    @staticmethod
    def abstract_object_to_subject(grouped):
        """
        Takes a dictionary containing abstract objects/subjects and applies a series of conventions to re-assign the
        main objects/subjects.
        :param grouped:
        :return:
        """

        # todo: write this neatly
        for main_parent in grouped.keys():

            parent = main_parent

            if main_parent.attrib['cat'] not in ('ssub', 'smain', 'sv1'):
                continue
            real_so = list(filter(lambda x: (x[1] == ['su', 'secondary'] or x[1] == ['obj1', 'secondary']),
                                  [x for x in grouped[main_parent]]))
            if not real_so:
                continue

            assert isinstance(real_so[0][1], list)
            parent_dep = real_so[0][1][0]
            real_so = real_so[0][0]

            # om te construction --  go one level lower
            ti = list(filter(lambda x: x[0].attrib['cat'] == 'ti',
                             [x for x in grouped[main_parent] if 'cat' in x[0].attrib]))
            if ti:
                parent = ti[0][0]

            ppart = list(filter(lambda x: (x[0].attrib['cat'] == 'ppart' or x[0].attrib['cat'] == 'inf'),
                                [x for x in grouped[parent] if 'cat' in x[0].attrib]))
            if not ppart:
                continue
            ppart = ppart[0][0]
            abstract_so = list(filter(lambda x: (x[1] == ['obj1', 'primary'] or x[1] == ['su', 'primary']
                                                 or x[1] == ['sup', 'primary']) and
                                                    (x[0].attrib['index'] == real_so.attrib['index']),
                                      [x for x in grouped[ppart] if 'index' in x[0].attrib]))

            # chained inf / ppart construction
            if not abstract_so:
                ppart = list(filter(lambda x: (x[0].attrib['cat'] == 'ppart' or x[0].attrib['cat'] == 'inf'),
                             [x for x in grouped[ppart] if 'cat' in x[0].attrib]))
                if ppart:
                    ppart = ppart[0][0]
                    abstract_so = list(filter(lambda x: (x[1] == ['obj1', 'primary'] or x[1] == ['su', 'primary']
                                                 or x[1] == ['sup', 'primary']) and
                                                        (x[0].attrib['index'] == real_so.attrib['index']),
                                              [x for x in grouped[ppart] if 'index' in x[0].attrib]))
            if not abstract_so:
                continue

            rel = abstract_so[0][1][0]
            abstract_so = abstract_so[0][0]
            # # # Dictionary changes
            # remove the abstract real_so / object from being a child of the ppart/inf
            grouped[ppart].remove([abstract_so, [rel, 'primary']])
            # remove the abstract so from being a child of the ssub with a secondary label
            grouped[main_parent].remove([abstract_so, [parent_dep, 'secondary']])
            # add it again with a primary label
            grouped[main_parent].append([abstract_so, [parent_dep, 'primary']])
            # # # Internal node changes (for consistency) # todo redundant
            # remove the primary edge property from abstract object
            del abstract_so.attrib['rel'][ppart.attrib['id']]
            # convert the secondary edge to primary internally
            abstract_so.attrib['rel'][main_parent.attrib['id']] = ['su', 'primary']
        return grouped

    @staticmethod
    def remove_abstract_so(grouped, candidates=('su', 'obj', 'obj1', 'obj2', 'sup')):
        """
        Takes a dictionary containing abstract subjects/objects and removes them. The main subject/object must have
        been properly assigned before.
        :param grouped:
        :return:
        """
        for parent in grouped.keys():
            if parent.attrib['cat'] != 'ppart' and parent.attrib['cat'] != 'inf':
                # this is a proper secondary edge (non-abstract) and should not be removed
                continue
            for child, rel in grouped[parent]:
                if not isinstance(rel, list):
                    # ignore non-coindexed
                    continue
                red_rel = Decompose.get_rel(rel)
                if red_rel not in candidates:
                    continue
                if rel[1] == 'secondary':
                    # # # Dictionary changes
                    # remove the abstract s/o from being a child of the ppart/inf
                    grouped[parent].remove([child, rel])
                    # # # Internal node changes (for consistency) # todo redundant
                    del child.attrib['rel'][parent.attrib['id']]
                else:
                    # Trying to remove main coindex
                    if 'index' in child.keys():
                        # todo?
                        continue
                        # raise ValueError('Found primary object between {} and {}'.format(parent.attrib['id'],
                        #                                                    child.attrib['id']))
                    #ToGraphViz()(grouped, output='abc')

        return grouped

    @staticmethod
    def order_siblings(siblings, exclude=None):
        """
        Sorts a list of sibling nodes. Sorting is done on the basis of (span: start, span:end, original id)
        :param siblings:
        :param exclude:
        :return:
        """
        if exclude is not None:
            siblings = list(filter(lambda x: x[0] != exclude, siblings))
        return sorted(siblings,
                      key=lambda x: tuple(map(int, (x[0].attrib['begin'], x[0].attrib['end'], x[0].attrib['id']))))

    def collapse_mwu(self, grouped):
        """
        Takes a dictionary, possibly containing multi-word units, and collapses these together. The type of the
        collapsed node is inferred by majority voting on its children.
        :param grouped:
        :param relabel:
        :return:
        """
        to_remove = []

        # find all mwu parents
        for key in grouped.keys():
            if key is not None:
                if 'cat' in key.attrib.keys():
                    if key.attrib['cat'] == 'mwu':
                        nodes = Decompose.order_siblings(grouped[key])
                        collapsed_text = ''.join([x[0].attrib['word'] + ' ' for x in nodes])

                        key.attrib['word'] = collapsed_text[0:-1]  # update the parent text
                        key.attrib['cat'] = self.majority_vote(key, grouped)
                        to_remove.append(key)

        # parent is not a parent anymore (since no children are inherited)
        for key in to_remove:
            del (grouped[key])
        return grouped

    def choose_head(self, children_rels):
        """
        Takes a list of siblings and selects their head.
        :param children_rels: a list of [node, rel] lists
        :return:
            if a head is found, return the head and the rel
            if structure is headless, return None, None
            if unspecified case, raise ValueError
        """
        for i, (candidate, rel) in enumerate(children_rels):
            if Decompose.get_rel(rel) in self.head_candidates:
                return candidate, rel
        return None, None

    @staticmethod
    def get_rel(rel):
        return rel[0] if isinstance(rel, list) else rel

    @staticmethod
    def collapse_single_non_terminals(grouped):
        # list of nodes containing a single child
        intermediate_nodes = [k for k in grouped.keys() if len(grouped[k]) == 1]
        if not intermediate_nodes:
            return grouped

        # for each intermediate node
        for k in intermediate_nodes:
            # find its only child
            points_to, old_rel = grouped[k][0]
            # for each other node
            for kk in grouped.keys():
                # find its dependencies pointing to the intermediate node
                rels = [r for n, r in grouped[kk] if n == k]
                # for each such dependency
                if not rels:
                    continue
                grouped[kk].remove([k, rels[0]])
                if isinstance(old_rel, list):
                    grouped[kk].append([points_to, [rels[0], old_rel[1]]])
                    points_to.attrib['rel'][kk] = [rels[0], old_rel[1]]
                else:
                    grouped[kk].append([points_to, old_rel])
                    points_to.attrib['rel'][kk] = old_rel
            del points_to.attrib['rel'][k.attrib['id']]
        for k in intermediate_nodes:
            del(grouped[k])
        return Decompose.collapse_single_non_terminals(grouped)

    def recursive_assignment(self, current, grouped, top_type, lexicon):
        """
        Takes a node, a dictionary, a word type from the above subtree and a lexicon. Updates the lexicon with inferred
        types for this node's children and iterates downwards.
        :param current:
        :param grouped:
        :param top_type:
        :param lexicon:
        :return:
        """
        def is_gap(node):
            # this node is involved in some sort of magic trickery if it has more than one incoming edges
            all_incoming_edges = list(map(self.get_rel, node.attrib['rel'].values()))
            if len(all_incoming_edges) > 1:
                num_heads = list(filter(lambda x: x in self.head_candidates, all_incoming_edges))
                if num_heads and len(num_heads) != len(all_incoming_edges):
                    return True
            else:
                return False

        # find all of the node's siblings
        siblings = grouped[current]
        # pick a head
        headchild, headrel = self.choose_head(siblings)

        # if no type given from above, assign one now (no nested types)
        if top_type is None:
            top_type = self.get_plain_type(current, grouped)

        # pose some linear order and exclude the picked head
        siblings = Decompose.order_siblings(siblings, exclude=headchild)

        if headchild is not None:
            # pick all the arguments
            arglist = [[self.get_plain_type(sib, grouped, rel), self.get_rel(rel)] for sib, rel in siblings]
            # convert the arglist to coloured arglist
            if arglist:
                argtypes, argdeps = list(zip(*arglist))
            else:

                # siblings must have been removed by processing, so chain of non-terminal nodes with single child
                # ending with a single word
                if not self.is_leaf(headchild):
                    # call self having made no adjustments to top_type (simply propagate it down)
                    # self.recursive_assignment(headchild, grouped, top_type, lexicon)
                    pass
                elif self.get_key(headchild) not in lexicon.keys():
                    lexicon[self.get_key(headchild)] = top_type
                elif lexicon[self.get_key(headchild)] != top_type:
                    raise KeyError('[No siblings] Trying to assign {} to node {}, when already assigned {}. '
                                   'Now iterating from parent {}.'.format(top_type, headchild.attrib['id'],
                                                                          lexicon[self.get_key(headchild)],
                                                                          current.attrib['id']))
                return lexicon

            # classify the headchild
            gap = is_gap(headchild)

            # case management
            if not gap:
                # standard type assignment
                headtype = ColoredType(arguments=argtypes, result=top_type, colors=argdeps)  # /W EXCHANGE
            else:
                # case of gap: project self within arglist

                # assert that there is one argument to project into
                if len(argtypes) != 1:
                    raise NotImplementedError('Case of non-terminal gap with many arguments.')

                # find the dependency which does not match the head
                internal_edge = [self.get_rel(r) for r in headchild.attrib['rel'].values()
                                 if self.get_rel(r) != self.get_rel(headrel)]

                # assert that there is just one (class) of those
                if len(set(internal_edge)) != 1:
                    raise NotImplementedError('Case of multiple external dependencies {}.'.
                                              format(headchild.attrib['id']))

                # constuct the internal type (which includes a hypothesis for the gap)
                internal_type = ColoredType(arguments=(self.get_plain_type(headchild, grouped, internal_edge),),
                                            result=argtypes[0], colors=(internal_edge[0],))
                # construct the external type (which takes the internal type back to the top type)
                headtype = ColoredType(arguments=(internal_type,), result=top_type, colors=(argdeps[0],))

            if Decompose.is_leaf(headchild):
                # assign the type to the lexicon
                if self.get_key(headchild) in lexicon.keys():
                    old_value = lexicon[self.get_key(headchild)]
                    if old_value != headtype:
                        headtype = CombinatorType((headtype, old_value), combinator='⊗')
                        # raise KeyError('[2] Trying to assign {} to node {}, when already assigned {}. '
                        #                'Now iterating from parent {}.'.format(headtype, headchild.attrib['id'],
                        #                                                       lexicon[self.get_key(headchild)],
                        #                                                       current.attrib['id']))
                lexicon[self.get_key(headchild)] = headtype
            else:
                # .. or iterate down
                self.recursive_assignment(headchild, grouped, headtype, lexicon)

        # now deal with the siblings
        for sib, rel in siblings:
            if not is_gap(sib):
                # simple type
                sib_type = self.get_plain_type(sib, grouped, rel)
                if Decompose.is_leaf(sib):
                    # assign to lexicon
                    lexicon[self.get_key(sib)] = sib_type
                else:
                    # .. or iterate down
                    self.recursive_assignment(sib, grouped, None, lexicon)

    def lexicon_to_list(self, sublex, grouped, to_sequences=True):
        """
        Take a sublexicon {word : WordType}, partially mapping leaves from the grouped dictionary to types, and convert
         it to a (word, WordType) list that respects the original linear order of the sentence
        :param sublex:
        :param grouped:
        :param to_sequences: if True, return a list of words and a corresponding list of their types
        :return:
        """
        # todo: sorting by keys properly

        all_leaves = set(list(filter(lambda x: 'word' in x.attrib.keys(),
                              map(lambda x: x[0], chain.from_iterable(grouped.values())))))

        all_leaves = sorted(all_leaves,
                            key=lambda x: tuple(map(int, (x.attrib['begin'], x.attrib['end'], x.attrib['id']))))

        # mapping from linear order to dictionary keys
        enum = {i: self.get_key(l) for i, l in enumerate(all_leaves)}

        ret = [(enum[i].split(self.separation_symbol)[0], sublex[enum[i]])
               for i in range(len(all_leaves)) if enum[i] in sublex.keys()]
        if to_sequences:
            return list(zip(*ret))
        return ret

    def __call__(self, grouped):
        if self.visualize:
            ToGraphViz()(grouped)
        top_nodes = Decompose.get_disconnected(grouped)

        top_node_types = map(lambda x: self.get_plain_type(x, grouped), top_nodes)

        if self.unify:
            # init lexicon here
            lexicon = dict()

            # recursively iterate from each top node
            for top_node in top_nodes:
                self.recursive_assignment(top_node, grouped, None, lexicon)
            if self.return_lists:
                return Decompose.lexicon_to_list(lexicon, grouped)  # return the dict transformation
            return lexicon  # or return the dict

        else:
            # one dict per disjoint sequence
            dicts = [dict() for _ in top_nodes]
            for i, top_node in enumerate(top_nodes):
                # recursively iterate each
                self.recursive_assignment(top_node, grouped, None, dicts[i])

            if self.return_lists:
                return list(map(lambda x: self.lexicon_to_list(x, grouped), dicts))  # return the dict transformation
            return dicts  # or return the dicts


def main(ignore=False, return_lists=False, viz=False):
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
                                                                rels_to_remove=('dp', 'sat', 'nucl', 'tag', '--',
                                                                                'top', 'mod', 'predm',))],
                           lambda x: [x[0], Decompose.abstract_object_to_subject(x[1])],  # relabel abstract so's
                           lambda x: [x[0], Decompose.remove_abstract_so(x[1])],  # remove abstract so's
                           lambda x: [x[0], Decompose.collapse_single_non_terminals(x[1])],
                           lambda x: [x[1], decomposer(x[1])],  # decompose into a lexicon
                           ])
    L = Lassy(transform=lexicalizer, ignore=ignore)
    return L0, L, ToGraphViz()


class ToGraphViz:
    def __init__(self, to_show=('id', 'word', 'pos', 'cat', 'index')):
        self.to_show = to_show

    def construct_node_label(self, child):
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

    def construct_edge_label(self, rel):
        return ' '.join(rel) if isinstance(rel, list) else rel

    def get_edge_style(self, rel):
        return 'dashed' if isinstance(rel, list) and rel[1] == 'secondary' else ''

    def xml_to_gv(self, xtree):
        nodes = list(Lassy.extract_nodes(xtree))  # a list of triples
        graph = graphviz.Digraph()

        graph.node('title', label=xtree.findtext('sentence'), shape='none')
        graph.node(nodes[0][0].attrib['id'], label='ROOT')
        graph.edge('title', nodes[0][0].attrib['id'], style='invis')

        for child, parent in nodes[1:]:
            node_label = self.construct_node_label(child.attrib)
            graph.node(child.attrib['id'], label=node_label)

            if type(child.attrib['rel']) == str:
                graph.edge(parent.attrib['id'], child.attrib['id'], label=child.attrib['rel'])
            else:
                for parent_id, dependency in child.attrib['rel'].items():
                    graph.edge(parent_id, child.attrib['id'], label=self.construct_edge_label(dependency))

        return graph

    def grouped_to_gv(self, grouped):
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

    def __call__(self, parse, output='gv_output', view=True):
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