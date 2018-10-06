import os
import xml.etree.cElementTree as ET
from glob import glob

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from warnings import warn

from torchvision.transforms import Compose

import graphviz

from itertools import groupby, chain
from copy import deepcopy

from pprint import pprint as print

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
        # todo: primary / secondary edge labels
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

class ToGraphViz():
    def __init__(self, to_show = ['id', 'word', 'pos', 'cat', 'index']):
        self.to_show = to_show

    def construct_node_label(self, child):
        """
        :param child:
        :return:
        """
        label = ''
        for key in self.to_show:
        #for key in child:
            if key != 'span':
                try:
                    label += child[key] + '\n'
                except KeyError:
                    pass
            else:
                label += child['begin'] + '-' + child['end'] + '\n'
        return label

    def construct_edge_label(self, rel):
        if type(rel) == list:
            return rel[0] + ' ' + rel[1]
        else:
            return rel

    def get_edge_style(self, rel):
        style = ''
        if type(rel) == list:
            if rel[1] == 'secondary':
                style = 'dashed'
        return style

    def xml_to_gv(self, xtree):
        nodes = list(Lassy.extract_nodes(xtree)) # a list of triples
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
        reduced_sentence = ''.join([x.attrib['word']+' ' for x in sorted(
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
        if type(parse) == ET.ElementTree:
            graph = self.xml_to_gv(parse)
        else:
            graph = self.grouped_to_gv(parse)
        if output:
            graph.render(output, view=view)

class Decompose():
    def __init__(self, **kwargs):
        # type_dict: POS → Type
        all_POS = {'--',  'adj', 'adv', 'comp', 'comparative', 'det', 'fixed', 'name', 'noun', 'num',
                    'part', 'pp', 'prefix', 'prep', 'pron', 'punct', 'tag', 'verb', 'vg', 'smain'}
        all_CAT = {'du', 'mwu', 'ssub', 'ppres', 'smain', 'detp', 'rel', 'ppart', 'oti', 'sv1', 'ap', 'svan', 'whq',
                   'pp', 'whsub', 'ti', 'ahi', 'whrel', 'advp', 'np', 'conj', 'cp', 'inf', 'top'}
        self.type_dict = {x: x for x in {*all_POS, *all_CAT}}

    @staticmethod
    def is_leaf(node):
        if 'word' in node.attrib.keys():
            return True
        else:
            return False

    @staticmethod
    def group_by_parent(xtree):
        """
        Converts the representation from ETree to a dictionary mapping parents to their children
        :param xtree:
        :return:
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
    def split_dag(grouped, cats_to_remove=['du'], rels_to_remove=['dp', 'sat', 'nucl', 'tag', '--', 'top']):
        # todo: write this neatly
        """
        take a dictionary that contains headless structures and return multiple dictionaries that don't
        :param grouped: the original dictionary to split
        :return:
        """
        keys_to_remove = list()

        for key in grouped.keys():
            if key is not None:
                if 'cat' in key.attrib.keys():
                    if key.attrib['cat'] in cats_to_remove: # or key.attrib['cat'] == 'conj':
                        keys_to_remove.append(key)

        for key in keys_to_remove:
            del grouped[key]  # here we delete the node from being a parent

        # here we remove children
        for key in grouped.keys():
            children_to_remove = list()
            for c, r in grouped[key]:
                if c in keys_to_remove:
                    children_to_remove.append([c, r])
                elif r in rels_to_remove:
                    children_to_remove.append([c, r])
            for c in children_to_remove:
                grouped[key].remove(c)

        repeat = 1
        while repeat:
            empty_keys = list()
            repeat = 0

            for key in grouped.keys():
                if not grouped[key]:
                    repeat = 1
                    empty_keys.append(key)
            for key in empty_keys:
                del grouped[key]
            for key in grouped.keys():
                for c, r in grouped[key]:
                    if c in empty_keys:
                        repeat = 1
                        grouped[key].remove([c, r])

        return grouped

    @staticmethod
    def abstract_object_to_subject(grouped):
        # todo: write this neatly
        for parent in grouped.keys():
            if parent.attrib['cat'] != 'ssub' and parent.attrib['cat'] != 'smain':
                continue
            subject = list(filter(lambda x: x[1] == ['su', 'secondary'], [x for x in grouped[parent]]))
            if not subject:
                continue
            subject = subject[0][0]
            ppart = list(filter(lambda x: (x[0].attrib['cat'] == 'ppart' or x[0].attrib['cat'] == 'inf'),
                                [x for x in grouped[parent] if 'cat' in x[0].attrib]))
            if not ppart:
                continue
            ppart = ppart[0][0]
            print(ppart.attrib)
            abstract_so = list(filter(lambda x: (x[1] == ['obj1', 'primary'] or x[1] == ['su', 'primary']) and
                                                    (x[0].attrib['index'] == subject.attrib['index']),
                                      [x for x in grouped[ppart] if 'index' in x[0].attrib]))
            if not abstract_so:
                continue
            rel = abstract_so[0][1][0]
            abstract_so = abstract_so[0][0]
            # # # Dictionary changes
            # remove the abstract subject / object from being a child of the ppart/inf
            grouped[ppart].remove([abstract_so, [rel, 'primary']])
            # remove the abstract so from being a child of the ssub with a secondary label
            grouped[parent].remove([abstract_so, ['su', 'secondary']])
            # add it again with a primary label
            grouped[parent].append([abstract_so, ['su', 'primary']])
            # # # Internal node changes (for consistency) # todo redundant
            # remove the primary edge property from abstract object
            del abstract_so.attrib['rel'][ppart.attrib['id']]
            # convert the secondary edge to primary internally
            abstract_so.attrib['rel'][parent.attrib['id']] = ['su', 'primary']
        return grouped

    @staticmethod
    def remove_abstract_so(grouped):
        for parent in grouped.keys():
            if parent.attrib['cat'] != 'ppart' and parent.attrib['cat'] != 'inf':
                continue
            for child, rel in grouped[parent]:
                if type(rel) != list:
                    continue
                if (rel[0] == 'su' or rel[0] == 'obj1'):
                    if rel[1] == 'secondary':
                        # # # Dictionary changes
                        # remove the abstract s/o from being a child of the ppart/inf
                        grouped[parent].remove([child, rel])
                        # # # Internal node changes (for consistency) # todo redundant
                        del child.attrib['rel'][parent.attrib['id']]
                    else:
                        if 'index' in child.keys():
                            warn('Found primary object between {} and {}'.format(parent.attrib['id'],
                                                                                 child.attrib['id']))
                        #ToGraphViz()(grouped, output='abc')

        return grouped

def main():
    # # # # # Example pipelines
    ### Gather all lemmas/pos/cat/...
    # lemma_transform = Compose([lambda x: x[2],
    #                            lambda x: Lassy.get_property_set(x, 'cat')])
    # L = Lassy(transform=lemma_transform)
    # lemmatizer = DataLoader(L, batch_size=256, shuffle=False, num_workers=8, collate_fn=Lassy.reduce_property)
    # lemmas = Lassy.reduce_property([batch for batch in tqdm(lemmatizer)])
    # return L, lemmatizer, lemmas
    #
    # ## Gather all trees. remove modifiers and punct and convert to DAGs
    tree_transform = Compose([lambda x: x])
                             # lambda x: Lassy.remove_subtree(x, {'pos': 'punct', 'rel': 'mod'}),
                             # lambda x: Lassy.remove_abstract_so(x),
                             # Lassy.tree_to_dag])
    L0 = Lassy(transform=tree_transform, ignore=False)
    # forester = DataLoader(L, batch_size=256, shuffle=False, num_workers=8, collate_fn=lambda x: list(chain(x)))
    # trees = list(chain(*[batch for batch in tqdm(forester)]))
    # return L, forester, trees

    ### Gather all sentences
    # text_transform = Compose([lambda x: x[2].getroot().find('sentence')])
    # L = Lassy(transform=text_transform)
    # sentencer = DataLoader(L, batch_size=256, shuffle=False, num_workers=8, collate_fn=lambda x: list(chain(x)))
    # sentences = list(chain(*[batch for batch in sentencer]))

    ### Find all same-level dependencies without a head
    # find_non_head = Compose([lambda x: [x[0], x[2]],
    #                          lambda x: [x[0], Lassy.remove_subtree(x[1], {'pos': 'punct', 'rel': 'mod'})],
    #                          lambda x: [x[0], Lassy.remove_abstract_so(x[1])],
    #                          lambda x: [x[0], Lassy.tree_to_dag(x[1])],
    #                          lambda x: [x[0], Decompose.group_by_parent(x[1])],
    #                          lambda x: [x[0], Decompose.find_non_head(x[1])]])
    # L = Lassy(transform=find_non_head)
    # finder = DataLoader(L, batch_size=256, shuffle=False, num_workers=8,
    #                     collate_fn=lambda y: set(chain.from_iterable(filter(lambda x: x[1], y))))
    # non_heads = set()
    # for batch in tqdm(finder):
    #     non_heads |= batch
    # return L, finder, non_heads

    ### Assert the grouping by parent
    decomposer = Decompose()
    find_non_head = Compose([lambda x: [x[0], x[2]],
                             # lambda x: [x[0], Lassy.remove_abstract_so(x[1], inline=False)],
                             lambda x: [x[0], Lassy.tree_to_dag(x[1], inline=False)],
                             lambda x: [x[0], Decompose.group_by_parent(x[1])],
                             # lambda x: [x[0], Decompose.sanitize(x[1])],
                             # lambda x: [x[0], Decompose.collapse_mwu(x[1])],
                             lambda x: [x[0], Decompose.split_dag(x[1],
                                                                  cats_to_remove=['du'],
                                                                  rels_to_remove=['dp', 'sat', 'nucl', 'tag', '--',
                                                                                  'top'])],
                             lambda x: [x[0], Decompose.abstract_object_to_subject(x[1])],
                             lambda x: [x[0], Decompose.remove_abstract_so(x[1])],
                             #lambda x: [x[0], Decompose.get_disconnected(x[1])]])
                             #lambda x: [x[0], decomposer(x[1])]])
                             #lambda x: [x[0], Decompose.test_iter_group(x[1])]])
                             ])
    L = Lassy(transform=find_non_head, ignore=False)
    asserter = DataLoader(L, batch_size=256, shuffle=False, num_workers=8,
                          collate_fn=lambda y: list(chain(y)))
    #                      collate_fn=lambda y: list((filter(lambda x: x[1] != [], y))))
    #bad_groups = list(chain.from_iterable(filter(lambda x: x != [], [i for i in tqdm(asserter)])))
    return L0, L, asserter #, bad_groups
