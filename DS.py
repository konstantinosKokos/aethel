import os
import xml.etree.cElementTree as ET
from glob import glob

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torchvision.transforms import Compose

import networkx as nx
import graphviz

from itertools import groupby, chain
from copy import deepcopy

class Lassy(Dataset):
    """
    Lassy dataset
    """

    def __init__(self, root_dir='/home/kokos/Documents/Projects/LassySmall 4.0', treebank_dir = '/Treebank',
                 transform=None):
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

        self.filelist = [y for x in os.walk(self.treebank_dir) for y in glob(os.path.join(x[0], '*.[xX][mM][lL]'))]
        self.transform = transform

        print('Dataset constructed with {} samples.'.format(len(self.filelist)))

    def __len__(self):
        """
        :return:
        """
        return len(self.filelist)

    def __getitem__(self, file):
        """
        :param file:
        :return:
        """

        if type(file) == int:
            file = self.filelist[file]
        elif type(file) == str:
            pass
        else:
            raise TypeError('file argument has to be int or str')

        parse = ET.parse(file)

        sample = (file, parse)

        if self.transform:
            return self.transform(sample)

        return sample

    def count_tokens(self):
        count = 0

        for i in tqdm(range(len(self))):
            count+= len(self[i][1].split(' '))

        return count

    def get_lemmas(self):
        #todo replace with map/reduce for efficiency?
        lemmas = set()

        for i in tqdm(range(len(self))):
            lemmas |= Lassy.get_lemmas(self[i][1])

        return lemmas

    @staticmethod
    def extract_nodes(xtree):
        """
        A simple iterator over an xml parse that returns the parse tree's nodes. This is necessary as the default ET
        iterator does not provide parent or depth info.
        :param xtree:
        :return:
        """
        root = xtree.getroot().find('node') # first child of root is actual root (there is an 'alpino' node at the top)
        parents = [root]
        depth = 0

        yield (root, None, depth)

        while parents:
            depth += 1
            children = []
            for parent in parents:
                for child in parent.findall('node'):
                    children.append(child)
                    yield (child, parent, depth)
                parents = children


    @staticmethod
    def tree_to_dag2(xtree):
        xtree = deepcopy(xtree)

        # todo rewrite this properly
        coindexed = list(filter(lambda x: 'index' in x.attrib.keys(), xtree.iter('node')))
        if not coindexed:
            return xtree
        coindex = {i: [node for node in group] for i, group in
                   groupby(sorted(coindexed, key=lambda x: x.attrib['index']), key=lambda x: x.attrib['index'])}
        # find the 'main' child for each set of siblings
        coindex = {i: list(filter(lambda x: 'cat' in x.attrib or 'word' in x.attrib, nodes))[0]
                   for i, nodes in coindex.items()}

        nodes = list(Lassy.extract_nodes(xtree))

        for node, parent, depth in nodes[1:]:
            if type(node.attrib['rel'] == str) and node in coindex.values():
                node.attrib['rel'] = {parent.attrib['id']: node.attrib['rel']}
            if 'index' in node.attrib.keys() and node not in coindex.values():
                #print('BEFORE {}'.format(coindex[node.attrib['index']].attrib['rel']))
                #print('ADDING {}'.format({parent.attrib['id']: node.attrib['rel']}))#

                coindex[node.attrib['index']].attrib['rel'] = {parent.attrib['id']: node.attrib['rel'],
                                                              **coindex[node.attrib['index']].attrib['rel']}
                #print('AFTER: {}'.format(coindex[node.attrib['index']].attrib['rel']))
                coindex[node.attrib['index']].attrib['rel']
                parent.remove(node)
        return xtree


    @staticmethod
    def tree_to_dag(xtree):
        # todo just rewrite the whole fucking thing because it's completely wrong
        # todo - bug: iterates twice through coindexed subtrees
        """
        finds all occurrences of co-indexing within a tree and removes them, constructing appropriate links when necessary
        :param xtree: the tree to convert to a DAG
        :return: the xml parse of the resulting DAG
        """
        xtree = deepcopy(xtree)
        root = xtree.getroot().find('node')
        parents = [root]

        coindexed = list(filter(lambda x: 'index' in x.attrib.keys(), xtree.iter('node')))
        if not coindexed:
            return xtree
        coindex = {i: [node for node in group] for i, group in
                   groupby(sorted(coindexed, key=lambda x: x.attrib['index']), key=lambda x: x.attrib['index'])}
        # find the 'main' child for each set of siblings
        coindex = {i: list(filter(lambda x: 'cat' in x.attrib or 'word' in x.attrib, nodes))[0]
                   for i, nodes in coindex.items()}

        while parents:
            children = []
            for parent in parents:
                for child in parent.findall('node'):
                    children.append(child)
                    # collapse and resolve coindexing
                    if 'index' in child.attrib:  # is the child coindexed?
                        if coindex[child.attrib['index']].attrib['id'] != child.attrib['id']:  # is it NOT the main?
                            #actual_child = ET.SubElement(parent, child.tag, coindex[child.attrib['index']].attrib)
                            actual_child = deepcopy(coindex[child.attrib['index']])
                            actual_child.attrib['rel'] = child.attrib['rel']
                            parent.remove(child)
                            parent.append(actual_child)
                            children.remove(child)

            parents = children
        return xtree

    @staticmethod
    def get_lemmas(xtree):
        """
        use as map()
        :param xtree:
        :return:
        """
        nodes = [n for n in xtree.iter('node')]
        nodes = filter(lambda x: 'lemma' in x.attrib, nodes)
        return set(map(lambda x: x.attrib['lemma'], nodes))

    @staticmethod
    def reduce_lemmas(set_of_lemmas):
        """
        use as reduce()
        :param set_of_lemmas:
        :return:
        """
        lemmas = set()
        return lemmas.union(*set_of_lemmas)

    @staticmethod
    def group_by_depth(nodes):
        """
        Takes a list of nodes and groups them by depth.
        :param nodes:
        :return:
        """
        nodes = sorted(nodes, key = lambda x: x[2])
        groupings = groupby(nodes, key = lambda x: x[2])
        return [[node for node in group] for key, group in groupings]

    @staticmethod
    def remove_subtree(xtree, criteria):
        """
        usecase:
            - remove_subtree(xtree, {'pos': 'punct', 'rel': 'mod'}) will remove punctuation and modifiers
        :param xtree: the xml tree to be modified
        :param criteria: a dictionary of key-value pairs
        :return: the pruned tree according to input criteria
        """
        xtree = deepcopy(xtree)
        root = xtree.getroot().find('node')
        parents = [root]

        while parents:
            children = []
            for parent in parents:
                for child in parent.findall('node'):
                    removed = False
                    for key in criteria.keys():
                        if key in child.attrib:  # is the child coindexed?
                            if child.attrib[key] == criteria[key]:
                                parent.remove(child)
                                removed = True
                                break
                    if not removed:
                        children.append(child)
            parents = children
        return xtree

    @staticmethod
    def remove_abstract_subject(xtree):
        suspects = filter(lambda x: 'cat' in x.attrib.keys(),
                          [node for node in xtree.iter('node')])
        suspects = list(filter(lambda x: x.attrib['cat'] == 'ppart' or x.attrib['cat'] == 'inf',
                          suspects))

        if not suspects:
            return xtree

        xtree = deepcopy(xtree)

        for suspect in suspects:
            abstract_subjects = filter(lambda x: 'rel' in x.attrib.keys(), suspect.findall('node'))
            abstract_subjects = filter(lambda x: x.attrib['rel'] == 'su', abstract_subjects)
            abstract_subjects = filter(lambda x: 'index' in x.attrib.keys(), abstract_subjects)
            for abstract_subject in abstract_subjects:
                suspect.remove(abstract_subject)

        return xtree

# todo: obsolete
def has_coindexed_subject(xtree):
    nodes = [node for node in xtree.iter('node')]
    for node in nodes:
        if 'index' in node.attrib.keys() and 'rel' in node.attrib.keys():
            if node.attrib['rel'] == 'su':
                return True
    return False

class ToGraphViz():
    #todo: allow multi-type input (tree / DAG)
    def __init__(self, to_show = {'node': ['id', 'index'], #''word', 'pos', 'cat', 'index'],
                                           'edge': {'rel'} }):

        self.name_property = 'id'
        self.to_show = to_show


    def get_name(self, node_attr):
        """
        :param node_triple: A "node" element as returned by extract_nodes()
        :return:
        """
        return node_attr[self.name_property]

    def construct_node_label(self, child):
        """
        :param child:
        :return:
        """
        label = ''
        for key in self.to_show['node']:
        #for key in child:
            if key != 'span':
                try:
                    label += child[key] + '\n'
                except KeyError:
                    pass
            else:
                label += child['begin'] + '-' + child['end'] + '\n'
        return label

    def construct_edge_label(self, parent, child):
        #todo
        """
        :param child:
        :param parent:
        :return:
        """
        label = ''
        for key in self.to_show['edge']:
            if type(child[key]) == dict:
                label += child[key][parent['id']]
            else:
                label += child[key] + '\n'
        return label

    def __call__(self, xtree, output='gv_output', view=True):
        nodes = list(Lassy.extract_nodes(xtree)) # a list of triples
        graph = graphviz.Digraph()

        graph.node('title', label=xtree.findtext('sentence'), shape='none')

        graph.node(self.get_name(nodes[0][0].attrib), label='ROOT')

        graph.edge('title', self.get_name(nodes[0][0].attrib), style='invis')

        for child, parent, _ in nodes[1:]:
            node_label = self.construct_node_label(child.attrib)
            graph.node(self.get_name(child.attrib), label=node_label)

            # todo: iterate over the many parents of a child to construct edges
            edge_label = self.construct_edge_label(parent.attrib, child.attrib)
            graph.edge(self.get_name(parent.attrib), self.get_name(child.attrib), label=edge_label)

        if output:
            graph.render(output, view=view)

        return graph

class ToNetworkX():
    def __init__(self):
        pass

    def __call__(self, xtree, view=False):
        nodes = list(Lassy.extract_nodes(xtree)) # todo node -> nodeattr
        tree = nx.DiGraph()
        labels = dict()

        # TODO remember the pos issue
        tree.add_node(nodes[0][0], **nodes[0][2])
        labels[nodes[0][0]] = 'ROOT'
        for name, parent, attrib in nodes[1:]:
            tree.add_node(name, **attrib)
            tree.add_edge(name, parent)
            try:
                labels[name] = attrib['word']
            except KeyError:
                labels[name] = ''

        if view:
            nx.draw(tree, labels=labels)

        return tree, labels

class Decompose():
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def assign_struct(leaf):
        pass

    def __call__(self, xtree):
        nodes = Lassy.extract_nodes(xtree) # todo node-> node attr

        # leaves are all nodes that are assigned a word
        leaves = filter(lambda x: 'word' in x[0].keys(), nodes)

        # todo: map each leave to a structure
        structures = map(self.assign_struct, leaves)

        return {leaf[0]['lemma']: struct for leaf, struct in zip(leaves, structures)}

def main():
    # # # # # Example pipelines
    ### Gather all lemmas
    # lemma_transform = Compose([lambda x: x[1], get_lemmas])
    # L = Lassy(transform=lemma_transform)
    # lemmatizer = DataLoader(L, batch_size=256, shuffle=False, num_workers=8, collate_fn=reduce_lemmas)
    # lemmas = reduce_lemmas([batch for batch in lemmatizer])
    ### Gather all trees. remove modifiers and punct and convert to DAGs
    # tree_transform = Compose([lambda x: x[1], lambda x: Lassy.remove_subtree(x, {'pos': 'punct', 'rel': 'mod'}),
    #                          lambda x: Lassy.remove_abstract_subject(x), Lassy.tree_to_dag])
    # L = Lassy(transform = tree_transform)
    # forester = DataLoader(L, batch_size=256, shuffle=False, num_workers=8, collate_fn=lambda x: list(chain(x)))
    # trees = list(chain(*[batch for batch in forester]))
    # return forester
    ### Gather all sentences
    # text_transform = Compose([lambda x: x[0]])
    # L = Lassy(transform=text_transform)
    # sentencer = DataLoader(L, batch_size=256, shuffle=False, num_workers=8, collate_fn=lambda x: list(chain(x)))
    # sentences = list(chain(*[batch for batch in sentencer]))


    tree = Compose([lambda x: x[1]])
    L = Lassy(transform = tree)
    samples = [L[i] for i in [10, 20, 30, 40, 50,100,500,1000,200]]
    faster = DataLoader(L, batch_size=256, shuffle=False, num_workers=8, collate_fn=lambda x: list(chain(x)))

    tg = ToGraphViz()

    return samples, ToGraphViz(), faster