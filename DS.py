import os
import xml.etree.cElementTree as ET
from glob import glob

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torchvision import transforms

import networkx as nx
import graphviz

from itertools import groupby
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

    yield (root.attrib, None, depth)

    while parents:
        depth += 1
        children = []
        for parent in parents:
            for child in parent.findall('node'):
                children.append(child)
                yield (child.attrib, parent.attrib, depth)
            parents = children

def group_by_depth(nodes):
    """
    Takes a list of nodes and groups them by depth.
    :param nodes:
    :return:
    """
    nodes = sorted(nodes, key = lambda x: x[2])
    groupings = groupby(nodes, key = lambda x: x[2])
    return [[node for node in group] for key, group in groupings]

def find_coindexed(nodes):
    pass

def is_tree(xtree):
    pass

def has_coindexed_subject(xtree):
    nodes = [node for node in xtree.iter('node')]
    for node in nodes:
        if 'index' in node.attrib.keys() and 'rel' in node.attrib.keys():
            if node.attrib['rel'] == 'su':
                return True
    return False

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
                    if key in child.attrib: # is the child coindexed?
                        if child.attrib[key] == criteria[key]:
                            parent.remove(child)
                            removed = True
                            break
                if not removed:
                    children.append(child)
        parents = children
    return xtree

def remove_abstract_subject(xtree):
    xtree = deepcopy(xtree)

    suspects = filter(lambda x: 'cat' in x.attrib.keys(),
                      [node for node in xtree.iter('node')])
    suspects = filter(lambda x: x.attrib['cat'] == 'ppart' or x.attrib['cat'] == 'inf',
                      suspects)

    has_abstract_subject = False

    for suspect in suspects:
        abstract_subjects = filter(lambda x: 'rel' in x.attrib.keys(), suspect.findall('node'))
        abstract_subjects = filter(lambda x: x.attrib['rel'] == 'su', abstract_subjects)
        for abstract_subject in abstract_subjects:
            has_abstract_subject = True
            suspect.remove(abstract_subject)

    return xtree, has_abstract_subject



def tree_to_dag(xtree):
    xtree = deepcopy(xtree)
    root = xtree.getroot().find('node')
    parents = [root]

    unique = {root.attrib['id'] : root} # a dictionary of all nodes visited
    coindex = dict() # a dictionary of mutually indexed nodes

    while parents:
        children = []
        for parent in parents:
            for child in parent.findall('node'):
                unique[child.attrib['id']] = child
                children.append(child)

                # collapse and resolve coindexing
                if 'index' in child.attrib: # is the child coindexed?
                    if not child.attrib['index'] in coindex.keys(): # is it already stored?
                        coindex[child.attrib['index']] = child  # if not, assign the first occurrence
                    actual_child = coindex[child.attrib['index']] # refer to the first coindex instance
                    parent.remove(child) # remove the fake
                    actual_child = deepcopy(actual_child)
                    actual_child.attrib['rel'] = child.attrib['rel']
                    parent.append(actual_child) # add the original

        parents = children
    return xtree

class ToGraphViz():
    def __init__(self, to_show = {'node': ['word', 'pos', 'cat', 'index'], 'edge': {'rel'}}):
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

    def construct_edge_label(self, child):
        """
        :param child:
        :param parent:
        :return:
        """
        label = ''
        for key in self.to_show['edge']:
            label += child[key] + '\n'
        return label

    def __call__(self, xtree, output='gv_output', view=True):
        nodes = list(extract_nodes(xtree)) # a list of triples
        graph = graphviz.Digraph()

        graph.node('title', label=xtree.findtext('sentence'), shape='none')

        graph.node(self.get_name(nodes[0][0]), label='ROOT')

        graph.edge('title', self.get_name(nodes[0][0]), style='invis')

        for child_attr, parent_attr, _ in nodes[1:]:
            node_label = self.construct_node_label(child_attr)
            graph.node(self.get_name(child_attr), label=node_label)
            edge_label = self.construct_edge_label(child_attr)
            graph.edge(self.get_name(parent_attr), self.get_name(child_attr), label=edge_label)

        if output:
            graph.render(output, view=view)

        return graph

class ToNetworkX():
    def __init__(self):
        pass

    def __call__(self, xtree, view=False):
        nodes = list(extract_nodes(xtree))
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

def main():
    # dummy transforms
    composed = transforms.Compose([lambda x: x[1].getroot()])
    text = transforms.Compose([lambda x: x[1].findtext('sentence')])
    L = Lassy()

    samples = [L[i][1] for i in [10,20,30,40,50,100,500,1000,200]]

    faster = DataLoader(L, batch_size=8, shuffle=False, num_workers=8)
    # full list of dataloader objects may be obtained via list(chain.from_iterable([text for text in faster]))

    tg = ToGraphViz()

    # v This is how to get all trees that were pruned by abstract subject clause removal
    #bad = list(filter(lambda x: DS.remove_abstract_subject(x)[1], samples))
    #tg(samples[1])
    return samples, ToGraphViz()