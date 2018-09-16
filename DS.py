import os
import xml.etree.cElementTree as ET
from glob import glob

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torchvision import transforms

import networkx as nx
import graphviz

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
    A simple iterator over an xml parse that returns the parse tree's nodes
    :param xtree:
    :return:
    """
    root = xtree.getroot().find('node') # first child of root is actual root
    yield (root, None, root.attrib)
    parents = [root]

    while parents:
        children = []
        for parent in parents:
            for child in parent.findall('node'):
                children.append(child)
                yield (child, parent, child.attrib)
            parents = children

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

class ToGraphViz():
    def __init__(self):
        pass

    def __call__(self, xtree, output='gv_output', view=True):
        nodes = list(extract_nodes(xtree))
        graph = graphviz.Digraph()
        graph.node(str(nodes[0][0]), label='ROOT')
        for name, parent, attr in nodes[1:]:
            try:
                label = attr['word']
            except KeyError:
                label = ''
            graph.node(str(name), label=label)
            graph.edge(str(parent), str(name))

        graph.render(output, view=view)

def main():
    # dummy transforms
    composed = transforms.Compose([lambda x: x[1].getroot()])
    text = transforms.Compose([lambda x: x[1].findtext('sentence')])
    L = Lassy(transform=text)

    faster = DataLoader(L, batch_size=8, shuffle=False, num_workers=8)
    # full list of dataloader objects may be obtained via list(chain.from_iterable([text for text in faster]))
    return L, faster