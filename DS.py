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

from pprint import pprint as print

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
        parse.getroot().set('type', 'Tree')

        sample = (file, parse)

        if self.transform:
            return self.transform(sample)

        return sample

    @staticmethod
    def count_tokens(xtree):
        """
        use as map()
        :param xtree:
        :return:
        """

        return len(xtree.split(' '))

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
    def extract_nodes(xtree):
        """
        A simple iterator over an xml parse that returns the parse tree's nodes. This is necessary as the default ET
        iterator does not provide parent or depth info.
        :param xtree:
        :return (child_node, parent_node, depth)
        """

        root = xtree.getroot().find('node')
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
    def rel_to_dict(xtree):
        xtree = deepcopy(xtree)
        nodegroups = list(Lassy.extract_nodes(xtree))
        for node, parent, _ in nodegroups[1:]:
            node.attrib['rel'] = {parent.attrib['id']: node.attrib['rel']}
        return xtree

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
    def replace_main_coindex(all_coindexed, main_child):
        """
        Returns a new main coindex and removes the old main from the list of coindexed
        :param all_coindex: all nodes sharing the same mutual index
        :param main_child: the previous main node
        :return:
        """
        new_candidates = list(filter(lambda x: x != main_child, all_coindexed))
        all_coindexed.remove(main_child)
        # make sure no descendant is lost
        for subchild in main_child.findall('node'):
            new_candidates[0].append(subchild)
        # replace all properties except for the id (??) # todo
        for key, value in main_child.attrib.items():
            new_candidates[0].set(key, value)
        return new_candidates[0]

    @staticmethod
    def tree_to_dag(xtree, inline=False):
        if not inline:
            xtree = deepcopy(xtree)

        nodes = list(Lassy.extract_nodes(xtree))
        
        xtree.getroot().set('type', 'DAG')

        _, main_coind = Lassy.find_main_coindex(xtree)

        for node, parent, _ in nodes[1:]:
            if node in main_coind.values():
                node.attrib['rel'] = {parent.attrib['id']: node.attrib['rel']}


        for node, parent, _ in nodes[1:]:
            if type(node.attrib['rel']) == str:
                if 'index' in node.attrib.keys():

                    main_coind[node.attrib['index']].attrib['rel'] = {parent.attrib['id']: node.attrib['rel'],
                                                                      **main_coind[node.attrib['index']].attrib['rel']}
                    parent.remove(node)
                else:
                    node.attrib['rel'] = {parent.attrib['id']: node.attrib['rel']}
        return xtree

    @staticmethod
    def remove_subtree(xtree, criteria, inline=False):
        """
        usecase:
            - remove_subtree(xtree, {'pos': 'punct', 'rel': 'mod'}) will remove punctuation and modifiers
        :param xtree: the xml tree to be modified
        :param criteria: a dictionary of key-value pairs
        :return: the pruned tree according to input criteria
        """
        if not inline:
            xtree = deepcopy(xtree)

        root = xtree.getroot().find('node')
        parents = [root]

        all_coind, main_coind = Lassy.find_main_coindex(xtree)

        while parents:
            children = []
            for parent in parents:
                for child in parent.findall('node'):
                    removed = False
                    for key in criteria.keys():
                        if key in child.attrib:
                            if child.attrib[key] == criteria[key]:
                                for subchild in child.iter('node'):
                                    if subchild in main_coind.values():
                                        # trying to remove main coindex
                                        coindex = subchild.attrib['index']
                                        if subchild == main_coind[coindex] and len(all_coind[coindex]) > 1:
                                            main_coind[coindex] = Lassy.replace_main_coindex(all_coind[coindex],
                                                                                             subchild)
                                            # if only 1 item left, it no longer coindexes anything
                                            if len(all_coind[coindex]) == 1:
                                                del main_coind[coindex].attrib['index']
                                                del main_coind[coindex]
                                                del all_coind[coindex]
                                        else:
                                            for key in all_coind.keys():
                                                try:
                                                    all_coind[key].remove(subchild)
                                                except ValueError:
                                                    continue
                                parent.remove(child)
                                removed = True
                                break

                    if not removed:
                        children.append(child)
            parents = children
        return xtree

    @staticmethod
    def remove_abstract_subject(xtree, inline=False):
        if not inline:
            xtree = deepcopy(xtree)

        all_coind, main_coind = Lassy.find_main_coindex(xtree)

        suspects = filter(lambda x: 'cat' in x.attrib.keys(),
                          [node for node in xtree.iter('node')])
        suspects = list(filter(lambda x: x.attrib['cat'] == 'ppart' or x.attrib['cat'] == 'inf',
                          suspects))
        #print('suspects:', [s.attrib['id'] for s in suspects])

        if not suspects:
            return xtree

        for suspect in suspects:
            abstract_subjects = filter(lambda x: 'rel' in x.attrib.keys(), suspect.findall('node'))
            abstract_subjects = list(filter(lambda x: x.attrib['rel'] == 'su', abstract_subjects))
            #abstract_subjects = list(filter(lambda x: 'index' in x.attrib.keys(), abstract_subjects))
            #print('abstract subjects: ', [a.attrib['id'] for a in abstract_subjects])
            for abstract_subject in abstract_subjects:
                for subchild in abstract_subject.iter('node'):
                    if subchild in main_coind.values():
                        # trying to remove main coindex
                        coindex = subchild.attrib['index']
                        if subchild == main_coind[coindex] and len(all_coind[coindex]) > 1:
                            main_coind[coindex] = Lassy.replace_main_coindex(all_coind[coindex], subchild)
                            # if only 1 item left, it no longer coindexes anything
                            if len(all_coind[coindex]) == 1:
                                del main_coind[coindex].attrib['index']
                                del main_coind[coindex]
                                del all_coind[coindex]
                    else:
                        for key in all_coind.keys():
                            try:
                                all_coind[key].remove(subchild)
                            except ValueError:
                                continue

                suspect.remove(abstract_subject)
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

    def __call__(self, xtree, output='gv_output', view=True):
        nodes = list(Lassy.extract_nodes(xtree)) # a list of triples
        graph = graphviz.Digraph()

        graph.node('title', label=xtree.findtext('sentence'), shape='none')
        graph.node(nodes[0][0].attrib['id'], label='ROOT')
        graph.edge('title', nodes[0][0].attrib['id'], style='invis')

        for child, parent, _ in nodes[1:]:
            node_label = self.construct_node_label(child.attrib)
            graph.node(child.attrib['id'], label=node_label)

            if type(child.attrib['rel']) == str:
                graph.edge(parent.attrib['id'], child.attrib['id'], label=child.attrib['rel'])
            else:
                for parent_id, dependency in child.attrib['rel'].items():
                    graph.edge(parent_id, child.attrib['id'], label=dependency)

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
        # type_dict: POS → Type
        type_dict = dict()

    @staticmethod
    def is_leaf(node):
        if 'word' in node.attrib.keys():
            return True
        else:
            return False

    @staticmethod
    def group_by_parent(xtree):
        # todo: consider the effect of obsolete ids here
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
        grouped = {k: [[v[0],v[2]] for v in V] for k, V in grouped}
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
    def find_non_head(grouped, start=None):
        # todo: iterate over a DAG and find all same-depth siblings where no 'hd' is present
        this_level_rels = []
        for child, rel in grouped[start]:
            this_level_rels.append([child.attrib['id'], rel])
            try:
                Decompose.find_non_head(grouped, start=child)
            except KeyError:
                continue
        if start:
            print(start.attrib['id'])
        else:
            print('None')
        print(this_level_rels)


    @staticmethod
    def test_iter_group(grouped):
        # todo some doctest / sanity check
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
    L = Lassy(transform=tree)
    samples = [L[i] for i in [10, 20, 30, 40, 50, 100, 500, 1000, 200]]
    faster = DataLoader(L, batch_size=256, shuffle=False, num_workers=8, collate_fn=lambda x: list(chain(x)))

    tg = ToGraphViz()

    return samples, ToGraphViz(), faster