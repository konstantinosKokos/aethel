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

    def __getitem__(self, id):
        """
        :param file:
        :return:
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
    def count_tokens(xtree):
        """
        use as map()
        :param xtree:
        :return:
        """

        return len(xtree.split(' '))

    @staticmethod
    def get_property_set(xtree, property):
        """
        use as map()
        :param property:
        :param xtree:
        :return:
        """
        nodes = list(xtree.iter('node'))
        nodes = filter(lambda x: property in x.attrib, nodes)
        return set(map(lambda x: x.attrib[property], nodes))

    @staticmethod
    def reduce_property(set_of_sets):
        """
        use as reduce()
        :param set_of_property_items:
        :return:
        """
        property_items = set()
        return property_items.union(*set_of_sets)

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
                                #break

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
    def sanitize(grouped):
        """
        ad-hoc post-processing cleanup
        :param grouped:
        :return:
        """
        for key in grouped:
            seen = []
            for child, rel in grouped[key]:
                if (child not in grouped.keys() and not Decompose.is_leaf(child) and rel != 'top') \
                        or (child in seen):
                    grouped[key].remove([child,rel])
                seen.append(child)
        return grouped


    @staticmethod
    def choose_head(children_rels):
        #todo
        """
        :param children_rels: a list of [node, rel] lists
        :return:
            if a head is found, return the head
            if structure is headless, return None
            if unspecified case, return -1
        """
        # todo: perhaps deeper iteration necessary
        # todo: conjunction, dp, nucl/sat
        # todo: mwp?
        candidates = ['hd', 'rhd', 'whd', 'cmp', 'crd', 'dlink']
        throwaway = ['top', '--', 'mwp']  # .. just ignore these
        for i, (candidate, rel) in enumerate(children_rels):
            if rel in candidates:
                return candidate
        for _, rel in children_rels:
            if rel in throwaway:
                return None
        return -1

    @staticmethod
    def find_non_head(grouped):
        non_head = []
        for key in grouped.keys():
            if Decompose.choose_head(grouped[key]) == -1:
                non_head.append(list(map(lambda x: x[1], grouped[key])))
        return non_head

    @staticmethod
    def test_iter_group(grouped):
        for key in grouped:
            seen = []
            for child, rel in grouped[key]:
                if child not in grouped.keys() and rel != 'top':
                    try:
                        assert(Decompose.is_leaf(child))
                    except AssertionError:
                        print(child.attrib)
                        return grouped
                    try:
                        assert(child not in seen)
                        seen.append(child)
                    except AssertionError:
                        print(child.attrib)
                        return grouped

        return []

    def recursive_call(self, parent, grouped, start_type, lexicon):
        def get_key(node):
            return node.attrib['lemma']

        def add_to_lexicon(lexicon, to_add):
            if set(to_add.keys()).intersection(set(lexicon.keys())):
                print(lexicon)
                print('---------------')
                print(to_add)
            return {**lexicon, **to_add}

        # called with a parent that is a leaf
        if parent not in grouped.keys():
            return add_to_lexicon(lexicon, {get_key(parent): {'primary': start_type}})

        # find the next functor
        headchild = Decompose.choose_head(grouped[parent])
        if headchild == -1:
            raise ValueError('No head found')
        elif headchild is None:
            raise NotImplementedError('Headless structure')

        # if headchild has siblings, it is a function that accepts the siblings types as arguments
        # and returns the type of its parent
        arglist = []

        # todo: do not kill of the dependencies yet
        # todo: is lemmatization necessary at this part? perhaps return both word and lemma
        # todo: optimization: no need to iterate twice over visited nodes
        # todo: secondary types?
        # todo: what if a lemma/word/whatever is used twice with different types? dictionary is the wrong datastruct

        # todo: is this the proper way of ordering? no..
        ordered_children = sorted(grouped[parent], key=lambda x: x[0].attrib['begin'])

        for child, rel in ordered_children:
            # ignore self during iteration
            if child == headchild:
                continue
            # find the type
            child_type = self.get_plain_type(child)
            # add the type and dependency as arguments
            arglist.append((child_type, rel))

            # did we reach the end?
            if Decompose.is_leaf(child):
                if child not in lexicon.keys():
                    lexicon = add_to_lexicon(lexicon, {get_key(child): {'secondary': child_type}})
                elif child in lexicon.keys():
                    assert(lexicon[child] == child_type)
            else:
                # flow downwards this path
                lexicon = self.recursive_call(child, grouped, child_type, lexicon)

        # time to flow down the headchild
        if arglist:
            lexicon = self.recursive_call(headchild, grouped, [arglist, start_type], lexicon)
        else:
            lexicon = self.recursive_call(headchild, grouped, start_type, lexicon)

        return lexicon

    @staticmethod
    def reduce_secondary(lexicon):
        #todo
        return lexicon
        for key in lexicon:
            if 'primary' not in lexicon[key].keys():
                lexicon[key]['primary'] = lexicon[key]['secondary']
                del(lexicon[key]['secondary'])
        return lexicon

    def get_plain_type(self, node):
        try:
            return self.type_dict[node.attrib['pos']]
        except KeyError:
            return self.type_dict[node.attrib['cat']]

    def __call__(self, grouped):
        # todo: some function to remove useless root nodes
        start, rel = grouped[None][0]
        while rel == 'top':
            start, rel = grouped[start][0]
        start_type = self.get_plain_type(start)
        lexicon = dict()
        lexicon = self.recursive_call(start, grouped, start_type, lexicon)
        return Decompose.reduce_secondary(lexicon)

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
    tree_transform = Compose([lambda x: x[2],
                              lambda x: Lassy.remove_subtree(x, {'pos': 'punct', 'rel': 'mod'}),
                              lambda x: Lassy.remove_abstract_subject(x),
                              Lassy.tree_to_dag])
    L0 = Lassy(transform = tree_transform)
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
    #                          lambda x: [x[0], Lassy.remove_abstract_subject(x[1])],
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
    find_non_head = Compose([lambda x: [x[0], x[2]],
                             lambda x: [x[0], Lassy.remove_subtree(x[1], {'pos': 'punct', 'rel': 'mod'}, inline=True)],
                             lambda x: [x[0], Lassy.remove_abstract_subject(x[1], inline=True)],
                             lambda x: [x[0], Lassy.tree_to_dag(x[1], inline=True)],
                             lambda x: [x[0], Decompose.group_by_parent(x[1])],
                             lambda x: [x[0], Decompose.sanitize(x[1])]])
                             #lambda x: [x[0], Decompose.test_iter_group(x[1])]])
    L = Lassy(transform=find_non_head)
    asserter = DataLoader(L, batch_size=256, shuffle=False, num_workers=8,
                          collate_fn=lambda y: list((filter(lambda x: x[1] != [], y))))
    #bad_groups = list(chain.from_iterable(filter(lambda x: x != [], [i for i in tqdm(asserter)])))
    return L0, L, asserter #, bad_groups




    tree = Compose([lambda x: x[1]])
    L = Lassy(transform=tree)
    samples = [L[i] for i in [10, 20, 30, 40, 50, 100, 500, 1000, 200]]
    faster = DataLoader(L, batch_size=256, shuffle=False, num_workers=8, collate_fn=lambda x: list(chain(x)))

    tg = ToGraphViz()

    return samples, ToGraphViz(), faster