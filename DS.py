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

    def get_filename(self, id):
        return self.filelist[id]

    @staticmethod
    def get_sentence(xtree):
        return xtree.getroot().findtext('sentence')

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
            # todo
            abstract_subjects = list(filter((lambda x: x.attrib['rel'] == 'su') or
                                            (lambda x: x.attrib['rel'] == 'obj1'),
                                            abstract_subjects))
            print([a.attrib['word'] for a in abstract_subjects])
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

    def xml_to_gv(self, xtree):
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
                graph.edge(parent.attrib['id'], child.attrib['id'], label=rel)
        return graph

    def __call__(self, parse, output='gv_output', view=True):
        if type(parse) == ET.ElementTree:
            graph = self.xml_to_gv(parse)
        else:
            graph = self.grouped_to_gv(parse)
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
        """
        :param children_rels: a list of [node, rel] lists
        :return:
            if a head is found, return the head
            if structure is headless, return None
            if unspecified case, return -1
        """
        candidates = ['hd', 'rhd', 'whd', 'cmp', 'crd', 'dlink']
        for i, (candidate, rel) in enumerate(children_rels):
            if rel in candidates:
                return candidate
        return -1

    @staticmethod
    def collapse_mwu(grouped):
        """
        placeholder function that collapses nodes with 'mwp' dependencies into a single node
        :param grouped:
        :return:
        """
        # todo: better subcase management for proper names etc. using external parser or alpino
        to_remove = []

        # find all mwu parents
        for key in grouped.keys():
            if key is not None:
                if 'cat' in key.attrib.keys():
                    if key.attrib['cat'] == 'mwu':
                        nodes = Decompose.order_siblings(grouped[key])
                        collapsed_text = ''.join([x[0].attrib['word']+' ' for x in nodes])
                        key.attrib['word'] = collapsed_text[0:-1] # update the parent text
                        to_remove.append(key)

        # parent is not a parent anymore (since no children are inherited)
        for key in to_remove:
            del(grouped[key])
        # this is the normal return type
        return grouped
        # alternative return type for when using as map()
        # return [tr.attrib['word'].lower() for tr in to_remove]

    @staticmethod
    def split_du(grouped):
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
                    if key.attrib['cat'] == 'du': # or key.attrib['cat'] == 'conj':
                        keys_to_remove.append(key)

        for key in keys_to_remove:
            del grouped[key]  # here we delete the conj/du from being a parent

        for key in grouped.keys():
            children_to_remove = list()
            for c, r in grouped[key]:
                if c in keys_to_remove:
                    children_to_remove.append([c, r])
                elif r in ['dp', 'sat', 'nucl', 'tag', '--', 'top']:
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
    def get_disconnected(grouped):
        all_keys = set(grouped.keys())
        all_children = set([x[0] for k in all_keys for x in grouped[k]])
        assert(all(map(lambda x: Decompose.is_leaf(x), all_children.difference(all_keys))))
        # except AssertionError:
        #     return list(filter(lambda x: not Decompose.is_leaf(x), all_children.difference(all_keys)))
        # return True
        return all_keys.difference(all_children)

    @staticmethod
    def order_siblings(siblings, exclude=None):
        # todo: this needs work (i.e. secondary criteria, perhaps taking arguments)
        if exclude is not None:
            siblings = list(filter(lambda x: x[0] != exclude, siblings))
        return sorted(siblings, key=lambda x: (int(x[0].attrib['begin']), int(x[0].attrib['end']),
                                               int(x[0].attrib['id'])))

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

    # def recursive_call(self, parent, grouped, start_type, lexicon):
    #     # todo: optimization: no need to iterate twice over visited nodes | IMPORTANT! skews stats
    #     # todo: secondary types?
    #     # todo: what if a lemma/word/whatever is used twice with different types? dictionary is the wrong datastruct
    #
    #     def get_key(node):
    #         return node.attrib['lemma']
    #
    #     def add_to_lexicon(lexicon, to_add):
    #         if set(to_add.keys()).intersection(set(lexicon.keys())):
    #             print(lexicon)
    #             print('---------------')
    #             print(to_add)
    #         return {**lexicon, **to_add}
    #
    #     # called with a parent that is a leaf
    #     if parent not in grouped.keys():
    #         if parent in lexicon.keys():
    #             return lexicon
    #         else:
    #             return add_to_lexicon(lexicon, {get_key(parent): {'primary': start_type}})
    #
    #     # find the next functor
    #     headchild = Decompose.choose_head(grouped[parent])
    #     assert(Decompose.is_leaf(headchild))
    #     if headchild == -1:
    #         raise ValueError('No head found')
    #
    #     # if headchild has siblings, it is a function that accepts the siblings types as arguments
    #     # and returns the type of its parent
    #     arglist = []
    #
    #     ordered_children = Decompose.order_siblings(grouped[parent], exclude=headchild)
    #
    #     for child, rel in ordered_children:
    #         # left until we find headchild, then right
    #
    #         # find the type, add the type and dependency as arguments
    #         child_type = self.get_plain_type(child)
    #         arglist.append((child_type, rel))
    #
    #         # did we reach the end?
    #         if Decompose.is_leaf(child):
    #             if child not in lexicon.keys():
    #                 lexicon = add_to_lexicon(lexicon, {get_key(child): {'secondary': child_type}})
    #             elif child in lexicon.keys():
    #                 assert (lexicon[child] == child_type)
    #         else:
    #             # flow downwards this path
    #             lexicon = self.recursive_call(child, grouped, child_type, lexicon)
    #
    #     # time to flow down the headchild
    #     if arglist:
    #         lexicon = self.recursive_call(headchild, grouped, [arglist, start_type], lexicon)
    #     else:
    #         lexicon = self.recursive_call(headchild, grouped, start_type, lexicon)
    #
    #     return lexicon


    def get_plain_type(self, node):
        try:
            return self.type_dict[node.attrib['pos']]
        except KeyError:
            return self.type_dict[node.attrib['cat']]

    def recursive_call(self, start, grouped, lexicon, top_type=None):
        # todo: what if the modality applies to a group of constituents?
        """

        :param start:
        :param grouped:
        :param seen:
        :param lexicon:
        :return:
        """
        # this will give us the key to be used in the lexicon
        def get_key(node):
            return node.attrib['word'] + ', ' + node.attrib['id']

        # is this a subtree or a leaf node?
        if start not in grouped.keys():
            if get_key(start) in lexicon.keys():  # has this node been passed through before?
                lexicon[get_key(start)] = lexicon[get_key(start)] + ' MODALITY: {' + self.get_plain_type(start) + '}'
            else:
                lexicon[get_key(start)] = self.get_plain_type(start)
        else:
            siblings = grouped[start]
            headchild = Decompose.choose_head(siblings)
            if headchild == -1:
                print(siblings)
                raise ValueError

            siblings = Decompose.order_siblings(siblings, exclude=headchild)  # todo: this ignores left/right position
            arglist = list(map(lambda x: [self.get_plain_type(x[0]), x[1]], siblings))

            if top_type:
                start_type = str(top_type)
            else:
                start_type = self.get_plain_type(start)
            if Decompose.is_leaf(headchild):
                if get_key(headchild) in lexicon.keys():  # has this node been passed through before?
                    lexicon[get_key(headchild)] = lexicon[get_key(headchild)] + ' MODALITY: {' \
                                                  + str(arglist) + '→ ' + start_type + '}'
                else:
                    lexicon[get_key(headchild)] = str(arglist) + '→ ' + start_type
            else:
                lexicon = self.recursive_call(headchild, grouped, lexicon, top_type=str(arglist) +
                                                                                          '→ ' + start_type)

            for child, _ in siblings:
                lexicon = self.recursive_call(child, grouped, lexicon)
        return lexicon


    def __call__(self, grouped):
        # init an empty type dict
        lexicon = dict()
        # get list of roots in case of disconnected tree
        roots = sorted(Decompose.get_disconnected(grouped), key=lambda x: x.attrib['id'])

        for key in roots:
            lexicon = self.recursive_call(key, grouped, lexicon)

        return lexicon


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
    tree_transform = Compose([lambda x: x[2]])
                             # lambda x: Lassy.remove_subtree(x, {'pos': 'punct', 'rel': 'mod'}),
                             # lambda x: Lassy.remove_abstract_subject(x),
                             # Lassy.tree_to_dag])
    L0 = Lassy(transform = tree_transform, ignore=False)
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
    decomposer = Decompose()
    find_non_head = Compose([lambda x: [x[0], x[2]],
                             lambda x: [x[0], Lassy.remove_subtree(x[1], {'pos': 'punct', 'rel': 'mod'}, inline=True)],
                             lambda x: [x[0], Lassy.remove_abstract_subject(x[1], inline=True)],
                             lambda x: [x[0], Lassy.tree_to_dag(x[1], inline=True)],
                             lambda x: [x[0], Decompose.group_by_parent(x[1])],
                             lambda x: [x[0], Decompose.sanitize(x[1])],
                             lambda x: [x[0], Decompose.collapse_mwu(x[1])],
                             lambda x: [x[0], Decompose.split_du(x[1])],])
                             #lambda x: [x[0], Decompose.get_disconnected(x[1])]])
                             #lambda x: [x[0], decomposer(x[1])]])
                             #lambda x: [x[0], Decompose.test_iter_group(x[1])]])
    L = Lassy(transform=find_non_head, ignore=False)
    asserter = DataLoader(L, batch_size=256, shuffle=False, num_workers=8,
                          collate_fn=lambda y: list(chain(y)))
    #                      collate_fn=lambda y: list((filter(lambda x: x[1] != [], y))))
    #bad_groups = list(chain.from_iterable(filter(lambda x: x != [], [i for i in tqdm(asserter)])))
    return L0, L, asserter #, bad_groups




    tree = Compose([lambda x: x[1]])
    L = Lassy(transform=tree)
    samples = [L[i] for i in [10, 20, 30, 40, 50, 100, 500, 1000, 200]]
    faster = DataLoader(L, batch_size=256, shuffle=False, num_workers=8, collate_fn=lambda x: list(chain(x)))

    tg = ToGraphViz()

    return samples, ToGraphViz(), faster