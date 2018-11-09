from itertools import permutations
from warnings import warn
from functools import reduce


class WordType:
    def __init__(self, arglist, result, modality=False, sanitize=True):
        """
        Constructor for WordType.
        :param arglist:
        :param result:
        :param modality:
        """
        # normalize the arglist
        if isinstance(arglist, str):
            # if str, assume singular type argument and convert it to tuple of WordType
            arglist = (WordType((), arglist),)
            warn('Implicit conversion of arglist from str to WordType.')
        elif isinstance(arglist, list) or isinstance(arglist, tuple):
            # if iterable, assert all items are WordTypes (also covers the case of empty iterable)
            if all(map(lambda t: isinstance(t, str), arglist)) and len(arglist):
                warn('Implicit conversion of arglist from (str,) to (WordType,).')
                arglist = tuple([WordType((), a) for a in arglist])
            if not all(map(lambda t: isinstance(t, WordType), arglist)):
                raise TypeError('Expected an iterable of WordTypes, received {} instead.'.
                                format(list(map(type, arglist))))
            arglist = tuple(arglist)
        elif isinstance(arglist, WordType):
            # if WordType, convert to tuple of WordType
            arglist = (arglist,)

        # normalize the result
        if not result:
            raise TypeError('Received empty result type.')
        if isinstance(result, str):
            pass
        elif isinstance(result, list) or isinstance(result, tuple):
            # if iterable, try to convert to WordType
            if len(result) == 1:
                if isinstance(result[0], WordType):
                    self.result = result[0]
                    warn('Implicit conversion from iterable to WordType.')
                else:
                    raise TypeError('Expected result type to be str or Wordtype, received {} instead.'
                                    .format(type(result)))
            else:
                raise TypeError('A WordType cannot have a sequence as a result.')
        elif not isinstance(result, WordType):
            raise TypeError('WordType result must be str or WordType, received {} instead.'.format(type(result)))
        self.result = result
        self.arglist = arglist
        if sanitize:
            self.sanitize()
        self.modality = modality
        self.arity = WordType.get_arity(result) if not self.arglist else self.find_arglist_arity() + 1 + \
                                                                WordType.get_arity(self.result)

    def sanitize(self):
        # final sanity check
        while isinstance(self.result, WordType):
            if not self.result.arglist:
                # remove empty internal arglist
                self.result = self.result.result
            elif not self.arglist:
                # remove empty external arglist
                self.arglist = self.result.arglist
                self.result = self.result.result
            else:
                break

    @staticmethod
    def get_arity(item):
        if isinstance(item, str):
            return 0
        elif isinstance(item, WordType):
            return item.arity
        else:
            raise TypeError('Unknown type.')

    def find_arglist_arity(self):
        return reduce(max, map(lambda x: x.arity, [a for a in self.arglist if isinstance(a, WordType)]), 0)

    def __str__(self):
        """
        Conversion to string
        :return:
        """
        def print_args(args):
            if isinstance(args, tuple):
                if len(args) > 1:
                    return '(' + ', '.join([a.__str__() for a in args]) + ') → '
                elif len(args) == 1:
                    if args[0].arity == 0:
                        return args[0].__str__() + ' → '
                    else:
                        return '(' + args[0].__str__() + ') → '
                else:
                    return ''
            elif isinstance(args, WordType):
                return args.__str__()
        if isinstance(self.result, WordType):
            s = print_args(self.arglist) + self.result.__str__()
            return '!' + s if self.modality else s
        elif isinstance(self.result, str):
            s = print_args(self.arglist) + self.result
            return '!' + s if self.modality else s

    def __repr__(self):
        return self.__str__()

    def __call__(self):
        return self.__str__()

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        if isinstance(other, WordType):
            return self.arglist == other.arglist and self.result == other.result and self.modality == other.modality
        return False


class ColouredType(WordType):
    def __init__(self, arglist, result, colors, modality=False):
        super(ColouredType, self).__init__(arglist, result, modality, sanitize=False)
        if isinstance(colors, str):
            warn('Implicit conversion of colors from str to tuple')
            colors = [colors]
        if isinstance(colors, list) or isinstance(colors, tuple):
            if len(colors) != len(self.arglist):
                raise TypeError('Expected {} colors, received {} instead.'.format(len(self.arglist), len(colors)))
            if not all(map(lambda x: isinstance(x, str), colors)):
                raise TypeError('Colors must be an iterable of strings')
            self.colors = tuple(colors)
        else:
            raise TypeError('Colors must be an iterable of strings')
        self.sanitize()

    def sanitize(self):
        # final sanity check
        while isinstance(self.result, WordType):
            if not self.result.arglist:
                # remove empty internal arglist
                self.result = self.result.result
            elif not self.arglist:
                # remove empty external arglist
                self.arglist = self.result.arglist
                self.colors = self.result.colors
                self.result = self.result.result
            else:
                break

    def __str__(self):
        # todo: just pprint this part a bit, otherwise we gucci
        castr = '(' + ', '.join([str(ca) for ca in zip(self.arglist, self.colors)]) + ') → ' if self.arglist else ''

        if isinstance(self.result, WordType):
            s = castr + self.result.__str__()
            return '!' + s if self.modality else s
        elif isinstance(self.result, str):
            s = castr + self.result
            return '!' + s if self.modality else s

    def remove_colors(self):
        return WordType(tuple(map(lambda x: x.remove_colors if isinstance(x, ColouredType) else x, self.arglist)),
                        self.result, self.modality)

    def __eq__(self, other):
        if isinstance(other, ColouredType):
            return self.colors == other.colors and self.remove_colors() == other.remove_colors()
        elif isinstance(other, WordType):
            return self.remove_colors() == other and not self.colors


#     def wcmp(self, other):
#         """
#         Weak comparison between a normal Type and a Type with a missing argument / result
#         :param other:
#         :return:
#         """
#         # fuck python
#         if len(self.arglist) != len(other.arglist):
#             return False
#         for i, a in enumerate(self.arglist):
#             if a == other.arglist[i] or (other.arglist[i][0] == '?' and a[1] == other.arglist[i][1]):
#                 continue
#             else:
#                 return False
#         # if other.result == '?': # todo
#         #     return True
#         if type(self.result)!=type(other.result):
#             print(1)
#             return False
#         elif type(self.result) == WordType:
#             return True and self.result.wcmp(other.result)
#         else:
#             return self.result == other.result
#
#     @staticmethod
#     def remove_deps(wordtype):
#         arglist = []
#         for a in wordtype.arglist:
#             if type(a) == WordType:
#                 arglist.append(WordType.remove_deps(a).__str__())
#             else:
#                 arglist.append(a[0].__str__())
#         if type(wordtype.result) == WordType:
#             result = WordType.remove_deps(wordtype.result)
#         else:
#             result = wordtype.result
#         return WordType(arglist, result, wordtype.modality)
#
#
# def collapse(seq, verbose=False):
#     # todo: intractable
#     def apply(t1, t2):
#         r = apply_left(t1, t2)
#         if r:
#             return r
#         else:
#             return apply_left(t2, t1)
#
#     def apply_left(t1, t2):
#         if not t1.arglist and t1.result in t2.arglist:
#             if isinstance(t2.arglist, list):
#                 return WordType([x for x in t2.arglist if x != t1.result], t2.result)
#             else:
#                 return WordType([], t2.result)
#         return None
#
#     perms = permutations(seq)  # take all permutations of the given type sequence
#     perms_ns = perms
#     # for p in perms:
#     #     if p[::-1] not in perms_ns:
#     #         perms_ns.append(p)  # ignore the symmetric ones
#
#     precomputed = dict()
#
#     for p in perms_ns:
#         current = list(p)
#         reduced = True
#         while reduced:
#             if verbose:
#                 print('starting with :', current)
#             reduced = False
#             if len(current) == 1:
#                 return current
#             for t1, t2 in zip(current, current[1:]):
#                 if tuple([t1, t2]) in precomputed.keys():
#                     t12 = precomputed[tuple([t1, t2])]
#                 else:
#                     t12 = apply(t1, t2)
#                     precomputed[tuple([t1, t2])] = t12
#                     precomputed[tuple([t2, t1])] = t12
#                 if t12:
#                     if verbose:
#                         print('removing ', t1, t2, 'and replacing with ', t12)
#                     current.remove(t1)
#                     current.remove(t2)
#                     current.append(t12)
#                     reduced = True
#                     break
#             if reduced==False and verbose:
#                 print('Failed.')
#     return None
