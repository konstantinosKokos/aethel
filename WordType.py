

class WordType:
    def __init__(self, arglist, result, modality=False):
        # make sure that [] -> A -> B is the same as A -> B
        while type(result) == WordType:
            if not result.arglist:
                result = result.result
            elif not arglist:
                arglist = result.arglist
                result = result.result
            else:
                break
        self.arglist = arglist
        self.result = result

        self.modality = modality
        # todo: arglist arity is ignored
        if type(self.result) == WordType:
            result_arity = result.arity
        else:
            result_arity = 0
        if not arglist:
            self.arity = result_arity
        else:
            arglist_arity = self.find_arglist_arity(arglist)
            self.arity = result_arity + arglist_arity

    def find_arglist_arity(self, arglist):
        arglist_arity = 0
        for a in arglist:
            if type(a) == WordType:
                arglist_arity = max(arglist_arity, a.arity)
        return arglist_arity

    @staticmethod
    def print_arg(arg):
        if type(arg) == list: # case of item w/ dep
            return '(' + arg[0].__str__() + ', ' + arg[1] + ')'
        elif type(arg) == str:  # case of item w/o dep
            return arg
        else:
            raise TypeError('Unknown type')

    def __str__(self):
        to_print = ''
        if self.arglist:
            # len check for parentheses
            if len(self.arglist) == 1:  # single item with dep
                argprint = self.print_arg(self.arglist[0])
            else:
                argprint = '(' + ', '.join(map(self.print_arg, self.arglist)) + ')'
            to_print = argprint + ' → '
        resprint = str(self.result)
        try:
            if self.result.arity:
                resprint = '(' + resprint + ')'
        except AttributeError:
            pass
        to_print = to_print + resprint
        if self.modality:
            if self.arglist:
                to_print = '(' + to_print + ')'
            to_print = ' ◇ □ ' + to_print
        return to_print

    def __repr__(self):
        return self.__str__()

    def __eq__(self, t):
        # two wordtypes can never be the same if their word types do not match
        if type(self) != type(t):
            return False
        # recursive calls within argument and result types
        return self.arglist == t.arglist and self.result == t.result and self.modality == t.modality

    def __hash__(self):
        # hash the string representation which should be unique for any given type
        return self.__repr__().__hash__()

    def wcmp(self, other):
        """
        Weak comparison between a normal Type and a Type with a missing argument / result
        :param other:
        :return:
        """
        if len(self.arglist) != len(other.arglist):
            return False
        for i, a in enumerate(self.arglist):
            if a == other.arglist[i] or (other.arglist[i][0] == '?' and a[1] == other.arglist[i][1]):
                continue
            else:
                return False
        # if other.result == '?': # todo
        #     return True
        if type(self.result)!=type(other.result):
            print(1)
            return False
        elif type(self.result) == WordType:
            return True and self.result.wcmp(other.result)
        else:
            return self.result == other.result

    @staticmethod
    def remove_deps(wordtype):
        arglist = []
        for a in wordtype.arglist:
            if type(a) == WordType:
                arglist.append(WordType.remove_deps(a).__str__())
            else:
                arglist.append(a[0].__str__())
        if type(wordtype.result) == WordType:
            result = WordType.remove_deps(wordtype.result)
        else:
            result = wordtype.result
        return WordType(arglist, result)

    def __main__(self):
        print(self.__str__)
