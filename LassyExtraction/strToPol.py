import string
from pyparsing import Word, Literal, opAssoc, infixNotation

atom_ = Word(string.ascii_uppercase)
typedec_ = Word(string.ascii_lowercase + '_')
arrow_ = Literal("->")
arrowexp_ = typedec_ + arrow_

typeexp_ = infixNotation(atom_, [(arrowexp_, 2, opAssoc.RIGHT)],)

def strToPol(str_):
    return unfoldExp(typeexp_.parseString(str_).asList()[0])


def unfoldExp(expL):
    if isinstance(expL, str):
        return expL
    else:
        return ' '.join([unfoldExp(expL[1]), unfoldExp(expL[0]), unfoldExp(expL[3])])
