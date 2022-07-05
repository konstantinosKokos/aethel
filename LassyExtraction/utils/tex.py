from ..mill.types import Type, Atom, Diamond, Box, Functor
from ..mill.types import needs_par as needs_par_type
from ..mill.terms import (TERM, Variable, Constant, ArrowElimination, ArrowIntroduction,
                          DiamondIntroduction, DiamondElimination, BoxIntroduction, BoxElimination,)
from ..mill.terms import needs_par as needs_par_term
from ..mill.proofs import Proof, Judgement, Rule, Logical, Structural
from ..mill.structures import Sequence, Unary

vdash = '\\vdash '
arrow = '\\multimap '
diamond = '\\diamondsuit '
box = '\\Box '
boxelim = '\\blacktriangledown '
boxintro = '\\blacktriangle '
diaelim = '\\vartriangle '
diaintro = '\\triangledown '
lam = '\\lambda '

prelude = '\\documentclass[preview=true,border={20pt 20pt 20pt 20pt}]{standalone}\n' \
          '\\usepackage{color}\n' \
          '\\usepackage{amsmath}\n' \
          '\\usepackage{proof}\n' \
          '\\usepackage{amssymb}\n' \
          '\\begin{document}\n'
closure = '\\end{document}\n'


def emph(x: str) -> str: return '\\textcolor{red}{' + x + '}'
def type_format(x: str) -> str: return x.lower()
def term_format(x: str) -> str: return x


def format_proof(proof: Proof, show_terms: bool = True) -> str:
    def go(_proof: Proof, focus: Variable | None = None, ) -> list[str]:
        premise_lines: list[list[str]] = [[*go(p, _proof.focus), '&'] for p in _proof.premises]
        return [f'\\infer[{format_rule(_proof.rule)}]',
                '\t{' + format_judgement(_proof.conclusion, focus, show_terms) + '}',
                '{',
                *['\t' + line for premise in premise_lines for line in premise],
                '}']
    return '\n'.join(go(proof))


def format_judgement(judgement: Judgement, focus: Variable | None, show_terms: bool) -> str:
    return (structure_to_tex(judgement.assumptions, focus)
            + vdash
            + (format_term(judgement.term, True) if show_terms else format_type(judgement.term.type)))


def format_rule(rule: Rule) -> str:
    match rule:
        case Logical.Variable: return 'Ax'
        case Logical.Constant: return 'Lex'
        case Logical.ArrowElimination: return f'{arrow}E'
        case Logical.ArrowIntroduction: return f'{arrow}I'
        case Logical.DiamondIntroduction: return f'{diamond}I'
        case Logical.DiamondElimination: return f'{diamond}E'
        case Logical.BoxIntroduction: return f'{box}I'
        case Logical.BoxElimination: return f'{box}E'
        case Structural.Extract: return '\\mathsf{X}'
        case _: raise ValueError


def structure_to_tex(structure: Sequence[Variable | Constant]
                                | Unary[Variable | Constant]
                                | Variable
                                | Constant,
                     focus: Variable | None) -> str:
    match structure:
        case Sequence(xs): return ',~'.join(structure_to_tex(x, focus) for x in xs)
        case Unary(x, bs): return f'\\langle {structure_to_tex(x, focus)}\\rangle' + '^{' + bs + '}'
        case Variable(_) | Constant(_):
            ret = format_term(structure, False)
            return emph(ret) if structure == focus else ret
        case _: raise ValueError


def format_term(term: TERM, show_types: bool) -> str:
    def par(_term: TERM) -> str:
        ret = go(_term)
        return '(' + ret + ')' if needs_par_term(_term) else ret

    def go(_term: TERM) -> str:
        match _term:
            case Variable(_type, i): return 'x_{' + str(i) + '}'
            case Constant(_type, i): return 'c_{' + str(i) + '}'
            case ArrowElimination(fn, arg): return f'{go(fn)}~{par(arg)}'
            case ArrowIntroduction(abstraction, body): return lam + go(abstraction) + '.' + par(body)
            case DiamondElimination(decoration, body): return diaelim + '^{' + decoration + '}' + par(body)
            case DiamondIntroduction(decoration, body): return diaintro + '^{' + decoration + '}' + par(body)
            case BoxElimination(decoration, body): return boxelim + '^{' + decoration + '}' + par(body)
            case BoxIntroduction(decoration, body): return boxintro + '^{' + decoration + '}' + par(body)
            case _: raise ValueError
    return go(term) + (':' + format_type(term.type)) * show_types


def format_type(_type: Type) -> str:
    def par(__type: Type) -> str:
        ret = format_type(__type)
        return '(' + ret + ')' if needs_par_type(__type) else ret

    match _type:
        case Atom(A): return type_format(A)
        case Functor(A, B): return par(A) + arrow + format_type(B)
        case Diamond(d, A): return diamond + '^{' + d + '}' + par(A)
        case Box(d, A): return box + '^{' + d + '}' + par(A)
        case _: raise ValueError


def proof_to_tex(proof: Proof) -> str:
    return prelude + format_proof(proof) + closure


def proofs_to_tex(proofs: list[Proof]) -> str:
    return prelude + '\\newpage'.join(format_proof(proof) for proof in proofs) + closure
