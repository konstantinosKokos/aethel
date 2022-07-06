from typing import Callable
import subprocess
import os

from ..mill.types import Type, Atom, Diamond, Box, Functor
from ..mill.types import needs_par as needs_par_type
from ..mill.terms import (TERM, Variable, Constant, ArrowElimination, ArrowIntroduction,
                          DiamondIntroduction, DiamondElimination, BoxIntroduction, BoxElimination,)
from ..mill.terms import needs_par as needs_par_term
from ..mill.proofs import Proof, Judgement, Rule, Logical, Structural
from ..mill.structures import Sequence, Unary
from ..frontend import Sample


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
          '\\usepackage[utf8]{inputenc}' \
          '\\begin{document}\n\n'
closure = '\\end{document}\n'


def emph(x: str) -> str: return '\\textcolor{red}{' + x + '}'
def mathrm(x: str) -> str: return '\\mathrm{' + x + '}'
def mathsf(x: str) -> str: return '\\mathsf{' + x + '}'
def textit(x: str) -> str: return '\\textit{' + x + '}'
def textsf(x: str) -> str: return '\\textsf{' + x + '}'
def text(x: str) -> str: return '\\text{' + x + '}'
def stylize_atom(x: str) -> str: return x.lower()
def stylize_term(x: str) -> str: return mathrm(x)
def empty_string(_) -> str: return ''
def center(x: str) -> str: return '\\begin{center}\n' + x + '\n\\end{center}\n'
def mathmode(x: str) -> str: return '\\[' + x + '\\]'
def wrap(x: str) -> str: return prelude + x + '\n' + closure
def tabulate(lines: list[str]) -> str: return '\\begin{tabular}{l}' + '\\\n'.join(lines) + '\\end{tabular}\n'


def format_sentence(sentence: str) -> str:
    return text(sanitize_text(sentence))


def format_word(sample: Sample, style: Callable[[str], str]) -> Callable[[int], str]:
    def f(idx: int) -> str:
        return style(sanitize_text(sample.lexical_phrases[idx].string))
    return f


def sanitize_text(x: str) -> str:
    return x.replace('\\', '\\textbackslash').replace('~', '\\raisebox{0.5ex}{\\texttildelow}').\
        replace('&', '\\&').replace('%', '\\%').replace('{', '\\{').replace('}', '\\}').\
        replace('^', '\\^').replace('_', '\\_')


def format_constant(idx: int) -> str:
    return 'c_{' + str(idx) + '}'


def format_proof(proof: Proof,
                 show_intermediate_terms: bool = True,
                 leaf_formatter: Callable[[int], str] = format_constant) -> str:
    def go(_proof: Proof, focus: Variable | None = None, ) -> list[str]:
        premises: list[list[str]]
        if (lps := len(_proof.premises)) > 0:
            premises = [go(p, _proof.focus) for p in _proof.premises]
            premises = [['{'] * (i == 0) + premise + ['&' if i < (lps - 1) else '}']
                        for i, premise in enumerate(premises)]
        elif _proof.rule == Logical.Constant:
            premises = [['{' + leaf_formatter(_proof.term.index) + '}']]
        else:
            premises = [['{}']]

        return [f'\\infer[{format_rule(_proof.rule)}]',
                '\t{' + format_judgement(_proof.conclusion, focus, show_intermediate_terms) + '}',
                *['\t' + line for premise in premises for line in premise]]
    return '\n'.join(go(proof))


def format_judgement(judgement: Judgement,
                     focus: Variable | None,
                     show_terms: bool) -> str:
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


def format_term(term: TERM,
                show_types: bool,
                constant_formatter: Callable[[int], str] = format_constant) -> str:
    def par(_term: TERM) -> str:
        ret = go(_term)
        return '(' + ret + ')' if needs_par_term(_term) else ret

    def go(_term: TERM) -> str:
        match _term:
            case Variable(_, i): return stylize_term('x_{' + str(i) + '}')
            case Constant(_, i): return constant_formatter(i)
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
        case Atom(A): return stylize_atom(A)
        case Functor(A, B): return par(A) + arrow + format_type(B)
        case Diamond(d, A): return diamond + '^{' + d + '}' + par(A)
        case Box(d, A): return box + '^{' + d + '}' + par(A)
        case _: raise ValueError


def proof_to_tex(proof: Proof, show_terms: bool) -> str:
    return wrap(format_proof(proof, show_intermediate_terms=show_terms))


def proofs_to_tex(proofs: list[Proof]) -> str:
    return wrap('\\newpage'.join(format_proof(proof) for proof in proofs))


def format_sample(sample: Sample,
                  show_intermediate_terms: bool = True,
                  show_words_at_leaves: bool = True,
                  show_sentence: bool = True,
                  show_final_term: bool = True) -> str:
    leaf_format = format_word(sample, textit) if show_words_at_leaves else empty_string
    body = format_proof(sample.proof, show_intermediate_terms, leaf_format)
    if show_sentence:
        body = tabulate([format_sentence(sample.sentence)]) + '\n' + body
    if show_final_term:
        body += '\n' + mathmode(emph(format_term(sample.proof.term,
                                                 show_types=False,
                                                 constant_formatter=format_word(sample, textsf))))
    return body


def sample_to_tex(sample: Sample,
                  show_intermediate_terms: bool = True,
                  show_words_at_leaves: bool = True,
                  show_sentence: bool = True,
                  show_final_term: bool = True) -> str:
    return wrap(format_sample(sample, show_intermediate_terms, show_words_at_leaves, show_sentence, show_final_term))


def samples_to_dir(samples: list[Sample], output_dir: str):
    for sample in samples:
        with open(os.path.join(output_dir, sample.name + '.tex'), 'w') as f:
            f.write(sample_to_tex(sample))
