from __future__ import annotations


from .structures import Structure, Sequence, Unary, bracket
from .types import Type, Atom, Diamond, Box
from .terms import (Term, Variable, Constant, ArrowElimination, ArrowIntroduction,
                    DiamondIntroduction, BoxElimination, BoxIntroduction, DiamondElimination,
                    substitute, term_repr)
from enum import Enum
from functools import partial


class ProofError(Exception):
    pass


class Judgement:
    __match_args__ = ('structure', 'term')

    structure:  Structure
    term:       Term

    def __init__(self, structure: Structure, term: Term) -> None:
        self.structure = structure
        self.term = term

    def __repr__(self) -> str: return judgement_repr(self)


def judgement_repr(judgement: Judgement) -> str:
    return f'{judgement.structure} |- {term_repr(judgement.term, False)}: {judgement.term.type}'


class Rule(Enum):
    def __repr__(self) -> str: return self.name
    def __call__(self, *args, **kwargs) -> Proof: return self.value(*args, **kwargs)


class Logical(Rule):
    @staticmethod
    def _init_var(variable: Variable) -> Proof:
        return Proof(rule=Logical.Variable, premises=(), conclusion=Judgement(Sequence(variable), variable))

    @staticmethod
    def _init_con(constant: Constant) -> Proof:
        return Proof(rule=Logical.Constant, premises=(), conclusion=Judgement(Sequence(constant), constant))

    @staticmethod
    def _init_arrow_elim(function: Proof, argument: Proof) -> Proof:
        return Proof(rule=Logical.ArrowElimination,
                     premises=(function, argument),
                     conclusion=Judgement(Sequence(*function.structure, *argument.structure),
                                          ArrowElimination(function.term, argument.term)))

    @staticmethod
    def _init_arrow_intro(abstraction: Term, body: Proof) -> Proof:
        if abstraction not in body.structure:
            raise ProofError(f'{abstraction} does not occur free in {body.structure}')
        return Proof(rule=Logical.ArrowIntroduction,
                     premises=(body,),
                     conclusion=Judgement(abs(Sequence(*[s for s in body.structure if s != abstraction])),
                                          ArrowIntroduction(abstraction, body.term)))

    @staticmethod
    def _init_dia_intro(body: Proof, diamond: str) -> Proof:
        succedent = DiamondIntroduction(diamond, body.term)
        return Proof(rule=Logical.DiamondIntroduction,
                     premises=(body,),
                     conclusion=Judgement(bracket(abs(Sequence(body.structure)), succedent.decoration), succedent))

    @staticmethod
    def _init_dia_elim(original: Proof, where: Proof, becomes: Proof) -> Proof:
        replacement = DiamondElimination(None, becomes.term)
        replaced_structure = where.structure
        if not isinstance(replaced_structure, Sequence) or not isinstance(replaced_structure[0], Variable):
            raise ProofError(f'{where.structure} is not a variable')
        return Proof(rule=Logical.DiamondElimination,
                     premises=(original, becomes),
                     conclusion=Judgement(
                         abs(Sequence(original.structure.replace(Unary(where.structure, replacement.decoration),
                                                                 becomes.structure))),
                         substitute(original.term, where.term, replacement)))

    @staticmethod
    def _init_box_intro(body: Proof, box: str) -> Proof:
        structure = body.structure
        if not isinstance(structure, Unary) or structure.brackets != box:
            raise ProofError(f'{body.structure} is not a {box}-bracketed structure')
        return Proof(rule=Logical.BoxIntroduction,
                     premises=(body,),
                     conclusion=Judgement(structure.content, BoxIntroduction(box, body.term)))

    @staticmethod
    def _init_box_elim(body: Proof, box: str | None) -> Proof:
        succedent = BoxElimination(box, body.term)
        return Proof(rule=Logical.BoxElimination,
                     premises=(body,),
                     conclusion=Judgement(bracket(abs(Sequence(body.structure)), succedent.decoration), succedent))

    Variable = partial(_init_var)
    Constant = partial(_init_con)
    ArrowElimination = partial(_init_arrow_elim)
    ArrowIntroduction = partial(_init_arrow_intro)
    DiamondIntroduction = partial(_init_dia_intro)
    BoxElimination = partial(_init_box_elim)
    BoxIntroduction = partial(_init_box_intro)
    DiamondElimination = partial(_init_dia_elim)


class Proof:
    __match_args__ = ('premises', 'conclusion', 'rule')

    premises:   tuple[Proof, ...]
    conclusion: Judgement
    rule:       Rule

    def __init__(self, premises: tuple[Proof, ...], conclusion: Judgement, rule: Rule) -> None:
        self.premises = premises
        self.conclusion = conclusion
        self.rule = rule

    @property
    def structure(self) -> Structure: return self.conclusion.structure
    @property
    def term(self) -> Term: return self.conclusion.term
    @property
    def type(self) -> Type: return self.term.type
    def __repr__(self) -> str: return judgement_repr(self.conclusion)
    def apply(self, other: Proof) -> Proof: return Logical.ArrowElimination(self, other)
    def diamond(self, diamond: str) -> Proof: return Logical.DiamondIntroduction(self, diamond)
    def box(self, box: str) -> Proof: return Logical.BoxIntroduction(self, box)
    def unbox(self, box: str | None = None) -> Proof: return Logical.BoxElimination(self, box)
    def undiamond(self, where: Proof, becomes: Proof) -> Proof: return Logical.DiamondElimination(self, where, becomes)
    def __matmul__(self, other) -> Proof: return self.apply(other)
    def vars(self) -> list[Variable]: return [v for v in self.structure.vars() if isinstance(v, Variable)]
    def abstract(self, variable: Variable) -> Proof: return Logical.ArrowIntroduction(variable, self)


def constant(type: Type, index: int) -> Proof: return Logical.Constant(Constant(type, index))
def variable(type: Type, index: int) -> Proof: return Logical.Variable(Variable(type, index))


########################################################################################################################
# # Examples / Tests
########################################################################################################################
A = Atom('A')

# Adjunction theorems

# ◇ □ A |- A
p0 = (x := variable(Box('a', A), 0)).unbox()
y = variable(Diamond('a', Box('a', A)), 0)
p0 = p0.undiamond(where=x, becomes=y)

# A |- □ ◇ A
p1 = variable(A, 0).diamond('a').box('a')

# □ A |- □ ◇ □ A
p2 = variable(Box('a', A), 0).diamond('a').box('a')

# ◇ A |- ◇ □ ◇ A
p3 = (x := variable(A, 0)).diamond('a').box('a').diamond('a')
y = variable(Diamond('a', A), 0)
p3 = p3.undiamond(where=x, becomes=y)

# ◇ □ ◇ A |- ◇ A
p4 = (x := variable(Box('a', Diamond('a', A)), 0)).unbox()
y = variable(Diamond('a', Box('a', Diamond('a', A))), 0)
p4 = p4.undiamond(where=x, becomes=y)

# □ ◇ □ A |- □ A
p5 = (x := variable(Box('a', A), 0)).unbox()
y = variable(Box('a', Diamond('a', Box('a', A))), 0).unbox()
p5 = p5.undiamond(where=x, becomes=y).box('a')


########################################################################################################################
# # # Term Syntax
########################################################################################################################
# # class Proof:
# #     rule: Rule
# #     decoration: str
# #     constant: int
# #     variable: int
# #     abstraction: Proof
# #     function: Proof
# #     argument: Proof
# #     body: Proof
# #     abstraction: Proof
# #
# #     ####################################################################################################################
# #     # Constructors
# #     ####################################################################################################################
# #     class Rule(Enum):
# #         @staticmethod
# #         def _init_lex(proof: Proof, constant: int) -> None:
# #             proof.constant = constant
# #
# #         @staticmethod
# #         def _init_var(proof: Proof, variable: int) -> None:
# #             proof.variable = variable
# #
# #         @staticmethod
# #         def _init_arrow_elim(proof: Proof, function: Proof, argument: Proof) -> None:
# #             rest = TypeInference.arrow_elim(function.type, argument.type)
# #             TypeInference.assert_equal(rest, proof.type)
# #             if function.is_negative():
# #                 raise TypeInference.TypeCheckError(f'{function} is a negative structure')
# #             if argument.is_negative():
# #                 raise TypeInference.TypeCheckError(f'{argument} is a negative structure')
# #             proof.function = function
# #             proof.argument = argument
# #
# #         @staticmethod
# #         def _init_arrow_intro(proof: Proof, abstraction: Proof, body: Proof) -> None:
# #             TypeInference.assert_equal(Functor(abstraction.type, body.type), proof.type)
# #             if abstraction.rule != Proof.Rule.Variable:
# #                 raise TypeInference.TypeCheckError(f'{abstraction} is not a variable')
# #             if abstraction not in body.vars():
# #                 raise TypeInference.TypeCheckError(f'{abstraction} does not occur in {body}')
# #             if abstraction not in body.free():
# #                 raise TypeInference.TypeCheckError(f'{abstraction} structurally locked in {body}')
# #             proof.abstraction = abstraction
# #             proof.body = body
# #
# #         @staticmethod
# #         def _init_box_elim(proof: Proof, box: str, body: Proof) -> None:
# #             TypeInference.assert_equal(Box(box, proof.type), body.type)
# #             proof.decoration = box
# #             proof.body = body
# #
# #         @staticmethod
# #         def _init_dia_intro(proof: Proof, diamond: str, body: Proof) -> None:
# #             TypeInference.assert_equal(Diamond(diamond, body.type), proof.type)
# #             proof.decoration = diamond
# #             proof.body = body
# #
# #         @staticmethod
# #         def _init_box_intro(proof: Proof, box: str, body: Proof) -> None:
# #             TypeInference.assert_equal(Box(box, body.type), proof.type)
# #             proof.decoration = box
# #             proof.body = body
# #
# #         @staticmethod
# #         def _init_dia_elim(proof: Proof, diamond: str, body: Proof) -> None:
# #             TypeInference.assert_equal(Diamond(diamond, proof.type), body.type)
# #             proof.decoration = diamond
# #             proof.body = body
# #
# #         # Enumeration of rules
# #         Lexicon = partial(_init_lex)
# #         Variable = partial(_init_var)
# #         ArrowElimination = partial(_init_arrow_elim)
# #         ArrowIntroduction = partial(_init_arrow_intro)
# #         BoxElimination = partial(_init_box_elim)
# #         DiamondIntroduction = partial(_init_dia_intro)
# #         BoxIntroduction = partial(_init_box_intro)
# #         DiamondElimination = partial(_init_dia_elim)
# #
# #     def __init__(self: Proof, rule: Rule, **kwargs) -> None:
# #         self.rule = rule
# #         self.rule.value(self, **kwargs)
# #
# #     ####################################################################################################################
# #     # Term-level computations
# #     ####################################################################################################################
# #     def apply(self: Proof, argument: Proof) -> Proof:
# #         res_type = TypeInference.arrow_elim(self.type, argument.type)
# #         return res_type(rule=Proof.Rule.ArrowElimination, function=self, argument=argument)
# #
# #     def abstract(self: Proof, variable: Proof) -> Proof:
# #         res_type = Functor(variable.type, self.type)
# #         return res_type(rule=Proof.Rule.ArrowIntroduction, abstraction=variable, body=self)
# #
# #     def box(self: Proof, box: str) -> Proof:
# #         res_type = Box(box, self.type)
# #         return res_type(rule=Proof.Rule.BoxIntroduction, box=box, body=self)
# #
# #     def diamond(self: Proof, diamond: str) -> Proof:
# #         res_type = Diamond(diamond, self.type)
# #         return res_type(rule=Proof.Rule.DiamondIntroduction, diamond=diamond, body=self)
# #
# #     def unbox(self: Proof, decoration: str = None) -> Proof:
# #         res_type, decoration = TypeInference.box_elim(self.type, decoration)
# #         return res_type(rule=Proof.Rule.BoxElimination, box=decoration, body=self)
# #
# #     def undiamond(self: Proof, decoration: str = None) -> Proof:
# #         res_type, decoration = TypeInference.dia_elim(self.type, decoration)
# #         return res_type(rule=Proof.Rule.DiamondElimination, diamond=decoration, body=self)
# #
# #     ####################################################################################################################
# #     # Utilities
# #     ####################################################################################################################
# #     def nested_body(self: Proof) -> Proof:
# #         match self.rule:
# #             case Proof.Rule.BoxElimination:
# #                 return self.body.nested_body()
# #             case Proof.Rule.DiamondElimination:
# #                 return self.body.nested_body()
# #             case Proof.Rule.DiamondIntroduction:
# #                 return self.body.nested_body()
# #             case Proof.Rule.BoxIntroduction:
# #                 return self.body.nested_body()
# #             case _:
# #                 return self
# #
# #     def brackets(self: Proof) -> list[IndexedBracket]:
# #         def go(proof: Proof, context: list[IndexedBracket]) -> list[IndexedBracket]:
# #             match proof.rule:
# #                 case Proof.Rule.BoxElimination:
# #                     return go(proof.body, context + [(Bracket.Lock, proof.decoration)])
# #                 case Proof.Rule.DiamondElimination:
# #                     return go(proof.body, context + [(Bracket.Inner, proof.decoration)])
# #                 case Proof.Rule.DiamondIntroduction:
# #                     return go(proof.body, context + [(Bracket.Lock, proof.decoration)])
# #                 case Proof.Rule.BoxIntroduction:
# #                     return go(proof.body, context + [(Bracket.Outer, proof.decoration)])
# #                 case _:
# #                     return context
# #
# #         return go(self, [])
# #
# #     def is_negative(self: Proof) -> bool:
# #         return False if (bs := self.brackets()) == [] else not (any(map(_is_positive, associahedron(bs))))
# #
# #     def unbracketed(self: Proof) -> bool:
# #         return _cancel_out(self.brackets())
# #
# #     def substructures(self: Proof) -> list[tuple[list[IndexedBracket], Proof]]:
# #         def go(proof: Proof, context: list[IndexedBracket]) -> list[tuple[list[IndexedBracket], Proof]]:
# #             match proof.rule:
# #                 case Proof.Rule.Variable | Proof.Rule.Lexicon:
# #                     return [(context, proof)]
# #                 case Proof.Rule.ArrowElimination:
# #                     return [(context, proof), *go(proof.function, context), *go(proof.argument, context)]
# #                 case Proof.Rule.ArrowIntroduction:
# #                     return [(context, proof), *go(proof.body, context)]
# #                 case _:
# #                     return [(context, proof), *go(proof.nested_body(), context + _minimal_brackets(proof.brackets()))]
# #
# #         return go(self, [])
# #
# #     def __repr__(self) -> str:
# #         return show_term(self)
# #
# #     def __str__(self) -> str:
# #         return show_term(self)
# #
# #     def __eq__(self, other):
# #         if self.type != other.type or self.rule != other.rule: return False
# #         match self.rule:
# #             case Proof.Rule.Lexicon:
# #                 return self.constant == other.constant
# #             case Proof.Rule.Variable:
# #                 return self.variable == other.variable
# #             case Proof.Rule.ArrowElimination:
# #                 return self.abstraction == other.abstraction and self.body == other.body
# #             case Proof.Rule.ArrowIntroduction:
# #                 return self.abstraction == other.abstraction and self.body == other.body
# #             case _:
# #                 return self.decoration == other.decoration and self.body == other.body
# #
# #     def __hash__(self) -> int:
# #         match self.rule:
# #             case Proof.Rule.Lexicon:
# #                 return hash((self.rule, self.type, self.constant))
# #             case Proof.Rule.Variable:
# #                 return hash((self.rule, self.type, self.variable))
# #             case Proof.Rule.ArrowElimination:
# #                 return hash((self.rule, self.type, self.function, self.argument))
# #             case Proof.Rule.ArrowIntroduction:
# #                 return hash((self.rule, self.type, self.abstraction, self.body))
# #             case _:
# #                 return hash((self.rule, self.type, self.decoration, self.body))
# #
# #     def vars(self) -> list[Proof]:
# #         match self.rule:
# #             case Proof.Rule.Variable:
# #                 return [self]
# #             case Proof.Rule.Lexicon:
# #                 return []
# #             case Proof.Rule.ArrowElimination:
# #                 return self.function.vars() + self.argument.vars()
# #             case Proof.Rule.ArrowIntroduction:
# #                 return [f for f in self.body.vars() if f != self.abstraction]
# #             case _:
# #                 return self.body.vars()
# #
# #     def free(self: Proof) -> list[Proof]:
# #         substructures = self.substructures()
# #         return [var for brackets, var in substructures if var.rule == Proof.Rule.Variable and _cancel_out(brackets)]
# #
# #     def serialize(self) -> SerializedProof:
# #         return serialize_proof(self)
# #
# #     @property
# #     def type(self) -> Type:
# #         return type(self)  # type: ignore
# #
# #     ####################################################################################################################
# #     # Meta-rules and rewrites
# #     ####################################################################################################################
# #     def eta_norm(self: Proof) -> Proof:
# #         if self.is_negative():
# #             return self
# #
# #         match self.rule:
# #             case Proof.Rule.Variable | Proof.Rule.Lexicon:
# #                 return self
# #             case Proof.Rule.ArrowElimination:
# #                 return self.function.eta_norm().apply(self.argument.eta_norm())
# #             case Proof.Rule.ArrowIntroduction:
# #                 body = self.body.eta_norm()
# #                 if body.rule == Proof.Rule.ArrowElimination and body.argument == self.abstraction:
# #                     return body.function
# #                 return body.abstract(self.abstraction)
# #             case Proof.Rule.BoxIntroduction:
# #                 body = self.body.eta_norm()
# #                 if body.rule == Proof.Rule.BoxElimination and body.decoration == self.decoration:
# #                     return body.body
# #                 return body.box(self.decoration)
# #             case Proof.Rule.DiamondIntroduction:
# #                 body = self.body.eta_norm()
# #                 if body.rule == Proof.Rule.DiamondElimination and body.decoration == self.decoration:
# #                     return body.body
# #                 return body.diamond(self.decoration)
# #             case Proof.Rule.DiamondElimination:
# #                 return self.body.eta_norm().undiamond()
# #             case Proof.Rule.BoxElimination:
# #                 return self.body.eta_norm().unbox()
# #
# #     def beta_norm(self: Proof) -> Proof:
# #         match self.rule:
# #             case Proof.Rule.Variable | Proof.Rule.Lexicon:
# #                 return self
# #             case Proof.Rule.ArrowElimination:
# #                 if self.function.rule == Proof.Rule.ArrowIntroduction:
# #                     return self.function.body.substitute(self.function.abstraction, self.argument).beta_norm()
# #                 return self.function.beta_norm().apply(self.argument.beta_norm())
# #             case Proof.Rule.ArrowIntroduction:
# #                 return self.body.beta_norm().abstract(self.abstraction)
# #             case Proof.Rule.BoxIntroduction:
# #                 return self.body.beta_norm().box(self.decoration)
# #             case Proof.Rule.DiamondIntroduction:
# #                 return self.body.beta_norm().diamond(self.decoration)
# #             case Proof.Rule.DiamondElimination:
# #                 if self.body.rule == Proof.Rule.DiamondIntroduction:
# #                     return self.body.body
# #                 return self.body.beta_norm().undiamond()
# #             case Proof.Rule.BoxElimination:
# #                 if self.body.rule == Proof.Rule.BoxIntroduction and self.body.decoration == self.decoration:
# #                     return self.body.body
# #                 return self.body.beta_norm().unbox()
# #
# #     def translate_lex(self: Proof, trans: dict[int, int]) -> Proof:
# #         def go(p: Proof) -> Proof:
# #             match p.rule:
# #                 case Proof.Rule.Lexicon:
# #                     return p.type.lex(trans[c]) if (c := p.constant) in trans.keys() else p
# #                 case Proof.Rule.Variable:
# #                     return p
# #                 case Proof.Rule.ArrowElimination:
# #                     return go(p.function).apply(go(p.argument))
# #                 case Proof.Rule.ArrowIntroduction:
# #                     return go(p.body).abstract(p.abstraction)
# #                 case Proof.Rule.BoxElimination:
# #                     return go(p.body).unbox()
# #                 case Proof.Rule.BoxIntroduction:
# #                     return go(p.body).box(p.decoration)
# #                 case Proof.Rule.DiamondElimination:
# #                     return go(p.body).undiamond()
# #                 case Proof.Rule.DiamondIntroduction:
# #                     return go(p.body).diamond(p.decoration)
# #
# #         return go(self)
# #
# #     def subproofs(self: Proof) -> list[Proof]:
# #         match self.rule:
# #             case Proof.Rule.Variable | Proof.Rule.Lexicon:
# #                 return [self]
# #             case Proof.Rule.ArrowElimination:
# #                 return [self, *self.function.subproofs(), *self.argument.subproofs()]
# #             case Proof.Rule.ArrowIntroduction:
# #                 return [self, *self.body.subproofs()]
# #             case _:
# #                 return [self, *self.body.subproofs()]
# #
# #     def canonicalize_var_names(self: Proof) -> Proof:
# #         def de_bruijn(proof: Proof, variable: Proof) -> int:
# #             match proof.rule:
# #                 case Proof.Rule.Variable:
# #                     return 0
# #                 case Proof.Rule.ArrowIntroduction:
# #                     return 1 + de_bruijn(proof.body, variable)
# #                 case Proof.Rule.ArrowElimination:
# #                     if variable in proof.function.vars():
# #                         return de_bruijn(proof.function, variable)
# #                     return de_bruijn(proof.argument, variable)
# #                 case _:
# #                     return de_bruijn(proof.body, variable)
# #
# #         def go(proof: Proof, trans: dict[int, int]) -> tuple[Proof, dict[int, int]]:
# #             match proof.rule:
# #                 case Proof.Rule.Lexicon:
# #                     return proof, trans
# #                 case Proof.Rule.Variable:
# #                     return proof.type.var(trans.pop(proof.variable)), trans
# #                 case Proof.Rule.ArrowElimination:
# #                     fn, trans = go(proof.function, trans)
# #                     arg, trans = go(proof.argument, trans)
# #                     return fn.apply(arg), trans
# #                 case Proof.Rule.ArrowIntroduction:
# #                     trans |= {proof.abstraction.variable: (db := de_bruijn(proof.body, proof.abstraction))}
# #                     body, trans = go(proof.body, trans)
# #                     return body.abstract(proof.abstraction.type.var(db)), trans
# #                 case Proof.Rule.BoxElimination:
# #                     body, trans = go(proof.body, trans)
# #                     return body.unbox(), trans
# #                 case Proof.Rule.BoxIntroduction:
# #                     body, trans = go(proof.body, trans)
# #                     return body.box(proof.decoration), trans
# #                 case Proof.Rule.DiamondIntroduction:
# #                     body, trans = go(proof.body, trans)
# #                     return body.diamond(proof.decoration), trans
# #                 case Proof.Rule.DiamondElimination:
# #                     body, trans = go(proof.body, trans)
# #                     return body.undiamond(), trans
# #
# #         return go(self, {})[0]
# #
# #     def substitute(self: Proof, replace: Proof, with_: Proof) -> Proof:
# #         TypeInference.assert_equal(replace.type, with_.type)
# #         if (c := self.subproofs().count(replace)) != 1:
# #             raise TypeInference.TypeCheckError(f"Expected to find one occurrence of {replace} in {self}, but found {c}")
# #
# #         def go(_proof: Proof) -> Proof:
# #             if _proof == replace:
# #                 return with_
# #             match _proof.rule:
# #                 case Proof.Rule.Variable | Proof.Rule.Lexicon:
# #                     return _proof
# #                 case Proof.Rule.ArrowElimination:
# #                     return go(_proof.function).apply(go(_proof.argument))
# #                 case Proof.Rule.ArrowIntroduction:
# #                     return go(_proof.body).abstract(go(_proof.abstraction))
# #                 case Proof.Rule.BoxIntroduction:
# #                     return go(_proof.body).box(_proof.decoration)
# #                 case Proof.Rule.DiamondIntroduction:
# #                     return go(_proof.body).diamond(_proof.decoration)
# #                 case Proof.Rule.DiamondElimination:
# #                     return go(_proof.body).undiamond()
# #                 case Proof.Rule.BoxElimination:
# #                     return go(_proof.body).unbox()
# #
# #         return go(self)
# #
# #     def dia_cut(self: Proof, replace: Proof, diamond: Proof) -> Proof:
# #         return self.substitute(replace, diamond.undiamond())
# #
# #
# # def show_term(
# #         proof: Proof,
# #         show_decorations: bool = True,
# #         show_types: bool = True,
# #         word_printer: Callable[[int], str] = str) -> str:
# #     def f(_proof: Proof) -> str:
# #         return show_term(_proof, show_decorations, show_types, word_printer)
# #
# #     def v(_proof: Proof) -> str:
# #         return show_term(_proof, show_decorations, False)
# #
# #     wp = word_printer
# #
# #     def needs_par(_proof: Proof) -> bool:
# #         match _proof.rule:
# #             case Proof.Rule.Variable | Proof.Rule.Lexicon:
# #                 return False
# #             case (Proof.Rule.BoxElimination | Proof.Rule.BoxIntroduction | Proof.Rule.DiamondElimination |
# #                   Proof.Rule.DiamondIntroduction):
# #                 return not show_decorations and needs_par(_proof.body)
# #             case _:
# #                 return True
# #
# #     match proof.rule:
# #         case Proof.Rule.Lexicon:
# #             return f'{wp(proof.constant)}' if not show_types else f'{wp(proof.constant)}::{type(proof)}'
# #         case Proof.Rule.Variable:
# #             return f'x{proof.variable}' if not show_types else f'x{proof.variable}::{type(proof)}'
# #         case Proof.Rule.ArrowElimination:
# #             fn, arg = proof.function, proof.argument
# #             return f'{f(fn)} ({f(arg)})' if needs_par(arg) else f'{f(fn)} {f(arg)}'
# #         case Proof.Rule.ArrowIntroduction:
# #             var, body = proof.abstraction, proof.body
# #             return f'λ{v(var)}.({f(body)})' if needs_par(body) else f'λ{v(var)}.({f(body)})'
# #         case Proof.Rule.BoxElimination:
# #             return f'▾{proof.decoration}({f(proof.body)})' if show_decorations else f(proof.body)
# #         case Proof.Rule.BoxIntroduction:
# #             return f'▴{proof.decoration}({f(proof.body)})' if show_decorations else f(proof.body)
# #         case Proof.Rule.DiamondElimination:
#             return f'▿{proof.decoration}({f(proof.body)})' if show_decorations else f(proof.body)
#         case Proof.Rule.DiamondIntroduction:
#             return f'▵{proof.decoration}({f(proof.body)})' if show_decorations else f(proof.body)
#
#
# def serialize_proof(proof: Proof) -> SerializedProof:
#     name = proof.rule.name
#     match proof.rule:
#         case Proof.Rule.Lexicon:
#             return name, (serialize_type(proof.type), proof.constant)
#         case Proof.Rule.Variable:
#             return name, (serialize_type(proof.type), proof.variable)
#         case Proof.Rule.ArrowElimination:
#             return name, (proof.function.serialize(), proof.argument.serialize())
#         case Proof.Rule.ArrowIntroduction:
#             return name, (proof.abstraction.serialize(), proof.body.serialize())
#         case _:
#             return name, (proof.decoration, proof.body.serialize())
#
#
# def deserialize_proof(args) -> Proof:
#     match args:
#         case Proof.Rule.Lexicon.name, (t, idx):
#             return deserialize_type(t).lex(idx)
#         case Proof.Rule.Variable.name, (t, idx):
#             return deserialize_type(t).var(idx)
#         case Proof.Rule.ArrowElimination.name, (fn, arg):
#             return deserialize_proof(fn).apply(deserialize_proof(arg))
#         case Proof.Rule.ArrowIntroduction.name, (var, body):
#             return deserialize_proof(body).abstract(deserialize_proof(var))
#         case Proof.Rule.BoxElimination.name, (box, body):
#             return deserialize_proof(body).unbox(box)
#         case Proof.Rule.BoxIntroduction.name, (box, body):
#             return deserialize_proof(body).box(box)
#         case Proof.Rule.DiamondElimination.name, (diamond, body):
#             return deserialize_proof(body).undiamond(diamond)
#         case Proof.Rule.DiamondIntroduction.name, (diamond, body):
#             return deserialize_proof(body).diamond(diamond)
#         case _:
#             raise ValueError(f'Cannot deserialize {args}')
