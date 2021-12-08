from LassyExtraction.aethel import (Term, Var, Lex, Application, Abstraction,
                                    BoxIntro, BoxElim, DiamondIntro, DiamondElim, ProofNet, ProofFrame,
                                    AxiomLinks, WordType, FunctorType, PolarizedType, ModalType)
from LassyExtraction.milltypes import polarize_and_index
from LassyExtraction.proofs import get_result, ProofError
from typing import Set


def is_mod(functor: Term) -> bool:
    if isinstance(functor, BoxElim):
        if functor.box == "mod" or functor.box == "predm" or functor.box == "app":
            # todo: check if it is a mandatory modifier: (▾ᵐ(▿ᵐ(σ))
            return not mandatory_mod(functor)
    return False


def mandatory_mod(functor: BoxElim) -> bool:
    body = functor.body
    return isinstance(body, DiamondElim) and body.diamond == functor.box


def remove_all_mods(term: Term) -> Term:
    # σ, τ := x | w | σ(τ) | λx.σ | ▴(b)σ | ▾(b)σ | ▵(d)σ | ▿(d)σ
    #
    # ▵ᵒᵇʲ¹((▾ᵐᵒᵈ(w₁₃)(▾ᵈᵉᵗ(w₁₁) w₁₂))) -> ▵ᵒᵇʲ¹(▾ᵈᵉᵗ(w₁₁) w₁₂))
    # σ = ▾ᵐᵒᵈ(w₁₃)                 |   swims (hairy(dog))
    # τ = (▾ᵈᵉᵗ(w₁₁) w₁₂)           |   ((happily) swim) (hairy(dog))    -> swims(dog)
    #(▾ᵐᵒᵈ((▾ᵐᵒᵈ(w₀:□ᵐᵒᵈ(□ᵐᵒᵈ(sᴍᴀɪɴ → sᴍᴀɪɴ) → □ᵐᵒᵈ(sᴍᴀɪɴ → sᴍᴀɪɴ)))
    #       ▴ᵐᵒᵈ(λx₀.(▾ᵐᵒᵈ((▾ᵈᵉᵗ((▾ᵐᵒᵈ(w₁:□ᵐᵒᵈ(□ᵈᵉᵗ(ɴ → □ᵐᵒᵈ(sᴍᴀɪɴ → sᴍᴀɪɴ)) → □ᵈᵉᵗ(ɴ → □ᵐᵒᵈ(sᴍᴀɪɴ → sᴍᴀɪɴ))))
    #           ▴ᵈᵉᵗ(λx₁.▴ᵐᵒᵈ(λx₂.(▾ᵐᵒᵈ((▾ᵈᵉᵗ(w₂:□ᵈᵉᵗ(ɴ → □ᵐᵒᵈ(sᴍᴀɪɴ → sᴍᴀɪɴ))) x₁))
    #               x₂))))) w₃:ɴ)) x₀))))
    #       (((w₄:◊ᵖᶜᴘᴘ → ◊ˢᵉᴠɴᴡ → ◊ˢᵘɴᴘ → sᴍᴀɪɴ
    #           ▵ᵖᶜ((w₈:◊ᵒᵇʲ¹ɴᴘ → ᴘᴘ
    #           ▵ᵒᵇʲ¹(((((w₁₂:◊ᶜⁿʲ(□ᵈᵉᵗ(ɴ → ɴᴘ) → ◊ᵒᵇʲ¹ɴ → ɴᴘ) → ◊ᶜⁿʲ(□ᵈᵉᵗ(ɴ → ɴᴘ) → ◊ᵒᵇʲ¹ɴ → ɴᴘ) → □ᵈᵉᵗ(ɴ → ɴᴘ) → ◊ᵒᵇʲ¹ɴ → ɴᴘ
    #           ▵ᶜⁿʲ(λx₃.λx₄.(▾ᵐᵒᵈ((w₁₁:◊ᵒᵇʲ¹ɴ → □ᵐᵒᵈ(ɴᴘ → ɴᴘ) ▵ᵒᵇʲ¹(▿ᵒᵇʲ¹(x₄)))) (▾ᵈᵉᵗ(x₃) w₁₀:ɴ))))
    #           ▵ᶜⁿʲ(λx₅.λx₆.(▾ᵐᵒᵈ((w₁₄:◊ᵒᵇʲ¹ɴ → □ᵐᵒᵈ(ɴᴘ → ɴᴘ) ▵ᵒᵇʲ¹(▿ᵒᵇʲ¹(x₆)))) (▾ᵈᵉᵗ(x₅) w₁₃:ɴ))))
    #           ▴ᵈᵉᵗ(λx₇.(▾ᵈᵉᵗ(w₉:□ᵈᵉᵗ(ɴ → ɴᴘ)) x₇))) ▵ᵒᵇʲ¹(w₁₅:ɴ)))))) ▵ˢᵉ(w₇:ᴠɴᴡ)) ▵ˢᵘ((▾ᵈᵉᵗ(w₅:□ᵈᵉᵗ(ɴ → ɴᴘ)) w₆:ɴ))))

    if isinstance(term, Var):
        return term
    if isinstance(term, Lex):
        return term
    if isinstance(term, Application):
        functor = term.functor
        argument = term.argument
        if is_mod(functor):
            return remove_all_mods(argument)
        else:
            functor = remove_all_mods(functor)
            argument = remove_all_mods(argument)
            return Application(functor, argument)
    if isinstance(term, Abstraction):
        # remember mistake!!
        body = remove_all_mods(term.body)
        return Abstraction(body, term.abstraction.idx)
    if isinstance(term, BoxIntro):
        body = remove_all_mods(term.body)
        return BoxIntro(body, term.box)
    if isinstance(term, BoxElim):
        body = remove_all_mods(term.body)
        return BoxElim(body)
    if isinstance(term, DiamondIntro):
        body = remove_all_mods(term.body)
        return DiamondIntro(body, term.diamond)
    if isinstance(term, DiamondElim):
        body = remove_all_mods(term.body)
        return DiamondElim(body)
    raise TypeError(f"it did not work :( {type(term)}")


### Make the new Proofnet for the unmodded data
def _remove_all_mods(pn: ProofNet) -> ProofNet:
    removed_term = remove_all_mods(pn.get_term())
    removed_pf = ...  # an ordered subset of the original PF obtained from participating_ids
    return term_to_axiom_links(removed_term, removed_pf)


def term_to_axiom_links(term: Term, proof_frame: ProofFrame) -> AxiomLinks:
    idx = -500

    def get_type_from_premise(idx: int) -> WordType:
        return proof_frame.premises[idx].type

    def invert_polarity(hatted: WordType) -> WordType:
        if isinstance(hatted, FunctorType):
            hatted_arg = invert_polarity(hatted.argument)
            hatted_res = invert_polarity(hatted.result)
            return FunctorType(hatted_arg, hatted_res)
        if isinstance(hatted, PolarizedType):
            return PolarizedType(hatted.type, not hatted.polarity, hatted.index)
        if isinstance(hatted, ModalType):
            return type(hatted)(invert_polarity(hatted.content), hatted.modality)
        raise TypeError(f"{hatted, type(hatted)}")

    def f(_term: Term, index: int) -> tuple[AxiomLinks, WordType, dict[int, WordType], int]:
        if isinstance(_term, Var):
            index, new_hat_type = polarize_and_index(_term.t(), True, index)
            return set(), new_hat_type, {_term.idx: new_hat_type}, index
        if isinstance(_term, Lex):
            return set(), get_type_from_premise(_term.idx), dict(), index
        if isinstance(_term, Application):
            axiomlinks_arg, output_arg, var_arg, index = f(_term.argument, index)
            axiomlinks_func, output_func, var_func, index = f(_term.functor, index)
            axiomlinks = axiomlinks_arg.union(axiomlinks_func)
            for atom_arg, atom_func in zip(output_arg, output_func):
                axiom_link = (atom_arg.index, atom_func.index) if atom_arg.polarity else (atom_func.index, atom_arg.index)
                axiomlinks.add(axiom_link)
            return axiomlinks, get_result(output_func), {**var_arg, **var_func}, index
        if isinstance(_term, Abstraction):
            axiomlinks, output, variables, index = f(_term.body, index)
            new_hat_type = variables.pop(_term.abstraction.idx)
            new_output = FunctorType(invert_polarity(new_hat_type), output)
            return axiomlinks, new_output, variables, index
        if isinstance(_term, (DiamondElim, DiamondIntro, BoxElim, BoxIntro)):
            return f(_term.body, index)
        raise TypeError(f':( what is this?? {type(_term)}')


    axiom_links, output_type, _, _ = f(term, idx)
    axiom_links.add((output_type.index, proof_frame.conclusion.index))

    # (a, b)       <~ (a, -c), (-c, b)      remove negative detours from hypotheses
    def fix_detours(axiom_links: AxiomLinks) -> AxiomLinks:
        left_match = {right: left for left, right in axiom_links if right < 0}
        new_links = {(left_match[left], right) for left, right in axiom_links if left < 0}
        var_sets = {(left, right) for left, right in axiom_links if left < 0 or right < 0}
        axiom_links = (axiom_links.union(new_links) - var_sets)

        return axiom_links

    # fix negative detours so that there collapsed and be used in fix detours
    def negative_detours(axiom_links:AxiomLinks) -> AxiomLinks:
        result_links = axiom_links.copy()
        for left, right in axiom_links:
            for links, rechts in axiom_links:
                if right == links:
                    result_links.add((left, rechts))
                    result_links.remove((links, rechts))
                    result_links.remove((left, right))
                    result_links = negative_detours(result_links)
                    return result_links

        return result_links

    neg_links = {(left, right) for left, right in axiom_links if left < 0 or right < 0}
    #print(f"neg links: {neg_links}")
    neg_axioms = negative_detours(neg_links)
    #print(f"neg_axioms: {neg_axioms}")
    axiom_links = (axiom_links - neg_links).union(neg_axioms)
    #axiom_links = axiom_links.union(neg_axioms)
    #print(axiom_links)


    #axiom_links = fix_detours(axiom_links)

    # todo: fix indices
    # todo: sanity check: should return original pf (& new pf should be smaller or the same length)
    return axiom_links


### Sanity check: term_to_axiom_links should return the original proofframe
# sanity check for whole dataset: list of proofnets, pns
def sanity_check_axiom_links(pns: list[ProofNet]):
    right = []
    wrong = []
    weird_key = {}
    weird_proof = {}
    counter = 0
    for pn in pns:
            try:
                new_axioms = term_to_axiom_links(pn.get_term(), pn.proof_frame)
                original_axioms = pn.axiom_links
                if not new_axioms==original_axioms:
                    wrong.append(counter)
                    counter += 1
                    continue
                else:
                    right.append(counter)
                    counter += 1
            except KeyError as e:
                weird_key[counter] = e
                counter += 1
                continue
            except ProofError as es:
                weird_proof[counter] = es
                counter += 1
                continue
    return right, wrong, weird_key, weird_proof

    # right = length 68795
    # wrong = length 0 --> these were the ones that had multiple 'layers' of negative detours ()
    # weird = length 2691  --> ProofError ; 30/11/2021 3816 Proof- and KeyError...

# sanity check for individual proofnets from dataset
def sanity_check_axioms(pn: ProofNet):
    new_axioms = term_to_axiom_links(pn.get_term(), pn.proof_frame)
    original_axioms = pn.axiom_links
    if not new_axioms==original_axioms:
        raise TypeError(f"Error, axiomslinks are not the same, \n new: {new_axioms} \n original: {original_axioms}")
    #else:
        #print("The right axiom links!!")




'''import pickle
with open('LassyExtraction//train_dev_test_0.4.dev0.p', 'rb') as f:
    train, dev, test = pickle.load(f)
pns = sum([train, dev, test], [])
print(pns[6])
print(pns[6].print_term())
from LassyExtraction.unmod import remove_all_mods
term = pns[6].get_term()
unmodded = remove_all_mods(term)
from LassyExtraction.terms import print_term
print_term(unmodded, True, lambda x: str(x))
print_term(term, True, lambda x: str(x))
pns[6].proof_frame.get_words()
list(enumerate(pns[6].proof_frame.get_words()))
from LassyExtraction.unmod import participating_ids
participating_ids(unmodded)
'''

# σ, τ := x | w | σ(τ) | λx.σ | ▴(b)σ | ▾(b)σ | ▵(d)σ | ▿(d)σ
# todo: run unmod and see if it works --> save number of sentences where it does not seem to work

'((4 ▵ᵛᶜ(((15 ▵ᵖʳᵉᵈᶜ((14 ▵ᵛᶜ((16 ▵ᶜᵐᵖᵇᵒᵈʸ((17 ▵ᶜᵐᵖᵇᵒᵈʸ(18)))))))) ▵ᵒᵇʲ¹((▾ᵈᵉᵗ(11) 12))))) ▵ˢᵘ((▾ᵈᵉᵗ(5) 7)))'
# zou hebben nodig om te reorg een aantal het legerkorps
'((4 ▵ᵛᶜ((▾ᵐᵒᵈ(10) (▾ᵐᵒᵈ((0 ▵ᵒᵇʲ¹((▾ᵐᵒᵈ(2) (▾ᵈᵉᵗ(1) 3))))) ((15 ▵ᵖʳᵉᵈᶜ((14 ▵ᵛᶜ((16 ▵ᶜᵐᵖᵇᵒᵈʸ((17 ▵ᶜᵐᵖᵇᵒᵈʸ(18)))))))) ▵ᵒᵇʲ¹((▾ᵐᵒᵈ(13) (▾ᵈᵉᵗ(11) 12)))))))) ▵ˢᵘ((▾ᵐᵒᵈ((8 ▵ᵒᵇʲ¹(9))) (▾ᵐᵒᵈ(6) (▾ᵈᵉᵗ(5) 7)))))'


def participating_ids(term: Term) -> Set[int]:
    if isinstance(term, Var):
        return set()
    if isinstance(term, Lex):
        return {term.idx}
    if isinstance(term, Application):
        functor = participating_ids(term.functor)
        argument = participating_ids(term.argument)
        return functor.union(argument)
    if isinstance(term, Abstraction):
        body = participating_ids(term.body)
        return body
    return participating_ids(term.body)

# ((w₄ ▵ᵛᶜ((▾ᵐᵒᵈ(x) (▾ᵐᵒᵈ((w₀ ▵ᵒᵇʲ¹((▾ᵐᵒᵈ(w₂) (▾ᵈᵉᵗ(w₁) w₃))))) ((w₁₅ ▵ᵖʳᵉᵈᶜ((w₁₄ ▵ᵛᶜ((w₁₆ ▵ᶜᵐᵖᵇᵒᵈʸ((w₁₇ ▵ᶜᵐᵖᵇᵒᵈʸ(w₁₈)))))))) ▵ᵒᵇʲ¹((▾ᵐᵒᵈ(w₁₃) (▾ᵈᵉᵗ(w₁₁) w₁₂)))))))) ▵ˢᵘ((▾ᵐᵒᵈ((w₈ ▵ᵒᵇʲ¹(w₉))) (▾ᵐᵒᵈ(w₆) (▾ᵈᵉᵗ(w₅) w₇)))))
