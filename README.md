# lassy-tlg-extraction

---

Source code and Python module for the representation of typelogical grammar derivations and their extraction from 
Lassy-style dependency graphs. 


**Cool things to look out for**:
* an interface to [Alpino](http://www.let.rug.nl/vannoord/alp/Alpino/) outputs -- convert dependency graphs to λ-terms and annoy your informal linguist friends!
* a faithful implementation of (temporal) modal linear logic proofs and its terms -- almost as good as doing it on paper!
---

## Installing & Using with æthel
This repository is required to access and play with the æthel dataset, which contains typelogical analyses
for the majority of the [Lassy Small](https://taalmaterialen.ivdnt.org/download/lassy-klein-corpus6/) corpus.
Begin by cloning the project locally and unzipping the dump file in `data/` (remember to unzip).
You can then install the project by running:

```shell
python3 -m pip install .
```

Afterwards, you can load the dump by running:

```python
from aethel import ProofBank

dataset = ProofBank.load_data("PATH_TO_DUMP_FILE")
```

*Note that loading might take a short while, as proofs are reconstructed bottom up and type-checked along the way.*

### Example usage:
Please refer to `examples/` for some hands-on guides on how to use the module.


## Major Changelog
Different major versions are not backward compatible. 
Train/dev/test segmentation is respected to the largest extent possible. 
If looking for older versions, take a look at other branches of this repository.

### 1.0.1 (02/2023)
> Tidier packaging and installation. Backwards compatible with 1.0.0x.
### 1.0.0a (06/2022) 
> Implementation of long-postponed changes to the type system. In practical terms, type assignments are a bit more
> complicated but all proofs are now sound wrt. to the underlying logic.
> >* **Adjunction**: diamonds and boxes are no longer treated as ad-hoc type constructors, and their adjoint nature is
>> made explicit (i.e. ◇□A⇒A⇒□◇A). The diamond elimination and arrow introduction rules are now explicit to which
>> variable is being abstracted or replaced.
>> * **Bracketing Structure**: brackets are now explicated to regulate the applicability of diamond eliminations 
>> and box introductions. 
>> * **Structural Rules**: rules are now split between logical and structural. For the purpose of allowing 
>> **extraction** of hypotheses nested within unary structures, we now use an extraction modality reserved for the
>> higher-order types that supply them. The modality is paired to a single structural rule which permits outwards movement:
>> `<Χ>!, <Γ, Δ> → <Γ, <Χ>!, Δ>`
>> * **Proofs & Terms**: Due to the presence of term-neutral structural rules (and the ambiguity inherent to the implicit
>> cut of the diamond elimination rule) proofs are now defined separately from syntactically valid terms.
>> * **Multi-word Phrases**: Multi-word phrases are heuristically fixed to the extent possible; this permits the 
>> extraction of correct proofs when (previously unstructured) constituents are elided under a conjunction. Coverage is
>> slightly increased.
> ---
> ### Minor Changelog
> #### 1.0.0a5 (01/2023)
> * Explicit objects are hypothesized from within the local phrase. 
> #### 1.0.0a4 (12/2022)
> * Past participles now have explicit objects.
> #### 1.0.0a3 (10/2022)
> * Structural extraction now properly renames variables.
> #### 1.0.0a2 (10/2022)
> * Diamond elimination terms now act as variable binders that explicate their position via 
> the `case [τ:term] of [x:variable] in [σ:term]` term constructor (meaning "replace any occurrence of x in σ with τ").
> * To ease comprehension, the left and right side occurrences of the "same" variable in diamond eliminations are no
> longer identified, i.e. the substitute and substituted variables have unique names.
> #### 1.0.0a1 (07/2022)
> * Proofs are delivered in η-normal form, and variables are assigned unique names.
> * Natural deduction is back-and-forth compatible with proof nets again.
> * A few leftover proofs failing the linearity criterion are removed.

## Project Structure
The package source code can be found in directory `src`.
* `aethel.frontend` is the high-level interface that allows user access to the processed corpus.
* `aethel.mill` contains an implementation of the grammar's type system, namely 
 Implication-Only Multiplicative Intuitionistic Linear Logic with Unary Temporal Modalities.
Each of the following modules provide implementations for specific aspects of the logic:
  * `aethel.mill.proofs` - classes & methods for the representation and manipulation of 
  judgements, rules and proofs in natural deduction format.
  * `aethel.mill.nets` - ditto for decomposition formulas, axiom links and proof nets,
  and conversion to and from natural deduction.
  * `aethel.mill.terms` - a rudimentary implementation of the term calculus.
  * `aethel.mill.types` - a metacass-based implementation of logical formulas.
  * `aethel.mill.structures` - a handy abstraction over the antecedent structure of logical judgements.
  * `aethel.mill.serialization` - parsers and serializers for the above.
* `aethel.alpino` contains utilities to interface with Alpino-style graphs
  * `aethel.alpino.transformations` contains the linguistic transformations necessary to make Alpino 
   analyses amenable to proof-theoretic analyses, and packs them into a single pipeline. *Warning: Not for the 
  faint of heart.* 
  * `aethel.alpino.extraction` implements the extraction algorithm, whereby transformed Alpino graphs are gradually
  proven in a bottom-up fashion.
  * `aethel.alpino.lassy` - a wrapper for reading Lassy xml trees.
* `aethel.utils` general utilities:
  * `aethel.utils.graph` - definitions of simple graph-theoretic operations.
  * `aethel.utils.viz` - visualization of Lassy dependency graphs, useful for debugging.
  * `aethel.utils.tex` - export proofs and samples to latex code.

Subdirectory `examples` contains (or will contain) introductory jupyter notebooks. 
Subdirectory `scripts` contains (or will contain) high-level scripts for processing the corpus and lexicon.


---

## Requirements
Python 3.10+

If you intend to use the visualization utilities you will also need GraphViz.

---
## Citing
If you found this software or data useful in your research, please cite the corresponding [paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.647.pdf):
```BibTeX
@inproceedings{kogkalidis2020aethel,
  title={{\AE}THEL: Automatically Extracted Typelogical Derivations for Dutch},
  author={Kogkalidis, Konstantinos and Moortgat, Michael and Moot, Richard},
  booktitle={Proceedings of the 12th Language Resources and Evaluation Conference},
  pages={5257--5266},
  year={2020}
}
```
