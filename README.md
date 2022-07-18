# lassy-tlg-extraction

---

Source code and Python module for the representation of typelogical grammar derivations and their extraction from 
Lassy-style dependency graphs. 


**Cool things to look out for**:
* an interface to [Alpino](http://www.let.rug.nl/vannoord/alp/Alpino/) outputs -- convert constituency graphs to λ-terms and annoy your informal linguist friends!
* a faithful implementation of modal linear logic proofs and its terms -- almost as good as doing it on paper!
---

## Installing & Using with æthel
This repository is required to access and play with the æthel dataset, which contains typelogical analyses
for the majority of the [Lassy Small](https://taalmaterialen.ivdnt.org/download/lassy-klein-corpus6/) corpus.
You will need to download the most recent binarized version of the dataset
([1.0.0a1](https://surfdrive.surf.nl/files/index.php/s/xfJfVLNPNTUXQpf)). 
Begin by cloning the project locally and placing the dump file in `data/` (remember to unzip).
You can then load the dump by running:

```python
import LassyExtraction
aethel = LassyExtraction.ProofBank.load_data(PATH_TO_DUMP_FILE)
```

*Note that loading might take a short while, as proofs are reconstructed bottom up and type-checked along the way.*

### Example usage:
Please refer to `examples/` for some hands-on guides on how to use the module.


## Major Changelog
Different major versions are not backward compatible. 
Train/dev/test segmentation is respected to the largest extent possible. 
If looking for older versions, take a look at other branches of this repository.


### 1.0.0a (06/2022) 
> Implementation of long-postponed changes to the type system. In practical terms, type assignments are a bit more 
complicated but all proofs are now sound wrt. to the underlying logic.
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
> #### 1.0.0a1 (07/2022)
> * Proofs are delivered in η-normal form, and variables are assigned unique names.
> * Natural deduction is back-and-forth compatible with proof nets again.
> * A few leftover proofs failing the linearity criterion are removed. 
 
### **0.9 (01/2022)**

> Big changes to the extraction algorithm and data representation.
>> * Proofs are now constructed and delivered in natural-deduction (≡ λ-term) format. This ensures type-safety during 
>> extraction and loading, and allows inspection of proofs at a sub-sentential level.
>> * Proofs are now instances of both the `Proof` protocol and an actual type metaclass (itself an instance of the abstract 
>> metaclass `Type`), allowing pythonic access to a proof's type and its attributes.
>> * **Punctuation handling**: punctuation marks are now retained and contained in æthel samples.  
>> Punctuations are assigned dummy types, and their terms do not participate in the derivation, unless 
>> dependency annotated.
>> * **Fewer but longer sentences**: sentence splitting is now reserved only for non-functional branches.
>> Some previously problematic cases are now handled by duplicating the missing material from the source 
>> sentence into all of the independent generated samples, maintaining grammaticality and increasing average  
>> sentence length.
---

## Project Structure
* `LassyExtraction.frontend` is the high-level interface that allows user access to the processed corpus.
* `LassyExtraction.mill` contains an implementation of the grammar's type system, namely 
 Implication-Only Multiplicative Intuitionistic Linear Logic with unary Modalities.
Each of the following modules provide implementations for specific aspects of the logic:
  * `LassyExtraction.mill.proofs` - classes & methods for the representation and manipulation of 
  judgements, rules and proofs in natural deduction format.
  * `LassyExtraction.mill.nets` - ditto for decomposition formulas, axiom links and proof nets,
  and conversion to and from natural deduction.
  * `LassyExtraction.mill.terms` - a rudimentary implementation of the term calculus.
  * `LassyExtraction.mill.types` - a metacass-based implementation of logical formulas.
  * `LassyExtraction.mill.structures` - a handy abstraction over the antecedent structure of logical judgements.
  * `LassyExtraction.mill.serialization` - parsers and serializers for the above.
* `LassyExtraction.transformations` contains the linguistic transformations necessary to make Lassy-style 
 analyses amenable to proof-theoretic analyses, and packs them into a single pipeline. *Warning: Not for the 
faint of heart.* 
* `LassyExtraction.extraction` implements the extraction algorithm, whereby transformed Lassy graphs are gradually
proven in a bottom-up fashion.
* `LassyExtraction.utils` general utilities:
  * `LassyExtraction.utils.graph` - definitions of simple graph-theoretic operations.
  * `LassyExtraction.utils.viz` - visualization of Lassy dependency graphs, useful for debugging.
  * `LassyExtraction.utils.lassy` - a wrapper for reading Lassy xml trees.
  * `LassyExtraction.utils.tex` - export proofs and samples to latex code.

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
