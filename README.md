## lassy-tlg-extraction

---

Source code and Python module for the representation of typelogical grammar derivations and their extraction from 
Lassy-style dependency graphs. 

---

**Cool things to look out for**:
* an interface to [Alpino](http://www.let.rug.nl/vannoord/alp/Alpino/) outputs -- convert constituency graphs to λ-terms and annoy your informal linguist friends!
* a working implementation of a type system and term calculus using metaclasses -- show it to Haskell 
programmers and watch them rage in disgust!

---

### Major Versioning Changelog
**v0.9.1.devx (05/2022)**

Implementation of long-postponed changes to the type system. In practical terms, type assignments are a bit more 
complicated but all proofs are now sound wrt. to the underlying logic.
* **Adjunction**: diamonds and boxes are no longer treated as ad-hoc type constructors, and their adjoint nature is
made explicit (i.e. ◇□A⇒A⇒□◇A). 
* **Bracketing structure**: brackets are now explicated to control the applicability of diamond eliminations 
and box introductions. A minimal set of structural rules are now defined to allow the abstraction of variables nested
deeply within constituent substructures (_their formulation is still WiP so anticipate changes_).


**v0.9.devx (01/2022)**

Big changes to the extraction algorithm, data representation, and the data.
* Proofs are now constructed and delivered in natural-deduction (≡ λ-term) format. This ensures type-safety during 
extraction and loading, and allows inspection of proofs at a sub-sentential level.
* Proofs are now instances of both the `Proof` protocol and an actual type metaclass (itself an insance of the abstract 
metaclass `Type`), allowing pythonic access to a proof's type and its attributes.
* **Punctuation handling**: punctuation marks are now retained and contained in æthel samples.  
Punctuations are assigned dummy types, and their terms do not show in the derivation.
* **Fewer but longer sentences**: sentence splitting is now reserved only for non-functional branches.
Some previously problematic cases are now handled by duplicating the missing material from the source 
sentence into all of the independent generated samples, maintaining grammaticality and increasing average  
sentence length.
* **Compatibility**: this and following versions of the extraction will *no longer be compatible* with 
older dataset releases (≤ 0.4.dev). Train/dev/test segmentation is respected to the largest extent possible. 
If looking for older versions, take a look at other branches of this repository.
---

### Project Structure
* `LassyExtraction.aethel` is the (sort-of) high-level interface that allows user access to the processed corpus.
* `LassyExtraction.mill` contains an implementation of the grammar's type system, namely 
 Multiplicative Intuitionistic Linear Logic with unary Modalities, and the corresponding λ-calculus.
* `LassyExtraction.transformations` contains the linguistic transformations necessary to make Lassy-style 
 analyses amenable to proof-theoretic analyses, and packs them into a single pipeline.
* `LassyExtraction.extraction` implements the extraction algorithm, whereby transformed Lassy graphs are gradually
proven in a bottom-up fashion.
* `LassyExtraction.utils` don't look in there.

Subdirectory `scripts` contains (or will contain) example scripts that attempt to convince that the code runs and is 
useful.

---

### Requirements
Python 3.10+

If you intend to use the visualization utilities you will also need GraphViz.

---

### Installing & Using with æthel
This repository is required to access and play around with the æthel dataset, which contains typelogical analyses
for the majority of the [Lassy Small](https://taalmaterialen.ivdnt.org/download/lassy-klein-corpus6/) corpus.
You will need to download the most recent binarized version of the dataset ([0.9.dev0](https://surfdrive.surf.nl/files/index.php/s/a3mySlereVtzDSF)). 
Begin by cloning the project locally and placing the dump file in `data/` (remember to unzip).
You can then load the dump by running:

```
import LassyExtraction
aethel = LassyExtraction.ProofBank.load_data(PATH_TO_DUMP_FILE)
```
where the dataset is a container of instances of the `Sample` class. 
Note that loading might take a short while, as proofs are reconstructed bottom up and type-checked along the way.

Example usage:
```
>>> print(aethel)
æthel dump version 0.9.dev1 containing 68768 samples.
>>> print(sample := aethel.samples[1312])
WR-P-E-I-0000050381.p.1.s.82.xml(2)
>>> print(sample.show_sentence())
De weg werd gemaakt door Spaanse krijgsgevangenen , soms onder erbarmelijke omstandigheden .
>>> print(sample.show_term())
werd::◇vc(PPART)⊸◇su(NP)⊸SMAIN ▵vc(▾mod(door::◇obj1(NP)⊸□mod(PPART⊸PPART) ▵obj1(▾mod(Spaanse::□mod(NP⊸NP)) krijgsgevangenen::NP)) (▾mod(▾mod(soms::□mod((□mod(PPART⊸PPART))⊸□mod(PPART⊸PPART))) (onder::◇obj1(NP)⊸□mod(PPART⊸PPART) ▵obj1(▾mod(erbarmelijke::□mod(NP⊸NP)) omstandigheden::NP))) gemaakt::PPART)) ▵su(▾det(De::□det(N⊸NP)) weg::N)
>>> print(sample.premises[3])
Premise(word='gemaakt', pos='verb', pt='ww', lemma='maken', type=PPART)
>>> print(sample.subset)
train
```
---
### Citing
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
