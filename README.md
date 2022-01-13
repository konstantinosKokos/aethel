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

### Changelog

**01/2022**

Big changes to the extraction algorithm, data representation, and the data.
* Proofs are now constructed and delivered in natural-deduction (≡ λ-term) format. This ensures type-safety during 
extraction and loading, and allows inspection of proofs at a sub-sentential level.
* Proofs are now instances of both the `Proof` protocol and an actual type object (itself an insance of `Type`), 
allowing pythonic access to a proof's type and its attributes.
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
from LassyExtraction.aethel import aethel
dataset = aethel.load_data(PATH_TO_DUMP_FILE)
```
where the dataset is a container of instances of the `Sample` class. 
Note that loading might take a short while, as proofs are reconstructed bottom up and type-checked along the way.

Example usage:
```
>>> print(dataset)
æthel dump version 0.9.dev0, containing 56858 training samples, 6095 dev samples, and 5766 test samples.
>>> sample = dataset.train[1312]
>>> print(sample)
WR-P-E-I-0000015007.p.1.s.120.xml(1)
>>> print(sample.show_sentence())
De toenemende spanning bereikte een voorlopig hoogtepunt met de Arabische opstand die begon in 1936 en in 1939 door bijzonder hardhandig ingrijpen van het Britse leger tot een einde kwam .
>>> print(sample.show_proof())
▾mod(met::◇obj1(NP)⊸□mod(SMAIN⊸SMAIN) ▵obj1(▾mod(Arabische::□mod(NP⊸NP)) (▾mod(die::(◇relcl(◇su(VNW)⊸◇su(VNW)⊸SSUB))⊸□mod(NP⊸NP) ▵relcl(λx0.(λx1.(en::◇cnj(SSUB)⊸◇cnj(SSUB)⊸SSUB ▵cnj(▾mod(in::◇obj1(TW)⊸□mod(SSUB⊸SSUB) ▵obj1(1939::TW)) (▾mod(door::◇obj1(NP)⊸□mod(SSUB⊸SSUB) ▵obj1(▾mod(▾mod(bijzonder::□mod((□mod(NP⊸NP))⊸□mod(NP⊸NP))) hardhandig::□mod(NP⊸NP)) (▾mod(van::◇obj1(NP)⊸□mod(NP⊸NP) ▵obj1(▾mod(Britse::□mod(NP⊸NP)) (▾det(het::□det(N⊸NP)) leger::N))) ingrijpen::NP))) (kwam::◇pc(PP)⊸◇su(VNW)⊸SSUB ▵pc(tot::◇obj1(NP)⊸PP ▵obj1(▾det(een::□det(N⊸NP)) einde::N)) x0::◇su(VNW)))) ▵cnj(▾mod(in::◇obj1(TW)⊸□mod(SSUB⊸SSUB) ▵obj1(1936::TW)) (begon::◇su(VNW)⊸SSUB x1::◇su(VNW))))))) (▾det(de::□det(N⊸NP)) opstand::N)))) (bereikte::◇obj1(NP)⊸◇su(NP)⊸SMAIN ▵obj1(▾mod(voorlopig::□mod(NP⊸NP)) (▾det(een::□det(N⊸NP)) hoogtepunt::N)) ▵su(▾mod(toenemende::□mod(NP⊸NP)) (▾det(De::□det(N⊸NP)) spanning::N)))
>>> print(sample.premises[12])
Premise(word='begon', pos='verb', pt='ww', lemma='beginnen', type=◇su(VNW)⊸SSUB)
>>> print(sample.subset)
dev
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
