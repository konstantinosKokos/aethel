# Lassy-TLG-Extraction
Source code for the conversion of Lassy dependency graphs into type-logical grammar derivations.

[Link to paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.647.pdf)

---

## Project Structure
* `LassyExtraction.milltypes` implements the type grammar.
* `LassyExtraction.graphutils` contains utility classes and functions for graph processing.
* `LassyExtraction.transformations` implements the preprocessing compatibility transformations for Lassy graphs.
* `LassyExtraction.extraction` implements the typing algorithm for dependency graph nodes.
* `LassyExtraction.proofs` implements the conversion from typed graphs to atomic type bijections (proofnet axiom links).
* `LassyExtraction.lambdas` implements the conversion from typed graphs and axiom links to lambda terms.
* `LassyExtraction.viz` contains utility classes for graph visualization. 
* `LassyExtraction.utils.tools` contains helper functions and shortcuts for parsed data.

---
### Requirements
Python3.8

If you intend to use the visualization utilities you will also need GraphViz.

---
### Using with Æthel
The code in this repository is necessary to open and edit the binarized dumps of 
[Æthel](https://github.com/konstantinosKokos/aethel) with Python.
Begin by cloning the project locally and placing the dump file in the outermost directory.
You can then access the data by running:
```
>>> import pickle
>>> with open('./train_dev_test.p', 'rb') as f: 
>>>     train, dev, test = pickle.load(f)
```
Example usage:
```
>>> from LassyExtraction.utils.tools import *
>>> from pprint import pprint
>>> sample = train[1312]
>>> dag, proof = sample
>>> pprint(get_context(dag))
[('In', <NP(-,0)> obj1 → [PPART(-,1) → PPART(+,2)] mod),
 ('het', [N(-,3) → NP(+,4)] det),
 ('laatste', [NP(-,5) → NP(+,6)] mod),
 ('hoofdstuk', N(+,7)),
 ('van', <NP(-,8)> obj1 → [NP(-,9) → NP(+,10)] mod),
 ('dit', [N(-,11) → NP(+,12)] det),
 ('tweede', [NP(-,13) → NP(+,14)] mod),
 ('deel', N(+,15)),
 ('worden', <PPART(-,16)> vc → <NP(-,17)> su → SMAIN(+,18)),
 ('door', <NP(-,19)> obj1 → [PPART(-,20) → PPART(+,21)] mod),
 ('Professor', NP(+,22)),
 ('Dimitri Mortelmans', [NP(-,23) → NP(+,24)] app),
 ('Koen Ponnet', NP(+,25)),
 ('en', <NP(-,26)> cnj → <NP(-,27)> cnj → <NP(-,28)> cnj → NP(+,29)),
 ('Koen Vleminckx', NP(+,30)),
 ('de', [N(-,31) → NP(+,32)] det),
 ('resultaten', N(+,33)),
 ('gepresenteerd', PPART(+,34)),
 ('van', <NP(-,35)> obj1 → [NP(-,36) → NP(+,37)] mod),
 ('een', [N(-,38) → NP(+,39)] det),
 ('analyse', N(+,40)),
 ('naar', <NP(-,41)> obj1 → [NP(-,42) → NP(+,43)] mod),
 ('de', [N(-,44) → NP(+,45)] det),
 ('noden', N(+,46)),
 ('en',
  <([N(-,47) → NP(+,48)] det) → NP(-,49)> cnj → <([N(-,50) → NP(+,51)] det) → NP(-,52)> cnj → ([N(+,53) → NP(-,54)] det) → NP(+,55)),
 ('behoeften', N(+,56)),
 ('van', <N(-,57)> obj1 → [NP(-,58) → NP(+,59)] mod),
 ('weduwen', N(+,60))]
>>> get_lambda(dag, proof, show_types=False, show_word_names=False)
'((w₈ ((w₉ (((w₁₃ (w₁₁ᵃᵖᵖ  w₁₀)ᶜⁿʲ)  w₁₂ᶜⁿʲ)  w₁₄ᶜⁿʲ)ᵒᵇʲ¹)ᵐᵒᵈ ((w₀ ((w₄ (w₆ᵐᵒᵈ (w₅ᵈᵉᵗ  w₇))ᵒᵇʲ¹)ᵐᵒᵈ (w₂ᵐᵒᵈ (w₁ᵈᵉᵗ  w₃)))ᵒᵇʲ¹)ᵐᵒᵈ  w₁₇))ᵛᶜ) ((w₁₈ ((w₂₁ ((w₂₆  w₂₇ᵒᵇʲ¹)ᵐᵒᵈ (((w₂₄ λx₀.(x₀ᵈᵉᵗ  w₂₃)ᶜⁿʲ) λx₀.(x₀ᵈᵉᵗ  w₂₅)ᶜⁿʲ) λx₀.(w₂₂ᵈᵉᵗ x₀)))ᵒᵇʲ¹)ᵐᵒᵈ (w₁₉ᵈᵉᵗ  w₂₀))ᵒᵇʲ¹)ᵐᵒᵈ (w₁₅ᵈᵉᵗ  w₁₆))ˢᵘ)'
``` 
---
If you have issues using the code or need help getting started, feel free to get in touch.