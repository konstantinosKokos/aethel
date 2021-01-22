# lassy-tlg-extraction
Source code for the conversion of Lassy dependency graphs into type-logical grammar derivations.

Read more about the process in our LREC [paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.647.pdf).

---

## Project Structure
* `LassyExtraction.milltypes` implements the type grammar.
* `LassyExtraction.terms` implements the term grammar.
* `LassyExtraction.graphutils` contains utility classes and functions for graph processing.
* `LassyExtraction.transformations` implements the preprocessing compatibility transformations for Lassy graphs.
* `LassyExtraction.extraction` implements the typing algorithm for dependency graph nodes.
* `LassyExtraction.proofs` implements the conversion from typed graphs to atomic type bijections (proofnet axiom links).
* `LassyExtraction.aethel` implements high-level classes intended for front-end use.
* `LassyExtraction.viz` contains utility classes for graph visualization.
* `LassyExtraction.utils.printing` contains pretty-printing functions for types and terms.

---
### Requirements
Python3.8

If you intend to use the visualization utilities you will also need GraphViz.

---
### Using with æthel
The code in this repository is necessary to open and edit the binarized dumps of 
[æthel](https://github.com/konstantinosKokos/aethel) with Python.
You can download the most recent binarized dump [here](https://surfdrive.surf.nl/files/index.php/s/zHEgwDJQ7jxnpCI) 
(current version is *0.4.dev0*).
Begin by cloning the project locally and placing the data file in the outermost directory.
You can then access the data by running:
```
>>> import pickle
>>> with open('./data/train_dev_test_0.4.dev0.p', 'rb') as f:
...    train, dev, test = pickle.load(f) 
```
where `train`, `dev` and `test` are lists of the [ProofNet](https://github.com/konstantinosKokos/lassy-tlg-extraction/blob/40b9223e9b13f35909e34c776b9aedcb4b1f3627/LassyExtraction/aethel.py#L65) class.

Example usage:
```
>>> sample = train[1312]
sample.proof_frame
Alle:□ᵈᵉᵗ(ɴ → ɴᴘ), films:ɴ, zijn:◊ᵛᶜᴘᴘᴀʀᴛ → ◊ˢᵘɴᴘ → sᴍᴀɪɴ, heel:□ᵐᵒᵈ(ᴀᴘ → ᴀᴘ), barok:ᴀᴘ, versierd:◊ᵖʳᵉᵈᶜᴀᴘ → ᴘᴘᴀʀᴛ ⊢ sᴍᴀɪɴ
>>> sample.proof_frame.get_words()
['Alle', 'films', 'zijn', 'heel', 'barok', 'versierd']
>>> sample.proof_frame.get_types()
[□ᵈᵉᵗ(ɴ(-,0) → ɴᴘ(+,1)), ɴ(+,2), ◊ᵛᶜᴘᴘᴀʀᴛ(-,3) → ◊ˢᵘɴᴘ(-,4) → sᴍᴀɪɴ(+,5), □ᵐᵒᵈ(ᴀᴘ(-,6) → ᴀᴘ(+,7)), ᴀᴘ(+,8), ◊ᵖʳᵉᵈᶜᴀᴘ(-,9) → ᴘᴘᴀʀᴛ(+,10)]
>>> sample.axiom_links
{(5, 11), (2, 0), (1, 4), (7, 9), (8, 6), (10, 3)}
>>> sample.print_term(show_words=True, show_types=False, show_decorations=True)
'zijn ▵ᵛᶜ(versierd ▵ᵖʳᵉᵈᶜ(▾ᵐᵒᵈ(heel) barok)) ▵ˢᵘ(▾ᵈᵉᵗ(Alle) films)'
``` 
---
If you need access to the processed Lassy trees, encounter issues using the code or just need help getting started, 
feel free to get in touch.
