# Lassy-TLG-Extraction
Source code for the conversion of Lassy dependency graphs into type-logical grammar derivations.

[Link to paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.647.pdf)

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
>>> sample = train[1312]
>>> sample.proof_frame
op:<ɴᴘ> obj1 → [ɪɴғ → ɪɴғ] mod, de:[ɴ → ɴᴘ] det, verjaardag:ɴ, laat:<ɪɴғ> vc → <ɴᴘ> obj1 → <ɴ> su → sᴍᴀɪɴ, Laslo:ɴ, de:[ɴ → ɴᴘ] det, wens:ɴ, van:<ɴᴘ> obj1 → [ɴᴘ → ɴᴘ] mod, zijn:[ɴ → ɴᴘ] det, moeder:ɴ, in vervulling:ɴᴘ, gaan:<ɴᴘ> svp → ɪɴғ ⊢ SMAIN
>>> sample.proof_frame.get_words()
['op', 'de', 'verjaardag', 'laat', 'Laslo', 'de', 'wens', 'van', 'zijn', 'moeder', 'in vervulling', 'gaan']
>>> sample.proof_frame.get_types()
'[<NP(-,0)> obj1 → [INF(-,1) → INF(+,2)] mod, [N(-,3) → NP(+,4)] det, N(+,5), <INF(-,6)> vc → <NP(-,7)> obj1 → <N(-,8)> su → SMAIN(+,9), N(+,10), [N(-,11) → NP(+,12)] det, N(+,13), <NP(-,14)> obj1 → [NP(-,15) → NP(+,16)] mod, [N(-,17) → NP(+,18)] det, N(+,19), NP(+,20), <NP(-,21)> svp → INF(+,22)]'
>>> sample.axiom_links
{(16, 7), (9, 23), (4, 0), (10, 8), (19, 17), (22, 1), (20, 21), (2, 6), (12, 15), (5, 3), (13, 11), (18, 14)}
>>> sample.print_term(show_words=True, show_types=False, show_decorations=True)
'(((laat ((op (deᵈᵉᵗ verjaardag)ᵒᵇʲ¹)ᵐᵒᵈ (gaan in vervullingˢᵛᵖ))ᵛᶜ) ((van (zijnᵈᵉᵗ moeder)ᵒᵇʲ¹)ᵐᵒᵈ (deᵈᵉᵗ wens))ᵒᵇʲ¹) Lasloˢᵘ)'
``` 
---
If you have issues using the code or need help getting started, feel free to get in touch.