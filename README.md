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
[ÆThel](https://github.com/konstantinosKokos/aethel) with Python.
Begin by cloning the project locally and placing the dump file in the outermost directory.
You can then access the data by running:
```
>>> import pickle
>>> with open('./dump_file.p', 'rb') as f: 
>>>     data = pickle.load(f)
```
Example usage:
```
>>> from LassyExtraction.utils.tools import *
>>> sample = data[1312]
>>> dag, proof = sample
>>> get_words(dag)
['Hij', 'werd', 'als', 'gevangene', 'aan', 'boord', 'van', 'het', 'marineschip', 'Northumberland', 'vervoerd', 'en', 'na', 'een', 'tocht', 'van', '70', 'dagen', 'afgezet', 'op', 'het', 'verlaten', 'eiland', 'Sint Helena', 'in', 'het', 'zuidelijke', 'deel', 'van', 'de', 'Atlantische Oceaan']
>>> get_types(dag)
[VNW(+,0), <PPART(-,1)> vc → <VNW(-,2)> su → SMAIN(+,3), <WW(-,4)> cmp_body → [SMAIN(-,5) → SMAIN(+,6)] predm, WW(+,7), <NP(-,8)> obj1 → [PPART(-,9) → PPART(+,10)] mod, NP(+,11), <NP(-,12)> obj1 → [NP(-,13) → NP(+,14)] mod, [N(-,15) → NP(+,16)] det, N(+,17), [NP(-,18) → NP(+,19)] app, PPART(+,20), <PPART(-,21)> cnj → <PPART(-,22)> cnj → PPART(+,23), <NP(-,24)> obj1 → [PPART(-,25) → PPART(+,26)] mod, [N(-,27) → NP(+,28)] det, N(+,29), <NP(-,30)> obj1 → [NP(-,31) → NP(+,32)] mod, [N(-,33) → NP(+,34)] det, N(+,35), PPART(+,36), <NP(-,37)> obj1 → [PPART(-,38) → PPART(+,39)] mod, [N(-,40) → NP(+,41)] det, [NP(-,42) → NP(+,43)] mod, N(+,44), [NP(-,45) → NP(+,46)] app, <NP(-,47)> obj1 → [NP(-,48) → NP(+,49)] mod, [N(-,50) → NP(+,51)] det, [NP(-,52) → NP(+,53)] mod, N(+,54), <NP(-,55)> obj1 → [NP(-,56) → NP(+,57)] mod, [ADJ(-,58) → NP(+,59)] det, ADJ(+,60)]
>>> get_lambda(dag, proof)
((als::ᴡᴡ → sᴍᴀɪɴ → sᴍᴀɪɴ  gevangene::ᴡᴡᶜᵐᵖ_ᵇᵒᵈʸ)ᵖʳᵉᵈᵐ ((werd::ᴘᴘᴀʀᴛ → ᴠɴᴡ → sᴍᴀɪɴ ((en::ᴘᴘᴀʀᴛ → ᴘᴘᴀʀᴛ → ᴘᴘᴀʀᴛ ((aan::ɴᴘ → ᴘᴘᴀʀᴛ → ᴘᴘᴀʀᴛ ((van::ɴᴘ → ɴᴘ → ɴᴘ (Northumberland::ɴᴘ → ɴᴘᵃᵖᵖ (het::ɴ → ɴᴘᵈᵉᵗ  marineschip::ɴ))ᵒᵇʲ¹)ᵐᵒᵈ  boord::ɴᴘ)ᵒᵇʲ¹)ᵐᵒᵈ  vervoerd::ᴘᴘᴀʀᴛ)ᶜⁿʲ) ((op::ɴᴘ → ᴘᴘᴀʀᴛ → ᴘᴘᴀʀᴛ ((in::ɴᴘ → ɴᴘ → ɴᴘ ((van::ɴᴘ → ɴᴘ → ɴᴘ (de::ᴀᴅᴊ → ɴᴘᵈᵉᵗ  Atlantische Oceaan::ᴀᴅᴊ)ᵒᵇʲ¹)ᵐᵒᵈ (zuidelijke::ɴᴘ → ɴᴘᵐᵒᵈ (het::ɴ → ɴᴘᵈᵉᵗ  deel::ɴ)))ᵒᵇʲ¹)ᵐᵒᵈ (Sint Helena::ɴᴘ → ɴᴘᵃᵖᵖ (verlaten::ɴᴘ → ɴᴘᵐᵒᵈ (het::ɴ → ɴᴘᵈᵉᵗ  eiland::ɴ))))ᵒᵇʲ¹)ᵐᵒᵈ ((na::ɴᴘ → ᴘᴘᴀʀᴛ → ᴘᴘᴀʀᴛ ((van::ɴᴘ → ɴᴘ → ɴᴘ (70::ɴ → ɴᴘᵈᵉᵗ  dagen::ɴ)ᵒᵇʲ¹)ᵐᵒᵈ (een::ɴ → ɴᴘᵈᵉᵗ  tocht::ɴ))ᵒᵇʲ¹)ᵐᵒᵈ  afgezet::ᴘᴘᴀʀᴛ))ᶜⁿʲ)ᵛᶜ)  Hij::ᴠɴᴡˢᵘ))
``` 
---
If you have issues using the code or need help getting started, feel free to get in touch.