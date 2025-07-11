# LLM4Docq
Collaborative project to add docstrings to MathComp.

_All informations are updated once every 15 minutes_
<div style="text-align: center;">
  <em>Current state of the project</em>
</div>
<img src="http://51.38.234.127:8000/global.png?" alt="Alt Text" width="100%">

 
<div style="text-align: center;">
  <em>Thanks to our contributors</em>
</div>
<div style="text-align: center;">
<img src="http://51.38.234.127:8000/leaderboard.png?" alt="Alt Text" width="50%">
</div>

Want more details? Jump to [progress](#progress) to see section-by-section status.

## Introduction
**LLM4Docq** is a collaborative research project with three main objectives:

1. **Docstring enrichment for MathComp:**  
   We aim to add detailed docstrings to all elements of the MathComp library:
   - Definitions: 3,067
   - Lemmas: 14,925
   - Notations: 3,386
   - Theorems: 61
   - Facts: 654
   - Fixpoints: 140
   - Records: 256

2. **Embedding-based retrieval system:**  
   A VSCode plugin for Rocq users, enabling retrieval from natural language query.

3. **Annotator–formalizer models:**  
   - Generate a docstring from a formal statement and its context.
   - Reconstruct a formal statement from a docstring and context.

To gather the underlying dataset, we use an iterative process:
1. LLMs generate initial docstrings.
2. Experts review and give feedback on a subpart of the generated docstrings.
3. Unseen or refused entries are regenerated with new instructions based on previous feedback.

### Progress

<details>
<summary>Algebra</summary>

![Algebra Progress](http://51.38.234.127:8000/algebra.png?)
</details>

<details>
<summary>Boot</summary>

![Boot Progress](http://51.38.234.127:8000/boot.png?)
</details>

<details>
<summary>Character</summary>

![Character Progress](http://51.38.234.127:8000/character.png?)
</details>

<details>
<summary>Field</summary>

![Field Progress](http://51.38.234.127:8000/field.png?)
</details>

<details>
<summary>Fingroup</summary>

![Fingroup Progress](http://51.38.234.127:8000/fingroup.png?)
</details>

<details>
<summary>Order</summary>

![Order Progress](http://51.38.234.127:8000/order.png?)
</details>

<details>
<summary>Solvable</summary>

![Solvable Progress](http://51.38.234.127:8000/solvable.png?)
</details>

<details>
<summary>Test Suite</summary>

![Test Suite Progress](http://51.38.234.127:8000/test_suite.png?)
</details>

## How to contribute
1. **Request access** to join the project. You can reach me on [rocq zulip](https://rocq-prover.zulipchat.com/) (Théo Stoskopf).
2. **Pick a source file:**  
   Each file is managed as a separate project in our collaborative platform (Label Studio). Find one in the [project hierarchy](#project-hierarchy) below.
3. **Annotate entries.**

### Entry Interactions

- **Annotate** (required):  
  Review the proposed docstring and select its status:
  - **Acceptable:** The docstring is correct and sufficiently detailed.
  - **Needs Improvement:** The docstring is mostly correct but could be clearer or more precise.
  - **Incorrect:** The docstring is wrong or irrelevant.

- **Suggest an improved version:**  
  If the annotation is "Needs Improvement" or "Incorrect", provide a better docstring.

- **Add comments:**  
  Share additional feedback, clarifications, or suggestions for future improvements.

- **Skip:**  
  If you are unsure how to annotate the entry, you can skip it.

- **Submit:**  
  Submit your review once you have finished annotating or commenting.

**See below for some examples:**

#### Acceptable case

The docstring is correct and complete.
Select "Acceptable" and submit.

<div style="text-align: left;">
<img src="img/correct_2.png" alt="Alt Text" width="70%">
</div>

#### Needs Improvement case

The docstring is mostly correct, but could be improved or clarified.
Select "Needs Improvement”, suggest a better version and leave a comment.

<div style="text-align: left;">
<img src="img/improvement_3.png" alt="Alt Text" width="70%">
</div>

#### Incorrect case

The docstring is incorrect or unrelated to the code.
Select "Incorrect" and provide a corrected version and comment.

<div style="text-align: left;">
<img src="img/incorrect_4.png" alt="Alt Text" width="70%">
</div>

### Project hierarchy

Below is the overall hierarchy. To contribute, click on a source file in an expandable section.

<details>
<summary>Algebra</summary>

* num_theory
  * [numdomain](http://51.38.234.127:8080/projects/132/data?tab=4&labeling=1)
  * [numfield](http://51.38.234.127:8080/projects/126/data?tab=4&labeling=1)
  * [orderedzmod](http://51.38.234.127:8080/projects/182/data?tab=4&labeling=1)
  * [ssrnum](http://51.38.234.127:8080/projects/173/data?tab=4&labeling=1)
* [archimedean](http://51.38.234.127:8080/projects/115/data?tab=4&labeling=1)
* [countalg](http://51.38.234.127:8080/projects/180/data?tab=4&labeling=1)
* [finalg](http://51.38.234.127:8080/projects/171/data?tab=4&labeling=1)
* [fraction](http://51.38.234.127:8080/projects/148/data?tab=4&labeling=1)
* [intdiv](http://51.38.234.127:8080/projects/137/data?tab=4&labeling=1)
* [interval](http://51.38.234.127:8080/projects/118/data?tab=4&labeling=1)
* [interval_inference](http://51.38.234.127:8080/projects/175/data?tab=4&labeling=1)
* [matrix](http://51.38.234.127:8080/projects/109/data?tab=4&labeling=1)
* [mxalgebra](http://51.38.234.127:8080/projects/146/data?tab=4&labeling=1)
* [mxpoly](http://51.38.234.127:8080/projects/113/data?tab=4&labeling=1)
* [mxred](http://51.38.234.127:8080/projects/129/data?tab=4&labeling=1)
* [poly](http://51.38.234.127:8080/projects/110/data?tab=4&labeling=1)
* [polyXY](http://51.38.234.127:8080/projects/159/data?tab=4&labeling=1)
* [polydiv](http://51.38.234.127:8080/projects/163/data?tab=4&labeling=1)
* [qpoly](http://51.38.234.127:8080/projects/104/data?tab=4&labeling=1)
* [rat](http://51.38.234.127:8080/projects/141/data?tab=4&labeling=1)
* [ring_quotient](http://51.38.234.127:8080/projects/177/data?tab=4&labeling=1)
* [sesquilinear](http://51.38.234.127:8080/projects/153/data?tab=4&labeling=1)
* [spectral](http://51.38.234.127:8080/projects/164/data?tab=4&labeling=1)
* [ssralg](http://51.38.234.127:8080/projects/106/data?tab=4&labeling=1)
* [ssrint](http://51.38.234.127:8080/projects/155/data?tab=4&labeling=1)
* [vector](http://51.38.234.127:8080/projects/114/data?tab=4&labeling=1)
* [zmodp](http://51.38.234.127:8080/projects/156/data?tab=4&labeling=1)
</details>

<details>
<summary>Boot</summary>

* [bigop](http://51.38.234.127:8080/projects/144/data?tab=4&labeling=1)
* [binomial](http://51.38.234.127:8080/projects/139/data?tab=4&labeling=1)
* [choice](http://51.38.234.127:8080/projects/167/data?tab=4&labeling=1)
* [div](http://51.38.234.127:8080/projects/124/data?tab=4&labeling=1)
* [eqtype](http://51.38.234.127:8080/projects/103/data?tab=4&labeling=1)
* [finfun](http://51.38.234.127:8080/projects/169/data?tab=4&labeling=1)
* [fingraph](http://51.38.234.127:8080/projects/158/data?tab=4&labeling=1)
* [finset](http://51.38.234.127:8080/projects/140/data?tab=4&labeling=1)
* [fintype](http://51.38.234.127:8080/projects/150/data?tab=4&labeling=1)
* [generic_quotient](http://51.38.234.127:8080/projects/178/data?tab=4&labeling=1)
* [monoid](http://51.38.234.127:8080/projects/143/data?tab=4&labeling=1)
* [nmodule](http://51.38.234.127:8080/projects/101/data?tab=4&labeling=1)
* [path](http://51.38.234.127:8080/projects/125/data?tab=4&labeling=1)
* [prime](http://51.38.234.127:8080/projects/157/data?tab=4&labeling=1)
* [seq](http://51.38.234.127:8080/projects/121/data?tab=4&labeling=1)
* [ssrAC](http://51.38.234.127:8080/projects/181/data?tab=4&labeling=1)
* [ssrbool](http://51.38.234.127:8080/projects/172/data?tab=4&labeling=1)
* [ssreflect](http://51.38.234.127:8080/projects/197/data?tab=4&labeling=1)
* [ssrfun](http://51.38.234.127:8080/projects/191/data?tab=4&labeling=1)
* [ssrnat](http://51.38.234.127:8080/projects/119/data?tab=4&labeling=1)
* [ssrnotations](http://51.38.234.127:8080/projects/190/data?tab=4&labeling=1)
* [tuple](http://51.38.234.127:8080/projects/130/data?tab=4&labeling=1)
</details>

<details>
<summary>Character</summary>

* [character](http://51.38.234.127:8080/projects/108/data?tab=4&labeling=1)
* [classfun](http://51.38.234.127:8080/projects/142/data?tab=4&labeling=1)
* [inertia](http://51.38.234.127:8080/projects/136/data?tab=4&labeling=1)
* [integral_char](http://51.38.234.127:8080/projects/193/data?tab=4&labeling=1)
* [mxabelem](http://51.38.234.127:8080/projects/120/data?tab=4&labeling=1)
* [mxrepresentation](http://51.38.234.127:8080/projects/102/data?tab=4&labeling=1)
* [vcharacter](http://51.38.234.127:8080/projects/152/data?tab=4&labeling=1)
</details>

<details>
<summary>Field</summary>

* [algC](http://51.38.234.127:8080/projects/135/data?tab=4&labeling=1)
* [algebraics_fundamentals](http://51.38.234.127:8080/projects/196/data?tab=4&labeling=1)
* [algnum](http://51.38.234.127:8080/projects/179/data?tab=4&labeling=1)
* [closed_field](http://51.38.234.127:8080/projects/166/data?tab=4&labeling=1)
* [cyclotomic](http://51.38.234.127:8080/projects/183/data?tab=4&labeling=1)
* [falgebra](http://51.38.234.127:8080/projects/154/data?tab=4&labeling=1)
* [fieldext](http://51.38.234.127:8080/projects/127/data?tab=4&labeling=1)
* [finfield](http://51.38.234.127:8080/projects/165/data?tab=4&labeling=1)
* [galois](http://51.38.234.127:8080/projects/174/data?tab=4&labeling=1)
* [qfpoly](http://51.38.234.127:8080/projects/186/data?tab=4&labeling=1)
* [separable](http://51.38.234.127:8080/projects/147/data?tab=4&labeling=1)
</details>

<details>
<summary>Fingroup</summary>

* [action](http://51.38.234.127:8080/projects/99/data?tab=4&labeling=1)
* [automorphism](http://51.38.234.127:8080/projects/185/data?tab=4&labeling=1)
* [fingroup](http://51.38.234.127:8080/projects/100/data?tab=4&labeling=1)
* [gproduct](http://51.38.234.127:8080/projects/145/data?tab=4&labeling=1)
* [morphism](http://51.38.234.127:8080/projects/116/data?tab=4&labeling=1)
* [perm](http://51.38.234.127:8080/projects/138/data?tab=4&labeling=1)
* [presentation](http://51.38.234.127:8080/projects/192/data?tab=4&labeling=1)
* [quotient](http://51.38.234.127:8080/projects/133/data?tab=4&labeling=1)
</details>

<details>
<summary>Order</summary>

* [order](http://51.38.234.127:8080/projects/111/data?tab=4&labeling=1)
* [preorder](http://51.38.234.127:8080/projects/107/data?tab=4&labeling=1)
</details>

<details>
<summary>Solvable</summary>

* [abelian](http://51.38.234.127:8080/projects/134/data?tab=4&labeling=1)
* [alt](http://51.38.234.127:8080/projects/168/data?tab=4&labeling=1)
* [burnside_app](http://51.38.234.127:8080/projects/131/data?tab=4&labeling=1)
* [center](http://51.38.234.127:8080/projects/123/data?tab=4&labeling=1)
* [commutator](http://51.38.234.127:8080/projects/170/data?tab=4&labeling=1)
* [cyclic](http://51.38.234.127:8080/projects/112/data?tab=4&labeling=1)
* [extraspecial](http://51.38.234.127:8080/projects/151/data?tab=4&labeling=1)
* [extremal](http://51.38.234.127:8080/projects/189/data?tab=4&labeling=1)
* [finmodule](http://51.38.234.127:8080/projects/187/data?tab=4&labeling=1)
* [frobenius](http://51.38.234.127:8080/projects/161/data?tab=4&labeling=1)
* [gfunctor](http://51.38.234.127:8080/projects/195/data?tab=4&labeling=1)
* [gseries](http://51.38.234.127:8080/projects/149/data?tab=4&labeling=1)
* [hall](http://51.38.234.127:8080/projects/188/data?tab=4&labeling=1)
* [jordanholder](http://51.38.234.127:8080/projects/105/data?tab=4&labeling=1)
* [maximal](http://51.38.234.127:8080/projects/117/data?tab=4&labeling=1)
* [nilpotent](http://51.38.234.127:8080/projects/162/data?tab=4&labeling=1)
* [pgroup](http://51.38.234.127:8080/projects/122/data?tab=4&labeling=1)
* [primitive_action](http://51.38.234.127:8080/projects/160/data?tab=4&labeling=1)
* [sylow](http://51.38.234.127:8080/projects/194/data?tab=4&labeling=1)
</details>

<details>
<summary>Test Suite</summary>

* [test_guard](http://51.38.234.127:8080/projects/184/data?tab=4&labeling=1)
* [test_intro_rw](http://51.38.234.127:8080/projects/176/data?tab=4&labeling=1)
* [test_ssrAC](http://51.38.234.127:8080/projects/128/data?tab=4&labeling=1)
</details>
