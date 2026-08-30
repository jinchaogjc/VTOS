# VTOS: Learning to Orchestrate Vision Tools by Co-Searching Solutions and Observers

Jinchao Ge<sup>1</sup>, Lingqiao Liu<sup>2\*</sup>, Shuwen Zhao<sup>3</sup>, Lei Wang<sup>1</sup>

<sup>1</sup>University of Wollongong &nbsp;·&nbsp;
<sup>2</sup>Adelaide University &nbsp;·&nbsp;
<sup>3</sup>Tianjin University of Technology &nbsp;·&nbsp;
<sup>\*</sup>Corresponding author

**EMNLP 2026, Main Conference** &nbsp;·&nbsp; [arXiv:2606.20728](https://arxiv.org/abs/2606.20728)

---

> **⚠️ Code release in progress.**
> This repository currently contains the project description and licence only.
> The full implementation — search engine, tool library, evaluation harness and
> the searched programs reported in the paper — will be released here **before
> the conference**. Watch or star the repository to be notified.

---

## What VTOS does

Vision foundation tools — open-vocabulary detectors, segmentation models,
post-processing operators — are strong building blocks, but how well they work
depends on how they are *orchestrated*: which tool runs, in what order, with
what parameters, and under which visual conditions. Existing visual-programming
agents emit one fixed pipeline, which breaks down under dense objects,
occlusion, small targets and domain shift.

VTOS searches for the orchestration instead of assuming it. It co-searches two
kinds of executable knowledge:

- a **solution program** that composes vision tools (Grounding DINO, SAM, NMS,
  slice-and-detect) to solve the task, and
- an **observer program** that inspects candidate solutions, identifies how they
  fail, and turns that into feedback for the next round.

Observations accumulate in a shared *VisionThoughts* knowledge base that
conditions later search. What the search returns is an ordinary Python program:
inspectable, and **free of any VLM call at deployment**.

## Scope

VTOS targets settings where a fixed pipeline still leaves headroom — dense,
occluded scenes and out-of-distribution segmentation. On simpler inputs, where
one well-calibrated detector already runs near its ceiling, a static pipeline is
the better choice: it needs no search budget and gives up nothing. The paper
states this limit explicitly rather than claiming general-purpose gains.

## Evaluation

Two case studies, both scored against existing annotations so that no subjective
labelling is involved:

| Task | Data | Source |
|---|---|---|
| Dense object counting | **LVIS-Count** — 180 images, 30 categories | sampled from [LVIS](https://www.lvisdataset.org/) |
| Zero-shot disease segmentation | **PlantSeg-OOD** — 180 images, 24 species, species-disjoint splits | sampled from PlantSeg |

Both source datasets are public; the sampling protocol and splits will be
released with the code so the subsets can be reconstructed exactly.

## Citation

```bibtex
@article{ge2026vtos,
  title   = {{VTOS}: Learning to Orchestrate Vision Tools by Co-Searching
             Solutions and Observers},
  author  = {Ge, Jinchao and Liu, Lingqiao and Zhao, Shuwen and Wang, Lei},
  journal = {arXiv preprint arXiv:2606.20728},
  year    = {2026},
}
```

<!--      Once the EMNLP proceedings are published, switch this to the
     @inproceedings form with the ACL Anthology identifier. -->

## Licence

[MIT](LICENSE). The vision foundation tools VTOS orchestrates carry their own
licences; check each before use.

## Contact

Jinchao Ge — jge@uow.edu.au
