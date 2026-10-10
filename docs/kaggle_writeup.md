# Shennong's Glass: training-free foundation-model phenotypic screening with calibrated uncertainty for organ-on-chip drug evaluation

## Category Declaration
**End-to-End System**

## Demo Video
https://youtu.be/W5NdRDSqj0U

## Code Repository Link
https://github.com/shreshth006/shennongs-glass (MIT licence; Kaggle feature-extraction notebooks are public and linked from the README)

> **Real data, no simulation.** Every number below is computed on 5,775 real microscopy fields (BBBC021, 113 compounds × 8 doses) with held-out-compound protocols. It is fully reproducible from the public repo.

## Project Summary

*Shennong, the Divine Farmer of Chinese legend, tasted hundreds of herbs to tell medicine from
poison. His body was transparent, so he could watch each herb act on his organs. An organ-on-chip
is that transparent body, and Shennong's Glass is the eye that watches it.*

Organ-on-chip and other microphysiological assays usually end with fluorescence images of nuclei,
β-tubulin and F-actin. Turning those images into decisions still relies on assay-specific analysis
and expert inspection, and AI classifiers give an answer even when they shouldn't. **Shennong's
Glass** is an end-to-end system that takes the three stains and returns:

1. a **phenotypic activity score**, calibrated against vehicle controls;
2. **viability and Hill-fitted dose-response**, giving a phenotype-to-cytotoxicity window and a
   triage label for each compound;
3. a **mechanism-of-action ranking with a conformal prediction set** that has guaranteed coverage;
4. a **novel-mechanism alarm**;
5. a **plain-language explanation** of what changed in the cells.

The image encoder is a frozen self-supervised foundation model (DINOv2), so **no training and no
labels are needed**. Moving to a new chip only means swapping the reference library. On the
BBBC021 benchmark (113 compounds × 8 doses) it reaches **85.4%** MoA accuracy on held-out
compounds and **77.2%** when imaging batches are held out as well. Its 90% conformal sets reach
**91.3%** coverage, and mechanisms missing from the library are detected with
**AUROC 0.79**. A four-tab Gradio app makes all of this usable without code.

## Technical Report

Full report (PDF, 9 pages): https://github.com/shreshth006/shennongs-glass/blob/main/report/shennongs_glass_report.pdf

**Team:** Shreshth Mishra (solo; AI / computer science; Vellore Institute of Technology)

Key points:

### Problem
Three gaps:
- **Assay-specific analysis.** Organ-on-chip datasets are too small to train deep models.
- **No reliability information.** Classifiers always output a mechanism.
- **Potency and toxicity judged separately**, although a phenotype only matters if it appears
  below the cytotoxic dose.

### Data
BBBC021v1 (Caie et al. 2010; Broad Bioimage Benchmark Collection): MCF-7 cells, 113 compounds at
8 doses, 3 channels, 55 plates, 10 batches. 103 treatments are annotated with 12 MoAs. The images
are © AstraZeneca and are not redistributed; scripts fetch them from the public sources. No human
or patient data is used.

### Method
- **Three views of each field:**
  - interpretable morphology from classical segmentation (23 features);
  - frozen DINOv2 embeddings of single-cell crops, per channel;
  - DINOv2 embeddings of the whole field, per channel.
- **Normalisation:** per-plate DMSO centring, then typical-variation normalisation (PCA whitening
  fitted on vehicle wells).
- **Mechanism:** nearest-neighbour MoA ranking, with split-conformal prediction sets calibrated
  leave-one-compound-out, and a novelty flag on library similarity.
- **Dose and toxicity:** viability from nuclei counts, and phenotypic activity calibrated on the
  vehicle null distribution, both fitted with Hill curves.

### Results (held-out compounds)

| Profile | NSC | NSCB |
|---|---|---|
| Classical morphology | 70.9% | 59.8% |
| DINOv2 single cells (no TVN) | 79.6% | 67.4% |
| DINOv2 whole field + TVN | 79.6% | 71.7% |
| DINOv2 single cells + TVN | 85.4% | 80.4% |
| **Fusion (Shennong's Glass)** | **85.4%** | **77.2%** |

- **Conformal sets:** 91.3% coverage at the 90% target, 1.18 MoAs on average.
- **Novel-MoA detection:** AUROC 0.79.
- **Robustness:** fused profile within 84.5%–88.3% NSC across every whitening/fusion setting tried (single views vary 78.6%–90.3%).
- **Replicate consistency:** 88.3% of annotated treatments.
- **Dose-response:** 112 compounds triaged by their phenotype-to-cytotoxicity window.
- **Finds neurotoxic mechanisms it was never shown:** vinblastine, a vinca alkaloid that causes chemotherapy-induced neuropathy, is absent from the annotated set. It is called a *microtubule destabiliser* at 7 of its 8 doses, starting at 3 nM.

For context, the best published pipelines on BBBC021, which are trained or hand-engineered, report about 90–96% NSC. Our training-free, CPU-only configuration is below that. Its value is that it needs no labels, calibrates its uncertainty and is stable across design choices.

### Reproduction
```bash
pip install -r requirements.txt && kaggle auth login
bash scripts/fetch_data.sh && python scripts/run_analysis.py && python app/app.py
```

### Limitations
- Validated on 2D MCF-7 cells, not yet on organ-on-chip images. The method only assumes the three
  stains, but 3D chips need validation.
- Cell count is a viability proxy, not a dedicated viability assay.
- The conformal guarantee is marginal.
- The reported run uses a reduced, CPU-only configuration (ViT-S, 2 fields per well, 7 of 8 extraction shards). A single field is noisier than replicate wells, and the app flags borderline fields as low-confidence.

### Impact
- **Fast deployment:** a lab can deploy it on a new chip in a day by imaging reference compounds
  and vehicle controls.
- **Safe triage:** calibrated sets and novelty flags send ambiguous or unprecedented phenotypes to
  an expert instead of mislabelling them.
- **Neural organ-on-chip:** tubulin and actin are exactly the readouts of axonal toxicity, so the
  system is a candidate early-warning tool for neurotoxic liabilities.

### Sources and licences
- BBBC021v1: Caie et al., *Mol. Cancer Ther.* 2010; Ljosa et al., *Nat. Methods* 2012
  (© AstraZeneca, research use).
- DINOv2: Oquab et al. 2023 (Apache-2.0).
- PyTorch, scikit-image, scikit-learn, UMAP, Gradio (open-source licences).
- Code: MIT licence.

## Optional Demo Link
No hosted demo. The four-tab Gradio app runs locally with `python app/app.py` (setup in the repo README) and is shown end to end in the demo video.
