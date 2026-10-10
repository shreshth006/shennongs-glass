# Demo video script: Shennong's Glass (target 4:30, hard limit 5:00)

**Setup**
- Start the app with `.venv/bin/python app/app.py` and open http://127.0.0.1:7860. Zoom the browser
  to about 110%.
- Record the screen at 1080p, either with Fedora's recorder (Ctrl+Shift+Alt+R) or OBS. Speak
  calmly and leave a short pause between sections, which you can trim later.
- **Keep on screen the whole time** (a corner caption, or the app header): *"Images: BBBC021v1,
  Caie et al. 2010, Broad Bioimage Benchmark Collection"*. Use no music, or royalty-free only.
- Before recording, run each example once so the model is loaded and nothing stalls on camera.

---

### 0:00–0:25 Hook (app header with the Shennong quote)
> "In Chinese legend, Shennong tasted hundreds of herbs to tell medicine from poison. His body
> was transparent, so he could watch each herb act on his organs. An organ-on-a-chip is that
> transparent body. **Shennong's Glass** is the AI that watches it."

### 0:25–0:55 The problem (README table, or talk over the app)
> "Organ-on-chip assays end with fluorescence images: nuclei, tubulin, actin. A scientist needs
> four answers. Is the compound doing anything? Is it just killing cells? How does it act? And
> can I trust that answer? Today that means manual analysis and models that never say 'I don't
> know'."

### 0:55–1:25 The idea (pipeline figure, report Figure 1)
> "Shennong's Glass needs **no training**. A frozen foundation model, DINOv2, describes every cell
> and every field, channel by channel. Classical segmentation adds measurements a biologist can
> read. Everything is calibrated against the untreated wells on the same plate. A mechanism comes
> from the most similar compounds in a reference library, so moving to a new chip means a new
> library, not new training. And everything you'll see runs on **real images, not simulations**."

### 1:25–2:40 Live demo (Tab 1, Analyze images)
1. Pick the **taxol** example. Note that taxol is removed from the library ("blind test"). Click
   Analyze.
   > "A real field of taxol-treated cells, and taxol itself has been removed from the library.
   > It finds every nucleus. Activity: stronger than 100% of untreated fields. Mechanism:
   > microtubule stabiliser. The 90% conformal set, which is guaranteed to contain the truth 90%
   > of the time on new compounds, has just this one answer. And the reason, in words: nuclear
   > shape is fragmented, the signature of cells stuck in mitosis. The nearest references are
   > docetaxel and epothilone, the same drug family."
2. Pick **cytochalasin B**, then Analyze.
   > "An actin poison. Correctly called an actin disruptor, matched to latrunculin and
   > cytochalasin D."
3. Pick **DMSO**, then Analyze.
   > "Untreated cells. Activity is normal, so the system **refuses** to name a mechanism."
4. Pick **colchicine**, then Analyze.
   > "And here's why that matters. This colchicine field is borderline, so the system flags it as
   > low confidence. Its guess is wrong, and it warns you. A model that knows when it doesn't
   > know."

### 2:40–3:30 Dose and toxicity (Tab 2, Compound explorer)
- Show **cytochalasin D**:
  > "For each of 112 compounds we fit dose-response curves for phenotype and for viability. The
  > gap between them is the window we triage on. Cytochalasin D changes cells more than ten
  > times below the dose that kills them."
- Show **etoposide**:
  > "Etoposide is active, but not cytotoxic in this range."
- Show **vinblastine**:
  > "Vinblastine was **never labelled** in this dataset. Shennong's Glass calls it a microtubule
  > destabiliser at seven of its eight doses, starting at 3 nanomolar. Vinblastine is a vinca
  > alkaloid, one of the drugs that damage nerves in chemotherapy patients by breaking
  > microtubules. That's exactly the readout a neural organ-on-chip needs."

### 3:30–3:55 Atlas (Tab 3)
> "A map of all 906 compound-dose treatments. Mechanisms form clusters that the model was never
> told about, and the 800 unlabelled doses are placed and labelled too."

### 3:55–4:35 Validation (Tab 4, Benchmark)
> "We validated on the standard BBBC021 benchmark: 5,775 real fields, each compound held out.
> **85% of mechanisms are identified correctly** with no training on this data at all, and 77%
> when entire imaging batches are held out as well. Across every design choice we tried, the
> result stays between 84 and 88%. The 90% conformal sets reach 91% coverage, with just over one
> mechanism per set. And when a mechanism is missing from the library, the system notices,
> with an AUROC of 0.79."

### 4:35–5:00 Close (GitHub page)
> "Everything is open source and reproducible on a free Kaggle account and a laptop. Next, we'd
> rebuild the reference library on neural organ-on-chip images, where tubulin and actin read out
> axonal damage. **Shennong's Glass**: see the remedy, and the poison, before they reach a
> patient. Thank you."

---
**Numbers to quote:** these come from `results/metrics.json`. Re-check them if the analysis is
re-run.
