# Demo video script: Shennong's Glass (target about 4:30, hard limit 5:00)

Say it the way you'd explain it to a friend in the lab. You don't need to read it word for word;
if a sentence doesn't sound like you, change it. Each block is one voice file, named as in the
heading (e.g. `01_hook`).

**Numbers to quote** come from `results/metrics.json`. Re-check them if the analysis is re-run.

---

### 01_hook (about 25 s)
> There's an old Chinese legend about Shennong, the Divine Farmer. He tasted hundreds of plants to
> work out which ones heal and which ones kill. And the story goes that his body was transparent,
> so he could actually watch each herb working inside him.
>
> We kind of have that now. It's called an organ-on-a-chip: living human tissue you can look
> straight through. What's missing is someone to watch it. So I built Shennong's Glass.

### 02_problem (about 30 s)
> At the end of most of these experiments you get a stack of microscope images: nuclei, tubulin,
> actin. And from those, a scientist wants to know four things. Is the drug doing anything? Is it
> just killing the cells? How is it working? And how sure can I be?
>
> Right now that's a lot of manual work, and the AI tools that do exist will always hand you an
> answer, even when they really shouldn't.

### 03_idea (about 30 s)
> So the idea is simple. I don't train anything. I take DINOv2, a vision model that's already
> learned what images look like, and use it to describe every cell and every field, one stain at a
> time. Then I add some classic measurements a biologist can actually read, and I compare everything
> against the untreated wells on the same plate.
>
> To guess a mechanism, it just looks up the most similar drugs in a reference library. New chip?
> Swap the library. And all of this is real data, nothing simulated.

### 04_taxol (about 25 s)
> Let's try it. This is a real field of cells treated with taxol, and I've pulled taxol out of the
> library, so it's never seen this drug.
>
> It finds the cells, and says the field looks more changed than any untreated one. Its call:
> microtubule stabiliser, which is right. The 90% prediction set has just that one answer. It
> explains that the nuclei are broken up, which is what happens when cells get stuck dividing, and
> the closest matches are docetaxel and epothilone, the same family of drugs.

### 05_cytoB (about 10 s)
> Cytochalasin B, which wrecks the actin skeleton. Called an actin disruptor, and it lines up with
> cytochalasin D and latrunculin. Good.

### 06_dmso (about 10 s)
> Now plain untreated cells. Nothing's changed, so it doesn't make up a mechanism. It just says
> there's nothing here.

### 07_colchicine (about 15 s)
> This one I like showing. It's a colchicine field that's pretty borderline. The guess is actually
> wrong, but look: it's marked low confidence. It tells you not to trust it. That's the behaviour
> I care about.

### 08_cytoD (about 20 s)
> The second tab is about dose. For 112 compounds I fit two curves: when the cells start to look
> different, and when they start dying. The gap between those is what matters. Cytochalasin D
> changes the cells at doses more than ten times lower than the ones that kill them.

### 09_etoposide (about 8 s)
> Etoposide is clearly active, but it isn't killing cells in this range.

### 10_vinblastine (about 25 s)
> And this is my favourite. Vinblastine has no label anywhere in this dataset. The system calls it a
> microtubule destabiliser at seven of its eight doses, starting from three nanomolar.
>
> That's correct. And vinblastine is one of the chemo drugs that damages patients' nerves, by
> breaking exactly those microtubules. Which is the kind of thing you'd want a nerve-on-a-chip to
> catch early.

### 11_atlas (about 20 s)
> This is every compound at every dose, about 900 treatments, laid out by how similar they look.
> Nobody told the model what the mechanisms are, but they still group together. And the 800 or so
> doses with no label get placed on the map too.

### 12_benchmark (about 35 s)
> To check it properly, I used the standard BBBC021 benchmark: about 5,800 real images, and every
> drug is hidden while it's being tested.
>
> It gets the mechanism right 85% of the time, with no training on this data, and 77% when whole
> imaging batches are held out as well. I tried a bunch of different settings and it stayed between
> 84 and 88%. The 90% prediction sets actually cover the right answer 91% of the time, usually with
> just one guess. And if a mechanism isn't in the library at all, it can tell, with an AUROC of 0.79.

### 13_close (about 25 s)
> It's all open source, and you can rerun the whole thing on a free Kaggle account and a laptop.
> The next step is to build the reference library from actual nerve-on-a-chip images, where
> tubulin and actin show you axon damage directly.
>
> That's Shennong's Glass. Thanks for watching.

---

**On screen throughout:** *"Images: BBBC021v1, Caie et al. 2010, Broad Bioimage Benchmark
Collection"*. No music, or royalty-free only.
