# 🏙 DL World

**A walkable city inside a neural network.**

A real two-layer neural network trains on real MNIST digits, live in your
browser — and the city *is* that computation. Walk around it, enter the
buildings, and inspect what is actually happening during the forward pass,
the backward pass, and inference: down to the individual multiply-adds of a
single matmul cell. South of the river, **the Frontier** leaves images
behind: training tricks, movie-taste embeddings, decision-tree forests,
text pipelines and a tiny language model — all running live too.

Built around the concepts of **fast.ai's _Deep Learning for Coders_
([fastbook](https://github.com/fastai/fastbook)), chapters 4–12**. A full
copy of the book's notebooks is vendored at `reference/fastbook-master.zip`
so features can always be checked against the source material (see
AGENTS.md). In-world, everything is named after *concepts*, never after
chapters — the chapter mapping below exists only in this README.

## Run it

```bash
npm install
npm run dev        # open http://localhost:5173 (desktop or mobile)
npm test           # engine gradient checks + training smoke tests
```

The MNIST subset (`public/data/mnist.bin`, 3,000 train + 600 test) is
committed; regenerate it with `npm run fetch-mnist`.

**Controls:** WASD / arrows to walk · **E** to enter buildings and inspect
machines · **Esc** to go back · Shift to run. On touch devices an on-screen
pad appears (with a run/walk toggle). Press **▶ train** in the top bar to
set the whole city in motion, and follow the numbered signs ① → ㉑ for the
guided tour through the training loop and on across the river.

## The city → curriculum map

| District | Buildings | fastai concepts |
|---|---|---|
| **Data Quarter** | Dataset Warehouse, Batch Depot | ch.4 — images as tensors (open a crate: every pixel as a number), shapes/rank/flattening, the shuffling DataLoader and mini-batches |
| **Forward Avenue** | Linear Mill №1, Activation Springs, Linear Mill №2 | ch.4 — `z = x·W + b` with a click-any-cell **dot-product microscope**, weights-as-images, bias, ReLU (and why nonlinearity matters, with a worked collapse of two linear layers) |
| **Loss District** | Cross-Entropy Foundry | ch.5 — softmax → log → NLL as an **assembly line** with the live batch's numbers at every station; a loss ledger of the currently-confusing images; the Return Belt verifying ∂L/∂logits = (p−y)/B numerically |
| **Gradient Row** | Backprop Works, Optimizer Depot | ch.4 — the recorded computation graph walked in reverse, one weight's gradient derived by hand against autograd, a finite-difference **Gradient Check Lab**, the SGD update floor with a live lr dial, and a 1-D **loss-landscape slice** |
| **Civic Center** | Metrics Observatory | ch.4/5 — loss vs metric, live curves, confusion matrix, top losses |
| **First Steps Quarter** (ch. 4) | Pixel Similarity Museum, Linear Cottage | the L1/L2 mean-image baseline, then the 785-parameter 3-vs-7 learner with `mnist_loss` — watch its weights become a picture |
| **Tuning Heights** (ch. 5) | LR Finder Tower | `lr_find()` — sweep a fresh model until the loss explodes |
| **Side Quest Yards** + **Deployment Dock** (ch. 6) | Multi-Label Workshop, Regression Studio, Inference Gallery | sigmoid+BCE per label, MSE regression to the digit's ink center, and a drawing canvas that runs your handwriting through the model — inference as "the forward pass with frozen weights" |
| **Refinement Row** (ch. 7) | Refinement Gym | training a state-of-the-art model: input normalization (real train-set stats), label smoothing (the (1−ε+ε/N) target vector with live loss values), mixup (the actual blended images from the live batch, λ ∈ 0.5–1 like the book's pseudocode), and TTA (averaging predictions over shifted views) — all racing an identically-initialized *plain twin* trained on the same batches |
| **Taste Quarter** (ch. 8) | Taste Cinema | collaborative filtering from scratch: the book's `DotProductBias` model (factors + biases + `sigmoid_range`) with weight decay, trained on a planted-structure ratings grid; a dot-product desk for single predictions, learned-bias interpretation, and a factor projector that reveals the planted genres the model rediscovers |
| **Table Grove** (ch. 9) | Decision Arboretum | tabular modeling without gradients: CART regression trees grown by variance reduction (the candidate-question audition shown with real numbers), a walkable grown tree, bagging into a random forest, out-of-bag error and split-gain feature importance |
| **Language Lane** (ch. 10–11) | Tokenizer Mill, Sentiment Studio | the text pipeline as mid-level plumbing: tokenization with special tokens (xxbos/xxunk), train-only vocab with a min-freq cutoff, numericalization into tensor rows — then a live bag-of-words classifier with one inspectable weight per word and weight decay |
| **Sequence Summit** (ch. 12) | Echo Tower | a language model from scratch on human numbers: the book's `LMModel2` (shared hidden state, one reused weight matrix — an unrolled RNN), quizzed on the held-out tail of the count, compared against the most-common-token baseline, and generating text greedily or by temperature sampling |

The Frontier's non-image datasets (ratings, reviews, rents, the counting
corpus) are generated deterministically in code with *planted* structure
plus noise — the models never see the generators and must rediscover the
structure live, which is what lets panels compare "planted truth" with
"learned factors" honestly.

## How it works

- **`src/engine/`** — a tiny define-by-run autograd engine written from
  scratch in TypeScript (`Tensor`, matmul, broadcasting, reductions,
  softmax/CE/BCE/MSE built **from primitives** so the recorded graph
  contains every station you walk through). Verified against numeric
  differentiation in `tests/engine.test.ts`.
- **`src/sim/`** — eight live training scenarios (10-class MLP, linear 3v7,
  multi-label, regression, the refinery twins, collaborative filtering,
  bag-of-words sentiment, the RNN language model), trainers, the LR finder,
  the pixel baseline, plus gradient-free CART trees/forests and the
  Frontier's generated datasets. After each SGD step a fresh recorded
  forward/backward pass keeps every panel's values and gradients mutually
  consistent.
- **`src/world/`** — the procedurally-drawn city, avatar, interiors
  (Canvas 2D, no game engine, no assets).
- **`src/ui/`** — the HUD and 55 inspection panels (DOM overlays with live
  heatmaps, charts and element-level expansions).

No backend, no GPU, no dependencies beyond Vite/TypeScript/Vitest.

## On "dynamic scaffolding"

Rather than generating explanations on the fly (which would need an LLM in
the loop and could hallucinate math), DL World gets the same effect
deterministically: because the engine records the *actual* computation
graph, every node can be expanded on demand — a matmul cell into its 784
products, the loss into its per-station values, any gradient into a
finite-difference re-derivation. Everything you zoom into is the real
number from the real training run happening in your tab.
