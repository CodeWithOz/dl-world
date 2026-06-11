# 🏙 DL World

**A walkable city inside a neural network.**

A real two-layer neural network trains on real MNIST digits, live in your
browser — and the city *is* that computation. Walk around it, enter the
buildings, and inspect what is actually happening during the forward pass,
the backward pass, and inference: down to the individual multiply-adds of a
single matmul cell.

Built around the concepts of **fast.ai's _Deep Learning for Coders_,
chapters 4–6**.

## Run it

```bash
npm install
npm run dev        # open http://localhost:5173 (desktop browser)
npm test           # engine gradient checks + training smoke tests
```

The MNIST subset (`public/data/mnist.bin`, 3,000 train + 600 test) is
committed; regenerate it with `npm run fetch-mnist`.

**Controls:** WASD / arrows to walk · **E** to enter buildings and inspect
machines · **Esc** to go back · Shift to run. Press **▶ train** in the top
bar to set the whole city in motion. On viewports narrower than ~980px the
city gates close (desktop-only by design; `?nogate` overrides for dev).

## The city → curriculum map

| District | Buildings | fastai concepts |
|---|---|---|
| **Data Quarter** | Dataset Warehouse, Batch Depot | ch.4 — images as tensors (open a crate: every pixel as a number), shapes/rank/flattening, the shuffling DataLoader and mini-batches |
| **Forward Avenue** | Linear Mill №1, Activation Springs, Linear Mill №2 | ch.4 — `z = x·W + b` with a click-any-cell **dot-product microscope**, weights-as-images, bias, ReLU (and why nonlinearity matters, with a worked collapse of two linear layers) |
| **Loss District** | Cross-Entropy Foundry | ch.5 — softmax → log → NLL as an **assembly line** with the live batch's numbers at every station; a loss ledger of the currently-confusing images; the Return Belt verifying ∂L/∂logits = (p−y)/B numerically |
| **Gradient Row** | Backprop Works, Optimizer Depot | ch.4 — the recorded computation graph walked in reverse, one weight's gradient derived by hand against autograd, a finite-difference **Gradient Check Lab**, the SGD update floor with a live lr dial, and a 1-D **loss-landscape slice** |
| **Civic Center** | Metrics Observatory | ch.4/5 — loss vs metric, live curves, confusion matrix, top losses |
| **Chapter 4 Quarter** | Pixel Similarity Museum, Linear Cottage | the L1/L2 mean-image baseline, then the 785-parameter 3-vs-7 learner with `mnist_loss` — watch its weights become a picture |
| **Ch. 5 Heights** | LR Finder Tower | `lr_find()` — sweep a fresh model until the loss explodes |
| **Chapter 6 Yards** | Multi-Label Workshop, Regression Studio, Inference Gallery | sigmoid+BCE per label, MSE regression to the digit's ink center, and a drawing canvas that runs your handwriting through the model — inference as "the forward pass with frozen weights" |

## How it works

- **`src/engine/`** — a tiny define-by-run autograd engine written from
  scratch in TypeScript (`Tensor`, matmul, broadcasting, reductions,
  softmax/CE/BCE/MSE built **from primitives** so the recorded graph
  contains every station you walk through). Verified against numeric
  differentiation in `tests/engine.test.ts`.
- **`src/sim/`** — four live training scenarios (10-class MLP, linear 3v7,
  multi-label, regression), trainers, the LR finder and the pixel baseline.
  After each SGD step a fresh recorded forward/backward pass keeps every
  panel's values and gradients mutually consistent.
- **`src/world/`** — the procedurally-drawn city, avatar, interiors
  (Canvas 2D, no game engine, no assets).
- **`src/ui/`** — the HUD and 36 inspection panels (DOM overlays with live
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
