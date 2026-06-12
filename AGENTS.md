# AGENTS.md — working on DL World

DL World is a walkable city where a real neural network trains live in the
browser, built for developing intuition about deep learning down to
element-level tensor ops. Read this before changing anything.

## Commands

```bash
npm run dev          # vite dev server (http://localhost:5173)
npm test             # vitest: engine gradient checks + training smoke tests
npm run build        # tsc --noEmit && vite build
npm run fetch-mnist  # regenerate public/data/mnist.bin from source MNIST
```

## Core principles (do not break these)

1. **Every number on screen is real.** Panels read live tensors from the
   actual training run — never hardcode, fake, or pre-bake a value. If a
   visualization can't be driven by real data, don't build it.
2. **Losses are composed from primitives on purpose.** `crossEntropy`,
   `bceWithLogits`, `mnistLoss`, `mseLoss` in `src/engine/tensor.ts` are
   built from `exp`/`sumRows`/`divCol`/`gather`/etc. so the recorded
   autograd graph contains every station the player walks through. Do not
   fuse them for performance; the graph IS the content.
3. **Tensor labels are load-bearing.** Panels look up graph nodes by their
   `.named("...")` label strings (e.g. `"z1 = x·W1 + b1"`, `"softmax"`,
   `"log probs"`). Renaming a label breaks the panel that reads it — grep
   `nodes[` before touching any `.named()` call.
4. **The display pass.** `Scenario.trainStep` runs the real SGD step, then
   a *second* recorded forward/backward with the updated weights. This
   keeps every panel's values, gradients, and on-demand recomputations
   (microscopes, gradient checks) mutually consistent. Don't "optimize"
   the second pass away.
5. **Gradients must check out.** Any new engine op needs a finite-difference
   test in `tests/engine.test.ts` (use the existing `gradCheck` helper).
   Backward functions must match the *clamped/actual* forward (see `log`).
6. **Concept-first naming, no chapter numbers in-world.** Building names,
   district names, panel titles, and blurbs describe concepts ("Loss
   District", "First Steps Quarter", "Sequence Summit"), never book
   chapters. The book is how the *curriculum* is organized, not how the
   city is organized: the fastai chapter mapping lives only in README.md.
7. **The book is the source material — read it, don't guess.** A full copy
   of fast.ai's *Deep Learning for Coders* notebooks is vendored at
   `reference/fastbook-master.zip`. Before building or changing any feature
   that demonstrates a book concept, unzip the relevant chapter and read
   it (notebooks are JSON; extract the markdown/code cells, e.g.
   `unzip -p reference/fastbook-master.zip fastbook-master/08_collab.ipynb`
   piped through a small script). Models, formulas, jargon, and named
   techniques in DL World must match what the book actually teaches — e.g.
   the Echo Tower is the book's `LMModel2`, the Taste Cinema is
   `DotProductBias` with `sigmoid_range` and weight decay, label smoothing
   uses the book's (1−ε+ε/N) targets. Simplifications are fine, but they
   must be acknowledged in the panel copy, never silent.

## UI conventions

- `liveRegion` (`src/ui/panels/common.ts`) re-renders a panel section on
  trainer steps (throttled ~300ms). **Interactive state must live in the
  panel's closure** (not the DOM) so it survives re-renders. It skips
  re-rendering while focus is on a SELECT/INPUT inside the region — that's
  the fix for dropdowns being yanked shut during training; keep it.
- Sliders that must survive continuous interaction live *outside*
  liveRegion roots (see `trainerControls`).
- Never use `innerHTML +=` on an element that contains a `<canvas>` — the
  re-serialization wipes the canvas bitmap. Build rows with `el()` + DOM
  `append`.
- `digitCanvas` requires an integer scale (it coerces, but pass integers).
  `heatmap` auto-downsamples large matrices and may stretch non-uniformly.
- `el(tag, cls, html)` sets innerHTML: only ever pass developer-authored
  strings (numbers via `fmt()` are fine). Anything user- or error-derived
  goes through `textContent` (see `showScreen` in `src/main.ts`).
- Panel renderers return a cleanup function; remove any window-level
  listeners and trainer subscriptions there.
- New panels: `registerPanel(id, ...)` (ids are `building.station`, must be
  unique — duplicates throw), then reference the id from a station in
  `src/world/buildings.ts`.

## World conventions

- Tiles are 32px; the map is drawn procedurally in `src/world/city.ts` —
  no image assets anywhere.
- Buildings carry a `tour` number (1–21) shown as a gold badge on their
  sign; the route is also painted as road chevrons. If you add a building,
  give it a tour stop and keep the route geographically sensible:
  data → forward → loss → backward → step → metrics, then extensions,
  then south across the river into the Frontier (non-image data).
- District names are lawn-sign placards (`DISTRICT_SIGNS` in
  `src/world/city.ts`): bright text on a dark board, planted on the grass
  strip *above* their building group. `generate()` keeps trees/flowers out
  of each sign's footprint — if you add or move a sign, keep it in
  `DISTRICT_SIGNS` so the decor clearing applies.
- The Frontier's datasets (`src/sim/datasets.ts`) are generated in code
  with *planted* structure + noise; models must rediscover the structure.
  Don't replace them with hardcoded "learned" results — panels compare
  planted truth vs. learned parameters, and that only works if training is
  real. The sentiment corpus uses identical templates for both classes so
  function words carry no class signal; keep that property.
- The avatar's head is deliberately a non-human mint color — keep it
  playful and unreal; never use a real skin tone.
- Works on touch devices: on-screen controls in `src/ui/touch.ts` write
  into `Game.keys` (the same set the keyboard uses). The run toggle holds
  `"shift"` in that set — don't bulk-`clear()` the key set casually.

## Debugging

- `window.dlworld` exposes `{ game, world, openPanel, closePanel }` —
  teleport with `dlworld.game.avatar.x/y`, open any panel by id, drive
  trainers via `dlworld.world.main.setRunning(true)` etc.
- MNIST pack format ("DLW1" magic) is documented in
  `scripts/fetch-mnist.mjs`; loaders validate the magic.
