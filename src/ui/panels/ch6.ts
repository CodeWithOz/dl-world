// Side Quest Yards: the Multi-Label Workshop (sigmoid + BCE per label) and
// the Regression Studio (predict coordinates with MSE).

import { Tensor } from "../../engine/tensor";
import { MULTILABEL_NAMES, digitProperties, inkCenter } from "../../sim/scenarios";
import { registerPanel } from "../panel";
import { chip, digitCanvas, el, fmt } from "../widgets";
import { liveRegion, picker, trainerControls } from "./common";

export function registerCh6Panels(): void {
  // ------------------------------------------------------------ workshop ---
  registerPanel("workshop.train", {
    title: "Training Bench — multi-label model",
    subtitle:
      "One image can answer several yes/no questions at once: has a loop? is even? is ≥ 5? (like a photo that contains several objects at once).",
    render(body, world) {
      const [controls, cleanup] = trainerControls(world.workshop);
      body.append(controls);
      body.append(
        el(
          "p",
          "explain",
          "Why not softmax here? Softmax forces the 10 classes to compete for one total of 1.0 — perfect when exactly one answer is true. With independent questions, each output gets its OWN sigmoid and its own binary cross-entropy; nothing competes. That swap of the final activation + loss is the entire difference between this workshop and the foundry.",
        ),
      );
      return cleanup;
    },
  });

  registerPanel("workshop.labels", {
    title: "Label Wall — three sigmoids per image",
    subtitle: "Each label is its own little yes/no classifier sharing the same body.",
    render(body, world) {
      let sample = 0;
      const [region, cleanup, refresh] = liveRegion(world.workshop, (root) => {
        const s = world.multi;
        const logits = s.nodes["logits (one per label)"];
        if (!logits) return;
        const batch = s.lastBatch;
        sample = Math.min(sample, batch.length - 1);
        const digit = world.data.trainLabels[batch[sample]];
        const truth = digitProperties(digit);
        root.append(
          picker("sample:", batch.length, sample, (v) => { sample = v; refresh(); },
            (i) => `#${i} (a “${world.data.trainLabels[batch[i]]}”)`),
        );
        const row = el("div", "hstack");
        const left = el("div", "vstack");
        left.append(digitCanvas(world.data.trainImages, batch[sample] * 784, 4), el("div", "caption", `the digit is a ${digit}`));
        row.append(left);
        const table = el("table", "num-table");
        table.innerHTML = `<thead><tr><th>question</th><th>logit</th><th>σ(logit)</th><th>≥ 0.5 ?</th><th>truth</th><th></th></tr></thead>`;
        const tb = el("tbody");
        for (let j = 0; j < 3; j++) {
          const lg = logits.at(sample, j);
          const p = 1 / (1 + Math.exp(-lg));
          const guess = p >= 0.5 ? 1 : 0;
          const ok = guess === truth[j];
          const tr = el("tr");
          tr.innerHTML = `<td>${MULTILABEL_NAMES[j]}</td><td>${fmt(lg)}</td><td>${fmt(p)}</td><td>${guess ? "yes" : "no"}</td><td>${truth[j] ? "yes" : "no"}</td><td class="${ok ? "pos" : "neg"}">${ok ? "✓" : "✗"}</td>`;
          tb.append(tr);
        }
        table.append(tb);
        row.append(table);
        root.append(row);
        root.append(
          el(
            "p",
            "explain",
            "The 0.5 threshold is OUR choice at prediction time, not the model's — a threshold is something you tune on validation data. Training never thresholds; it works with the smooth probabilities.",
          ),
        );
      });
      body.append(region);
      return cleanup;
    },
  });

  registerPanel("workshop.bce", {
    title: "BCE Bench — binary cross-entropy, term by term",
    subtitle: "loss = −[ y·log p + (1−y)·log(1−p) ], averaged over every label of every sample.",
    render(body, world) {
      let sample = 0;
      const [region, cleanup, refresh] = liveRegion(world.workshop, (root) => {
        const s = world.multi;
        const sig = s.nodes["sigmoid"];
        const logP = s.nodes["log p"];
        const log1mP = s.nodes["log(1-p)"];
        const per = s.nodes["per-item BCE"];
        const loss = s.nodes["loss"];
        if (!sig || !logP || !log1mP || !per || !loss) return;
        const batch = s.lastBatch;
        sample = Math.min(sample, batch.length - 1);
        const digit = world.data.trainLabels[batch[sample]];
        const truth = digitProperties(digit);
        root.append(
          picker("sample:", batch.length, sample, (v) => { sample = v; refresh(); },
            (i) => `#${i} (a “${world.data.trainLabels[batch[i]]}”)`),
        );
        const table = el("table", "num-table");
        table.innerHTML = `<thead><tr><th>label</th><th>y</th><th>p</th><th>log p</th><th>log(1−p)</th><th>−[y·log p + (1−y)·log(1−p)]</th></tr></thead>`;
        const tb = el("tbody");
        for (let j = 0; j < 3; j++) {
          const tr = el("tr");
          tr.innerHTML = `<td>${MULTILABEL_NAMES[j]}</td><td>${truth[j]}</td><td>${fmt(sig.at(sample, j))}</td><td>${fmt(logP.at(sample, j))}</td><td>${fmt(log1mP.at(sample, j))}</td><td class="${per.at(sample, j) < 0.3 ? "pos" : "neg"}">${fmt(per.at(sample, j))}</td>`;
          tb.append(tr);
        }
        table.append(tb);
        root.append(table);
        root.append(
          el("div", "bigmath", `mean over all ${per.size} label-answers in the batch = <b>${fmt(loss.item(), 5)}</b>`),
          el(
            "p",
            "explain",
            "Exactly the cross-entropy idea from the foundry, specialized to two outcomes: when y=1 only −log p matters; when y=0 only −log(1−p). One of the two log terms is always switched off by the y/(1−y) gate.",
          ),
        );
        root.append(el("div"));
      });
      body.append(region);
      return cleanup;
    },
  });

  // -------------------------------------------------------------- studio ---
  registerPanel("studio.train", {
    title: "Training Bench — regression model",
    subtitle: "Predict two continuous numbers (the ink's center) instead of a category — regression with the very same network body.",
    render(body, world) {
      const [controls, cleanup] = trainerControls(world.studio);
      body.append(controls);
      body.append(
        el(
          "p",
          "explain",
          "Same mills, same springs, same SGD — only two things changed: the head outputs 2 numbers squeezed into (0,1) by a sigmoid (since coordinates live in the image), and the loss is MSE: mean((pred − target)²). The metric above is the average miss distance in pixels.",
        ),
      );
      return cleanup;
    },
  });

  registerPanel("studio.preds", {
    title: "Crosshair Desk — where does the model think the ink is?",
    subtitle: "Green ✛ = true center of mass. Gold ✛ = the model's prediction. They converge as it trains.",
    render(body, world) {
      let pick = 0;
      const [region, cleanup, refresh] = liveRegion(world.studio, (root) => {
        const s = world.reg;
        const data = world.data;
        root.append(picker("test image:", 24, pick, (v) => { pick = v; refresh(); },
          (i) => `#${i} (a “${data.testLabels[i]}”)`));
        const idx = pick;
        // forward this single test image
        const xArr = new Float32Array(784);
        for (let j = 0; j < 784; j++) xArr[j] = data.testImages[idx * 784 + j] / 255;
        const p = s.predsFor(new Tensor(xArr, [1, 784]));
        const [tx, ty] = inkCenter(data.testImages, idx);
        const px = p.at(0, 0);
        const py = p.at(0, 1);

        const scale = 9;
        const canvas = digitCanvas(data.testImages, idx * 784, scale);
        const ctx = canvas.getContext("2d")!;
        const cross = (cx: number, cy: number, color: string) => {
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(cx * 27 * scale - 9, cy * 27 * scale);
          ctx.lineTo(cx * 27 * scale + 9, cy * 27 * scale);
          ctx.moveTo(cx * 27 * scale, cy * 27 * scale - 9);
          ctx.lineTo(cx * 27 * scale, cy * 27 * scale + 9);
          ctx.stroke();
        };
        cross(tx, ty, "#6ee87a");
        cross(px, py, "#ffd34d");
        const row = el("div", "hstack");
        row.append(canvas);
        const right = el("div", "vstack");
        const missPx = Math.hypot((px - tx) * 27, (py - ty) * 27);
        right.append(
          el("div", "bigmath", `target  y = (${fmt(tx, 4)}, ${fmt(ty, 4)})`),
          el("div", "bigmath", `predict ŷ = (${fmt(px, 4)}, ${fmt(py, 4)})`),
          el("div", "bigmath", `squared error = (${fmt(px - tx, 4)})² + (${fmt(py - ty, 4)})² = ${fmt((px - tx) ** 2 + (py - ty) ** 2, 6)}`),
          chip("miss distance", `${fmt(missPx, 2)} px`, true),
          el(
            "p",
            "explain",
            "MSE has no logs and no classes — error grows quadratically with distance, so big misses dominate the gradient. Train the studio and watch the gold crosshair glide onto the green one.",
          ),
        );
        row.append(right);
        root.append(row);
      });
      body.append(region);
      return cleanup;
    },
  });
}
