// Chapter 4 Quarter: the Pixel Similarity Museum (the pre-learning baseline)
// and the Linear Cottage (the first real learner, 3 vs 7).

import { imageAsFloats } from "../../engine/data";
import { distL1, distL2, pixelSimilarityBaseline } from "../../sim/baseline";
import { registerPanel } from "../panel";
import { chip, digitCanvas, el, fmt, heatmap, section } from "../widgets";
import { liveRegion, picker, trainerControls } from "./common";

export function registerCh4Panels(): void {
  registerPanel("museum.means", {
    title: "Hall of Averages — the 'ideal' 3 and 7",
    subtitle: "Before any learning: stack every training 3 (and 7) and average the pixels. Ch.4's first classifier.",
    render(body, world) {
      const r = pixelSimilarityBaseline(world.data);
      const s = section("The exhibits");
      const row = el("div", "hstack");
      const left = el("div", "vstack");
      left.append(digitCanvas(r.mean3, 0, 6), el("div", "caption", "mean of all training 3s"));
      const right = el("div", "vstack");
      right.append(digitCanvas(r.mean7, 0, 6), el("div", "caption", "mean of all training 7s"));
      row.append(left, right);
      s.append(row);
      s.append(
        el(
          "p",
          "explain",
          "Blurry, ghostly — an average over many handwriting styles. To classify a new image: measure which ghost it is closer to, pixel by pixel. No parameters, no gradients, no learning.",
        ),
      );
      body.append(s);
      const chips = el("div", "chips-row");
      chips.append(
        chip("L1 (mean |diff|) accuracy", `${(r.accuracyL1 * 100).toFixed(1)}%`, true),
        chip("L2 (RMSE) accuracy", `${(r.accuracyL2 * 100).toFixed(1)}%`, true),
      );
      body.append(chips);
      body.append(
        el(
          "p",
          "explain",
          "Surprisingly strong! This is the baseline the Linear Cottage next door must beat — fastai's advice: always start with a dumb baseline so you know your fancy model is actually earning its keep.",
        ),
      );
    },
  });

  registerPanel("museum.classify", {
    title: "Distance Desk — classify by hand",
    subtitle: "Pick a test image and measure its distance to each ideal digit.",
    render(body, world) {
      const data = world.data;
      const candidates: number[] = [];
      for (let i = 0; i < data.nTest; i++)
        if (data.testLabels[i] === 3 || data.testLabels[i] === 7) candidates.push(i);
      let pick = 0;
      const wrap = el("div");
      const render = () => {
        wrap.innerHTML = "";
        const r = pixelSimilarityBaseline(data);
        const idx = candidates[pick];
        const img = imageAsFloats(data.testImages, idx);
        const d3L1 = distL1(img, r.mean3);
        const d7L1 = distL1(img, r.mean7);
        const d3L2 = distL2(img, r.mean3);
        const d7L2 = distL2(img, r.mean7);
        const verdict = d3L2 < d7L2 ? 3 : 7;
        const truth = data.testLabels[idx];
        wrap.append(
          picker("test image:", Math.min(candidates.length, 60), pick, (v) => { pick = v; render(); },
            (i) => `#${i} (a “${data.testLabels[candidates[i]]}”)`),
        );
        const row = el("div", "hstack");
        const left = el("div", "vstack");
        left.append(digitCanvas(data.testImages, idx * 784, 5), el("div", "caption", `true label: ${truth}`));
        row.append(left);
        const table = el("table", "num-table");
        table.innerHTML = `<thead><tr><th></th><th>distance to mean-3</th><th>distance to mean-7</th></tr></thead>
        <tbody>
          <tr><td>L1 = mean |pixel diff|</td><td class="${d3L1 < d7L1 ? "pos" : ""}">${fmt(d3L1, 4)}</td><td class="${d7L1 < d3L1 ? "pos" : ""}">${fmt(d7L1, 4)}</td></tr>
          <tr><td>L2 = √mean(diff²)</td><td class="${d3L2 < d7L2 ? "pos" : ""}">${fmt(d3L2, 4)}</td><td class="${d7L2 < d3L2 ? "pos" : ""}">${fmt(d7L2, 4)}</td></tr>
        </tbody>`;
        row.append(table);
        wrap.append(row);
        wrap.append(
          el("div", "bigmath", `verdict: closer to the ideal <b>${verdict}</b> → ${verdict === truth ? '<span class="pos">correct ✓</span>' : '<span class="neg">wrong ✗</span>'}`),
          el("p", "explain", "L1 vs L2 (ch.4): L2 punishes big single-pixel differences harder; L1 treats all differences evenly. Both are legitimate 'distances' — choosing one is a modeling decision."),
        );
      };
      render();
      body.append(wrap);
    },
  });

  registerPanel("cottage.train", {
    title: "Training Bench — the 3-vs-7 linear learner",
    subtitle: "784 weights + 1 bias, sigmoid, mnist_loss, SGD — chapter 4's complete learner, training live in your browser.",
    render(body, world) {
      const [controls, cleanup] = trainerControls(world.cottage);
      body.append(controls);
      body.append(
        el(
          "p",
          "explain",
          "The seven steps of ch.4 are all here: init → predict → loss → gradient → step → repeat → stop. Open The Learned Eye next to this bench while it trains and watch the weights become a picture. The pixel museum's baseline was ~94% — can this learner beat it?",
        ),
      );
      return cleanup;
    },
  });

  registerPanel("cottage.eye", {
    title: "The Learned Eye — w reshaped to 28×28",
    subtitle: "One weight per pixel. Red pixels vote 'it's a 3', blue pixels vote 'it's a 7'.",
    render(body, world) {
      const [region, cleanup] = liveRegion(world.cottage, (root) => {
        const w = world.lin37.w;
        const b = world.lin37.b;
        const s = section("The weights, live", "This image IS the model. Watch it sharpen as SGD runs:");
        s.append(heatmap(w.data, 28, 28, { cellSize: 11, symmetric: true }));
        const chips = el("div", "chips-row");
        chips.append(
          chip("bias b", fmt(b.data[0], 4)),
          chip("parameters", "785"),
          chip("score", "x · w + b  (positive ⇒ 3, negative ⇒ 7)"),
        );
        s.append(chips);
        s.append(
          el(
            "p",
            "explain",
            "Why does it look like that? A 3's middle-left region has ink where a 7 doesn't (red there), while a 7's diagonal stroke region turns blue. The model literally learned where 3s and 7s disagree — no one told it; the gradients did.",
          ),
        );
        root.append(s);
        if (w.grad) {
          const s2 = section("Current gradient on w (also 28×28)", "Where the next SGD step wants to push each pixel-weight:");
          s2.append(heatmap(w.grad, 28, 28, { cellSize: 7, symmetric: true }));
          root.append(s2);
        }
      });
      body.append(region);
      return cleanup;
    },
  });

  registerPanel("cottage.loss", {
    title: "mnist_loss Corner — where(y==1, 1−p, p)",
    subtitle: "Chapter 4's hand-rolled loss, computed live on the cottage's current batch.",
    render(body, world) {
      const [region, cleanup] = liveRegion(world.cottage, (root) => {
        const s = world.lin37;
        const preds = s.nodes["preds = x·w + b"];
        const sig = s.nodes["sigmoid preds"];
        const per = s.nodes["per-item loss"];
        const y = s.nodes["y (1 if 3)"];
        const loss = s.nodes["loss"];
        if (!preds || !sig || !per || !y || !loss) return;
        const batchIdx = s.resolve(s.lastBatch);

        const s1 = section(
          "The assembly, sample by sample (first 8)",
          "p = σ(score) is 'how much the model believes it's a 3'. If the truth is 3 (y=1) the loss is 1−p; if 7 (y=0) the loss is p. Either way: smaller = better.",
        );
        const table = el("table", "num-table");
        table.innerHTML = `<thead><tr><th>image</th><th>y</th><th>score x·w+b</th><th>p = σ(score)</th><th>where(y==1, 1−p, p)</th></tr></thead>`;
        const tb = el("tbody");
        for (let i = 0; i < Math.min(8, preds.rows); i++) {
          const tr = el("tr");
          const td = el("td");
          td.append(digitCanvas(world.data.trainImages, batchIdx[i] * 784, 1.3));
          tr.append(td);
          tr.innerHTML += `<td>${y.at(i, 0)}</td><td>${fmt(preds.at(i, 0))}</td><td>${fmt(sig.at(i, 0))}</td><td class="${per.at(i, 0) < 0.2 ? "pos" : "neg"}">${fmt(per.at(i, 0))}</td>`;
          tb.append(tr);
        }
        table.append(tb);
        s1.append(table);
        s1.append(el("div", "bigmath", `batch mean = <b>${fmt(loss.item(), 5)}</b> ← this is what backward() starts from`));
        root.append(s1);

        const s2 = section(
          "Why not just use accuracy as the loss?",
          "Accuracy only changes when a prediction crosses 0.5 — nudge a weight slightly and accuracy stays identical, so its gradient is zero almost everywhere and SGD gets no signal. mnist_loss rewards every tiny improvement in confidence, giving a usable slope. (Ch.4's loss-vs-metric lesson, the same reason the foundry uses cross-entropy.)",
        );
        root.append(s2);
      });
      body.append(region);
      return cleanup;
    },
  });
}
