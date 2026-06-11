// Optimizer Depot: the SGD update itself, and a 1-D slice of the loss
// landscape to show what "downhill" actually means.

import { registerPanel } from "../panel";
import { chip, el, fmt, lineChart, section, slider } from "../widgets";
import { liveRegion, picker } from "./common";

export function registerOptimPanels(): void {
  registerPanel("optim.sgd", {
    title: "The Update Floor — w ← w − lr · gradient",
    subtitle: "The entire learning rule. Everything else in the city exists to compute the `gradient` in this line.",
    render(body, world) {
      let paramIdx = 0;
      const lrRow = el("div", "controls-row");
      const lrVal = el("span", "mono", fmt(world.mlp.opt.lr, 3));
      const lrLabel = el("label", "control-label", "learning rate (live — try breaking it) ");
      lrLabel.append(
        slider(-3, 0.9, 0.05, Math.log10(world.mlp.opt.lr), (v) => {
          world.mlp.opt.lr = Math.pow(10, v);
          lrVal.textContent = fmt(world.mlp.opt.lr, 3);
        }),
        lrVal,
      );
      lrRow.append(lrLabel);
      body.append(lrRow);

      const [region, cleanup, refresh] = liveRegion(world.main, (root) => {
        const opt = world.mlp.opt;
        const last = opt.lastStep;
        root.append(picker("watch parameter:", world.mlp.params.length, paramIdx, (v) => { paramIdx = v; refresh(); },
          (i) => `${world.mlp.params[i].name} [${world.mlp.params[i].tensor.shape.join("×")}]`));
        const entry = last.find((e) => e.param === world.mlp.params[paramIdx].tensor);
        const s = section(
          "The most recent update, element by element",
          "Eight sample elements of this tensor. `before` is the value going into the step, `after` is what the mills are using right now:",
        );
        if (!entry) {
          s.append(el("p", "explain", "Take a training step to see an update."));
        } else {
          const table = el("table", "num-table");
          table.innerHTML = `<thead><tr><th>element</th><th>w before</th><th>gradient g</th><th>− lr·g</th><th>w after</th></tr></thead>`;
          const tb = el("tbody");
          const N = Math.min(entry.before.length, entry.grad.length);
          const stride = Math.max(1, Math.floor(N / 8));
          for (let k = 0; k < 8 && k * stride < N; k++) {
            const i = k * stride;
            const w0 = entry.before[i];
            const g = entry.grad[i];
            const delta = -opt.lr * g;
            const tr = el("tr");
            tr.innerHTML = `<td>[${i}]</td><td>${fmt(w0, 6)}</td><td>${fmt(g, 6)}</td><td class="${delta >= 0 ? "pos" : "neg"}">${fmt(delta, 6)}</td><td><b>${fmt(w0 + delta, 6)}</b></td>`;
            tb.append(tr);
          }
          table.append(tb);
          s.append(table);
        }
        const chips = el("div", "chips-row");
        chips.append(
          chip("lr right now", fmt(opt.lr, 4), true),
          chip("parameters updated per step", "50,890"),
          chip("rule", "w -= lr * w.grad"),
        );
        s.append(chips);
        s.append(
          el(
            "p",
            "explain",
            "Push the learning rate to ~5 and watch the Observatory: the loss will bounce or explode because each step overshoots the valley. Drop it to 0.001 and learning slows to a crawl. The LR Finder Tower automates finding the sweet spot.",
          ),
        );
        root.append(s);
      });
      body.append(region);
      return cleanup;
    },
  });

  registerPanel("optim.landscape", {
    title: "Loss Slice Viewer — what 'downhill' means",
    subtitle:
      "Hold every parameter fixed except ONE weight, sweep it left and right, and recompute the real loss on the current batch at each point. The gradient is exactly the slope of this curve at the dot.",
    render(body, world) {
      const m = world.mlp;
      let paramIdx = 2;
      let elemIdx = 11;
      const wrap = el("div");
      const controls = el("div", "hstack");
      const result = el("div");

      const run = () => {
        result.innerHTML = "";
        const p = m.params[paramIdx].tensor;
        const name = m.params[paramIdx].name;
        const i = Math.min(elemIdx, p.size - 1);
        const orig = p.data[i];
        // fresh gradient at the current point
        m.opt.zeroGrad();
        const lossT = m.buildLoss(m.lastBatch, false);
        lossT.backward();
        const grad = p.grad![i];
        m.opt.zeroGrad();
        const range = Math.max(0.6, Math.abs(orig) * 3);
        const pts: { x: number; y: number }[] = [];
        for (let k = 0; k <= 40; k++) {
          const w = orig - range + (2 * range * k) / 40;
          p.data[i] = w;
          pts.push({ x: w, y: m.lossOnLastBatch() });
        }
        p.data[i] = orig;
        const here = m.lossOnLastBatch();

        const s = section(`Sweeping ${name}[${i}] — every point is a full forward pass (41 of them)`);
        s.append(
          lineChart(pts, {
            width: 640,
            height: 230,
            yLabel: "loss on current batch",
            markers: [{ x: orig, label: "current w", color: "#ffd34d" }],
          }),
        );
        const goes = grad > 0 ? "left (smaller w)" : "right (larger w)";
        s.append(
          el("div", "bigmath", `loss here = ${fmt(here, 5)} · ∂loss/∂w = ${fmt(grad, 6)} → downhill is to the ${goes}`),
          el("div", "bigmath", `SGD will move w by −lr·g = −${fmt(m.opt.lr, 3)} × ${fmt(grad, 5)} = ${fmt(-m.opt.lr * grad, 6)}`),
          el(
            "p",
            "explain",
            "Training never sees this curve — computing it took 41 forward passes for ONE weight. The miracle of backprop: one backward pass reveals the slope of all 50,890 such curves simultaneously.",
          ),
        );
        result.append(s);
      };

      controls.append(
        picker("parameter:", m.params.length, paramIdx, (v) => { paramIdx = v; run(); }, (k) => `${m.params[k].name} [${m.params[k].tensor.shape.join("×")}]`),
        picker("element #:", 50, elemIdx, (v) => { elemIdx = v * 23; run(); }, (k) => String(k * 23)),
      );
      const note = el("p", "explain", "Sweeps are computed on demand — change a selector (or train a bit, then reselect) to recompute.");
      wrap.append(controls, note, result);
      run();
      body.append(wrap);
    },
  });
}
