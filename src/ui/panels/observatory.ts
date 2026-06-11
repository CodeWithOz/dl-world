// Metrics Observatory: loss/accuracy curves and the live confusion matrix.

import { registerPanel } from "../panel";
import { chip, digitCanvas, el, fmt, lineChart, section } from "../widgets";
import { liveRegion, refreshable } from "./common";

export function registerObservatoryPanels(): void {
  registerPanel("observatory.curves", {
    title: "Curve Wall — is it learning?",
    subtitle: "Loss is what SGD minimizes; accuracy is what we actually care about (ch.4: metric ≠ loss).",
    render(body, world) {
      const [region, cleanup] = liveRegion(world.main, (root) => {
        const s = world.mlp;
        const chips = el("div", "chips-row");
        chips.append(
          chip("epoch", String(s.epoch)),
          chip("step", String(s.step)),
          chip("last batch loss", fmt(world.main.lastLossValue, 4), true),
          chip("test accuracy", `${(world.main.lastMetricValue * 100).toFixed(1)}%`, true),
          chip("lr", fmt(s.opt.lr, 3)),
        );
        root.append(chips);
        const s1 = section("Training loss, every step", "Noisy by nature — each point is a different random batch of 64. The trend is what matters.");
        const pts = s.lossHistory.map((p) => ({ x: p.step, y: p.value }));
        const sampled = pts.length > 600 ? pts.filter((_, i) => i % Math.ceil(pts.length / 600) === 0) : pts;
        s1.append(lineChart(sampled, { width: 860, height: 200, yLabel: "loss" }));
        root.append(s1);
        const s2 = section("Test accuracy, measured each epoch", "Computed on 600 held-out images the model never trains on — the honest score.");
        const mpts = s.metricHistory.map((p) => ({ x: p.step, y: p.value }));
        s2.append(lineChart(mpts.length ? mpts : [{ x: 0, y: 0 }], { width: 860, height: 180, color: "#ffd34d", yLabel: "accuracy" }));
        s2.append(
          el(
            "p",
            "explain",
            "Why train on loss but report accuracy? Accuracy barely moves when a weight nudges — its gradient is ~zero almost everywhere, so SGD would get no signal. Cross-entropy is a smooth stand-in that improves accuracy as a side effect (ch.4's key distinction between loss and metric).",
          ),
        );
        root.append(s2);
      });
      body.append(region);
      return cleanup;
    },
  });

  registerPanel("observatory.confusion", {
    title: "Confusion Matrix — where the model goes wrong",
    subtitle: "Rows = true digit, columns = model's guess, over all 600 test images. Recompute after training more.",
    render(body, world) {
      const wrap = refreshable((root) => {
        const { preds, correct } = world.mlp.predictTestAll();
        const data = world.data;
        const M = new Array(100).fill(0) as number[];
        for (let i = 0; i < data.nTest; i++) M[data.testLabels[i] * 10 + preds[i]]++;
        const chips = el("div", "chips-row");
        chips.append(
          chip("test accuracy now", `${((correct / data.nTest) * 100).toFixed(1)}%`, true),
          chip("errors", `${data.nTest - correct} / ${data.nTest}`),
        );
        root.append(chips);

        // matrix with counts
        const cell = 40;
        const canvas = document.createElement("canvas");
        canvas.width = (cell * 11 + 10) * 2;
        canvas.height = (cell * 11 + 10) * 2;
        canvas.style.width = `${cell * 11 + 10}px`;
        canvas.style.height = `${cell * 11 + 10}px`;
        const ctx = canvas.getContext("2d")!;
        ctx.scale(2, 2);
        ctx.font = "12px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const maxOff = Math.max(1, ...M.filter((_, i) => Math.floor(i / 10) !== i % 10));
        for (let r = 0; r < 10; r++) {
          ctx.fillStyle = "rgba(255,255,255,0.7)";
          ctx.fillText(String(r), cell / 2, cell * (r + 1) + cell / 2);
          ctx.fillText(String(r), cell * (r + 1) + cell / 2, cell / 2);
          for (let c = 0; c < 10; c++) {
            const v = M[r * 10 + c];
            const diag = r === c;
            const intensity = diag ? v / 60 : v / maxOff;
            ctx.fillStyle = diag
              ? `rgba(90, 209, 200, ${0.15 + 0.85 * Math.min(intensity, 1)})`
              : `rgba(232, 128, 122, ${v === 0 ? 0.06 : 0.25 + 0.75 * Math.min(intensity, 1)})`;
            ctx.fillRect(cell * (c + 1) + 1, cell * (r + 1) + 1, cell - 2, cell - 2);
            ctx.fillStyle = v > 0 ? "#fff" : "rgba(255,255,255,0.25)";
            ctx.fillText(String(v), cell * (c + 1) + cell / 2, cell * (r + 1) + cell / 2);
          }
        }
        const s1 = section("The matrix", "Teal diagonal = correct. Red off-diagonal = confusions (e.g. 4↔9, 3↔5 are classic).");
        s1.append(canvas);
        root.append(s1);

        const s2 = section("The actual mistakes (first 14)", "fastai's plot_top_losses in spirit — always look at the data:");
        const strip = el("div", "digit-grid");
        let shown = 0;
        for (let i = 0; i < data.nTest && shown < 14; i++) {
          if (preds[i] === data.testLabels[i]) continue;
          const cellEl = el("div", "digit-cell");
          cellEl.append(digitCanvas(data.testImages, i * 784, 1.8));
          cellEl.append(el("div", "digit-label neg", `${data.testLabels[i]}→${preds[i]}`));
          strip.append(cellEl);
          shown++;
        }
        if (!shown) strip.append(el("p", "explain", "No errors on the test set — remarkable!"));
        s2.append(strip);
        root.append(s2);
      });
      body.append(wrap);
    },
  });
}
