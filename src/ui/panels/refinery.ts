// Refinement Gym: training tricks (normalization, label smoothing, mixup)
// applied to a digit classifier that races an identically-initialized plain
// twin on the very same batches — a live controlled experiment.

import { registerPanel } from "../panel";
import { chip, digitCanvas, el, fmt, lineChart, numberStrip, section } from "../widgets";
import { liveRegion, refreshable, trainerControls } from "./common";
import { World } from "../../sim/world";
import { Refinery } from "../../sim/scenarios2";

function r(world: World): Refinery {
  return world.refine;
}

/** the tricks switchboard — shared by the gym's stations; state lives on the
 *  scenario itself so every station (and the twin race) stays in sync */
function trickSwitches(world: World, show: ("norm" | "smooth" | "mixup")[]): HTMLElement {
  const s = r(world);
  const row = el("div", "controls-row");
  const checkbox = (label: string, get: () => boolean, set: (v: boolean) => void) => {
    const wrap = el("label", "control-label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = get();
    box.addEventListener("change", () => set(box.checked));
    wrap.append(box, document.createTextNode(` ${label}`));
    return wrap;
  };
  if (show.includes("norm"))
    row.append(checkbox("normalize inputs", () => s.useNorm, (v) => (s.useNorm = v)));
  if (show.includes("smooth")) {
    const wrap = el("label", "control-label", "label smoothing ε ");
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "0.2";
    slider.step = "0.01";
    slider.value = String(s.smoothEps);
    const val = el("span", "mono", fmt(s.smoothEps, 2));
    slider.addEventListener("input", () => {
      s.smoothEps = parseFloat(slider.value);
      val.textContent = fmt(s.smoothEps, 2);
    });
    wrap.append(slider, val);
    row.append(wrap);
  }
  if (show.includes("mixup"))
    row.append(checkbox("mixup", () => s.useMixup, (v) => (s.useMixup = v)));
  return row;
}

export function registerRefineryPanels(): void {
  registerPanel("gym.bench", {
    title: "Twin Race Bench — do the tricks actually help?",
    subtitle:
      "Two identical networks were born from the same random init. The refined one gets your chosen tricks; the plain twin trains on the exact same batches with none. Everything below is the live score of that race.",
    render(body, world) {
      const s = r(world);
      body.append(trickSwitches(world, ["norm", "smooth", "mixup"]));
      const [controls, cleanCtl] = trainerControls(world.refinery);
      body.append(controls);
      const [live, cleanLive] = liveRegion(world.refinery, (root) => {
        const chips = el("div", "chips-row");
        chips.append(
          chip("refined test acc", `${(s.accuracy(true) * 100).toFixed(1)}%`, true),
          chip("plain twin test acc", `${(s.accuracy(false) * 100).toFixed(1)}%`),
          chip("refined loss", fmt(s.lossHistory.at(-1)?.value ?? NaN, 4)),
          chip("plain loss", fmt(s.lastPlainLoss, 4)),
        );
        root.append(chips);
        const sample = (pts: { step: number; value: number }[]) => {
          const mapped = pts.map((p) => ({ x: p.step, y: p.value }));
          return mapped.length > 300
            ? mapped.filter((_, i) => i % Math.ceil(mapped.length / 300) === 0)
            : mapped;
        };
        root.append(el("div", "caption", "training loss — refined (teal) vs plain twin (gold):"));
        root.append(
          lineChart(sample(s.lossHistory), {
            width: 520,
            height: 150,
            series2: sample(s.plainLossHistory),
          }),
        );
        if (s.metricHistory.length > 1) {
          root.append(el("div", "caption", "test accuracy per epoch — refined (teal) vs plain (gold):"));
          root.append(
            lineChart(sample(s.metricHistory), {
              width: 520,
              height: 150,
              series2: sample(s.plainMetricHistory),
            }),
          );
        }
      }, 600);
      body.append(live);
      body.append(
        el(
          "p",
          "explain",
          "A note on honesty: with the label-smoothing loss the refined model's loss value is not directly comparable to the twin's plain cross-entropy (smoothing adds a constant floor — even a perfect model can't reach 0). The accuracy curves are the fair comparison.",
        ),
      );
      return () => {
        cleanCtl();
        cleanLive();
      };
    },
  });

  registerPanel("gym.norm", {
    title: "Normalization Bar — put every pixel on the same scale",
    subtitle:
      "Raw pixels live in 0..1 with a mean far from zero. Normalizing to mean 0 / std 1 makes the loss surface rounder, so the same learning rate behaves better in every direction.",
    render(body, world) {
      const s = r(world);
      const s1 = section("The real statistics of the 3,000 training digits");
      s1.append(
        el(
          "div",
          "bigmath",
          `x_norm = (x − <b>${fmt(s.pixMean, 4)}</b>) / <b>${fmt(s.pixStd, 4)}</b>`,
        ),
        el(
          "p",
          "explain",
          "Those two numbers were computed once over every pixel of the training set — exactly what a Normalize() transform stores. The test set is normalized with the training statistics, never its own — and if you ever ship a model, you must ship these two numbers with it, or everyone running inference feeds it data on the wrong scale.",
        ),
      );
      body.append(s1);
      body.append(trickSwitches(world, ["norm"]));
      const [live, cleanLive] = liveRegion(world.refinery, (root) => {
        const node = s.nodes["x (normalized)"] ?? s.nodes["x (raw pixels)"];
        if (!node) return;
        const isNorm = !!s.nodes["x (normalized)"];
        let mn = Infinity,
          mx = -Infinity,
          sum = 0;
        for (let i = 0; i < node.size; i++) {
          mn = Math.min(mn, node.data[i]);
          mx = Math.max(mx, node.data[i]);
          sum += node.data[i];
        }
        const mean = sum / node.size;
        let q = 0;
        for (let i = 0; i < node.size; i++) q += (node.data[i] - mean) ** 2;
        const chips = el("div", "chips-row");
        chips.append(
          chip("current batch input", isNorm ? "normalized" : "raw", true),
          chip("batch mean", fmt(mean, 3)),
          chip("batch std", fmt(Math.sqrt(q / node.size), 3)),
          chip("min", fmt(mn, 3)),
          chip("max", fmt(mx, 3)),
        );
        root.append(chips);
        root.append(
          el(
            "div",
            "caption",
            isNorm
              ? "mean ≈ 0, std ≈ 1: the ink is positive, the background is a small negative number instead of exactly zero."
              : "raw: mean is far from 0 because most pixels are background. Flip the switch above and watch these chips move.",
          ),
        );
      });
      body.append(live);
      return cleanLive;
    },
  });

  registerPanel("gym.smooth", {
    title: "Label-Smoothing Bench — stop demanding 100% certainty",
    subtitle:
      "Plain cross-entropy asks the model to push the target probability to exactly 1, which means logits racing to infinity and overconfidence on mistakes. Smoothing redistributes a little ε of the target onto every class.",
    render(body, world) {
      const s = r(world);
      body.append(trickSwitches(world, ["smooth"]));
      const [live, cleanLive] = liveRegion(world.refinery, (root) => {
        const eps = s.smoothEps;
        const target = s.lastBatch.length > 0 ? s.data.trainLabels[s.lastBatch[0]] : 0;
        const sec = section(`The target vector for the batch's first image (a real ${target})`);
        const vec = [...Array(10)].map((_, c) => (c === target ? 1 - eps + eps / 10 : eps / 10));
        const row = el("div", "hstack wrap-row");
        row.append(numberStrip(Float32Array.from(vec), target));
        const expl = el("div", "vstack");
        expl.append(
          el(
            "div",
            "bigmath",
            `q[target] = 1 − ε + ε/10 = <b>${fmt(1 - eps + eps / 10, 3)}</b><br>q[other] = ε/10 = <b>${fmt(eps / 10, 3)}</b>`,
          ),
        );
        const nll = s.nodes["nll"];
        const smoothed = s.nodes["smoothed loss"];
        const chips = el("div", "chips-row");
        if (nll) chips.append(chip("plain NLL on this batch", fmt(nll.item(), 4)));
        if (smoothed) chips.append(chip("smoothed loss on this batch", fmt(smoothed.item(), 4), true));
        if (!smoothed) chips.append(chip("smoothing", "off (ε = 0)"));
        expl.append(chips);
        row.append(expl);
        sec.append(row);
        root.append(sec);
        root.append(
          el(
            "p",
            "explain",
            "Both numbers are read from the live graph of the most recent step: the smoothed loss is (1−ε)·NLL + (ε/10)·Σ(−log p) over all ten classes, composed from the same softmax→log nodes the Foundry uses.",
          ),
        );
      });
      body.append(live);
      return cleanLive;
    },
  });

  registerPanel("gym.mixup", {
    title: "Mixup Blender — train on images that don't exist",
    subtitle:
      "Mixup blends two training images (and their labels) with a random λ. The model must stay calibrated between classes instead of memorizing islands — a strong regularizer on small data.",
    render(body, world) {
      const s = r(world);
      body.append(trickSwitches(world, ["mixup"]));
      const [live, cleanLive] = liveRegion(world.refinery, (root) => {
        if (!s.useMixup || !s.nodes["mixed input"]) {
          root.append(
            el(
              "p",
              "explain",
              "Mixup is off. Flip the switch above, take a training step, and this station will show the actual blended image the network just trained on.",
            ),
          );
          return;
        }
        const lam = s.lastMixLam;
        const mixed = s.nodes["mixed input"];
        const i1 = s.lastBatch[0];
        const i2 = s.lastBatch[s.lastBatch.length - 1];
        const y1 = s.data.trainLabels[i1];
        const y2 = s.data.trainLabels[i2];
        const sec = section("The first sample of the live batch, blended for real");
        const row = el("div", "hstack wrap-row");
        const cell = (canvas: HTMLCanvasElement, label: string) => {
          const v = el("div", "vstack");
          v.append(canvas, el("div", "caption", label));
          return v;
        };
        row.append(cell(digitCanvas(s.data.trainImages, i1 * 784, 3), `image A (a ${y1}) × λ=${fmt(lam, 2)}`));
        row.append(el("div", "flow-arrow", "+"));
        row.append(cell(digitCanvas(s.data.trainImages, i2 * 784, 3), `image B (a ${y2}) × ${fmt(1 - lam, 2)}`));
        row.append(el("div", "flow-arrow", "="));
        // the mixed row may be normalized — rescale to 0..1 just for display
        const view = new Float32Array(784);
        let mn = Infinity,
          mx = -Infinity;
        for (let j = 0; j < 784; j++) {
          mn = Math.min(mn, mixed.data[j]);
          mx = Math.max(mx, mixed.data[j]);
        }
        for (let j = 0; j < 784; j++) view[j] = (mixed.data[j] - mn) / (mx - mn || 1);
        row.append(cell(digitCanvas(view, 0, 3), "what the network saw"));
        sec.append(row);
        root.append(sec);
        root.append(
          el(
            "div",
            "bigmath",
            `loss = λ·CE(logits, ${y1}) + (1−λ)·CE(logits, ${y2})  with λ = <b>${fmt(lam, 3)}</b>`,
          ),
          el(
            "p",
            "explain",
            "The blended pixels above are read straight out of the recorded graph node the last training step actually used (un-normalized only for display). A new λ is drawn every step.",
          ),
        );
      });
      body.append(live);
      return cleanLive;
    },
  });

  registerPanel("gym.tta", {
    title: "TTA Mirror Hall — better answers, zero extra training",
    subtitle:
      "Test-time augmentation: show the model several augmented versions of each test image and average the predicted probabilities. The book uses the original plus four augmented views; here the views are the digit nudged one pixel in each direction.",
    render(body, world) {
      const s = r(world);
      body.append(
        refreshable((root) => {
          const t0 = performance.now();
          const res = s.ttaAccuracy();
          const ms = performance.now() - t0;
          const chips = el("div", "chips-row");
          chips.append(
            chip("plain test accuracy", `${(res.plain * 100).toFixed(2)}%`),
            chip(`TTA over ${res.views} views`, `${(res.tta * 100).toFixed(2)}%`, true),
            chip("extra training needed", "none"),
            chip("inference cost", `${res.views}× (${ms.toFixed(0)}ms for all 600)`),
          );
          root.append(chips);
          root.append(
            el(
              "p",
              "explain",
              "Both numbers come from running all 600 held-out digits through the live weights just now — once plainly, once as the average over five views. TTA usually rescues a handful of borderline digits whose ink sits slightly off-center; the price is paid at inference time, not training time. (The book's other sizing trick, progressive resizing, needs variable-size images — a 28×28 digit has nowhere to grow, so it has no honest demo here.)",
            ),
          );
        }, "↻ re-run both evaluations"),
      );
    },
  });
}
