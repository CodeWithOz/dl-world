// LR Finder Tower: sweep the learning rate on a fresh model until the loss
// explodes, then read the good zone off the curve.

import { runLrFinder } from "../../sim/lrfinder";
import { registerPanel } from "../panel";
import { button, chip, el, fmt, lineChart, section } from "../widgets";

export function registerTowerPanels(): void {
  registerPanel("tower.sweep", {
    title: "The Sweep Console — learning rate finder",
    subtitle:
      "lr_find(): start a FRESH model at lr=0.0001 and multiply the lr by 1.18 every step until the loss blows up.",
    render(body, world) {
      const wrap = el("div");
      const result = el("div");
      let sweeping = false;

      const display = (r: NonNullable<typeof world.lrResult>) => {
        result.innerHTML = "";
        const s = section(
          "Loss vs learning rate (log scale)",
          "Teal = smoothed loss, gold = raw per-batch loss. Left: too small, nothing happens. Middle: the productive slope. Right: divergence.",
        );
        // cap the divergence spike so the productive region stays readable
        const cap = Math.max(...r.points.slice(0, 5).map((p) => p.smoothed)) * 1.6;
        s.append(
          lineChart(
            r.points.map((p) => ({ x: p.lr, y: Math.min(p.smoothed, cap) })),
            {
              width: 860,
              height: 260,
              logX: true,
              yLabel: "loss (capped for display)",
              series2: r.points.map((p) => ({ x: p.lr, y: Math.min(p.loss, cap) })),
              markers: [
                { x: r.suggestedSteepest, label: "steepest", color: "#5ad1c8" },
                { x: r.suggestedMinOver10, label: "min/10", color: "#e8807a" },
              ],
            },
          ),
        );
        const chips = el("div", "chips-row");
        chips.append(
          chip("steepest-slope suggestion", fmt(r.suggestedSteepest, 4), true),
          chip("min-loss ÷ 10 suggestion", fmt(r.suggestedMinOver10, 4), true),
          chip("steps until explosion", String(r.points.length)),
          chip("main pipeline currently uses", fmt(world.mlp.opt.lr, 3)),
        );
        s.append(chips);
        s.append(
          el(
            "p",
            "explain",
            "Each sweep trains a brand-new model for ~60 mini-batches, so suggestions vary slightly run to run. Take a suggestion to the Optimizer Depot, set it on the live lr dial, and watch the Observatory to judge it yourself.",
          ),
        );
        result.append(s);
      };

      const runBtn = button("🎚 run a fresh sweep", () => {
        if (sweeping) return; // ignore double clicks while a sweep is in flight
        sweeping = true;
        runBtn.disabled = true;
        result.innerHTML = "";
        result.append(el("p", "explain", "sweeping…"));
        // let the browser paint before the synchronous sweep
        setTimeout(() => {
          try {
            const r = runLrFinder(world.data);
            world.lrResult = r;
            display(r);
          } finally {
            sweeping = false;
            runBtn.disabled = false;
          }
        }, 30);
      }, "btn-play");

      wrap.append(runBtn, result);
      body.append(wrap);
      // show the cached sweep instantly; the button forces a fresh one
      if (world.lrResult) display(world.lrResult);
    },
  });
}
