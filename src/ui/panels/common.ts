// Shared plumbing for panels: live regions that re-render on training steps,
// and the reusable training-control bench.

import { Trainer } from "../../sim/trainer";
import { button, chip, el, fmt, lineChart, slider } from "../widgets";

/**
 * Re-runs `render` into a container on every trainer step (throttled).
 * Returns [root, cleanup]. Interactive state should live in the caller's
 * closure so it survives re-renders.
 */
export function liveRegion(
  trainers: Trainer | Trainer[],
  render: (root: HTMLElement) => void,
  intervalMs = 300,
): [HTMLElement, () => void, () => void] {
  const root = el("div", "live-region");
  const list = Array.isArray(trainers) ? trainers : [trainers];
  let dirty = false;
  let last = 0;
  const doRender = () => {
    root.innerHTML = "";
    render(root);
  };
  doRender();
  const offs = list.map((t) =>
    t.on("step", () => {
      dirty = true;
    }),
  );
  // while the user is interacting with a form control inside this region
  // (e.g. an open <select>), rebuilding the DOM would yank it shut — hold
  // the update until focus leaves; pickers call the refresh themselves
  const userIsInteracting = () => {
    const a = document.activeElement;
    return (
      !!a && root.contains(a) && ["SELECT", "INPUT", "TEXTAREA"].includes(a.tagName)
    );
  };
  const timer = window.setInterval(() => {
    if (!dirty || userIsInteracting()) return;
    const now = performance.now();
    if (now - last < intervalMs) return;
    last = now;
    dirty = false;
    doRender();
  }, 100);
  return [
    root,
    () => {
      offs.forEach((off) => off());
      window.clearInterval(timer);
    },
    doRender,
  ];
}

/** manual refresh wrapper for panels that are expensive to re-render */
export function refreshable(
  render: (root: HTMLElement) => void,
  label = "↻ recompute",
): HTMLElement {
  const wrap = el("div");
  const root = el("div");
  const btn = button(label, () => {
    root.innerHTML = "";
    render(root);
  }, "btn-small");
  wrap.append(btn, root);
  render(root);
  return wrap;
}

/** format a metric by its name: accuracies as %, pixel errors as px,
 *  anything else (RMSE in stars, …) as a plain number */
export function fmtMetric(name: string, value: number): string {
  if (name.includes("accuracy")) return `${(value * 100).toFixed(1)}%`;
  if (name.includes("px")) return `${fmt(value, 2)}px`;
  return fmt(value, 3);
}

/** play / pause / step / speed / lr — the training bench used by several buildings */
export function trainerControls(trainer: Trainer, opts: { showLr?: boolean } = {}): [HTMLElement, () => void] {
  const wrap = el("div", "trainer-controls");
  const row = el("div", "controls-row");
  // ︎ keeps the glyphs as text on iOS (no emoji presentation)
  const playBtn = button(trainer.running ? "⏸︎ pause" : "▶︎ train", () => trainer.toggle(), "btn-play");
  const stepBtn = button("⏭︎ one step", () => trainer.stepOnce());
  row.append(playBtn, stepBtn);

  const speedWrap = el("label", "control-label", "speed ");
  const speedVal = el("span", "mono", `${trainer.speed}/s`);
  speedWrap.append(
    slider(1, 60, 1, trainer.speed, (v) => {
      trainer.speed = v;
      speedVal.textContent = `${v}/s`;
    }),
    speedVal,
  );
  row.append(speedWrap);

  if (opts.showLr !== false) {
    const lrWrap = el("label", "control-label", "learning rate ");
    const lrVal = el("span", "mono", fmt(trainer.scenario.opt.lr, 3));
    // log-scale slider
    const cur = Math.log10(trainer.scenario.opt.lr);
    lrWrap.append(
      slider(-3, 0.7, 0.05, cur, (v) => {
        trainer.scenario.opt.lr = Math.pow(10, v);
        lrVal.textContent = fmt(trainer.scenario.opt.lr, 3);
      }),
      lrVal,
    );
    row.append(lrWrap);
  }
  wrap.append(row);

  const statsRow = el("div", "chips-row");
  wrap.append(statsRow);
  const chartWrap = el("div");
  wrap.append(chartWrap);

  let dirty = true;
  const off = trainer.on("step", () => {
    dirty = true;
  });
  const offState = trainer.on("state", () => {
    playBtn.textContent = trainer.running ? "⏸︎ pause" : "▶︎ train";
  });
  const update = () => {
    if (!dirty) return;
    dirty = false;
    const s = trainer.scenario;
    statsRow.innerHTML = "";
    statsRow.append(
      chip("epoch", String(s.epoch)),
      chip("step", String(s.step)),
      chip("loss", fmt(trainer.lastLossValue, 4), true),
      chip(s.metricName, fmtMetric(s.metricName, trainer.lastMetricValue), true),
    );
    chartWrap.innerHTML = "";
    const pts = s.lossHistory.map((p) => ({ x: p.step, y: p.value }));
    const sampled = pts.length > 400 ? pts.filter((_, i) => i % Math.ceil(pts.length / 400) === 0) : pts;
    chartWrap.append(lineChart(sampled, { width: 520, height: 130, yLabel: "training loss" }));
  };
  update();
  const timer = window.setInterval(update, 350);
  return [
    wrap,
    () => {
      off();
      offState();
      window.clearInterval(timer);
    },
  ];
}

/** numeric <select> for picking samples / rows / columns */
export function picker(
  label: string,
  count: number,
  value: number,
  onChange: (v: number) => void,
  describe?: (i: number) => string,
): HTMLElement {
  const wrap = el("label", "control-label", `${label} `);
  const sel = document.createElement("select");
  for (let i = 0; i < count; i++) {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = describe ? describe(i) : String(i);
    sel.append(o);
  }
  sel.value = String(Math.min(value, count - 1));
  sel.addEventListener("change", () => onChange(parseInt(sel.value)));
  wrap.append(sel);
  return wrap;
}
