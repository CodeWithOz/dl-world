// Activation Springs: ReLU, why nonlinearity matters, and the sigmoid pool.

import { registerPanel } from "../panel";
import { chip, el, fmt, heatmap, section } from "../widgets";
import { liveRegion, picker } from "./common";

/** plot y=f(x) with an optional marked point */
function funcPlot(
  f: (x: number) => number,
  xmin: number,
  xmax: number,
  mark?: { x: number; y: number },
  w = 300,
  h = 170,
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = w * 2;
  canvas.height = h * 2;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(2, 2);
  let ymin = Infinity,
    ymax = -Infinity;
  const N = 120;
  const ys: number[] = [];
  for (let i = 0; i <= N; i++) {
    const y = f(xmin + ((xmax - xmin) * i) / N);
    ys.push(y);
    ymin = Math.min(ymin, y);
    ymax = Math.max(ymax, y);
  }
  if (ymax - ymin < 1e-9) ymax = ymin + 1;
  const pad = 0.15 * (ymax - ymin);
  ymin -= pad;
  ymax += pad;
  const px = (x: number) => ((x - xmin) / (xmax - xmin)) * w;
  const py = (y: number) => h - ((y - ymin) / (ymax - ymin)) * h;
  // axes
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.beginPath();
  ctx.moveTo(px(0), 0);
  ctx.lineTo(px(0), h);
  ctx.moveTo(0, py(0));
  ctx.lineTo(w, py(0));
  ctx.stroke();
  ctx.strokeStyle = "#5ad1c8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= N; i++) {
    const x = xmin + ((xmax - xmin) * i) / N;
    if (i === 0) ctx.moveTo(px(x), py(ys[i]));
    else ctx.lineTo(px(x), py(ys[i]));
  }
  ctx.stroke();
  if (mark) {
    ctx.fillStyle = "#ffd34d";
    ctx.beginPath();
    ctx.arc(px(mark.x), py(mark.y), 5, 0, Math.PI * 2);
    ctx.fill();
  }
  return canvas;
}

export function registerSpringPanels(): void {
  registerPanel("springs.relu", {
    title: "ReLU Spring — a1 = max(z1, 0)",
    subtitle: "The simplest nonlinearity: keep positives, flatten negatives to zero.",
    render(body, world) {
      let sel = { r: 0, c: 0 };
      const [region, cleanup, refresh] = liveRegion(world.main, (root) => {
        const z1 = world.mlp.nodes["z1 = x·W1 + b1"];
        const a1 = world.mlp.nodes["a1 = ReLU(z1)"];
        if (!z1 || !a1) return;
        let zeros = 0;
        for (let i = 0; i < a1.size; i++) if (a1.data[i] === 0) zeros++;
        const chips = el("div", "chips-row");
        chips.append(
          chip("shape", `[${z1.rows} × ${z1.cols}]`),
          chip("zeroed this batch", `${((zeros / a1.size) * 100).toFixed(1)}%`),
        );
        root.append(chips);

        const zv = z1.at(sel.r, sel.c);
        const av = a1.at(sel.r, sel.c);
        const s1 = section("The function", "Click any cell of z1 below to trace one element through the spring:");
        const row = el("div", "hstack wrap-row");
        row.append(funcPlot((x) => Math.max(x, 0), -4, 4, { x: Math.max(-4, Math.min(4, zv)), y: av }));
        const trace = el("div", "vstack");
        trace.append(
          el("div", "bigmath", `z1[${sel.r},${sel.c}] = ${fmt(zv)}`),
          el("div", "bigmath", `a1[${sel.r},${sel.c}] = max(${fmt(zv)}, 0) = <b>${fmt(av)}</b>`),
          el(
            "div",
            "bigmath",
            `∂a/∂z = ${zv > 0 ? "1 — gradient flows through unchanged" : "0 — gradient is blocked here"}`,
          ),
          el(
            "p",
            "explain",
            "That derivative is the whole backward story of ReLU: each element is either a wire (1) or a cut (0).",
          ),
        );
        row.append(trace);
        s1.append(row);
        root.append(s1);

        const s2 = section("Before and after, whole batch");
        const hrow = el("div", "hstack wrap-row");
        const zBox = el("div", "vstack");
        zBox.append(
          el("div", "caption", "z1 (blue = negative — about to be flattened)"),
          heatmap(z1.data, z1.rows, z1.cols, {
            maxWidth: 420, maxHeight: 200, symmetric: true,
            highlight: sel,
            onClickCell: (r, c) => { sel = { r, c }; refresh(); },
          }),
        );
        const aBox = el("div", "vstack");
        aBox.append(
          el("div", "caption", "a1 = ReLU(z1) (nothing below zero survives)"),
          heatmap(a1.data, a1.rows, a1.cols, { maxWidth: 420, maxHeight: 200, symmetric: true, highlight: sel }),
        );
        hrow.append(zBox, aBox);
        s2.append(hrow);
        root.append(s2);
      });
      body.append(region);
      return cleanup;
    },
  });

  registerPanel("springs.why", {
    title: "Why Nonlinearity?",
    subtitle: "Two linear layers without a spring between them collapse into one.",
    render(body) {
      const A = [
        [1.0, -2.0],
        [0.5, 1.5],
      ];
      const B = [
        [2.0, 1.0],
        [-1.0, 0.5],
      ];
      const x = [1.0, -1.0];
      const mulVec = (v: number[], M: number[][]) => [
        v[0] * M[0][0] + v[1] * M[1][0],
        v[0] * M[0][1] + v[1] * M[1][1],
      ];
      const mulMat = (P: number[][], Q: number[][]) => [
        [P[0][0] * Q[0][0] + P[0][1] * Q[1][0], P[0][0] * Q[0][1] + P[0][1] * Q[1][1]],
        [P[1][0] * Q[0][0] + P[1][1] * Q[1][0], P[1][0] * Q[0][1] + P[1][1] * Q[1][1]],
      ];
      const xA = mulVec(x, A);
      const xAB = mulVec(xA, B);
      const AB = mulMat(A, B);
      const xAB2 = mulVec(x, AB);
      const relu = (v: number[]) => v.map((u) => Math.max(u, 0));
      const xAr = relu(xA);
      const xArB = mulVec(xAr, B);

      const s1 = section(
        "The collapse, with real numbers",
        "Take x = [1, −1] and two tiny weight matrices A and B. Composing two linear maps is itself a linear map — so stacking linear layers buys nothing:",
      );
      s1.append(
        el("div", "bigmath", `(x·A)·B = [${xAB.map((v) => fmt(v)).join(", ")}]`),
        el("div", "bigmath", `x·(A·B) = [${xAB2.map((v) => fmt(v)).join(", ")}] &nbsp;← identical. A·B = [[${fmt(AB[0][0])}, ${fmt(AB[0][1])}], [${fmt(AB[1][0])}, ${fmt(AB[1][1])}]] is just one matrix.`),
      );
      body.append(s1);

      const s2 = section("Insert the spring", "Put ReLU between them and the shortcut is destroyed:");
      s2.append(
        el("div", "bigmath", `x·A = [${xA.map((v) => fmt(v)).join(", ")}]`),
        el("div", "bigmath", `ReLU(x·A) = [${xAr.map((v) => fmt(v)).join(", ")}] &nbsp;← the −0.5 was flattened`),
        el("div", "bigmath", `ReLU(x·A)·B = [${xArB.map((v) => fmt(v)).join(", ")}] ≠ (x·A)·B = [${xAB.map((v) => fmt(v)).join(", ")}]`),
        el(
          "p",
          "explain",
          "This is the universal approximation insight: linear layers + nonlinearities can approximate any function, given enough hidden units. The mills do the heavy arithmetic; this little spring is what makes depth meaningful.",
        ),
      );
      body.append(s2);
    },
  });

  registerPanel("springs.sigmoid", {
    title: "Sigmoid Pool — σ(x) = 1 / (1 + e⁻ˣ)",
    subtitle: "Squashes any number into (0, 1) — perfect for probabilities of yes/no questions.",
    render(body, world) {
      let markX = 2.0;
      const [region, cleanup, refresh] = liveRegion([world.cottage], (root) => {
        const sig = (x: number) => 1 / (1 + Math.exp(-x));
        const s1 = section("The function", "");
        const row = el("div", "hstack wrap-row");
        row.append(funcPlot(sig, -6, 6, { x: markX, y: sig(markX) }));
        const right = el("div", "vstack");
        right.append(
          picker("probe x =", 25, Math.round((markX + 6) * 2), (v) => { markX = v / 2 - 6; refresh(); }, (i) => fmt(i / 2 - 6, 1)),
          el("div", "bigmath", `σ(${fmt(markX, 2)}) = ${fmt(sig(markX))}`),
          el("div", "bigmath", `σ′(x) = σ(x)·(1−σ(x)) = ${fmt(sig(markX) * (1 - sig(markX)))}`),
          el(
            "p",
            "explain",
            "Note how the derivative shrinks toward 0 at both ends — a saturated sigmoid learns slowly. That's why mnist_loss training starts fast and then crawls.",
          ),
        );
        row.append(right);
        s1.append(row);
        root.append(s1);

        const preds = world.lin37.nodes["preds = x·w + b"];
        const sp = world.lin37.nodes["sigmoid preds"];
        if (preds && sp) {
          const s2 = section(
            "Used live in the Linear Cottage",
            "The 3-vs-7 cottage pushes its raw scores through this pool. First 12 of the current batch:",
          );
          const table = el("table", "num-table");
          table.innerHTML = "<thead><tr><th>raw score x·w+b</th><th>σ(score) = P(it's a 3)</th></tr></thead>";
          const tb = el("tbody");
          for (let i = 0; i < Math.min(12, preds.rows); i++) {
            const tr = el("tr");
            tr.innerHTML = `<td>${fmt(preds.at(i, 0))}</td><td>${fmt(sp.at(i, 0))}</td>`;
            tb.append(tr);
          }
          table.append(tb);
          s2.append(table);
          root.append(s2);
        }
      });
      body.append(region);
      return cleanup;
    },
  });
}
