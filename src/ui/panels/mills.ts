// Forward Avenue: the two Linear Mills. The Matmul Floor lets you click any
// output cell and zoom into the element-level dot product that produced it.

import { Tensor } from "../../engine/tensor";
import { World } from "../../sim/world";
import { registerPanel } from "../panel";
import { barChart, chip, el, fmt, heatmap, section, stats } from "../widgets";
import { liveRegion, picker } from "./common";

interface MillCfg {
  layer: 1 | 2;
  inLabel: string; // label of the input tensor in the graph
  outLabel: string;
  inName: string;
  outName: string;
  inDim: number;
  outDim: number;
}

const MILL1: MillCfg = {
  layer: 1,
  inLabel: "x (batch images)",
  outLabel: "z1 = x·W1 + b1",
  inName: "x",
  outName: "z1",
  inDim: 784,
  outDim: 64,
};
const MILL2: MillCfg = {
  layer: 2,
  inLabel: "a1 = ReLU(z1)",
  outLabel: "logits = a1·W2 + b2",
  inName: "a1",
  outName: "logits",
  inDim: 64,
  outDim: 10,
};

function weights(world: World, layer: 1 | 2): { W: Tensor; b: Tensor } {
  const m = world.mlp;
  return layer === 1 ? { W: m.w1, b: m.b1 } : { W: m.w2, b: m.b2 };
}

/** the element-level zoom: out[i,j] = Σ_k in[i,k] · W[k,j] + b[j] */
function dotProductMicroscope(
  world: World,
  cfg: MillCfg,
  i: number,
  j: number,
): HTMLElement {
  const { W, b } = weights(world, cfg.layer);
  const input = world.mlp.nodes[cfg.inLabel];
  const out = world.mlp.nodes[cfg.outLabel];
  const wrap = el("div", "microscope");
  if (!input || !out) return wrap;
  const K = cfg.inDim;
  const terms: { k: number; x: number; w: number; prod: number }[] = [];
  let total = 0;
  for (let k = 0; k < K; k++) {
    const xv = input.at(i, k);
    const wv = W.at(k, j);
    const prod = xv * wv;
    total += prod;
    terms.push({ k, x: xv, w: wv, prod });
  }
  const bias = b.data[j];
  const final = total + bias;
  wrap.append(
    el(
      "div",
      "bigmath",
      `${cfg.outName}[${i},${j}] = Σ<sub>k=0…${K - 1}</sub> ${cfg.inName}[${i},k] · W${cfg.layer}[k,${j}] &nbsp;+&nbsp; b${cfg.layer}[${j}]`,
    ),
  );
  const sorted = [...terms].sort((a, z) => Math.abs(z.prod) - Math.abs(a.prod));
  const top = sorted.slice(0, 14);
  const table = el("table", "num-table");
  table.innerHTML = `<thead><tr><th>k</th><th>${cfg.inName}[${i},k]</th><th>W${cfg.layer}[k,${j}]</th><th>product</th></tr></thead>`;
  const tbody = el("tbody");
  let runningShown = 0;
  for (const t of top) {
    runningShown += t.prod;
    const tr = el("tr");
    tr.innerHTML = `<td>${t.k}</td><td>${fmt(t.x)}</td><td>${fmt(t.w)}</td><td class="${t.prod >= 0 ? "pos" : "neg"}">${fmt(t.prod)}</td>`;
    tbody.append(tr);
  }
  const restCount = K - top.length;
  const rest = total - runningShown;
  const tr = el("tr", "muted-row");
  tr.innerHTML = `<td colspan="3">… ${restCount} smaller terms</td><td>${fmt(rest)}</td>`;
  tbody.append(tr);
  table.append(tbody);
  wrap.append(
    el("p", "explain", `The ${K}-term dot product, largest contributions first (every term is just one input value times one learned weight):`),
    table,
    el(
      "div",
      "bigmath",
      `Σ products = <b>${fmt(total)}</b> &nbsp;+&nbsp; bias ${fmt(bias)} &nbsp;=&nbsp; <b>${fmt(final)}</b> &nbsp; ${Math.abs(final - out.at(i, j)) < 1e-3 ? "✓ matches the tensor" : `(tensor holds ${fmt(out.at(i, j))})`}`,
    ),
  );
  return wrap;
}

function registerMill(cfg: MillCfg): void {
  const L = cfg.layer;
  registerPanel(`mill${L}.matmul`, {
    title: `Matmul Floor — ${cfg.outLabel}`,
    subtitle:
      L === 1
        ? "784 pixel values become 64 hidden features. Each output cell is one dot product — click any cell of z1 to zoom all the way in."
        : "64 hidden features become 10 class scores (logits). Click any logits cell to zoom into its dot product.",
    render(body, world) {
      let sel: { r: number; c: number } = { r: 0, c: 0 };
      const [region, cleanup, refresh] = liveRegion(world.main, (root) => {
        const input = world.mlp.nodes[cfg.inLabel];
        const out = world.mlp.nodes[cfg.outLabel];
        const { W } = weights(world, L);
        if (!input || !out) return;
        const shapes = el("div", "chips-row");
        shapes.append(
          chip(`${cfg.inName} (input)`, `[${input.rows} × ${input.cols}]`),
          chip(`W${L}`, `[${W.rows} × ${W.cols}]`),
          chip(`${cfg.outName} (output)`, `[${out.rows} × ${out.cols}]`),
          chip("multiply-adds", `${(input.rows * input.cols * W.cols).toLocaleString()}`),
        );
        root.append(shapes);

        const s = section(
          "The three tensors, live",
          "Batch flows top-to-bottom (64 rows). These update as the model trains — click a cell in the output to open the microscope.",
        );
        const row = el("div", "hstack wrap-row");
        const inBox = el("div", "vstack");
        inBox.append(el("div", "caption", `${cfg.inName} — input batch`), heatmap(input.data, input.rows, input.cols, { maxWidth: 330, maxHeight: 170 }));
        const wBox = el("div", "vstack");
        wBox.append(el("div", "caption", `W${L} — learned weights (blue −, red +)`), heatmap(W.data, W.rows, W.cols, { maxWidth: 280, maxHeight: 170, symmetric: true }));
        const outBox = el("div", "vstack");
        outBox.append(
          el("div", "caption", `${cfg.outName} — output (click a cell!)`),
          heatmap(out.data, out.rows, out.cols, {
            maxWidth: 300,
            maxHeight: 170,
            symmetric: true,
            highlight: sel,
            onClickCell: (r, c) => {
              sel = { r, c };
              refresh();
            },
          }),
        );
        row.append(inBox, wBox, outBox);
        s.append(row);
        root.append(s);

        const micro = section(
          `🔬 Microscope: ${cfg.outName}[${sel.r}, ${sel.c}]`,
          `Row ${sel.r} of the batch (one ${L === 1 ? "image" : "set of hidden features"}) · column ${sel.c} of W${L} (one ${L === 1 ? "hidden feature" : "digit class"}).`,
        );
        micro.append(dotProductMicroscope(world, cfg, sel.r, sel.c));
        root.append(micro);
      });
      body.append(region);
      return cleanup;
    },
  });

  registerPanel(`mill${L}.weights`, {
    title: `Weight Vault — W${L}`,
    subtitle:
      L === 1
        ? "Each of the 64 columns of W1 is 784 numbers — one per pixel — so each hidden feature can be drawn as a 28×28 image."
        : "W2 maps 64 hidden features to 10 digit scores: each column says which features vote for that digit.",
    render(body, world) {
      let selCol = 0;
      const [region, cleanup, refresh] = liveRegion(world.main, (root) => {
        const { W } = weights(world, L);
        const st = stats(W.data);
        const gs = W.grad ? stats(W.grad) : null;
        const chips = el("div", "chips-row");
        chips.append(
          chip("shape", `[${W.rows} × ${W.cols}]`),
          chip("parameters", W.size.toLocaleString()),
          chip("min / mean / max", `${fmt(st.min, 2)} / ${fmt(st.mean, 3)} / ${fmt(st.max, 2)}`),
        );
        if (gs) chips.append(chip("last grad |max|", fmt(Math.max(Math.abs(gs.min), Math.abs(gs.max)), 4)));
        root.append(chips);

        const s1 = section("The whole matrix", "Every learned parameter of this layer (blue = negative, red = positive). Watch it organize itself as training runs.");
        s1.append(heatmap(W.data, W.rows, W.cols, { maxWidth: 900, maxHeight: 230, symmetric: true }));
        root.append(s1);

        if (L === 1) {
          const s2 = section(
            "One hidden feature as an image",
            "Column j reshaped back to 28×28: red pixels excite this feature, blue pixels inhibit it. Early in training it's noise; later, stroke detectors appear.",
          );
          s2.append(picker("hidden feature j =", 64, selCol, (v) => { selCol = v; refresh(); }));
          const col = new Float32Array(784);
          for (let k = 0; k < 784; k++) col[k] = W.at(k, selCol);
          s2.append(el("div"), heatmap(col, 28, 28, { cellSize: 8, symmetric: true }));
          root.append(s2);
        } else {
          const s2 = section(
            "One digit's column",
            "Which of the 64 hidden features vote for this digit (positive bars) or against it (negative bars):",
          );
          s2.append(picker("digit class j =", 10, selCol, (v) => { selCol = v; refresh(); }));
          const col = new Float32Array(64);
          for (let k = 0; k < 64; k++) col[k] = W.at(k, selCol);
          s2.append(el("div"), barChart(col, { width: 700, height: 140, showValues: false }));
          root.append(s2);
        }
      });
      body.append(region);
      return cleanup;
    },
  });

  registerPanel(`mill${L}.bias`, {
    title: `Bias Bench — b${L}`,
    subtitle: "One extra learnable number per output: shifts the neuron's threshold. y = x·W + b.",
    render(body, world) {
      const [region, cleanup] = liveRegion(world.main, (root) => {
        const { b } = weights(world, L);
        const s = section(
          `b${L} — ${b.size} values, live`,
          "Without a bias, every neuron would be forced to output 0 for a zero input. The bias lets each neuron choose its own baseline.",
        );
        s.append(barChart(b.data, { width: 760, height: 150, showValues: b.size <= 10 }));
        if (b.grad) {
          s.append(el("div", "caption", "gradient on each bias (what the last backward pass said):"));
          s.append(barChart(b.grad, { width: 760, height: 110, showValues: false }));
        }
        root.append(s);
      });
      body.append(region);
      return cleanup;
    },
  });
}

export function registerMillPanels(): void {
  registerMill(MILL1);
  registerMill(MILL2);
}
