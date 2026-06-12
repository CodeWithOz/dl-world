// Small DOM/canvas widgets shared by every inspection panel.

export function el(
  tag: string,
  cls = "",
  html = "",
): HTMLElement {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  return e;
}

/** compact number formatting for tensor values */
export function fmt(n: number, digits = 4): string {
  if (!isFinite(n)) return String(n);
  if (n === 0) return "0";
  const a = Math.abs(n);
  if (a >= 10000 || a < 0.001) return n.toExponential(2);
  // strip trailing zeros only after a decimal point — a bare /\.?0+$/
  // would also eat the zeros of round integers (1700 → "17")
  return n
    .toFixed(digits)
    .replace(/(\.\d*?)0+$/, "$1")
    .replace(/\.$/, "");
}

export function stats(data: Float32Array): { min: number; max: number; mean: number } {
  let min = Infinity,
    max = -Infinity,
    sum = 0;
  for (let i = 0; i < data.length; i++) {
    min = Math.min(min, data[i]);
    max = Math.max(max, data[i]);
    sum += data[i];
  }
  return { min, max, mean: sum / data.length };
}

/** blue-white-red diverging colormap for weights/grads (neutral at zero) */
function divergingColor(t: number): [number, number, number] {
  // t in [-1, 1]; 0 maps to a dark neutral so panels read sign at a glance
  const mid: [number, number, number] = [34, 38, 52];
  const pos: [number, number, number] = [255, 120, 70];
  const neg: [number, number, number] = [90, 160, 255];
  const u = Math.min(1, Math.abs(t));
  const target = t >= 0 ? pos : neg;
  return [
    mid[0] + (target[0] - mid[0]) * u,
    mid[1] + (target[1] - mid[1]) * u,
    mid[2] + (target[2] - mid[2]) * u,
  ];
}

export interface HeatmapOpts {
  cellSize?: number;
  maxWidth?: number;
  maxHeight?: number;
  /** center colormap at zero (for weights/gradients) */
  symmetric?: boolean;
  onClickCell?: (r: number, c: number) => void;
  highlight?: { r: number; c: number } | null;
}

/** render a [rows, cols] matrix as a heatmap canvas (large matrices are strided) */
export function heatmap(
  data: Float32Array,
  rows: number,
  cols: number,
  opts: HeatmapOpts = {},
): HTMLCanvasElement {
  const maxW = opts.maxWidth ?? 480;
  const maxH = opts.maxHeight ?? 300;
  // stride so the matrix always fits inside maxW x maxH
  const rStride = Math.max(1, Math.ceil(rows / maxH));
  const cStride = Math.max(1, Math.ceil(cols / maxW));
  const dRows = Math.ceil(rows / rStride);
  const dCols = Math.ceil(cols / cStride);
  let cell = opts.cellSize ?? Math.max(1, Math.floor(Math.min(maxW / dCols, maxH / dRows)));
  cell = Math.max(1, Math.min(cell, 26));
  const canvas = document.createElement("canvas");
  canvas.width = dCols * cell;
  canvas.height = dRows * cell;
  canvas.className = "heatmap";
  const ctx = canvas.getContext("2d")!;
  const s = stats(data);
  const lim = opts.symmetric
    ? Math.max(Math.abs(s.min), Math.abs(s.max), 1e-9)
    : Math.max(s.max - s.min, 1e-9);
  const img = ctx.createImageData(canvas.width, canvas.height);
  for (let r = 0; r < dRows; r++)
    for (let c = 0; c < dCols; c++) {
      // take the max-|v| value in the strided block so signal isn't lost
      let v = 0;
      let best = -1;
      for (let rr = r * rStride; rr < Math.min((r + 1) * rStride, rows); rr++)
        for (let cc = c * cStride; cc < Math.min((c + 1) * cStride, cols); cc++) {
          const u = data[rr * cols + cc];
          if (Math.abs(u) > best) {
            best = Math.abs(u);
            v = u;
          }
        }
      let rgb: [number, number, number];
      if (opts.symmetric) rgb = divergingColor(v / lim);
      else {
        const u = (v - s.min) / lim;
        rgb = [18 + u * 220, 24 + u * 215, 38 + u * 200];
      }
      for (let py = 0; py < cell; py++)
        for (let px = 0; px < cell; px++) {
          const o = ((r * cell + py) * canvas.width + c * cell + px) * 4;
          img.data[o] = rgb[0];
          img.data[o + 1] = rgb[1];
          img.data[o + 2] = rgb[2];
          img.data[o + 3] = 255;
        }
    }
  ctx.putImageData(img, 0, 0);
  if (rStride > 1 || cStride > 1) canvas.title = `downsampled ×${cStride}×${rStride} for display`;
  // stretch small canvases up to the display budget (pixelated, possibly non-square)
  const wUp = Math.max(1, Math.min(6, Math.floor(maxW / canvas.width)));
  const hUp = Math.max(1, Math.min(6, Math.floor(maxH / canvas.height)));
  if (wUp > 1 || hUp > 1) {
    canvas.style.width = `${canvas.width * wUp}px`;
    canvas.style.height = `${canvas.height * hUp}px`;
  }
  if (opts.highlight) {
    ctx.strokeStyle = "#ffe44d";
    ctx.lineWidth = 2;
    ctx.strokeRect(
      Math.floor(opts.highlight.c / cStride) * cell,
      Math.floor(opts.highlight.r / rStride) * cell,
      Math.max(cell, 3),
      Math.max(cell, 3),
    );
  }
  if (opts.onClickCell) {
    canvas.style.cursor = "crosshair";
    canvas.addEventListener("click", (ev) => {
      const rect = canvas.getBoundingClientRect();
      const c = Math.floor(((ev.clientX - rect.left) / rect.width) * dCols) * cStride;
      const r = Math.floor(((ev.clientY - rect.top) / rect.height) * dRows) * rStride;
      if (r >= 0 && r < rows && c >= 0 && c < cols) opts.onClickCell!(r, c);
    });
  }
  return canvas;
}

/** render a 784-length digit (0..1 floats or 0..255 bytes) as a crisp canvas */
export function digitCanvas(
  pixels: Float32Array | Uint8Array,
  offset = 0,
  scale = 3,
): HTMLCanvasElement {
  // fractional scales would compute fractional ImageData offsets, whose
  // writes are silently dropped — coerce to a whole number of pixels
  scale = Math.max(1, Math.round(scale));
  const canvas = document.createElement("canvas");
  canvas.width = 28 * scale;
  canvas.height = 28 * scale;
  canvas.className = "digit";
  const ctx = canvas.getContext("2d")!;
  const isByte = pixels instanceof Uint8Array;
  const img = ctx.createImageData(canvas.width, canvas.height);
  for (let r = 0; r < 28; r++)
    for (let c = 0; c < 28; c++) {
      const raw = pixels[offset + r * 28 + c];
      const v = isByte ? raw : Math.max(0, Math.min(1, raw)) * 255;
      for (let py = 0; py < scale; py++)
        for (let px = 0; px < scale; px++) {
          const o = ((r * scale + py) * canvas.width + c * scale + px) * 4;
          img.data[o] = v * 0.95 + 12;
          img.data[o + 1] = v * 0.98 + 10;
          img.data[o + 2] = v + 18;
          img.data[o + 3] = 255;
        }
    }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

export interface BarChartOpts {
  labels?: string[];
  highlight?: number;
  width?: number;
  height?: number;
  showValues?: boolean;
}

/** horizontal-axis bar chart for logits / probabilities */
export function barChart(values: number[] | Float32Array, opts: BarChartOpts = {}): HTMLCanvasElement {
  const w = opts.width ?? 420;
  const h = opts.height ?? 150;
  const canvas = document.createElement("canvas");
  canvas.width = w * 2;
  canvas.height = h * 2;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(2, 2);
  const n = values.length;
  const min = Math.min(0, ...values);
  const max = Math.max(0.0001, ...values);
  const range = max - min;
  const bw = (w - 10) / n;
  const zeroY = h - 24 - ((0 - min) / range) * (h - 40);
  ctx.strokeStyle = "rgba(255,255,255,0.25)";
  ctx.beginPath();
  ctx.moveTo(0, zeroY);
  ctx.lineTo(w, zeroY);
  ctx.stroke();
  ctx.font = "10px ui-monospace, monospace";
  ctx.textAlign = "center";
  for (let i = 0; i < n; i++) {
    const v = values[i];
    const barH = (Math.abs(v) / range) * (h - 40);
    const x = 5 + i * bw;
    ctx.fillStyle = i === opts.highlight ? "#ffd34d" : v >= 0 ? "#5ad1c8" : "#e8807a";
    const y = v >= 0 ? zeroY - barH : zeroY;
    ctx.fillRect(x + 2, y, bw - 4, Math.max(barH, 1));
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    if (opts.labels) ctx.fillText(opts.labels[i], x + bw / 2, h - 10);
    if (opts.showValues !== false && n <= 12)
      ctx.fillText(fmt(v, 2), x + bw / 2, v >= 0 ? y - 4 : y + barH + 11);
  }
  return canvas;
}

export interface LineChartOpts {
  width?: number;
  height?: number;
  logX?: boolean;
  color?: string;
  yLabel?: string;
  markers?: { x: number; label: string; color: string }[];
  series2?: { x: number; y: number }[];
  color2?: string;
}

export function lineChart(
  points: { x: number; y: number }[],
  opts: LineChartOpts = {},
): HTMLCanvasElement {
  const w = opts.width ?? 460;
  const h = opts.height ?? 180;
  const canvas = document.createElement("canvas");
  canvas.width = w * 2;
  canvas.height = h * 2;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(2, 2);
  ctx.font = "10px ui-monospace, monospace";
  if (points.length < 2) {
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.fillText("not enough data yet — train a little first", 20, h / 2);
    return canvas;
  }
  const tx = (x: number) => (opts.logX ? Math.log10(Math.max(x, 1e-12)) : x);
  const all = [...points, ...(opts.series2 ?? [])];
  const xs = all.map((p) => tx(p.x));
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  let ymin = Math.min(...all.map((p) => p.y));
  let ymax = Math.max(...all.map((p) => p.y));
  if (ymax - ymin < 1e-9) {
    ymax += 1;
    ymin -= 1;
  }
  const pad = { l: 44, r: 10, t: 10, b: 24 };
  const px = (x: number) => pad.l + ((tx(x) - xmin) / (xmax - xmin || 1)) * (w - pad.l - pad.r);
  const py = (y: number) => pad.t + (1 - (y - ymin) / (ymax - ymin)) * (h - pad.t - pad.b);
  // axes
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();
  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.textAlign = "right";
  ctx.fillText(fmt(ymax, 3), pad.l - 4, pad.t + 8);
  ctx.fillText(fmt(ymin, 3), pad.l - 4, h - pad.b);
  ctx.textAlign = "center";
  const xLabelFmt = (v: number) => (opts.logX ? `1e${fmt(v, 1)}` : fmt(v, 0));
  ctx.fillText(xLabelFmt(xmin), pad.l, h - 8);
  ctx.fillText(xLabelFmt(xmax), w - pad.r - 10, h - 8);
  if (opts.yLabel) {
    ctx.textAlign = "left";
    ctx.fillText(opts.yLabel, pad.l + 6, pad.t + 8);
  }
  const drawSeries = (pts: { x: number; y: number }[], color: string) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    pts.forEach((p, i) => {
      if (i === 0) ctx.moveTo(px(p.x), py(p.y));
      else ctx.lineTo(px(p.x), py(p.y));
    });
    ctx.stroke();
  };
  drawSeries(points, opts.color ?? "#5ad1c8");
  if (opts.series2) drawSeries(opts.series2, opts.color2 ?? "#ffd34d");
  for (const m of opts.markers ?? []) {
    ctx.strokeStyle = m.color;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(px(m.x), pad.t);
    ctx.lineTo(px(m.x), h - pad.b);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = m.color;
    ctx.textAlign = "center";
    ctx.fillText(m.label, px(m.x), pad.t + 8);
  }
  return canvas;
}

/** a labeled stat chip */
export function chip(label: string, value: string, accent = false): HTMLElement {
  const c = el("div", `chip${accent ? " chip-accent" : ""}`);
  c.append(el("span", "chip-label", label), el("span", "chip-value", value));
  return c;
}

/** section with a heading and explainer text */
export function section(title: string, explainer = ""): HTMLElement {
  const s = el("div", "panel-section");
  s.append(el("h3", "", title));
  if (explainer) s.append(el("p", "explain", explainer));
  return s;
}

/** row of flow boxes with arrows between them (factory pipeline visual) */
export function flowRow(items: { name: string; body: HTMLElement | string; accent?: boolean }[]): HTMLElement {
  const row = el("div", "flow-row");
  items.forEach((it, i) => {
    if (i > 0) row.append(el("div", "flow-arrow", "→"));
    const box = el("div", `flow-box${it.accent ? " flow-accent" : ""}`);
    box.append(el("div", "flow-name", it.name));
    const body = el("div", "flow-body");
    if (typeof it.body === "string") body.innerHTML = it.body;
    else body.append(it.body);
    box.append(body);
    row.append(box);
  });
  return row;
}

/** vertical list of numbers with optional highlight, for showing a tensor slice */
export function numberStrip(values: number[] | Float32Array, highlight = -1, digits = 3): HTMLElement {
  const wrap = el("div", "numstrip");
  const n = values.length;
  for (let i = 0; i < n; i++) {
    const v = el("div", `numcell${i === highlight ? " numcell-hl" : ""}`);
    v.textContent = fmt(values[i], digits);
    wrap.append(v);
  }
  return wrap;
}

export function button(label: string, onClick: () => void, cls = ""): HTMLButtonElement {
  const b = document.createElement("button");
  b.className = `btn ${cls}`;
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}

export function slider(
  min: number,
  max: number,
  step: number,
  value: number,
  onInput: (v: number) => void,
): HTMLInputElement {
  const s = document.createElement("input");
  s.type = "range";
  s.min = String(min);
  s.max = String(max);
  s.step = String(step);
  s.value = String(value);
  s.addEventListener("input", () => onInput(parseFloat(s.value)));
  return s;
}
