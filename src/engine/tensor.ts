// DL World's tiny define-by-run autograd engine.
//
// Every op records its inputs, so each training step produces a real
// computation graph. The world UI walks this graph: buildings map to labeled
// nodes, and "zooming in" expands a node into the element-level arithmetic
// using the live values stored here. Tensors are 1D or 2D ([rows, cols]),
// which covers everything in fastai ch. 4-6 territory.

export class Tensor {
  data: Float32Array;
  shape: number[];
  grad: Float32Array | null = null;
  requiresGrad: boolean;
  /** op that produced this tensor, "" for leaves (params / data) */
  op: string;
  inputs: Tensor[];
  /** human-readable name shown in the world ("logits", "softmax", ...) */
  label = "";
  backwardFn: (() => void) | null = null;

  constructor(
    data: Float32Array,
    shape: number[],
    opts: { requiresGrad?: boolean; op?: string; inputs?: Tensor[] } = {},
  ) {
    const size = shape.reduce((a, b) => a * b, 1);
    if (data.length !== size)
      throw new Error(`data length ${data.length} != shape ${shape.join("x")}`);
    this.data = data;
    this.shape = shape;
    this.requiresGrad = opts.requiresGrad ?? false;
    this.op = opts.op ?? "";
    this.inputs = opts.inputs ?? [];
  }

  get size(): number {
    return this.data.length;
  }
  get rows(): number {
    return this.shape.length === 2 ? this.shape[0] : 1;
  }
  get cols(): number {
    return this.shape.length === 2 ? this.shape[1] : this.shape[0];
  }

  static zeros(shape: number[], requiresGrad = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size), shape, { requiresGrad });
  }

  static fromArray(arr: number[], shape?: number[]): Tensor {
    return new Tensor(new Float32Array(arr), shape ?? [arr.length]);
  }

  /** Gaussian init, like torch.randn * scale */
  static randn(shape: number[], scale = 1, rand: () => number = Math.random): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i += 2) {
      // Box-Muller
      const u1 = Math.max(rand(), 1e-12);
      const u2 = rand();
      const r = Math.sqrt(-2 * Math.log(u1));
      data[i] = r * Math.cos(2 * Math.PI * u2) * scale;
      if (i + 1 < size) data[i + 1] = r * Math.sin(2 * Math.PI * u2) * scale;
    }
    return new Tensor(data, shape, { requiresGrad: false });
  }

  named(label: string): Tensor {
    this.label = label;
    return this;
  }

  at(r: number, c = 0): number {
    return this.data[r * this.cols + c];
  }

  gradAt(r: number, c = 0): number {
    return this.grad ? this.grad[r * this.cols + c] : 0;
  }

  ensureGrad(): Float32Array {
    if (!this.grad) this.grad = new Float32Array(this.size);
    return this.grad;
  }

  zeroGrad(): void {
    if (this.grad) this.grad.fill(0);
  }

  /** Run backprop from this (scalar) tensor through the recorded graph. */
  backward(): void {
    if (this.size !== 1)
      throw new Error("backward() must start from a scalar (the loss)");
    const topo: Tensor[] = [];
    const seen = new Set<Tensor>();
    const visit = (t: Tensor) => {
      if (seen.has(t)) return;
      seen.add(t);
      for (const inp of t.inputs) visit(inp);
      topo.push(t);
    };
    visit(this);
    this.ensureGrad().fill(1);
    for (let i = topo.length - 1; i >= 0; i--) topo[i].backwardFn?.();
  }

  /** Topologically sorted list of every node feeding this tensor (leaves first). */
  graphNodes(): Tensor[] {
    const topo: Tensor[] = [];
    const seen = new Set<Tensor>();
    const visit = (t: Tensor) => {
      if (seen.has(t)) return;
      seen.add(t);
      for (const inp of t.inputs) visit(inp);
      topo.push(t);
    };
    visit(this);
    return topo;
  }

  item(): number {
    if (this.size !== 1) throw new Error("item() needs a scalar");
    return this.data[0];
  }
}

function out(
  data: Float32Array,
  shape: number[],
  op: string,
  inputs: Tensor[],
): Tensor {
  const needsGrad = inputs.some((t) => t.requiresGrad);
  const t = new Tensor(data, shape, { op, inputs });
  t.requiresGrad = needsGrad;
  return t;
}

function sameShape(a: Tensor, b: Tensor, op: string): void {
  if (a.rows !== b.rows || a.cols !== b.cols)
    throw new Error(
      `${op}: shape mismatch ${a.shape.join("x")} vs ${b.shape.join("x")}`,
    );
}

// ---------------------------------------------------------------- matmul ---

/** A[m,k] @ B[k,n] -> [m,n] */
export function matmul(a: Tensor, b: Tensor): Tensor {
  const [m, k] = [a.rows, a.cols];
  const [k2, n] = [b.rows, b.cols];
  if (k !== k2) throw new Error(`matmul: inner dims ${k} vs ${k2}`);
  const data = new Float32Array(m * n);
  const ad = a.data, bd = b.data;
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const av = ad[i * k + p];
      if (av === 0) continue;
      const bOff = p * n;
      const oOff = i * n;
      for (let j = 0; j < n; j++) data[oOff + j] += av * bd[bOff + j];
    }
  }
  const t = out(data, [m, n], "matmul", [a, b]);
  if (t.requiresGrad)
    t.backwardFn = () => {
      const g = t.grad!;
      if (a.requiresGrad) {
        const ga = a.ensureGrad();
        // dA = dOut @ B^T
        for (let i = 0; i < m; i++)
          for (let p = 0; p < k; p++) {
            let s = 0;
            for (let j = 0; j < n; j++) s += g[i * n + j] * bd[p * n + j];
            ga[i * k + p] += s;
          }
      }
      if (b.requiresGrad) {
        const gb = b.ensureGrad();
        // dB = A^T @ dOut
        for (let p = 0; p < k; p++)
          for (let j = 0; j < n; j++) {
            let s = 0;
            for (let i = 0; i < m; i++) s += ad[i * k + p] * g[i * n + j];
            gb[p * n + j] += s;
          }
      }
    };
  return t;
}

// ----------------------------------------------------------- elementwise ---

function ew(
  a: Tensor,
  b: Tensor,
  op: string,
  f: (x: number, y: number) => number,
  dfa: (x: number, y: number) => number,
  dfb: (x: number, y: number) => number,
): Tensor {
  sameShape(a, b, op);
  const data = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) data[i] = f(a.data[i], b.data[i]);
  const t = out(data, a.shape, op, [a, b]);
  if (t.requiresGrad)
    t.backwardFn = () => {
      const g = t.grad!;
      if (a.requiresGrad) {
        const ga = a.ensureGrad();
        for (let i = 0; i < a.size; i++) ga[i] += g[i] * dfa(a.data[i], b.data[i]);
      }
      if (b.requiresGrad) {
        const gb = b.ensureGrad();
        for (let i = 0; i < b.size; i++) gb[i] += g[i] * dfb(a.data[i], b.data[i]);
      }
    };
  return t;
}

export function add(a: Tensor, b: Tensor): Tensor {
  return ew(a, b, "add", (x, y) => x + y, () => 1, () => 1);
}

export function sub(a: Tensor, b: Tensor): Tensor {
  return ew(a, b, "sub", (x, y) => x - y, () => 1, () => -1);
}

export function mul(a: Tensor, b: Tensor): Tensor {
  return ew(a, b, "mul", (x, y) => x * y, (_x, y) => y, (x) => x);
}

function unary(
  a: Tensor,
  op: string,
  f: (x: number) => number,
  df: (x: number, fx: number) => number,
): Tensor {
  const data = new Float32Array(a.size);
  for (let i = 0; i < a.size; i++) data[i] = f(a.data[i]);
  const t = out(data, a.shape, op, [a]);
  if (t.requiresGrad)
    t.backwardFn = () => {
      const g = t.grad!;
      const ga = a.ensureGrad();
      for (let i = 0; i < a.size; i++) ga[i] += g[i] * df(a.data[i], data[i]);
    };
  return t;
}

export function relu(a: Tensor): Tensor {
  return unary(a, "relu", (x) => (x > 0 ? x : 0), (x) => (x > 0 ? 1 : 0));
}

export function sigmoid(a: Tensor): Tensor {
  return unary(a, "sigmoid", (x) => 1 / (1 + Math.exp(-x)), (_x, fx) => fx * (1 - fx));
}

export function exp(a: Tensor): Tensor {
  return unary(a, "exp", Math.exp, (_x, fx) => fx);
}

/** log with epsilon clamp so log(0) stays finite for teaching-scale models */
export function log(a: Tensor): Tensor {
  const EPS = 1e-12;
  return unary(a, "log", (x) => Math.log(Math.max(x, EPS)), (x) => 1 / Math.max(x, EPS));
}

export function neg(a: Tensor): Tensor {
  return unary(a, "neg", (x) => -x, () => -1);
}

export function square(a: Tensor): Tensor {
  return unary(a, "square", (x) => x * x, (x) => 2 * x);
}

export function scale(a: Tensor, s: number): Tensor {
  return unary(a, `scale(${s})`, (x) => x * s, () => s);
}

export function addScalar(a: Tensor, s: number): Tensor {
  return unary(a, `add(${s})`, (x) => x + s, () => 1);
}

// ------------------------------------------------------------- broadcast ---

/** A[m,n] + b[n] (bias broadcast over rows) */
export function addRow(a: Tensor, b: Tensor): Tensor {
  const [m, n] = [a.rows, a.cols];
  if (b.size !== n) throw new Error(`addRow: bias size ${b.size} != cols ${n}`);
  const data = new Float32Array(m * n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) data[i * n + j] = a.data[i * n + j] + b.data[j];
  const t = out(data, [m, n], "addRow", [a, b]);
  if (t.requiresGrad)
    t.backwardFn = () => {
      const g = t.grad!;
      if (a.requiresGrad) {
        const ga = a.ensureGrad();
        for (let i = 0; i < a.size; i++) ga[i] += g[i];
      }
      if (b.requiresGrad) {
        const gb = b.ensureGrad();
        for (let i = 0; i < m; i++)
          for (let j = 0; j < n; j++) gb[j] += g[i * n + j];
      }
    };
  return t;
}

/** A[m,n] / d[m,1] (divide each row by its own scalar) */
export function divCol(a: Tensor, d: Tensor): Tensor {
  const [m, n] = [a.rows, a.cols];
  if (d.rows !== m || d.cols !== 1)
    throw new Error(`divCol: divisor must be [${m},1], got ${d.shape.join("x")}`);
  const data = new Float32Array(m * n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) data[i * n + j] = a.data[i * n + j] / d.data[i];
  const t = out(data, [m, n], "divCol", [a, d]);
  if (t.requiresGrad)
    t.backwardFn = () => {
      const g = t.grad!;
      if (a.requiresGrad) {
        const ga = a.ensureGrad();
        for (let i = 0; i < m; i++)
          for (let j = 0; j < n; j++) ga[i * n + j] += g[i * n + j] / d.data[i];
      }
      if (d.requiresGrad) {
        const gd = d.ensureGrad();
        for (let i = 0; i < m; i++) {
          let s = 0;
          for (let j = 0; j < n; j++)
            s += g[i * n + j] * (-a.data[i * n + j] / (d.data[i] * d.data[i]));
          gd[i] += s;
        }
      }
    };
  return t;
}

// ------------------------------------------------------------ reductions ---

export function sum(a: Tensor): Tensor {
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i];
  const t = out(new Float32Array([s]), [1], "sum", [a]);
  if (t.requiresGrad)
    t.backwardFn = () => {
      const g = t.grad![0];
      const ga = a.ensureGrad();
      for (let i = 0; i < a.size; i++) ga[i] += g;
    };
  return t;
}

export function mean(a: Tensor): Tensor {
  const n = a.size;
  let s = 0;
  for (let i = 0; i < n; i++) s += a.data[i];
  const t = out(new Float32Array([s / n]), [1], "mean", [a]);
  if (t.requiresGrad)
    t.backwardFn = () => {
      const g = t.grad![0] / n;
      const ga = a.ensureGrad();
      for (let i = 0; i < n; i++) ga[i] += g;
    };
  return t;
}

/** row-wise sum: A[m,n] -> [m,1] */
export function sumRows(a: Tensor): Tensor {
  const [m, n] = [a.rows, a.cols];
  const data = new Float32Array(m);
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += a.data[i * n + j];
    data[i] = s;
  }
  const t = out(data, [m, 1], "sumRows", [a]);
  if (t.requiresGrad)
    t.backwardFn = () => {
      const g = t.grad!;
      const ga = a.ensureGrad();
      for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++) ga[i * n + j] += g[i];
    };
  return t;
}

/** row-wise max as a constant (no grad path; used for stable softmax) */
export function rowMaxDetached(a: Tensor): Tensor {
  const [m, n] = [a.rows, a.cols];
  const data = new Float32Array(m);
  for (let i = 0; i < m; i++) {
    let mx = -Infinity;
    for (let j = 0; j < n; j++) mx = Math.max(mx, a.data[i * n + j]);
    data[i] = mx;
  }
  return new Tensor(data, [m, 1], { op: "rowMax (detached)" });
}

/** A[m,n] - c[m,1] broadcast */
export function subCol(a: Tensor, c: Tensor): Tensor {
  const [m, n] = [a.rows, a.cols];
  if (c.rows !== m || c.cols !== 1)
    throw new Error(`subCol: expected [${m},1], got ${c.shape.join("x")}`);
  const data = new Float32Array(m * n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) data[i * n + j] = a.data[i * n + j] - c.data[i];
  const t = out(data, [m, n], "subCol", [a, c]);
  if (t.requiresGrad)
    t.backwardFn = () => {
      const g = t.grad!;
      if (a.requiresGrad) {
        const ga = a.ensureGrad();
        for (let i = 0; i < a.size; i++) ga[i] += g[i];
      }
      if (c.requiresGrad) {
        const gc = c.ensureGrad();
        for (let i = 0; i < m; i++) {
          let s = 0;
          for (let j = 0; j < n; j++) s += g[i * n + j];
          gc[i] -= s;
        }
      }
    };
  return t;
}

/** pick one column per row: A[m,n], idx[m] -> [m,1] (e.g. prob of the target class) */
export function gatherCols(a: Tensor, idx: Int32Array | number[]): Tensor {
  const [m, n] = [a.rows, a.cols];
  if (idx.length !== m) throw new Error(`gatherCols: idx length ${idx.length} != rows ${m}`);
  const data = new Float32Array(m);
  for (let i = 0; i < m; i++) data[i] = a.data[i * n + idx[i]];
  const t = out(data, [m, 1], "gather", [a]);
  (t as Tensor & { gatherIdx?: Int32Array }).gatherIdx = Int32Array.from(idx);
  if (t.requiresGrad)
    t.backwardFn = () => {
      const g = t.grad!;
      const ga = a.ensureGrad();
      for (let i = 0; i < m; i++) ga[i * n + idx[i]] += g[i];
    };
  return t;
}

// --------------------------------------------------------- compositions ---
// These are deliberately built from the primitives above so the recorded
// graph contains every station the player can walk through.

/** row-wise stable softmax, graph: subCol(rowMax) -> exp -> sumRows -> divCol */
export function softmax(a: Tensor): Tensor {
  const shifted = subCol(a, rowMaxDetached(a)).named("shifted logits");
  const e = exp(shifted).named("exp");
  const z = sumRows(e).named("row sums");
  return divCol(e, z).named("softmax");
}

/** cross-entropy from primitives: softmax -> log -> gather target -> neg -> mean */
export function crossEntropy(logits: Tensor, targets: Int32Array): Tensor {
  const p = softmax(logits);
  const logp = log(p).named("log probs");
  const picked = gatherCols(logp, targets).named("log p[target]");
  const nll = neg(picked).named("negative log likelihood");
  return mean(nll).named("loss");
}

/** fastai ch.4 mnist_loss: where(y==1, 1-p, p).mean(), built as y(1-p)+(1-y)p */
export function mnistLoss(preds: Tensor, targets: Tensor): Tensor {
  const p = sigmoid(preds).named("sigmoid preds");
  const oneMinusP = addScalar(neg(p), 1).named("1 - p");
  const oneMinusY = addScalar(neg(targets), 1).named("1 - y");
  const term1 = mul(targets, oneMinusP).named("y * (1-p)");
  const term2 = mul(oneMinusY, p).named("(1-y) * p");
  const perItem = add(term1, term2).named("per-item loss");
  return mean(perItem).named("loss");
}

/** binary cross-entropy with logits, from primitives (ch.6 multi-label) */
export function bceWithLogits(logits: Tensor, targets: Tensor): Tensor {
  const p = sigmoid(logits).named("sigmoid");
  const logP = log(p).named("log p");
  const logOneMinusP = log(addScalar(neg(p), 1).named("1 - p")).named("log(1-p)");
  const oneMinusY = addScalar(neg(targets), 1).named("1 - y");
  const term1 = mul(targets, logP).named("y log p");
  const term2 = mul(oneMinusY, logOneMinusP).named("(1-y) log(1-p)");
  const perItem = neg(add(term1, term2)).named("per-item BCE");
  return mean(perItem).named("loss");
}

/** mean squared error (ch.6 regression) */
export function mseLoss(preds: Tensor, targets: Tensor): Tensor {
  const diff = sub(preds, targets).named("error");
  const sq = square(diff).named("squared error");
  return mean(sq).named("loss");
}
