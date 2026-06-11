// The four live training scenarios of DL World, mapped to fastai chapters:
//   mlp10      - ch.4/5: two-layer net, softmax + cross-entropy on 10 digits
//                (the city's central pipeline)
//   linear37   - ch.4: linear model, sigmoid + mnist_loss on 3s vs 7s
//   multilabel - ch.6: one image, several yes/no labels, sigmoid + BCE
//   regression - ch.6: predict the ink's center of mass, MSE
//
// Every forward pass is built from engine primitives and labeled, so the
// buildings can show the actual graph of the most recent training step.

import {
  Tensor,
  addRow,
  bceWithLogits,
  crossEntropy,
  matmul,
  mnistLoss,
  mseLoss,
  relu,
  sigmoid,
  softmax,
} from "../engine/tensor";
import { SGD } from "../engine/optim";
import { DataLoader, MnistData, mulberry32 } from "../engine/data";

export type ScenarioId = "mlp10" | "linear37" | "multilabel" | "regression";

export interface ParamInfo {
  name: string;
  tensor: Tensor;
}

export interface HistoryPoint {
  step: number;
  value: number;
}

/** multi-label properties derived from the digit (ch.6 style multi-label task) */
export const MULTILABEL_NAMES = ["has a loop", "is even", "is ≥ 5"];
export function digitProperties(d: number): [number, number, number] {
  const loop = [0, 6, 8, 9].includes(d) ? 1 : 0;
  const even = d % 2 === 0 ? 1 : 0;
  const big = d >= 5 ? 1 : 0;
  return [loop, even, big];
}

/** normalized (x, y) center of mass of the ink, in 0..1 (ch.6 regression target) */
export function inkCenter(images: Uint8Array, i: number): [number, number] {
  let sx = 0,
    sy = 0,
    total = 0;
  const off = i * 784;
  for (let r = 0; r < 28; r++)
    for (let c = 0; c < 28; c++) {
      const v = images[off + r * 28 + c];
      sx += v * c;
      sy += v * r;
      total += v;
    }
  if (total === 0) return [0.5, 0.5];
  return [sx / total / 27, sy / total / 27];
}

export abstract class Scenario {
  abstract readonly id: ScenarioId;
  abstract readonly title: string;
  abstract readonly metricName: string;

  data: MnistData;
  params: ParamInfo[] = [];
  loader!: DataLoader;
  opt!: SGD;
  step = 0;
  lossHistory: HistoryPoint[] = [];
  metricHistory: HistoryPoint[] = [];
  /** root of the most recent step's computation graph */
  lastLoss: Tensor | null = null;
  /** labeled tensors from the most recent forward, keyed by label */
  nodes: Record<string, Tensor> = {};
  lastBatch: Int32Array = new Int32Array(0);
  protected rand = mulberry32(1234);

  constructor(data: MnistData) {
    this.data = data;
  }

  get epoch(): number {
    return this.loader.epoch;
  }

  protected param(name: string, t: Tensor): Tensor {
    t.requiresGrad = true;
    t.named(name);
    this.params.push({ name, tensor: t });
    return t;
  }

  protected grab(t: Tensor): Tensor {
    if (t.label) this.nodes[t.label] = t;
    return t;
  }

  /** build loss graph for a batch; record=true stashes labeled nodes for the UI */
  abstract buildLoss(batch: Int32Array, record: boolean): Tensor;
  /** metric on the held-out test set, 0..1 */
  abstract evaluate(): number;

  trainStep(): number {
    const batch = this.loader.next();
    this.lastBatch = batch;
    // 1) the actual training step
    this.opt.zeroGrad();
    const loss = this.buildLoss(batch, false);
    loss.backward();
    this.opt.step();
    this.step++;
    const l = loss.item();
    this.lossHistory.push({ step: this.step, value: l });
    if (this.lossHistory.length > 5000) this.lossHistory.splice(0, 1000);
    // 2) a fresh recorded pass with the *updated* weights, so every panel
    //    (values, gradients, microscopes) shows one mutually consistent state
    this.opt.zeroGrad();
    this.nodes = {};
    const display = this.buildLoss(batch, true);
    this.lastLoss = display;
    display.backward();
    return l;
  }

  recordMetric(): number {
    const m = this.evaluate();
    this.metricHistory.push({ step: this.step, value: m });
    return m;
  }

  /** pure loss recomputation on the last batch (Gradient Lab / loss landscape) */
  lossOnLastBatch(): number {
    if (this.lastBatch.length === 0) return NaN;
    return this.buildLoss(this.lastBatch, false).item();
  }

  /** batch of normalized images as a [b, 784] tensor */
  protected batchX(idx: Int32Array, images = this.data.trainImages): Tensor {
    const b = idx.length;
    const x = new Float32Array(b * 784);
    for (let i = 0; i < b; i++) {
      const off = idx[i] * 784;
      for (let j = 0; j < 784; j++) x[i * 784 + j] = images[off + j] / 255;
    }
    return new Tensor(x, [b, 784]);
  }
}

// --------------------------------------------------------------- mlp10 ----

export class Mlp10 extends Scenario {
  readonly id = "mlp10" as const;
  readonly title = "Digit Classifier (2-layer net)";
  readonly metricName = "test accuracy";
  readonly hidden = 64;
  w1: Tensor;
  b1: Tensor;
  w2: Tensor;
  b2: Tensor;

  constructor(data: MnistData, lr = 0.5, batchSize = 64) {
    super(data);
    // Kaiming-ish init for the ReLU layer
    this.w1 = this.param("W1", Tensor.randn([784, this.hidden], Math.sqrt(2 / 784), this.rand));
    this.b1 = this.param("b1", Tensor.zeros([this.hidden]));
    this.w2 = this.param("W2", Tensor.randn([this.hidden, 10], Math.sqrt(1 / this.hidden), this.rand));
    this.b2 = this.param("b2", Tensor.zeros([10]));
    this.loader = new DataLoader(data.nTrain, batchSize, true, 99);
    this.opt = new SGD(this.params.map((p) => p.tensor), lr);
  }

  targets(idx: Int32Array): Int32Array {
    const t = new Int32Array(idx.length);
    for (let i = 0; i < idx.length; i++) t[i] = this.data.trainLabels[idx[i]];
    return t;
  }

  buildLoss(batch: Int32Array, record: boolean): Tensor {
    const g = record ? this.grab.bind(this) : (t: Tensor) => t;
    const x = g(this.batchX(batch).named("x (batch images)"));
    const z1 = g(addRow(matmul(x, this.w1), this.b1).named("z1 = x·W1 + b1"));
    const a1 = g(relu(z1).named("a1 = ReLU(z1)"));
    const logits = g(addRow(matmul(a1, this.w2), this.b2).named("logits = a1·W2 + b2"));
    const loss = crossEntropy(logits, this.targets(batch));
    if (record) for (const n of loss.graphNodes()) if (n.label) this.nodes[n.label] = n;
    return loss;
  }

  /** forward pass for one image (inference), capturing every stage */
  infer(pixels: Float32Array): {
    z1: Float32Array;
    a1: Float32Array;
    logits: Float32Array;
    probs: Float32Array;
    pred: number;
  } {
    const x = new Tensor(pixels.slice(), [1, 784]);
    const z1 = addRow(matmul(x, this.w1), this.b1);
    const a1 = relu(z1);
    const logits = addRow(matmul(a1, this.w2), this.b2);
    const probs = softmax(logits);
    let pred = 0;
    for (let j = 1; j < 10; j++) if (probs.data[j] > probs.data[pred]) pred = j;
    return {
      z1: z1.data,
      a1: a1.data,
      logits: logits.data,
      probs: probs.data,
      pred,
    };
  }

  predictTestAll(): { preds: Uint8Array; correct: number } {
    const n = this.data.nTest;
    const preds = new Uint8Array(n);
    let correct = 0;
    const CHUNK = 200;
    for (let start = 0; start < n; start += CHUNK) {
      const m = Math.min(CHUNK, n - start);
      const idx = new Int32Array(m);
      for (let i = 0; i < m; i++) idx[i] = start + i;
      const x = this.batchX(idx, this.data.testImages);
      const a1 = relu(addRow(matmul(x, this.w1), this.b1));
      const logits = addRow(matmul(a1, this.w2), this.b2);
      for (let i = 0; i < m; i++) {
        let best = 0;
        for (let j = 1; j < 10; j++)
          if (logits.at(i, j) > logits.at(i, best)) best = j;
        preds[start + i] = best;
        if (best === this.data.testLabels[start + i]) correct++;
      }
    }
    return { preds, correct };
  }

  evaluate(): number {
    return this.predictTestAll().correct / this.data.nTest;
  }
}

// ------------------------------------------------------------- linear37 ---

export class Linear37 extends Scenario {
  readonly id = "linear37" as const;
  readonly title = "Is it a 3 or a 7? (linear model)";
  readonly metricName = "test accuracy";
  w: Tensor;
  b: Tensor;
  /** dataset indices (into train images) that are 3s or 7s */
  subset: Int32Array;
  testSubset: Int32Array;

  constructor(data: MnistData, lr = 1.0, batchSize = 64) {
    super(data);
    const pick = (labels: Uint8Array, n: number) => {
      const keep: number[] = [];
      for (let i = 0; i < n; i++)
        if (labels[i] === 3 || labels[i] === 7) keep.push(i);
      return Int32Array.from(keep);
    };
    this.subset = pick(data.trainLabels, data.nTrain);
    this.testSubset = pick(data.testLabels, data.nTest);
    this.w = this.param("w (weights)", Tensor.randn([784, 1], 0.05, this.rand));
    this.b = this.param("b (bias)", Tensor.zeros([1]));
    this.loader = new DataLoader(this.subset.length, batchSize, true, 7);
    this.opt = new SGD(this.params.map((p) => p.tensor), lr);
  }

  /** loader yields positions in `subset`; resolve to dataset indices */
  resolve(batch: Int32Array): Int32Array {
    const r = new Int32Array(batch.length);
    for (let i = 0; i < batch.length; i++) r[i] = this.subset[batch[i]];
    return r;
  }

  buildLoss(batch: Int32Array, record: boolean): Tensor {
    const idx = this.resolve(batch);
    const y = new Float32Array(idx.length);
    for (let i = 0; i < idx.length; i++)
      y[i] = this.data.trainLabels[idx[i]] === 3 ? 1 : 0;
    const x = this.batchX(idx).named("x (batch images)");
    const preds = addRow(matmul(x, this.w), this.b).named("preds = x·w + b");
    const loss = mnistLoss(preds, new Tensor(y, [idx.length, 1]).named("y (1 if 3)"));
    if (record) {
      this.nodes = {};
      for (const n of loss.graphNodes()) if (n.label) this.nodes[n.label] = n;
    }
    return loss;
  }

  evaluate(): number {
    const idx = this.testSubset;
    const x = this.batchX(idx, this.data.testImages);
    const preds = addRow(matmul(x, this.w), this.b);
    let correct = 0;
    for (let i = 0; i < idx.length; i++) {
      const is3 = preds.at(i, 0) > 0;
      if (is3 === (this.data.testLabels[idx[i]] === 3)) correct++;
    }
    return correct / idx.length;
  }
}

// ----------------------------------------------------------- multilabel ---

export class MultiLabel extends Scenario {
  readonly id = "multilabel" as const;
  readonly title = "Digit Properties (multi-label)";
  readonly metricName = "label accuracy";
  readonly hidden = 48;
  w1: Tensor;
  b1: Tensor;
  w2: Tensor;
  b2: Tensor;

  constructor(data: MnistData, lr = 0.5, batchSize = 64) {
    super(data);
    this.w1 = this.param("W1", Tensor.randn([784, this.hidden], Math.sqrt(2 / 784), this.rand));
    this.b1 = this.param("b1", Tensor.zeros([this.hidden]));
    this.w2 = this.param("W2", Tensor.randn([this.hidden, 3], Math.sqrt(1 / this.hidden), this.rand));
    this.b2 = this.param("b2", Tensor.zeros([3]));
    this.loader = new DataLoader(data.nTrain, batchSize, true, 13);
    this.opt = new SGD(this.params.map((p) => p.tensor), lr);
  }

  targets(idx: Int32Array, labels = this.data.trainLabels): Tensor {
    const y = new Float32Array(idx.length * 3);
    for (let i = 0; i < idx.length; i++) {
      const props = digitProperties(labels[idx[i]]);
      y[i * 3] = props[0];
      y[i * 3 + 1] = props[1];
      y[i * 3 + 2] = props[2];
    }
    return new Tensor(y, [idx.length, 3]);
  }

  logitsFor(x: Tensor): Tensor {
    const a1 = relu(addRow(matmul(x, this.w1), this.b1).named("z1 = x·W1 + b1")).named("a1 = ReLU(z1)");
    return addRow(matmul(a1, this.w2), this.b2).named("logits (one per label)");
  }

  buildLoss(batch: Int32Array, record: boolean): Tensor {
    const x = this.batchX(batch).named("x (batch images)");
    const loss = bceWithLogits(this.logitsFor(x), this.targets(batch).named("y (3 labels each)"));
    if (record) {
      this.nodes = {};
      for (const n of loss.graphNodes()) if (n.label) this.nodes[n.label] = n;
    }
    return loss;
  }

  evaluate(): number {
    const n = this.data.nTest;
    const idx = new Int32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    const x = this.batchX(idx, this.data.testImages);
    const logits = this.logitsFor(x);
    const y = this.targets(idx, this.data.testLabels);
    let correct = 0;
    for (let i = 0; i < n * 3; i++)
      if (logits.data[i] > 0 === (y.data[i] === 1)) correct++;
    return correct / (n * 3);
  }
}

// ----------------------------------------------------------- regression ---

export class Regression extends Scenario {
  readonly id = "regression" as const;
  readonly title = "Ink Center Finder (regression)";
  readonly metricName = "avg error (px)";
  readonly hidden = 48;
  w1: Tensor;
  b1: Tensor;
  w2: Tensor;
  b2: Tensor;

  constructor(data: MnistData, lr = 0.3, batchSize = 64) {
    super(data);
    this.w1 = this.param("W1", Tensor.randn([784, this.hidden], Math.sqrt(2 / 784), this.rand));
    this.b1 = this.param("b1", Tensor.zeros([this.hidden]));
    this.w2 = this.param("W2", Tensor.randn([this.hidden, 2], Math.sqrt(1 / this.hidden), this.rand));
    this.b2 = this.param("b2", Tensor.zeros([2]));
    this.loader = new DataLoader(data.nTrain, batchSize, true, 21);
    this.opt = new SGD(this.params.map((p) => p.tensor), lr);
  }

  targets(idx: Int32Array, images = this.data.trainImages): Tensor {
    const y = new Float32Array(idx.length * 2);
    for (let i = 0; i < idx.length; i++) {
      const [cx, cy] = inkCenter(images, idx[i]);
      y[i * 2] = cx;
      y[i * 2 + 1] = cy;
    }
    return new Tensor(y, [idx.length, 2]);
  }

  predsFor(x: Tensor): Tensor {
    const a1 = relu(addRow(matmul(x, this.w1), this.b1).named("z1 = x·W1 + b1")).named("a1 = ReLU(z1)");
    return sigmoid(addRow(matmul(a1, this.w2), this.b2).named("z2 = a1·W2 + b2")).named("preds (x̂, ŷ in 0..1)");
  }

  buildLoss(batch: Int32Array, record: boolean): Tensor {
    const x = this.batchX(batch).named("x (batch images)");
    const loss = mseLoss(this.predsFor(x), this.targets(batch).named("y (true centers)"));
    if (record) {
      this.nodes = {};
      for (const n of loss.graphNodes()) if (n.label) this.nodes[n.label] = n;
    }
    return loss;
  }

  /** average distance between predicted and true center, in pixels (28px image) */
  evaluate(): number {
    const n = this.data.nTest;
    const idx = new Int32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    const x = this.batchX(idx, this.data.testImages);
    const p = this.predsFor(x);
    const y = this.targets(idx, this.data.testImages);
    let total = 0;
    for (let i = 0; i < n; i++) {
      const dx = (p.at(i, 0) - y.at(i, 0)) * 27;
      const dy = (p.at(i, 1) - y.at(i, 1)) * 27;
      total += Math.hypot(dx, dy);
    }
    return total / n;
  }
}
