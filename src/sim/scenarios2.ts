// The frontier's live training scenarios, mapped to the later chapters of
// the curriculum (the chapter mapping itself lives only in README.md):
//   refinery  - training tricks: input normalization, label smoothing,
//               mixup — racing a plain twin trained on the same batches
//   collab    - collaborative filtering: user/movie embeddings learned
//               from star ratings alone
//   sentiment - text classification: bag-of-words logistic regression,
//               one learned weight per vocabulary word
//   rnnlm     - a language model from scratch: read 3 words, predict the
//               4th, with a shared hidden state (an unrolled RNN)
//
// Embeddings are deliberately built as one-hot × matrix products: the
// recorded graph then *shows* that "an embedding layer is just a matmul
// with a one-hot vector" — the array-lookup is only the fast version.

import {
  Tensor,
  add,
  addRow,
  addScalar,
  bceWithLogits,
  crossEntropy,
  gatherCols,
  log,
  matmul,
  mean,
  mseLoss,
  mul,
  neg,
  relu,
  scale,
  sigmoid,
  softmax,
  square,
  sum,
  sumRows,
} from "../engine/tensor";
import { SGD } from "../engine/optim";
import { DataLoader, MnistData, mulberry32 } from "../engine/data";
import { Scenario } from "./scenarios";
import type { HistoryPoint } from "./scenarios";
import {
  CollabData,
  HumanNumbers,
  TextData,
  makeCollabData,
  makeHumanNumbers,
  makeTextData,
  numericalize,
  tokenize,
} from "./datasets";

/** [rows, width] one-hot matrix as a constant tensor (no grad needed) */
function oneHot(ids: ArrayLike<number>, width: number): Tensor {
  const n = ids.length;
  const data = new Float32Array(n * width);
  for (let i = 0; i < n; i++) data[i * width + ids[i]] = 1;
  return new Tensor(data, [n, width]);
}

// ------------------------------------------------------------- refinery ---

/**
 * The same 2-layer digit classifier as the main pipeline, plus the tricks —
 * and a "plain twin" with identical init, trained on the identical batches
 * with no tricks, so the comparison curves are a real controlled experiment.
 */
export class Refinery extends Scenario {
  readonly id = "refinery" as const;
  readonly title = "Refined Digit Classifier (tricks on)";
  readonly metricName = "test accuracy";
  readonly hidden = 48;

  // the tricks; panels flip these live (state lives here, not in the DOM)
  useNorm = true;
  smoothEps = 0.1;
  useMixup = false;
  /** λ used by the most recent mixup step (1 = mixup off) */
  lastMixLam = 1;

  /** train-set pixel statistics, the real numbers behind Normalize() */
  pixMean: number;
  pixStd: number;

  w1: Tensor;
  b1: Tensor;
  w2: Tensor;
  b2: Tensor;
  // the plain twin (same init — data arrays are copied at construction)
  pw1: Tensor;
  pb1: Tensor;
  pw2: Tensor;
  pb2: Tensor;
  plainOpt: SGD;
  plainLossHistory: HistoryPoint[] = [];
  plainMetricHistory: HistoryPoint[] = [];
  lastPlainLoss = NaN;

  constructor(data: MnistData, lr = 0.5, batchSize = 48) {
    super(data);
    this.w1 = this.param("W1", Tensor.randn([784, this.hidden], Math.sqrt(2 / 784), this.rand));
    this.b1 = this.param("b1", Tensor.zeros([this.hidden]));
    this.w2 = this.param("W2", Tensor.randn([this.hidden, 10], Math.sqrt(1 / this.hidden), this.rand));
    this.b2 = this.param("b2", Tensor.zeros([10]));
    const clone = (t: Tensor) => {
      const c = new Tensor(t.data.slice(), t.shape.slice());
      c.requiresGrad = true;
      return c;
    };
    this.pw1 = clone(this.w1);
    this.pb1 = clone(this.b1);
    this.pw2 = clone(this.w2);
    this.pb2 = clone(this.b2);
    this.loader = new DataLoader(data.nTrain, batchSize, true, 77);
    this.opt = new SGD(this.params.map((p) => p.tensor), lr);
    this.plainOpt = new SGD([this.pw1, this.pb1, this.pw2, this.pb2], lr);
    // dataset-wide mean/std — computed once, from the real training pixels
    let s = 0;
    const n = data.nTrain * 784;
    for (let i = 0; i < n; i++) s += data.trainImages[i];
    this.pixMean = s / n / 255;
    let q = 0;
    for (let i = 0; i < n; i++) {
      const d = data.trainImages[i] / 255 - this.pixMean;
      q += d * d;
    }
    this.pixStd = Math.sqrt(q / n);
  }

  targets(idx: Int32Array): Int32Array {
    const t = new Int32Array(idx.length);
    for (let i = 0; i < idx.length; i++) t[i] = this.data.trainLabels[idx[i]];
    return t;
  }

  private normalized(x: Tensor): Tensor {
    return scale(addScalar(x, -this.pixMean).named("x − mean"), 1 / this.pixStd).named(
      "x (normalized)",
    );
  }

  private logitsFor(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor): Tensor {
    const a1 = relu(addRow(matmul(x, w1), b1).named("z1 = x·W1 + b1")).named("a1 = ReLU(z1)");
    return addRow(matmul(a1, w2), b2).named("logits");
  }

  /**
   * cross-entropy with optional label smoothing, composed from primitives:
   * (1−ε)·NLL(target) + (ε/C)·Σ_classes(−log p) — for ε=0 this is plain CE.
   */
  private smoothedCE(logits: Tensor, t: Int32Array, suffix = ""): Tensor {
    const p = softmax(logits);
    const logp = log(p).named(`log probs${suffix}`);
    const nll = mean(neg(gatherCols(logp, t).named(`log p[target]${suffix}`))).named(`nll${suffix}`);
    if (this.smoothEps === 0) return nll;
    const uniform = mean(neg(sumRows(logp))).named(`Σ −log p (all classes)${suffix}`);
    return add(scale(nll, 1 - this.smoothEps), scale(uniform, this.smoothEps / 10)).named(
      `smoothed loss${suffix}`,
    );
  }

  buildLoss(batch: Int32Array, record: boolean): Tensor {
    let x = this.batchX(batch).named("x (raw pixels)");
    if (this.useNorm) x = this.normalized(x);
    const y1 = this.targets(batch);
    let loss: Tensor;
    if (this.useMixup && this.lastMixLam < 1) {
      // mixup pairs each image with the batch reversed; the loss is the
      // same λ-blend as the inputs, so the graph carries both targets
      const partner = Int32Array.from(batch).reverse();
      let x2 = this.batchX(partner);
      if (this.useNorm) x2 = scale(addScalar(x2, -this.pixMean), 1 / this.pixStd);
      const lam = this.lastMixLam;
      const mixed = add(scale(x, lam).named("λ·x"), scale(x2, 1 - lam).named("(1−λ)·x'")).named(
        "mixed input",
      );
      const logits = this.logitsFor(mixed, this.w1, this.b1, this.w2, this.b2);
      const y2 = this.targets(partner);
      loss = add(
        scale(this.smoothedCE(logits, y1), lam),
        scale(this.smoothedCE(logits, y2, " (partner)"), 1 - lam),
      ).named("loss");
    } else {
      const logits = this.logitsFor(x, this.w1, this.b1, this.w2, this.b2);
      loss = this.smoothedCE(logits, y1).named("loss");
    }
    if (record) {
      this.nodes = {};
      for (const n of loss.graphNodes()) if (n.label) this.nodes[n.label] = n;
    }
    return loss;
  }

  /** the twin's loss: same batch, no normalization, no smoothing, no mixup */
  private plainLoss(batch: Int32Array): Tensor {
    const x = this.batchX(batch);
    const a1 = relu(addRow(matmul(x, this.pw1), this.pb1));
    const logits = addRow(matmul(a1, this.pw2), this.pb2);
    return crossEntropy(logits, this.targets(batch));
  }

  trainStep(): number {
    const batch = this.loader.next();
    this.lastBatch = batch;
    // the book's mixup pseudocode draws the weight from 0.5..1.0
    this.lastMixLam = this.useMixup ? 0.5 + 0.5 * this.rand() : 1;
    // tuned model
    this.opt.zeroGrad();
    const loss = this.buildLoss(batch, false);
    loss.backward();
    this.opt.step();
    // plain twin: identical batch, identical lr — the control group
    this.plainOpt.zeroGrad();
    const ploss = this.plainLoss(batch);
    ploss.backward();
    this.plainOpt.step();
    this.lastPlainLoss = ploss.item();
    this.step++;
    const l = loss.item();
    this.lossHistory.push({ step: this.step, value: l });
    this.plainLossHistory.push({ step: this.step, value: this.lastPlainLoss });
    if (this.lossHistory.length > 5000) {
      this.lossHistory.splice(0, 1000);
      this.plainLossHistory.splice(0, 1000);
    }
    // display pass with the updated weights (see Scenario.trainStep)
    this.opt.zeroGrad();
    this.nodes = {};
    const display = this.buildLoss(batch, true);
    this.lastLoss = display;
    display.backward();
    return l;
  }

  recordMetric(): number {
    const m = super.recordMetric();
    this.plainMetricHistory.push({ step: this.step, value: this.accuracy(false) });
    return m;
  }

  /** test accuracy of either model (the tuned one normalizes if enabled) */
  accuracy(tuned: boolean): number {
    const [w1, b1, w2, b2] = tuned
      ? [this.w1, this.b1, this.w2, this.b2]
      : [this.pw1, this.pb1, this.pw2, this.pb2];
    const n = this.data.nTest;
    let correct = 0;
    const CHUNK = 200;
    for (let start = 0; start < n; start += CHUNK) {
      const m = Math.min(CHUNK, n - start);
      const idx = new Int32Array(m);
      for (let i = 0; i < m; i++) idx[i] = start + i;
      let x = this.batchX(idx, this.data.testImages);
      if (tuned && this.useNorm) x = scale(addScalar(x, -this.pixMean), 1 / this.pixStd);
      const a1 = relu(addRow(matmul(x, w1), b1));
      const logits = addRow(matmul(a1, w2), b2);
      for (let i = 0; i < m; i++) {
        let best = 0;
        for (let j = 1; j < 10; j++) if (logits.at(i, j) > logits.at(i, best)) best = j;
        if (best === this.data.testLabels[start + i]) correct++;
      }
    }
    return correct / n;
  }

  evaluate(): number {
    return this.accuracy(true);
  }

  /** one test image shifted by (dx, dy) pixels, zero-filled at the edges */
  private shiftedImage(i: number, dx: number, dy: number): Float32Array {
    const out = new Float32Array(784);
    const imgs = this.data.testImages;
    for (let r = 0; r < 28; r++)
      for (let c = 0; c < 28; c++) {
        const sr = r - dy;
        const sc = c - dx;
        if (sr < 0 || sr >= 28 || sc < 0 || sc >= 28) continue;
        out[r * 28 + c] = imgs[i * 784 + sr * 28 + sc] / 255;
      }
    return out;
  }

  /**
   * Test-time augmentation, the book's recipe scaled to digits: predict on
   * the original image plus four 1-pixel shifts, then average the softmax
   * probabilities. No training involved — same weights, better answers.
   */
  ttaAccuracy(): { plain: number; tta: number; views: number } {
    const shifts: [number, number][] = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]];
    const n = this.data.nTest;
    let plainCorrect = 0;
    let ttaCorrect = 0;
    const CHUNK = 100;
    for (let start = 0; start < n; start += CHUNK) {
      const m = Math.min(CHUNK, n - start);
      const summed = new Float32Array(m * 10);
      let plainProbs: Float32Array | null = null;
      for (const [dx, dy] of shifts) {
        const xd = new Float32Array(m * 784);
        for (let i = 0; i < m; i++) xd.set(this.shiftedImage(start + i, dx, dy), i * 784);
        let x = new Tensor(xd, [m, 784]);
        if (this.useNorm) x = scale(addScalar(x, -this.pixMean), 1 / this.pixStd);
        const a1 = relu(addRow(matmul(x, this.w1), this.b1));
        const probs = softmax(addRow(matmul(a1, this.w2), this.b2));
        for (let i = 0; i < m * 10; i++) summed[i] += probs.data[i];
        if (dx === 0 && dy === 0) plainProbs = probs.data;
      }
      for (let i = 0; i < m; i++) {
        let bestPlain = 0;
        let bestTta = 0;
        for (let j = 1; j < 10; j++) {
          if (plainProbs![i * 10 + j] > plainProbs![i * 10 + bestPlain]) bestPlain = j;
          if (summed[i * 10 + j] > summed[i * 10 + bestTta]) bestTta = j;
        }
        const truth = this.data.testLabels[start + i];
        if (bestPlain === truth) plainCorrect++;
        if (bestTta === truth) ttaCorrect++;
      }
    }
    return { plain: plainCorrect / n, tta: ttaCorrect / n, views: shifts.length };
  }
}

// --------------------------------------------------------------- collab ---

/** Collaborative filtering on the planted ratings: factors + biases. */
export class CollabFilter extends Scenario {
  readonly id = "collab" as const;
  readonly title = "Movie Taste Learner (collaborative filtering)";
  readonly metricName = "test RMSE (stars)";
  readonly k = 3;
  cd: CollabData;
  U: Tensor;
  M: Tensor;
  Ub: Tensor;
  Mb: Tensor;

  constructor(data: MnistData, lr = 1.2, batchSize = 32) {
    super(data);
    this.cd = makeCollabData();
    const nU = this.cd.users.length;
    const nM = this.cd.movies.length;
    this.U = this.param("user factors", Tensor.randn([nU, this.k], 0.25, this.rand));
    this.M = this.param("movie factors", Tensor.randn([nM, this.k], 0.25, this.rand));
    this.Ub = this.param("user biases", Tensor.zeros([nU, 1]));
    this.Mb = this.param("movie biases", Tensor.zeros([nM, 1]));
    this.loader = new DataLoader(this.cd.train.length, batchSize, true, 31);
    this.opt = new SGD(this.params.map((p) => p.tensor), lr);
  }

  /** weight decay — the regularizer the book reaches for when the dot
   *  product model starts overfitting: loss + wd·Σ(params²) */
  wd = 0.01;

  /** the model: rating ≈ sigmoid(u·m + b_u + b_m) stretched onto 0..5.5 —
   *  the range runs a little past 5 so a perfect score stays reachable */
  predFor(uIds: number[], mIds: number[]): Tensor {
    const uoh = oneHot(uIds, this.cd.users.length).named("user one-hots");
    const moh = oneHot(mIds, this.cd.movies.length).named("movie one-hots");
    const uf = matmul(uoh, this.U).named("picked user factors");
    const mf = matmul(moh, this.M).named("picked movie factors");
    const dot = sumRows(mul(uf, mf).named("factor × factor")).named("dot product");
    const ub = matmul(uoh, this.Ub).named("user bias");
    const mb = matmul(moh, this.Mb).named("movie bias");
    const raw = add(add(dot, ub), mb).named("score = dot + biases");
    return scale(sigmoid(raw).named("sigmoid(score)"), 5.5).named("pred rating (0..5.5)");
  }

  buildLoss(batch: Int32Array, record: boolean): Tensor {
    const us: number[] = [];
    const ms: number[] = [];
    const y = new Float32Array(batch.length);
    for (let i = 0; i < batch.length; i++) {
      const r = this.cd.train[batch[i]];
      us.push(r.u);
      ms.push(r.m);
      y[i] = r.r;
    }
    const pred = this.predFor(us, ms);
    const mse = mseLoss(pred, new Tensor(y, [batch.length, 1]).named("true rating")).named(
      "MSE (data loss)",
    );
    // weight decay on the factors, composed from primitives (the biases are
    // left free — they have honest work to do)
    const penalty = scale(
      add(sum(square(this.U)), sum(square(this.M))),
      this.wd,
    ).named("λ·Σfactors² (weight decay)");
    const loss = add(mse, penalty).named("loss");
    if (record) {
      this.nodes = {};
      for (const n of loss.graphNodes()) if (n.label) this.nodes[n.label] = n;
    }
    return loss;
  }

  /** root-mean-square error on the held-out ratings, in stars */
  evaluate(): number {
    const pred = this.predFor(
      this.cd.test.map((r) => r.u),
      this.cd.test.map((r) => r.m),
    );
    let q = 0;
    for (let i = 0; i < this.cd.test.length; i++) {
      const d = pred.data[i] - this.cd.test[i].r;
      q += d * d;
    }
    return Math.sqrt(q / this.cd.test.length);
  }

  /** one (user, movie) forward, with every intermediate value for the desk */
  inspect(u: number, m: number): {
    uf: number[];
    mf: number[];
    products: number[];
    dot: number;
    ub: number;
    mb: number;
    pred: number;
    known: number | null;
  } {
    const uf = [...Array(this.k)].map((_, j) => this.U.at(u, j));
    const mf = [...Array(this.k)].map((_, j) => this.M.at(m, j));
    const products = uf.map((v, j) => v * mf[j]);
    const dot = products.reduce((a, b) => a + b, 0);
    const ub = this.Ub.at(u, 0);
    const mb = this.Mb.at(m, 0);
    const raw = dot + ub + mb;
    const pred = (1 / (1 + Math.exp(-raw))) * 5.5;
    const found =
      this.cd.train.find((r) => r.u === u && r.m === m) ??
      this.cd.test.find((r) => r.u === u && r.m === m);
    return { uf, mf, products, dot, ub, mb, pred, known: found ? found.r : null };
  }
}

// ------------------------------------------------------------ sentiment ---

/** Bag-of-words sentiment: one learned weight per vocabulary word. */
export class SentimentNet extends Scenario {
  readonly id = "sentiment" as const;
  readonly title = "Review Sentiment (bag of words)";
  readonly metricName = "test accuracy";
  /** weight decay — with ~70 reviews and a weight per word, the model
   *  memorizes stopword noise without it (watch the Word-Weight Wall) */
  wd = 0.05;
  td: TextData;
  w: Tensor;
  b: Tensor;
  /** cached numericalized train/test token ids */
  private trainIds: number[][];
  private testIds: number[][];

  constructor(data: MnistData, lr = 0.5, batchSize = 16) {
    super(data);
    this.td = makeTextData();
    const V = this.td.vocab.length;
    this.w = this.param("w (one per word)", Tensor.randn([V, 1], 0.01, this.rand));
    this.b = this.param("b (bias)", Tensor.zeros([1]));
    this.trainIds = this.td.train.map((r) => numericalize(tokenize(r.text), this.td.vocab));
    this.testIds = this.td.test.map((r) => numericalize(tokenize(r.text), this.td.vocab));
    this.loader = new DataLoader(this.td.train.length, batchSize, true, 11);
    this.opt = new SGD(this.params.map((p) => p.tensor), lr);
  }

  /** bag-of-words features: how often each vocab word appears in the review */
  private featRows(idsList: number[][]): Float32Array {
    const V = this.td.vocab.length;
    const x = new Float32Array(idsList.length * V);
    idsList.forEach((ids, i) => {
      for (const id of ids) x[i * V + id] += 1;
    });
    return x;
  }

  buildLoss(batch: Int32Array, record: boolean): Tensor {
    const idsList = [...batch].map((i) => this.trainIds[i]);
    const y = new Float32Array(batch.length);
    for (let i = 0; i < batch.length; i++) y[i] = this.td.train[batch[i]].label;
    const x = new Tensor(this.featRows(idsList), [batch.length, this.td.vocab.length]).named(
      "x (word counts)",
    );
    const logit = addRow(matmul(x, this.w).named("x·w"), this.b).named("logit (one per review)");
    const bce = bceWithLogits(logit, new Tensor(y, [batch.length, 1]).named("y (1 = positive)")).named(
      "BCE (data loss)",
    );
    // weight decay, composed from primitives so it shows up in the graph
    const penalty = scale(sum(square(this.w).named("w²")), this.wd).named("λ·Σw² (weight decay)");
    const loss = add(bce, penalty).named("loss");
    if (record) {
      this.nodes = {};
      for (const n of loss.graphNodes()) if (n.label) this.nodes[n.label] = n;
    }
    return loss;
  }

  evaluate(): number {
    const x = new Tensor(this.featRows(this.testIds), [this.testIds.length, this.td.vocab.length]);
    const logit = addRow(matmul(x, this.w), this.b);
    let correct = 0;
    for (let i = 0; i < this.testIds.length; i++)
      if (logit.data[i] > 0 === (this.td.test[i].label === 1)) correct++;
    return correct / this.testIds.length;
  }

  /** classify any text and explain it: per-word contribution to the logit */
  classify(text: string): {
    tokens: string[];
    ids: number[];
    p: number;
    logit: number;
    contribs: { token: string; known: boolean; contrib: number }[];
  } {
    const tokens = tokenize(text);
    const ids = numericalize(tokens, this.td.vocab);
    let logit = this.b.data[0];
    const contribs = tokens.map((token, i) => {
      const contrib = this.w.at(ids[i], 0);
      logit += contrib;
      return { token, known: !(ids[i] === 0 && token !== "xxunk"), contrib };
    });
    return { tokens, ids, p: 1 / (1 + Math.exp(-logit)), logit, contribs };
  }
}

// ---------------------------------------------------------------- rnnlm ---

/** Language model from scratch: 3 words in, the 4th word out, one shared
 *  hidden state threaded through the sequence (an unrolled RNN). */
export class RnnLm extends Scenario {
  readonly id = "rnnlm" as const;
  readonly title = "Counting Language Model (RNN)";
  readonly metricName = "next-word accuracy";
  readonly seqLen = 3;
  readonly dim = 24;
  hn: HumanNumbers;
  E: Tensor;
  Wh: Tensor;
  bh: Tensor;
  Wo: Tensor;
  bo: Tensor;
  /** start offsets of the non-overlapping (3 in, 1 out) training samples */
  private starts: number[];
  private nTrainSamples: number;

  constructor(data: MnistData, lr = 0.4, batchSize = 64) {
    super(data);
    this.hn = makeHumanNumbers(2000);
    const V = this.hn.vocab.length;
    this.E = this.param("E (embeddings)", Tensor.randn([V, this.dim], 0.3, this.rand));
    this.Wh = this.param("Wh (hidden→hidden)", Tensor.randn([this.dim, this.dim], Math.sqrt(1 / this.dim), this.rand));
    this.bh = this.param("bh", Tensor.zeros([this.dim]));
    this.Wo = this.param("Wo (hidden→vocab)", Tensor.randn([this.dim, V], Math.sqrt(1 / this.dim), this.rand));
    this.bo = this.param("bo", Tensor.zeros([V]));
    this.starts = [];
    for (let i = 0; i + this.seqLen < this.hn.ids.length; i += this.seqLen) this.starts.push(i);
    // sequential split, like a real LM: the model is asked to continue
    // parts of the count it has never read
    this.nTrainSamples = Math.floor(this.starts.length * 0.85);
    this.loader = new DataLoader(this.nTrainSamples, batchSize, true, 5);
    this.opt = new SGD(this.params.map((p) => p.tensor), lr);
  }

  get vocab(): string[] {
    return this.hn.vocab;
  }

  sampleAt(s: number): { ctx: number[]; target: number } {
    const start = this.starts[s];
    return {
      ctx: [...this.hn.ids.subarray(start, start + this.seqLen)],
      target: this.hn.ids[start + this.seqLen],
    };
  }

  get validSamples(): number[] {
    const v: number[] = [];
    for (let s = this.nTrainSamples; s < this.starts.length; s++) v.push(s);
    return v;
  }

  /** the unrolled loop: h ← ReLU((h + emb(word_t))·Wh + bh), then read out */
  private logitsFor(ctxRows: number[][], record: boolean): Tensor {
    const V = this.hn.vocab.length;
    let h: Tensor | null = null;
    for (let t = 0; t < this.seqLen; t++) {
      const ids = ctxRows.map((c) => c[t]);
      const oh = oneHot(ids, V).named(record ? `word ${t + 1} (one-hot)` : "");
      const emb = matmul(oh, this.E).named(record ? `embedding of word ${t + 1}` : "");
      const pre: Tensor = h ? add(h, emb).named(record ? `h + embedding ${t + 1}` : "") : emb;
      h = relu(addRow(matmul(pre, this.Wh), this.bh)).named(
        record ? `hidden state after word ${t + 1}` : "",
      );
    }
    return addRow(matmul(h!, this.Wo), this.bo).named(record ? "next-word logits" : "");
  }

  buildLoss(batch: Int32Array, record: boolean): Tensor {
    const ctxRows: number[][] = [];
    const targets = new Int32Array(batch.length);
    for (let i = 0; i < batch.length; i++) {
      const { ctx, target } = this.sampleAt(batch[i]);
      ctxRows.push(ctx);
      targets[i] = target;
    }
    const loss = crossEntropy(this.logitsFor(ctxRows, record), targets);
    if (record) {
      this.nodes = {};
      for (const n of loss.graphNodes()) if (n.label) this.nodes[n.label] = n;
    }
    return loss;
  }

  /** next-word accuracy on the held-out tail of the count */
  evaluate(): number {
    const samples = this.validSamples;
    let correct = 0;
    const CHUNK = 256;
    for (let start = 0; start < samples.length; start += CHUNK) {
      const part = samples.slice(start, start + CHUNK);
      const ctxRows = part.map((s) => this.sampleAt(s).ctx);
      const logits = this.logitsFor(ctxRows, false);
      const V = this.hn.vocab.length;
      part.forEach((s, i) => {
        let best = 0;
        for (let j = 1; j < V; j++) if (logits.data[i * V + j] > logits.data[i * V + best]) best = j;
        if (best === this.sampleAt(s).target) correct++;
      });
    }
    return correct / samples.length;
  }

  /** what would "always guess the most common token" score? (the baseline) */
  baseline(): { token: string; acc: number } {
    const V = this.hn.vocab.length;
    const counts = new Array(V).fill(0);
    for (const id of this.hn.ids) counts[id]++;
    let best = 0;
    for (let j = 1; j < V; j++) if (counts[j] > counts[best]) best = j;
    let hits = 0;
    const samples = this.validSamples;
    for (const s of samples) if (this.sampleAt(s).target === best) hits++;
    return { token: this.hn.vocab[best], acc: hits / samples.length };
  }

  /** probabilities for the next word after a 3-token context */
  predictNext(ctx: number[]): Float32Array {
    const logits = this.logitsFor([ctx], false);
    return softmax(logits).data;
  }

  /** continue the count from a context; temp 0 = greedy argmax */
  generate(ctx: number[], n: number, temp = 0, seed = 1234): string[] {
    const rand = mulberry32(seed);
    const out: string[] = [];
    let cur = [...ctx];
    for (let i = 0; i < n; i++) {
      const probs = this.predictNext(cur);
      let pick = 0;
      if (temp <= 0) {
        for (let j = 1; j < probs.length; j++) if (probs[j] > probs[pick]) pick = j;
      } else {
        // temperature sampling over the real distribution
        const weights = [...probs].map((p) => Math.pow(p, 1 / temp));
        const total = weights.reduce((a, b) => a + b, 0);
        let r = rand() * total;
        for (let j = 0; j < weights.length; j++) {
          r -= weights[j];
          if (r <= 0) {
            pick = j;
            break;
          }
        }
      }
      out.push(this.hn.vocab[pick]);
      cur = [cur[1], cur[2], pick];
    }
    return out;
  }
}
