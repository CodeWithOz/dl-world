import { describe, expect, it } from "vitest";
import {
  Tensor,
  add,
  addRow,
  bceWithLogits,
  crossEntropy,
  gatherCols,
  matmul,
  mean,
  mnistLoss,
  mseLoss,
  mul,
  relu,
  sigmoid,
  softmax,
  sub,
} from "../src/engine/tensor";
import { SGD } from "../src/engine/optim";
import { mulberry32 } from "../src/engine/data";

/**
 * Numeric gradient check: nudge each input element by eps, recompute the
 * scalar loss, and compare (f(x+e)-f(x-e))/2e to the autograd gradient.
 * This is the same check the Gradient Lab building performs live.
 */
function gradCheck(
  makeLoss: (inputs: Tensor[]) => Tensor,
  inputs: Tensor[],
  eps = 1e-3,
  tol = 2e-2,
): void {
  for (const t of inputs) t.requiresGrad = true;
  const loss = makeLoss(inputs);
  loss.backward();
  for (const t of inputs) {
    for (let i = 0; i < t.size; i++) {
      const orig = t.data[i];
      t.data[i] = orig + eps;
      const up = makeLoss(inputs).item();
      t.data[i] = orig - eps;
      const down = makeLoss(inputs).item();
      t.data[i] = orig;
      const numeric = (up - down) / (2 * eps);
      const analytic = t.grad![i];
      const denom = Math.max(1, Math.abs(numeric), Math.abs(analytic));
      expect(
        Math.abs(numeric - analytic) / denom,
        `grad mismatch at ${i}: numeric=${numeric} analytic=${analytic}`,
      ).toBeLessThan(tol);
    }
  }
}

const rand = mulberry32(7);

describe("forward correctness", () => {
  it("matmul matches hand computation", () => {
    const a = Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = Tensor.fromArray([7, 8, 9, 10, 11, 12], [3, 2]);
    const c = matmul(a, b);
    expect(Array.from(c.data)).toEqual([58, 64, 139, 154]);
  });

  it("softmax rows sum to 1 and are stable for big logits", () => {
    const x = Tensor.fromArray([1000, 1001, 999, -5, 0, 5], [2, 3]);
    const s = softmax(x);
    for (let i = 0; i < 2; i++) {
      let total = 0;
      for (let j = 0; j < 3; j++) total += s.at(i, j);
      expect(total).toBeCloseTo(1, 5);
    }
    expect(s.at(0, 1)).toBeGreaterThan(s.at(0, 0));
  });

  it("cross-entropy of uniform logits is log(C)", () => {
    const logits = Tensor.zeros([4, 10]);
    const loss = crossEntropy(logits, new Int32Array([0, 3, 5, 9]));
    expect(loss.item()).toBeCloseTo(Math.log(10), 4);
  });

  it("mnist_loss matches where(y==1, 1-p, p).mean()", () => {
    const preds = Tensor.fromArray([2, -1, 0.5], [3, 1]);
    const y = Tensor.fromArray([1, 0, 1], [3, 1]);
    const loss = mnistLoss(preds, y);
    const sig = (v: number) => 1 / (1 + Math.exp(-v));
    const expected = ((1 - sig(2)) + sig(-1) + (1 - sig(0.5))) / 3;
    expect(loss.item()).toBeCloseTo(expected, 5);
  });
});

describe("gradients vs numeric differentiation", () => {
  it("matmul + addRow + mean", () => {
    const a = Tensor.randn([3, 4], 1, rand);
    const w = Tensor.randn([4, 2], 1, rand);
    const b = Tensor.randn([2], 1, rand);
    gradCheck((inp) => mean(addRow(matmul(inp[0], inp[1]), inp[2])), [a, w, b]);
  });

  it("relu / sigmoid chains", () => {
    const x = Tensor.randn([2, 5], 1, rand);
    gradCheck((inp) => mean(relu(inp[0])), [x]);
    const y = Tensor.randn([2, 5], 1, rand);
    gradCheck((inp) => mean(sigmoid(inp[0])), [y]);
  });

  it("elementwise add/sub/mul", () => {
    const a = Tensor.randn([2, 3], 1, rand);
    const b = Tensor.randn([2, 3], 1, rand);
    gradCheck((inp) => mean(mul(add(inp[0], inp[1]), sub(inp[0], inp[1]))), [a, b]);
  });

  it("cross-entropy through softmax/log/gather", () => {
    const logits = Tensor.randn([4, 5], 1, rand);
    const targets = new Int32Array([1, 0, 4, 2]);
    gradCheck((inp) => crossEntropy(inp[0], targets), [logits]);
  });

  it("mnist_loss", () => {
    const preds = Tensor.randn([6, 1], 1, rand);
    const y = Tensor.fromArray([1, 0, 1, 1, 0, 0], [6, 1]);
    gradCheck((inp) => mnistLoss(inp[0], y), [preds]);
  });

  it("bce with logits", () => {
    const logits = Tensor.randn([3, 4], 1, rand);
    const y = Tensor.fromArray([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], [3, 4]);
    gradCheck((inp) => bceWithLogits(inp[0], y), [logits]);
  });

  it("mse", () => {
    const preds = Tensor.randn([5, 2], 1, rand);
    const targets = Tensor.randn([5, 2], 1, rand);
    gradCheck((inp) => mseLoss(inp[0], inp[1]), [preds, targets]);
  });

  it("gatherCols routes gradients to picked elements only", () => {
    const a = Tensor.randn([3, 4], 1, rand);
    a.requiresGrad = true;
    const idx = new Int32Array([2, 0, 3]);
    const loss = mean(gatherCols(a, idx));
    loss.backward();
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 4; j++)
        expect(a.gradAt(i, j)).toBeCloseTo(j === idx[i] ? 1 / 3 : 0, 6);
  });
});

describe("end-to-end learning", () => {
  it("linear model learns a separable 2D problem with SGD", () => {
    // points above the line y=x get label 1
    const n = 64;
    const xs = new Float32Array(n * 2);
    const ys = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const x0 = rand() * 2 - 1;
      const x1 = rand() * 2 - 1;
      xs[i * 2] = x0;
      xs[i * 2 + 1] = x1;
      ys[i] = x1 > x0 ? 1 : 0;
    }
    const X = new Tensor(xs, [n, 2]);
    const Y = new Tensor(ys, [n, 1]);
    const w = Tensor.randn([2, 1], 0.1, rand);
    const b = Tensor.zeros([1]);
    w.requiresGrad = true;
    b.requiresGrad = true;
    const opt = new SGD([w, b], 1.0);
    let firstLoss = 0;
    let lastLoss = 0;
    for (let step = 0; step < 800; step++) {
      opt.zeroGrad();
      const loss = mnistLoss(addRow(matmul(X, w), b), Y);
      if (step === 0) firstLoss = loss.item();
      lastLoss = loss.item();
      loss.backward();
      opt.step();
    }
    expect(lastLoss).toBeLessThan(firstLoss * 0.3);
    expect(lastLoss).toBeLessThan(0.1);
  });
});
