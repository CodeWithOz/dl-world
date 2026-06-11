import { Tensor } from "./tensor";

/** Plain SGD, exactly the ch.4 step: p.data -= lr * p.grad */
export class SGD {
  params: Tensor[];
  lr: number;
  /** record of the last step, so the Optimizer Depot can show real updates */
  lastStep: { param: Tensor; before: Float32Array; grad: Float32Array }[] = [];

  constructor(params: Tensor[], lr: number) {
    this.params = params;
    this.lr = lr;
  }

  step(): void {
    this.lastStep = [];
    for (const p of this.params) {
      if (!p.grad) continue;
      this.lastStep.push({
        param: p,
        before: p.data.slice(0, Math.min(p.data.length, 4096)),
        grad: p.grad.slice(0, Math.min(p.grad.length, 4096)),
      });
      for (let i = 0; i < p.data.length; i++) p.data[i] -= this.lr * p.grad[i];
    }
  }

  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad();
  }
}
