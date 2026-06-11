// Ch.5's learning rate finder: train a *fresh* model while exponentially
// increasing the LR each step; plot loss vs LR; the good zone is on the
// steep downward slope, well before the explosion.

import { MnistData } from "../engine/data";
import { Mlp10 } from "./scenarios";

export interface LrPoint {
  lr: number;
  loss: number;
  smoothed: number;
}

export interface LrFinderResult {
  points: LrPoint[];
  /** lr at the steepest downward slope of the smoothed curve */
  suggestedSteepest: number;
  /** (lr at minimum loss) / 10, fastai's other rule of thumb */
  suggestedMinOver10: number;
}

export function runLrFinder(data: MnistData): LrFinderResult {
  const model = new Mlp10(data, 1e-4, 64);
  const points: LrPoint[] = [];
  let lr = 1e-4;
  let smoothed = NaN;
  let best = Infinity;
  const beta = 0.75;
  while (lr < 12) {
    model.opt.lr = lr;
    const loss = model.trainStep();
    smoothed = isNaN(smoothed) ? loss : beta * smoothed + (1 - beta) * loss;
    points.push({ lr, loss, smoothed });
    best = Math.min(best, smoothed);
    if (smoothed > 4 * best && points.length > 12) break; // diverged
    lr *= 1.18;
  }
  let steepest = points[0]?.lr ?? 1e-3;
  let bestSlope = 0;
  let minLr = points[0]?.lr ?? 1e-3;
  let minLoss = Infinity;
  for (let i = 1; i < points.length; i++) {
    const slope =
      (points[i].smoothed - points[i - 1].smoothed) /
      (Math.log(points[i].lr) - Math.log(points[i - 1].lr));
    if (slope < bestSlope) {
      bestSlope = slope;
      steepest = points[i].lr;
    }
    if (points[i].smoothed < minLoss) {
      minLoss = points[i].smoothed;
      minLr = points[i].lr;
    }
  }
  return {
    points,
    suggestedSteepest: steepest,
    suggestedMinOver10: minLr / 10,
  };
}
