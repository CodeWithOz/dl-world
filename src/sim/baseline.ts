// Ch.4's pre-deep-learning baseline: average all the 3s into an "ideal 3",
// average all the 7s into an "ideal 7", classify by pixel distance.

import { MnistData } from "../engine/data";

export interface BaselineResult {
  mean3: Float32Array; // 784, in 0..1
  mean7: Float32Array;
  accuracyL1: number;
  accuracyL2: number;
}

export function distL1(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s / a.length;
}

export function distL2(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
  return Math.sqrt(s / a.length);
}

let cached: BaselineResult | null = null;

export function pixelSimilarityBaseline(data: MnistData): BaselineResult {
  if (cached) return cached;
  const mean3 = new Float32Array(784);
  const mean7 = new Float32Array(784);
  let n3 = 0,
    n7 = 0;
  for (let i = 0; i < data.nTrain; i++) {
    const label = data.trainLabels[i];
    if (label !== 3 && label !== 7) continue;
    const target = label === 3 ? mean3 : mean7;
    const off = i * 784;
    for (let j = 0; j < 784; j++) target[j] += data.trainImages[off + j] / 255;
    if (label === 3) n3++;
    else n7++;
  }
  for (let j = 0; j < 784; j++) {
    mean3[j] /= n3;
    mean7[j] /= n7;
  }
  let okL1 = 0,
    okL2 = 0,
    total = 0;
  const img = new Float32Array(784);
  for (let i = 0; i < data.nTest; i++) {
    const label = data.testLabels[i];
    if (label !== 3 && label !== 7) continue;
    total++;
    const off = i * 784;
    for (let j = 0; j < 784; j++) img[j] = data.testImages[off + j] / 255;
    if (distL1(img, mean3) < distL1(img, mean7) === (label === 3)) okL1++;
    if (distL2(img, mean3) < distL2(img, mean7) === (label === 3)) okL2++;
  }
  cached = { mean3, mean7, accuracyL1: okL1 / total, accuracyL2: okL2 / total };
  return cached;
}
