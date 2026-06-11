// Trains every scenario on the real packed MNIST subset and checks they
// actually learn. Slower than the engine tests but still a few seconds.
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { MnistData } from "../src/engine/data";
import { Linear37, Mlp10, MultiLabel, Regression } from "../src/sim/scenarios";
import { pixelSimilarityBaseline } from "../src/sim/baseline";
import { runLrFinder } from "../src/sim/lrfinder";

function loadMnistFromDisk(): MnistData {
  const here = dirname(fileURLToPath(import.meta.url));
  const buf = new Uint8Array(
    readFileSync(join(here, "..", "public", "data", "mnist.bin")),
  );
  const view = new DataView(buf.buffer, buf.byteOffset);
  const nTrain = view.getUint32(4, true);
  const nTest = view.getUint32(8, true);
  let off = 12;
  const trainImages = buf.subarray(off, (off += nTrain * 784));
  const trainLabels = buf.subarray(off, (off += nTrain));
  const testImages = buf.subarray(off, (off += nTest * 784));
  const testLabels = buf.subarray(off, (off += nTest));
  return { trainImages, trainLabels, testImages, testLabels, nTrain, nTest };
}

const data = loadMnistFromDisk();

describe("scenarios learn on the real MNIST subset", () => {
  it("linear 3v7 reaches >90% in 3 epochs", () => {
    const s = new Linear37(data);
    for (let e = 0; e < 3; e++)
      for (let b = 0; b < s.loader.batchesPerEpoch; b++) s.trainStep();
    expect(s.evaluate()).toBeGreaterThan(0.9);
  });

  it("mlp10 reaches >85% in 6 epochs", () => {
    const s = new Mlp10(data);
    for (let e = 0; e < 6; e++)
      for (let b = 0; b < s.loader.batchesPerEpoch; b++) s.trainStep();
    expect(s.evaluate()).toBeGreaterThan(0.85);
  });

  it("multilabel reaches >85% label accuracy in 4 epochs", () => {
    const s = new MultiLabel(data);
    for (let e = 0; e < 4; e++)
      for (let b = 0; b < s.loader.batchesPerEpoch; b++) s.trainStep();
    expect(s.evaluate()).toBeGreaterThan(0.85);
  });

  it("regression gets average center error under 1.5px in 4 epochs", () => {
    const s = new Regression(data);
    const before = s.evaluate();
    for (let e = 0; e < 4; e++)
      for (let b = 0; b < s.loader.batchesPerEpoch; b++) s.trainStep();
    const after = s.evaluate();
    expect(after).toBeLessThan(before);
    expect(after).toBeLessThan(1.5);
  });

  it("pixel similarity baseline is decent but beatable (~90s%)", () => {
    const r = pixelSimilarityBaseline(data);
    expect(r.accuracyL1).toBeGreaterThan(0.8);
    expect(r.accuracyL2).toBeGreaterThan(0.8);
    expect(r.accuracyL2).toBeLessThan(1.0);
  });

  it("lr finder produces a curve and a sane suggestion", () => {
    const r = runLrFinder(data);
    expect(r.points.length).toBeGreaterThan(20);
    expect(r.suggestedSteepest).toBeGreaterThan(1e-4);
    expect(r.suggestedSteepest).toBeLessThan(12);
  });
});
