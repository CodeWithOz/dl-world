// The frontier's scenarios and data: collaborative filtering, sentiment,
// the RNN language model, the refinery twins, the rents decision tree.
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { MnistData } from "../src/engine/data";
import { CollabFilter, Refinery, RnnLm, SentimentNet } from "../src/sim/scenarios2";
import {
  makeCollabData,
  makeHumanNumbers,
  makeTabularData,
  makeTextData,
  numberToWords,
  numericalize,
  tokenize,
} from "../src/sim/datasets";
import {
  featureImportance,
  fitForest,
  fitTree,
  maeOn,
  predictForest,
  predictTree,
  splitCandidates,
} from "../src/sim/tabular";

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

describe("human numbers generator", () => {
  it("writes numbers the fastai way", () => {
    expect(numberToWords(7)).toEqual(["seven"]);
    expect(numberToWords(15)).toEqual(["fifteen"]);
    expect(numberToWords(42)).toEqual(["forty", "two"]);
    expect(numberToWords(110)).toEqual(["one", "hundred", "ten"]);
    expect(numberToWords(1234)).toEqual(["one", "thousand", "two", "hundred", "thirty", "four"]);
  });

  it("builds a small closed vocabulary", () => {
    const hn = makeHumanNumbers(2000);
    expect(hn.vocab.length).toBeLessThan(40);
    expect(hn.ids.length).toBe(hn.tokens.length);
    // round-trip: ids decode back to the tokens
    for (let i = 0; i < 200; i++) expect(hn.vocab[hn.ids[i]]).toBe(hn.tokens[i]);
  });
});

describe("text data", () => {
  it("tokenizes with xxbos and numericalizes unknowns to 0", () => {
    const td = makeTextData();
    const toks = tokenize("The UNHEARDOFWORD was stunning");
    expect(toks[0]).toBe("xxbos");
    const ids = numericalize(toks, td.vocab);
    expect(ids[0]).toBe(1); // xxbos
    expect(ids[1]).toBeGreaterThan(1); // "the" is in the vocab
    expect(ids[2]).toBe(0); // unknown word -> xxunk
    expect(td.vocab[0]).toBe("xxunk");
  });
});

describe("frontier scenarios learn", () => {
  it("collab filtering: test RMSE improves and beats 0.8 stars", () => {
    const s = new CollabFilter(data);
    const before = s.evaluate();
    for (let e = 0; e < 30; e++)
      for (let b = 0; b < s.loader.batchesPerEpoch; b++) s.trainStep();
    const after = s.evaluate();
    expect(after).toBeLessThan(before);
    expect(after).toBeLessThan(0.8);
  });

  it("sentiment: held-out accuracy beats 80% quickly", () => {
    const s = new SentimentNet(data);
    for (let e = 0; e < 25; e++)
      for (let b = 0; b < s.loader.batchesPerEpoch; b++) s.trainStep();
    expect(s.evaluate()).toBeGreaterThan(0.8);
    const pos = s.classify("stunning acting and gorgeous direction");
    const neg = s.classify("a boring film with sloppy pacing");
    expect(pos.p).toBeGreaterThan(0.5);
    expect(neg.p).toBeLessThan(0.5);
  });

  it("rnn lm: beats the most-common-token baseline on the held-out tail", () => {
    const s = new RnnLm(data);
    for (let e = 0; e < 4; e++)
      for (let b = 0; b < s.loader.batchesPerEpoch; b++) s.trainStep();
    const acc = s.evaluate();
    expect(acc).toBeGreaterThan(s.baseline().acc);
    expect(acc).toBeGreaterThan(0.3);
    // generation runs and emits known vocab tokens
    const gen = s.generate([...s.hn.ids.subarray(0, 3)], 10);
    expect(gen.length).toBe(10);
    for (const t of gen) expect(s.hn.vocab).toContain(t);
  });

  it("refinery: both twins learn, and the display graph carries the tricks", () => {
    const s = new Refinery(data);
    s.useMixup = true;
    for (let i = 0; i < 150; i++) s.trainStep();
    // mixup + smoothing slow the early steps — that's expected; both nets
    // must still clearly beat the 10% random baseline
    expect(s.accuracy(true)).toBeGreaterThan(0.45);
    expect(s.accuracy(false)).toBeGreaterThan(0.6);
    // labels the gym panels rely on
    expect(s.nodes["x (normalized)"]).toBeDefined();
    expect(s.nodes["mixed input"]).toBeDefined();
    expect(s.nodes["smoothed loss"]).toBeDefined();
    expect(s.lossHistory.length).toBe(s.plainLossHistory.length);
    // TTA runs over all 5 views and stays in sane accuracy territory
    const tta = s.ttaAccuracy();
    expect(tta.views).toBe(5);
    expect(tta.tta).toBeGreaterThan(0.3);
    expect(Math.abs(tta.tta - tta.plain)).toBeLessThan(0.2);
  });
});

describe("decision trees on the rents table", () => {
  const td = makeTabularData();
  const { rows: X, rent: y, trainIdx, validIdx } = td;

  it("the best first split has positive gain on every splittable feature", () => {
    const cands = splitCandidates(X, y, trainIdx);
    const gains = cands.filter((c) => c !== null).map((c) => c!.gain);
    expect(gains.length).toBeGreaterThan(3);
    for (const g of gains) expect(g).toBeGreaterThan(0);
  });

  it("a deeper tree fits better than the root mean, and the forest beats one tree", () => {
    const tree = fitTree(X, y, trainIdx, { maxDepth: 4, minLeaf: 8 });
    const rootOnly = fitTree(X, y, trainIdx, { maxDepth: 0 });
    const maeTree = maeOn(y, validIdx, (r) => predictTree(tree, r), X);
    const maeRoot = maeOn(y, validIdx, (r) => predictTree(rootOnly, r), X);
    expect(maeTree).toBeLessThan(maeRoot);
    const forest = fitForest(X, y, trainIdx, 20);
    const maeForest = maeOn(y, validIdx, (r) => predictForest(forest, r), X);
    expect(maeForest).toBeLessThanOrEqual(maeTree);
    const imp = featureImportance(forest, 5);
    expect(imp.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    // size m² was planted as the dominant driver — the forest must find that
    expect(Math.max(...imp)).toBe(imp[1]);
  });
});

describe("collab data generator", () => {
  it("plants learnable structure with sparse coverage", () => {
    const cd = makeCollabData();
    expect(cd.users.length).toBe(40);
    expect(cd.movies.length).toBe(24);
    expect(cd.train.length).toBeGreaterThan(300);
    expect(cd.test.length).toBeGreaterThan(40);
    for (const r of [...cd.train, ...cd.test]) {
      expect(r.r).toBeGreaterThanOrEqual(0.5);
      expect(r.r).toBeLessThanOrEqual(5);
    }
  });
});
