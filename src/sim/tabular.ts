// Decision trees and a small random forest for the Decision Arboretum.
//
// No gradients here on purpose: the point of the building is that tabular
// data is often best handled by asking questions, not by SGD. Everything
// is computed for real on the generated rents dataset — split gains, node
// means, validation errors, feature importances — using classic CART
// (greedy variance reduction).

import { makeTabularData, TabularData } from "./datasets";
import { mulberry32 } from "../engine/data";

export interface TreeNode {
  n: number;
  /** mean target of the rows that reached this node (= its prediction) */
  mean: number;
  /** mean squared error around that mean */
  mse: number;
  depth: number;
  /** split: feature index + threshold; feat = -1 marks a leaf */
  feat: number;
  thresh: number;
  /** total squared error removed by this split (importance unit) */
  gain: number;
  left: TreeNode | null;
  right: TreeNode | null;
}

export interface SplitCandidate {
  feat: number;
  thresh: number;
  /** weighted MSE of the two halves if we split here */
  mseAfter: number;
  gain: number;
}

function meanMse(y: number[], idx: number[]): { mean: number; mse: number } {
  let s = 0;
  for (const i of idx) s += y[i];
  const mean = s / idx.length;
  let q = 0;
  for (const i of idx) q += (y[i] - mean) * (y[i] - mean);
  return { mean, mse: q / idx.length };
}

/** best threshold for one feature (or null if it can't split this node) */
function bestSplitForFeature(
  X: number[][],
  y: number[],
  idx: number[],
  feat: number,
  minLeaf: number,
): SplitCandidate | null {
  const order = [...idx].sort((a, b) => X[a][feat] - X[b][feat]);
  const n = order.length;
  // prefix sums let every candidate threshold be scored in O(1)
  const pref = new Float64Array(n + 1);
  const prefSq = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const v = y[order[i]];
    pref[i + 1] = pref[i] + v;
    prefSq[i + 1] = prefSq[i] + v * v;
  }
  const total = pref[n];
  const totalSq = prefSq[n];
  const { mse: mseBefore } = meanMse(y, idx);
  let best: SplitCandidate | null = null;
  for (let i = minLeaf; i <= n - minLeaf; i++) {
    // only between distinct feature values
    if (X[order[i - 1]][feat] === X[order[i]][feat]) continue;
    const lN = i;
    const rN = n - i;
    // var = E[y²] − E[y]² per side, weighted
    const lMean = pref[i] / lN;
    const rMean = (total - pref[i]) / rN;
    const lVar = prefSq[i] / lN - lMean * lMean;
    const rVar = (totalSq - prefSq[i]) / rN - rMean * rMean;
    const mseAfter = (lVar * lN + rVar * rN) / n;
    if (!best || mseAfter < best.mseAfter) {
      best = {
        feat,
        thresh: (X[order[i - 1]][feat] + X[order[i]][feat]) / 2,
        mseAfter,
        gain: (mseBefore - mseAfter) * n,
      };
    }
  }
  return best;
}

/** the best split of each feature at this node — the "which question?" table */
export function splitCandidates(
  X: number[][],
  y: number[],
  idx: number[],
  minLeaf = 8,
): (SplitCandidate | null)[] {
  const nFeats = X[0].length;
  const out: (SplitCandidate | null)[] = [];
  for (let f = 0; f < nFeats; f++) out.push(bestSplitForFeature(X, y, idx, f, minLeaf));
  return out;
}

export function fitTree(
  X: number[][],
  y: number[],
  idx: number[],
  opts: { maxDepth?: number; minLeaf?: number; featSubset?: number[] } = {},
  depth = 0,
): TreeNode {
  const maxDepth = opts.maxDepth ?? 4;
  const minLeaf = opts.minLeaf ?? 8;
  const { mean, mse } = meanMse(y, idx);
  const node: TreeNode = { n: idx.length, mean, mse, depth, feat: -1, thresh: 0, gain: 0, left: null, right: null };
  if (depth >= maxDepth || idx.length < minLeaf * 2) return node;
  const feats = opts.featSubset ?? X[0].map((_, f) => f);
  let best: SplitCandidate | null = null;
  for (const f of feats) {
    const c = bestSplitForFeature(X, y, idx, f, minLeaf);
    if (c && (!best || c.mseAfter < best.mseAfter)) best = c;
  }
  if (!best || best.gain <= 0) return node;
  node.feat = best.feat;
  node.thresh = best.thresh;
  node.gain = best.gain;
  const li = idx.filter((i) => X[i][best!.feat] <= best!.thresh);
  const ri = idx.filter((i) => X[i][best!.feat] > best!.thresh);
  node.left = fitTree(X, y, li, opts, depth + 1);
  node.right = fitTree(X, y, ri, opts, depth + 1);
  return node;
}

export function predictTree(node: TreeNode, row: number[]): number {
  let cur = node;
  while (cur.feat >= 0) cur = row[cur.feat] <= cur.thresh ? cur.left! : cur.right!;
  return cur.mean;
}

/** the path a row takes through the tree (for the sample walker) */
export function treePath(node: TreeNode, row: number[]): TreeNode[] {
  const path = [node];
  let cur = node;
  while (cur.feat >= 0) {
    cur = row[cur.feat] <= cur.thresh ? cur.left! : cur.right!;
    path.push(cur);
  }
  return path;
}

export interface Forest {
  trees: TreeNode[];
  /** which training rows each tree actually saw (for out-of-bag error) */
  bags: Set<number>[];
}

export function fitForestDetailed(
  X: number[][],
  y: number[],
  idx: number[],
  nTrees = 20,
  seed = 7,
): Forest {
  const rand = mulberry32(seed);
  const trees: TreeNode[] = [];
  const bags: Set<number>[] = [];
  for (let t = 0; t < nTrees; t++) {
    // bagging: bootstrap-sample the training rows for each tree
    const bag: number[] = [];
    for (let i = 0; i < idx.length; i++) bag.push(idx[Math.floor(rand() * idx.length)]);
    trees.push(fitTree(X, y, bag, { maxDepth: 5, minLeaf: 5 }));
    bags.push(new Set(bag));
  }
  return { trees, bags };
}

export function fitForest(
  X: number[][],
  y: number[],
  idx: number[],
  nTrees = 20,
  seed = 7,
): TreeNode[] {
  return fitForestDetailed(X, y, idx, nTrees, seed).trees;
}

/** out-of-bag MAE: each training row is predicted only by the trees that
 *  never saw it — a free validation set, no rows withheld */
export function oobMae(forest: Forest, X: number[][], y: number[], idx: number[]): number {
  let total = 0;
  let counted = 0;
  for (const i of idx) {
    let s = 0;
    let n = 0;
    forest.trees.forEach((t, k) => {
      if (forest.bags[k].has(i)) return;
      s += predictTree(t, X[i]);
      n++;
    });
    if (n === 0) continue;
    total += Math.abs(s / n - y[i]);
    counted++;
  }
  return total / counted;
}

export function predictForest(trees: TreeNode[], row: number[]): number {
  let s = 0;
  for (const t of trees) s += predictTree(t, row);
  return s / trees.length;
}

/** split-gain feature importance, normalized to sum to 1 */
export function featureImportance(trees: TreeNode[], nFeats: number): number[] {
  const imp = new Array(nFeats).fill(0);
  const walk = (n: TreeNode | null) => {
    if (!n || n.feat < 0) return;
    imp[n.feat] += n.gain;
    walk(n.left);
    walk(n.right);
  };
  for (const t of trees) walk(t);
  const total = imp.reduce((a, b) => a + b, 0) || 1;
  return imp.map((v) => v / total);
}

export function maeOn(y: number[], idx: number[], predict: (row: number[]) => number, X: number[][]): number {
  let s = 0;
  for (const i of idx) s += Math.abs(predict(X[i]) - y[i]);
  return s / idx.length;
}

// ------------------------------------------------- the arboretum singleton

export interface Arboretum {
  data: TabularData;
  tree: TreeNode;
  forest: TreeNode[];
  treeMae: number;
  forestMae: number;
  oobMae: number;
  importance: number[];
  firstSplits: (SplitCandidate | null)[];
  /** how many forests have been grown so far (regrowing bumps it) */
  growth: number;
}

let arboretum: Arboretum | null = null;
let forestSeed = 7;

/** everything the Decision Arboretum shows, computed once on first entry */
export function getArboretum(): Arboretum {
  if (arboretum) return arboretum;
  const data = makeTabularData();
  const { rows: X, rent: y, trainIdx, validIdx } = data;
  const tree = fitTree(X, y, trainIdx, { maxDepth: 4, minLeaf: 8 });
  const forest = fitForestDetailed(X, y, trainIdx, 20, forestSeed);
  arboretum = {
    data,
    tree,
    forest: forest.trees,
    treeMae: maeOn(y, validIdx, (r) => predictTree(tree, r), X),
    forestMae: maeOn(y, validIdx, (r) => predictForest(forest.trees, r), X),
    oobMae: oobMae(forest, X, y, trainIdx),
    importance: featureImportance(forest.trees, data.featNames.length),
    firstSplits: splitCandidates(X, y, trainIdx),
    growth: 1,
  };
  return arboretum;
}

/**
 * Regrow the forest with fresh bootstrap draws. The single tree is
 * deterministic (same data, same questions — nothing to redo), but the
 * forest's randomness is real: every regrowth gives slightly different
 * errors and importances. That wobble is sampling variance, live.
 */
export function regrowForest(): Arboretum {
  const a = getArboretum();
  forestSeed += 1;
  const { rows: X, rent: y, trainIdx, validIdx } = a.data;
  const forest = fitForestDetailed(X, y, trainIdx, 20, forestSeed);
  a.forest = forest.trees;
  a.forestMae = maeOn(y, validIdx, (r) => predictForest(forest.trees, r), X);
  a.oobMae = oobMae(forest, X, y, trainIdx);
  a.importance = featureImportance(forest.trees, a.data.featNames.length);
  a.growth += 1;
  return a;
}
