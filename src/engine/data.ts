// MNIST subset loading + the ch.4 DataLoader (shuffle into mini-batches).

export interface MnistData {
  trainImages: Uint8Array; // n * 784, raw pixels 0..255
  trainLabels: Uint8Array;
  testImages: Uint8Array;
  testLabels: Uint8Array;
  nTrain: number;
  nTest: number;
}

export async function loadMnist(url = "data/mnist.bin"): Promise<MnistData> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}: HTTP ${res.status}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  const magic = String.fromCharCode(buf[0], buf[1], buf[2], buf[3]);
  if (magic !== "DLW1") throw new Error("bad mnist.bin magic");
  const view = new DataView(buf.buffer);
  const nTrain = view.getUint32(4, true);
  const nTest = view.getUint32(8, true);
  let off = 12;
  const trainImages = buf.subarray(off, (off += nTrain * 784));
  const trainLabels = buf.subarray(off, (off += nTrain));
  const testImages = buf.subarray(off, (off += nTest * 784));
  const testLabels = buf.subarray(off, (off += nTest));
  return { trainImages, trainLabels, testImages, testLabels, nTrain, nTest };
}

/** image i as normalized floats (0..1), the x of the dataset */
export function imageAsFloats(images: Uint8Array, i: number): Float32Array {
  const f = new Float32Array(784);
  const off = i * 784;
  for (let j = 0; j < 784; j++) f[j] = images[off + j] / 255;
  return f;
}

/** deterministic PRNG so training runs are reproducible inside the world */
export function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The ch.4 DataLoader: holds dataset indices, reshuffles every epoch,
 * hands out mini-batches. The Batch Depot building visualizes this object.
 */
export class DataLoader {
  indices: Int32Array;
  batchSize: number;
  cursor = 0;
  epoch = 0;
  shuffle: boolean;
  private rand: () => number;
  /** the most recent batch, for display */
  lastBatch: Int32Array = new Int32Array(0);

  constructor(n: number, batchSize: number, shuffle = true, seed = 42) {
    if (!Number.isInteger(batchSize) || batchSize <= 0)
      throw new Error(`DataLoader: batchSize must be a positive integer, got ${batchSize}`);
    this.indices = new Int32Array(n);
    for (let i = 0; i < n; i++) this.indices[i] = i;
    this.batchSize = batchSize;
    this.shuffle = shuffle;
    this.rand = mulberry32(seed);
    if (shuffle) this.reshuffle();
  }

  private reshuffle(): void {
    for (let i = this.indices.length - 1; i > 0; i--) {
      const j = Math.floor(this.rand() * (i + 1));
      [this.indices[i], this.indices[j]] = [this.indices[j], this.indices[i]];
    }
  }

  /** next mini-batch of dataset indices; wraps + reshuffles at epoch end */
  next(): Int32Array {
    if (this.cursor >= this.indices.length) {
      this.cursor = 0;
      this.epoch++;
      if (this.shuffle) this.reshuffle();
    }
    const end = Math.min(this.cursor + this.batchSize, this.indices.length);
    this.lastBatch = this.indices.slice(this.cursor, end);
    this.cursor = end;
    return this.lastBatch;
  }

  get batchesPerEpoch(): number {
    return Math.ceil(this.indices.length / this.batchSize);
  }
}
