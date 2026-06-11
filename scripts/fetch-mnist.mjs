// Downloads MNIST, samples a balanced subset, and packs it into public/data/mnist.bin
// Format (little-endian):
//   [0..3]   magic "DLW1"
//   [4..7]   uint32 nTrain
//   [8..11]  uint32 nTest
//   then nTrain*784 bytes train images (uint8, row-major 28x28)
//   then nTrain bytes train labels
//   then nTest*784 bytes test images
//   then nTest bytes test labels
import { mkdirSync, writeFileSync, existsSync, readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const cacheDir = join(root, ".mnist-cache");
const outDir = join(root, "public", "data");
mkdirSync(cacheDir, { recursive: true });
mkdirSync(outDir, { recursive: true });

const BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const FILES = [
  "train-images-idx3-ubyte.gz",
  "train-labels-idx1-ubyte.gz",
  "t10k-images-idx3-ubyte.gz",
  "t10k-labels-idx1-ubyte.gz",
];

const PER_CLASS_TRAIN = 300; // 3000 train images
const PER_CLASS_TEST = 60; // 600 test images

async function fetchFile(name) {
  const cached = join(cacheDir, name);
  if (existsSync(cached)) return readFileSync(cached);
  process.stdout.write(`downloading ${name}... `);
  const res = await fetch(BASE + name);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${name}`);
  const buf = Buffer.from(await res.arrayBuffer());
  writeFileSync(cached, buf);
  console.log(`${(buf.length / 1e6).toFixed(1)}MB`);
  return buf;
}

function parseImages(gz) {
  const b = gunzipSync(gz);
  const n = b.readUInt32BE(4);
  const rows = b.readUInt32BE(8);
  const cols = b.readUInt32BE(12);
  if (rows !== 28 || cols !== 28) throw new Error("unexpected image size");
  return { n, pixels: b.subarray(16) };
}

function parseLabels(gz) {
  const b = gunzipSync(gz);
  const n = b.readUInt32BE(4);
  return { n, labels: b.subarray(8) };
}

// Deterministic PRNG so the packed subset is reproducible
function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleBalanced(images, labels, perClass, rand) {
  const byClass = Array.from({ length: 10 }, () => []);
  for (let i = 0; i < labels.n; i++) byClass[labels.labels[i]].push(i);
  const chosen = [];
  for (let c = 0; c < 10; c++) {
    const pool = byClass[c];
    // partial Fisher-Yates: pick perClass distinct indices
    for (let k = 0; k < perClass; k++) {
      const j = k + Math.floor(rand() * (pool.length - k));
      [pool[k], pool[j]] = [pool[j], pool[k]];
      chosen.push(pool[k]);
    }
  }
  // shuffle the combined set so classes are interleaved
  for (let i = chosen.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [chosen[i], chosen[j]] = [chosen[j], chosen[i]];
  }
  const imgs = Buffer.alloc(chosen.length * 784);
  const labs = Buffer.alloc(chosen.length);
  chosen.forEach((srcIdx, dst) => {
    images.pixels.copy(imgs, dst * 784, srcIdx * 784, (srcIdx + 1) * 784);
    labs[dst] = labels.labels[srcIdx];
  });
  return { imgs, labs, n: chosen.length };
}

const [trainImgGz, trainLabGz, testImgGz, testLabGz] = await Promise.all(
  FILES.map(fetchFile),
);
const trainImages = parseImages(trainImgGz);
const trainLabels = parseLabels(trainLabGz);
const testImages = parseImages(testImgGz);
const testLabels = parseLabels(testLabGz);

const rand = mulberry32(20260611);
const train = sampleBalanced(trainImages, trainLabels, PER_CLASS_TRAIN, rand);
const test = sampleBalanced(testImages, testLabels, PER_CLASS_TEST, rand);

const header = Buffer.alloc(12);
header.write("DLW1", 0, "ascii");
header.writeUInt32LE(train.n, 4);
header.writeUInt32LE(test.n, 8);
const out = Buffer.concat([header, train.imgs, train.labs, test.imgs, test.labs]);
writeFileSync(join(outDir, "mnist.bin"), out);
console.log(
  `wrote public/data/mnist.bin: ${train.n} train + ${test.n} test, ${(out.length / 1e6).toFixed(2)}MB`,
);
