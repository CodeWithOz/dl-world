// The frontier's datasets. None of these are images — that's the point.
//
// They are generated deterministically in code (no binary assets), with
// *planted* structure plus noise: the ratings really do come from hidden
// user/movie factors, the rents really do follow the size/floor/age rules.
// The models never see the generators — they must rediscover the structure
// from the data alone, live. Panels can then compare "what was planted"
// with "what was learned", which is only possible because we own the
// generator. Every number shown in a panel still comes from real tensors
// or real arithmetic on this data.

import { mulberry32 } from "../engine/data";

/** Box-Muller gaussian on top of a deterministic uniform PRNG */
function gauss(rand: () => number): number {
  const u1 = Math.max(rand(), 1e-12);
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ================================================== collaborative filtering

export interface CollabRating {
  u: number;
  m: number;
  r: number; // 0.5 .. 5 stars, in half-star steps
}

export interface CollabData {
  users: string[];
  /** fictional movies; `genre` is the planted hidden factor [action, romance, sci-fi] */
  movies: { title: string; genre: [number, number, number] }[];
  genreNames: string[];
  /** planted user tastes, same 3 axes, in [-1, 1] */
  tastes: [number, number, number][];
  train: CollabRating[];
  test: CollabRating[];
}

const MOVIE_DEFS: [string, number, number, number][] = [
  // title, action, romance, sci-fi (all fictional)
  ["Explosion Season 4", 1.0, 0.05, 0.15],
  ["The Last Stuntman", 0.95, 0.15, 0.05],
  ["Fist of Mayhem", 0.9, 0.0, 0.1],
  ["Turbo Heist", 0.85, 0.1, 0.05],
  ["Skyfire Protocol", 0.8, 0.05, 0.45],
  ["Maximum Detonation", 1.0, 0.0, 0.05],
  ["Rogue Convoy", 0.75, 0.2, 0.0],
  ["The Demolition Kid", 0.7, 0.3, 0.05],
  ["Love in the Rain", 0.05, 1.0, 0.0],
  ["The Letter I Never Sent", 0.0, 0.95, 0.05],
  ["Two Hearts in Venice", 0.05, 0.9, 0.0],
  ["Slow Dance", 0.0, 0.85, 0.05],
  ["The Wedding Detour", 0.25, 0.8, 0.0],
  ["Autumn Promises", 0.0, 0.9, 0.1],
  ["Paris, Eventually", 0.05, 0.95, 0.0],
  ["Coffee for Two", 0.1, 0.8, 0.05],
  ["Starship Meridian", 0.35, 0.1, 0.95],
  ["The Quantum Garden", 0.05, 0.25, 0.9],
  ["Signal from Europa", 0.15, 0.05, 1.0],
  ["Clone Protocol Seven", 0.45, 0.0, 0.9],
  ["The Mars Archive", 0.1, 0.15, 0.85],
  ["Neutron Dawn", 0.5, 0.05, 0.8],
  ["Orbit City", 0.2, 0.4, 0.75],
  ["The Android's Dream", 0.1, 0.55, 0.85],
];

const USER_NAMES = [
  "Ada", "Ben", "Cleo", "Dev", "Ella", "Femi", "Gus", "Hana",
  "Igor", "June", "Kofi", "Lena", "Mio", "Nora", "Omar", "Pia",
  "Quinn", "Rosa", "Sam", "Tessa", "Udo", "Vera", "Wes", "Xena",
  "Yara", "Zane", "Abe", "Bibi", "Cyrus", "Dina", "Ezra", "Fay",
  "Gita", "Hugo", "Ines", "Jude", "Kira", "Liam", "Mara", "Nils",
];

export function makeCollabData(): CollabData {
  const rand = mulberry32(2024);
  const movies = MOVIE_DEFS.map(([title, a, r, s]) => ({
    title,
    genre: [a, r, s] as [number, number, number],
  }));
  const tastes: [number, number, number][] = USER_NAMES.map(() => [
    rand() * 2 - 1,
    rand() * 2 - 1,
    rand() * 2 - 1,
  ]);
  const train: CollabRating[] = [];
  const test: CollabRating[] = [];
  let seen = 0;
  for (let u = 0; u < USER_NAMES.length; u++)
    for (let m = 0; m < movies.length; m++) {
      if (rand() > 0.62) continue; // sparse: nobody has seen everything
      const g = movies[m].genre;
      // planted rule: rating = affinity between taste and (centered) genre
      const dot =
        tastes[u][0] * (g[0] * 2 - 1) +
        tastes[u][1] * (g[1] * 2 - 1) +
        tastes[u][2] * (g[2] * 2 - 1);
      let r = 2.75 + 0.75 * dot + gauss(rand) * 0.25;
      r = Math.round(Math.max(0.5, Math.min(5, r)) * 2) / 2;
      // hold out every 7th observed rating to measure generalization
      (seen++ % 7 === 6 ? test : train).push({ u, m, r });
    }
  return { users: USER_NAMES, movies, genreNames: ["action", "romance", "sci-fi"], tastes, train, test };
}

// ============================================================ text reviews

export interface TextData {
  /** label: 1 = positive, 0 = negative */
  train: { text: string; label: number }[];
  test: { text: string; label: number }[];
  /** id 0 is xxunk (unknown word), id 1 is xxbos (beginning of stream) */
  vocab: string[];
  /** train-corpus frequency of each vocab entry */
  freq: number[];
}

// The review corpus is generated from shared templates: both polarities use
// the SAME sentence skeletons and the same neutral nouns, so function words
// ("the", "was", "and") are class-balanced by construction and carry no
// signal — only the sentiment adjectives separate the classes. Each
// adjective recurs across several reviews, so it survives the min-freq
// cutoff and earns a meaningful weight (a word used once teaches nothing).
const POS_ADJ = [
  "superb", "brilliant", "wonderful", "gorgeous", "delightful", "gripping",
  "charming", "hilarious", "moving", "clever", "fresh", "stunning",
];
const NEG_ADJ = [
  "boring", "dull", "clumsy", "lazy", "pointless", "tedious",
  "predictable", "hollow", "forgettable", "sloppy", "exhausting", "lifeless",
];
const REVIEW_NOUNS = [
  "acting", "plot", "dialogue", "pacing", "soundtrack", "ending",
  "script", "cast", "humor", "story", "finale", "direction",
];

function makeReviews(positive: boolean, n: number, seed: number): { text: string; label: number }[] {
  const rand = mulberry32(seed);
  const adj = positive ? POS_ADJ : NEG_ADJ;
  const pickTwo = <T>(arr: T[]): [T, T] => {
    const a = Math.floor(rand() * arr.length);
    let b = Math.floor(rand() * (arr.length - 1));
    if (b >= a) b++;
    return [arr[a], arr[b]];
  };
  const out: { text: string; label: number }[] = [];
  for (let i = 0; i < n; i++) {
    const [a1, a2] = pickTwo(adj);
    const [n1, n2] = pickTwo(REVIEW_NOUNS);
    const templates = [
      `the ${n1} was ${a1}`,
      `the ${n1} was ${a1} and the ${n2} was ${a2}`,
      `a ${a1} film with ${a2} ${n1}`,
      `${a1} ${n1} and ${a2} ${n2}`,
      `the ${n1} felt ${a1} the whole way through`,
      `utterly ${a1} from start to finish`,
    ];
    out.push({ text: templates[i % templates.length], label: positive ? 1 : 0 });
  }
  return out;
}

/** lowercase word tokenizer + the xxbos marker, fastai-style */
export function tokenize(text: string): string[] {
  const words = text.toLowerCase().match(/[a-z']+/g) ?? [];
  return ["xxbos", ...words];
}

export function makeTextData(): TextData {
  const pos = makeReviews(true, 60, 41);
  const neg = makeReviews(false, 60, 42);
  // interleave so the train/test split stays balanced
  const all: { text: string; label: number }[] = [];
  for (let i = 0; i < 60; i++) all.push(pos[i], neg[i]);
  // every 5th review is held out; vocab is built from train only so the
  // model genuinely meets unknown words (xxunk) at test time
  const train = all.filter((_, i) => i % 5 !== 4);
  const test = all.filter((_, i) => i % 5 === 4);
  const counts = new Map<string, number>();
  for (const r of train)
    for (const t of tokenize(r.text)) counts.set(t, (counts.get(t) ?? 0) + 1);
  counts.delete("xxbos");
  // fastai-style min_freq: words seen only once are memorization handles,
  // not signal — they fall back to xxunk
  const sorted = [...counts.entries()].filter(([, n]) => n >= 2).sort((a, b) => b[1] - a[1]);
  const vocab = ["xxunk", "xxbos", ...sorted.map(([w]) => w)];
  const freqMap = new Map(sorted);
  const freq = vocab.map((w) =>
    w === "xxunk" ? 0 : w === "xxbos" ? train.length : freqMap.get(w) ?? 0,
  );
  return { train, test, vocab, freq };
}

const vocabIndexCache = new WeakMap<string[], Map<string, number>>();

/** token list -> vocab ids (0 = xxunk for words outside the vocab) */
export function numericalize(tokens: string[], vocab: string[]): number[] {
  let index = vocabIndexCache.get(vocab);
  if (!index) {
    index = new Map(vocab.map((w, i) => [w, i]));
    vocabIndexCache.set(vocab, index);
  }
  return tokens.map((t) => index!.get(t) ?? 0);
}

// ========================================================== human numbers

export interface HumanNumbers {
  /** the whole corpus as one token stream: "one . two . three . …" */
  tokens: string[];
  ids: Int32Array;
  vocab: string[];
}

const ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
const TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

/** 1..9999 in english words, one token per word ("one hundred ten") */
export function numberToWords(n: number): string[] {
  if (n <= 0 || n >= 10000) throw new Error(`numberToWords: ${n} out of range`);
  const words: string[] = [];
  if (n >= 1000) {
    words.push(ONES[Math.floor(n / 1000)], "thousand");
    n %= 1000;
  }
  if (n >= 100) {
    words.push(ONES[Math.floor(n / 100)], "hundred");
    n %= 100;
  }
  if (n >= 20) {
    words.push(TENS[Math.floor(n / 10)]);
    n %= 10;
  } else if (n >= 10) {
    words.push(TEENS[n - 10]);
    n = 0;
  }
  if (n > 0) words.push(ONES[n]);
  return words;
}

export function makeHumanNumbers(max = 2000): HumanNumbers {
  const tokens: string[] = [];
  for (let n = 1; n <= max; n++) {
    tokens.push(...numberToWords(n));
    tokens.push(".");
  }
  const seen = new Set<string>();
  const vocab: string[] = [];
  for (const t of tokens)
    if (!seen.has(t)) {
      seen.add(t);
      vocab.push(t);
    }
  const index = new Map(vocab.map((w, i) => [w, i]));
  const ids = new Int32Array(tokens.length);
  for (let i = 0; i < tokens.length; i++) ids[i] = index.get(tokens[i])!;
  return { tokens, ids, vocab };
}

// ============================================================ tabular rents

export interface TabularData {
  featNames: string[];
  districtNames: string[];
  /** feature matrix, rows[i][f]; district is a label-encoded category */
  rows: number[][];
  /** target: monthly rent */
  rent: number[];
  trainIdx: number[];
  validIdx: number[];
}

export const TAB_FEATS = ["district", "size m²", "floor", "age (years)", "balcony"];
export const TAB_DISTRICTS = ["Outskirts", "Garden Heights", "Riverside", "Old Town"];

export function makeTabularData(n = 400): TabularData {
  const rand = mulberry32(909);
  const base = [520, 700, 820, 950]; // planted: per-district base rent
  const rows: number[][] = [];
  const rent: number[] = [];
  for (let i = 0; i < n; i++) {
    const district = Math.floor(rand() * 4);
    const size = Math.round(22 + rand() * 96);
    const floor = Math.floor(rand() * 16);
    const age = Math.round(rand() * 75);
    const balcony = rand() < 0.45 ? 1 : 0;
    rows.push([district, size, floor, age, balcony]);
    // planted rule + noise; the tree never sees these coefficients
    const r = base[district] + 10.5 * size + 7 * floor - 2.6 * age + 85 * balcony + gauss(rand) * 55;
    rent.push(Math.round(Math.max(250, r)));
  }
  const trainIdx: number[] = [];
  const validIdx: number[] = [];
  for (let i = 0; i < n; i++) (i % 5 === 4 ? validIdx : trainIdx).push(i);
  return { featNames: TAB_FEATS, districtNames: TAB_DISTRICTS, rows, rent, trainIdx, validIdx };
}
