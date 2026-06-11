// Every building in DL World: where it stands, what it looks like, and the
// stations you can inspect inside. Stations reference panel ids registered
// by src/ui/panels/*.

export interface StationDef {
  id: string; // panel id
  name: string;
  icon: string;
  /** position inside the interior room, in tiles */
  x: number;
  y: number;
}

export interface BuildingDef {
  id: string;
  name: string;
  district: string;
  icon: string;
  /** footprint in city tiles */
  x: number;
  y: number;
  w: number;
  h: number;
  /** wall color */
  color: string;
  roof: string;
  /** which trainer powers this building's "machines are on" glow */
  trainer: "main" | "cottage" | "workshop" | "studio" | null;
  blurb: string;
  interior: { w: number; h: number; stations: StationDef[] };
}

// interior helper: spread stations along the back wall and sides
function room(w: number, h: number, stations: Omit<StationDef, "x" | "y">[]): BuildingDef["interior"] {
  const placed: StationDef[] = stations.map((s, i) => {
    const n = stations.length;
    if (n <= 4) {
      const gap = (w - 4) / Math.max(n - 1, 1);
      return { ...s, x: n === 1 ? Math.floor(w / 2) : Math.round(2 + i * gap), y: 2 };
    }
    // two rows
    const perRow = Math.ceil(n / 2);
    const row = Math.floor(i / perRow);
    const col = i % perRow;
    const gap = (w - 4) / Math.max(perRow - 1, 1);
    return { ...s, x: Math.round(2 + col * gap), y: row === 0 ? 2 : 5 };
  });
  return { w, h, stations: placed };
}

export const BUILDINGS: BuildingDef[] = [
  // ------------------------------------------------- Data Quarter (north-west)
  {
    id: "warehouse",
    name: "Dataset Warehouse",
    district: "Data Quarter",
    icon: "📦",
    x: 6, y: 7, w: 8, h: 6,
    color: "#c9a36a", roof: "#9a7544",
    trainer: "main",
    blurb: "Where the 3,000 handwritten digits live. A digit is just a 28×28 grid of numbers.",
    interior: room(16, 10, [
      { id: "warehouse.crates", name: "Crate Shelves", icon: "🖼" },
      { id: "warehouse.tensor", name: "The Tensor Plaque", icon: "🔢" },
    ]),
  },
  {
    id: "depot",
    name: "Batch Depot",
    district: "Data Quarter",
    icon: "🚉",
    x: 16, y: 7, w: 7, h: 6,
    color: "#b9b3a4", roof: "#82796a",
    trainer: "main",
    blurb: "The DataLoader: shuffles the dataset every epoch and ships out mini-batches of 64.",
    interior: room(16, 10, [
      { id: "depot.loader", name: "Shuffle Machine", icon: "🎲" },
      { id: "depot.batch", name: "Outgoing Batch", icon: "📤" },
    ]),
  },
  // ----------------------------------------------- Forward Avenue (north-center)
  {
    id: "mill1",
    name: "Linear Mill №1",
    district: "Forward Avenue",
    icon: "⚙️",
    x: 25, y: 7, w: 7, h: 6,
    color: "#8fb4d9", roof: "#5a7fa6",
    trainer: "main",
    blurb: "z1 = x·W1 + b1 — the first matrix multiply: 784 pixels → 64 hidden features.",
    interior: room(17, 10, [
      { id: "mill1.matmul", name: "Matmul Floor", icon: "✖️" },
      { id: "mill1.weights", name: "Weight Vault W1", icon: "🗄" },
      { id: "mill1.bias", name: "Bias Bench b1", icon: "➕" },
    ]),
  },
  {
    id: "springs",
    name: "Activation Springs",
    district: "Forward Avenue",
    icon: "⛲",
    x: 34, y: 7, w: 6, h: 6,
    color: "#7ecfc4", roof: "#4a9a90",
    trainer: "main",
    blurb: "ReLU — the nonlinearity. Negative values are clipped to zero; that's what makes depth matter.",
    interior: room(15, 10, [
      { id: "springs.relu", name: "ReLU Spring", icon: "📐" },
      { id: "springs.why", name: "Why Nonlinearity?", icon: "❓" },
      { id: "springs.sigmoid", name: "Sigmoid Pool", icon: "〰️" },
    ]),
  },
  {
    id: "mill2",
    name: "Linear Mill №2",
    district: "Forward Avenue",
    icon: "⚙️",
    x: 42, y: 7, w: 7, h: 6,
    color: "#8fb4d9", roof: "#5a7fa6",
    trainer: "main",
    blurb: "logits = a1·W2 + b2 — 64 hidden features → 10 scores, one per digit.",
    interior: room(17, 10, [
      { id: "mill2.matmul", name: "Matmul Floor", icon: "✖️" },
      { id: "mill2.weights", name: "Weight Vault W2", icon: "🗄" },
      { id: "mill2.bias", name: "Bias Bench b2", icon: "➕" },
    ]),
  },
  // -------------------------------------------------- Loss District (north-east)
  {
    id: "foundry",
    name: "Cross-Entropy Foundry",
    district: "Loss District",
    icon: "🏭",
    x: 51, y: 7, w: 9, h: 6,
    color: "#d98f8f", roof: "#a65a5a",
    trainer: "main",
    blurb: "The loss function as an assembly line: softmax → log → pick target → negate → mean.",
    interior: room(20, 11, [
      { id: "foundry.line", name: "The Assembly Line", icon: "🏗" },
      { id: "foundry.ledger", name: "Loss Ledger", icon: "📒" },
      { id: "foundry.backward", name: "The Return Belt", icon: "↩️" },
    ]),
  },
  // ------------------------------------------------ mid row: the backward loop
  {
    id: "backprop",
    name: "Backprop Works",
    district: "Gradient Row",
    icon: "🔧",
    x: 48, y: 16, w: 8, h: 6,
    color: "#b48fd9", roof: "#7e5aa6",
    trainer: "main",
    blurb: "The backward pass: walk the computation graph in reverse and watch the chain rule at work.",
    interior: room(18, 10, [
      { id: "backprop.chain", name: "Chain Rule Walk", icon: "⛓" },
      { id: "backprop.story", name: "One Weight's Story", icon: "📖" },
      { id: "backprop.check", name: "Gradient Check Lab", icon: "🔬" },
    ]),
  },
  {
    id: "optim",
    name: "Optimizer Depot",
    district: "Gradient Row",
    icon: "🔩",
    x: 39, y: 16, w: 7, h: 6,
    color: "#d9c08f", roof: "#a68a5a",
    trainer: "main",
    blurb: "SGD lives here: w ← w − lr·gradient, the whole secret of learning.",
    interior: room(16, 10, [
      { id: "optim.sgd", name: "The Update Floor", icon: "🛠" },
      { id: "optim.landscape", name: "Loss Slice Viewer", icon: "🏔" },
    ]),
  },
  {
    id: "observatory",
    name: "Metrics Observatory",
    district: "Civic Center",
    icon: "🔭",
    x: 6, y: 16, w: 7, h: 6,
    color: "#9fb6c9", roof: "#5e7d96",
    trainer: "main",
    blurb: "Loss curves, test accuracy and the confusion matrix — how training is actually going.",
    interior: room(16, 10, [
      { id: "observatory.curves", name: "Curve Wall", icon: "📈" },
      { id: "observatory.confusion", name: "Confusion Matrix", icon: "🧩" },
    ]),
  },
  // ----------------------------------------------------------- south: chapter 4
  {
    id: "museum",
    name: "Pixel Similarity Museum",
    district: "Chapter 4 Quarter",
    icon: "🏛",
    x: 6, y: 25, w: 7, h: 6,
    color: "#cfc4ae", roof: "#9c917b",
    trainer: null,
    blurb: "The pre-learning baseline: average all 3s and 7s, classify by pixel distance.",
    interior: room(16, 10, [
      { id: "museum.means", name: "Hall of Averages", icon: "🖼" },
      { id: "museum.classify", name: "Distance Desk", icon: "📏" },
    ]),
  },
  {
    id: "cottage",
    name: "Linear Cottage",
    district: "Chapter 4 Quarter",
    icon: "🏡",
    x: 15, y: 25, w: 7, h: 6,
    color: "#a8c98f", roof: "#6f9655",
    trainer: "cottage",
    blurb: "Ch.4's first learner: one weight per pixel decides — 3 or 7? Watch SGD shape it.",
    interior: room(16, 10, [
      { id: "cottage.train", name: "Training Bench", icon: "🎛" },
      { id: "cottage.eye", name: "The Learned Eye", icon: "👁" },
      { id: "cottage.loss", name: "mnist_loss Corner", icon: "🧮" },
    ]),
  },
  {
    id: "tower",
    name: "LR Finder Tower",
    district: "Chapter 5 Heights",
    icon: "🗼",
    x: 24, y: 23, w: 5, h: 8,
    color: "#c9a3c4", roof: "#96678f",
    trainer: null,
    blurb: "Ch.5's learning-rate finder: sweep the LR upward until the loss explodes; pick the steep part.",
    interior: room(14, 10, [
      { id: "tower.sweep", name: "The Sweep Console", icon: "🎚" },
    ]),
  },
  // ----------------------------------------------------------- south: chapter 6
  {
    id: "workshop",
    name: "Multi-Label Workshop",
    district: "Chapter 6 Yards",
    icon: "🏷",
    x: 32, y: 25, w: 7, h: 6,
    color: "#d9a98f", roof: "#a6735a",
    trainer: "workshop",
    blurb: "Ch.6: several yes/no questions per image — sigmoid + binary cross-entropy per label.",
    interior: room(16, 10, [
      { id: "workshop.train", name: "Training Bench", icon: "🎛" },
      { id: "workshop.labels", name: "Label Wall", icon: "🏷" },
      { id: "workshop.bce", name: "BCE Bench", icon: "🧮" },
    ]),
  },
  {
    id: "studio",
    name: "Regression Studio",
    district: "Chapter 6 Yards",
    icon: "🎯",
    x: 41, y: 25, w: 7, h: 6,
    color: "#8fd9b0", roof: "#5aa67e",
    trainer: "studio",
    blurb: "Ch.6: predict numbers, not categories — find the digit's center of ink with MSE loss.",
    interior: room(16, 10, [
      { id: "studio.train", name: "Training Bench", icon: "🎛" },
      { id: "studio.preds", name: "Crosshair Desk", icon: "🎯" },
    ]),
  },
  {
    id: "gallery",
    name: "Inference Gallery",
    district: "Chapter 6 Yards",
    icon: "🖌",
    x: 50, y: 25, w: 8, h: 6,
    color: "#e0d08a", roof: "#b09a45",
    trainer: "main",
    blurb: "Draw your own digit and watch it flow through the trained model — inference is just the forward pass.",
    interior: room(18, 10, [
      { id: "gallery.draw", name: "The Drawing Canvas", icon: "🖌" },
      { id: "gallery.about", name: "On Frozen Weights", icon: "🧊" },
    ]),
  },
];

export function buildingById(id: string): BuildingDef {
  const b = BUILDINGS.find((x) => x.id === id);
  if (!b) throw new Error(`no building ${id}`);
  return b;
}

/** door tile (city coords): bottom wall, centered */
export function doorTile(b: BuildingDef): { x: number; y: number } {
  return { x: b.x + Math.floor(b.w / 2), y: b.y + b.h - 1 };
}
