// Data Quarter panels: the Dataset Warehouse and the Batch Depot.

import { registerPanel } from "../panel";
import { chip, digitCanvas, el, fmt, heatmap, section } from "../widgets";
import { liveRegion } from "./common";
import { imageAsFloats } from "../../engine/data";

/** the raw 28x28 byte grid rendered as readable numbers */
function pixelNumberGrid(bytes: Uint8Array, offset: number): HTMLCanvasElement {
  const cell = 21;
  const canvas = document.createElement("canvas");
  canvas.width = 28 * cell * 2;
  canvas.height = 28 * cell * 2;
  canvas.style.width = `${28 * cell}px`;
  canvas.style.height = `${28 * cell}px`;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(2, 2);
  ctx.font = "7px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let r = 0; r < 28; r++)
    for (let c = 0; c < 28; c++) {
      const v = bytes[offset + r * 28 + c];
      ctx.fillStyle = `rgb(${18 + v * 0.85}, ${22 + v * 0.85}, ${34 + v * 0.8})`;
      ctx.fillRect(c * cell, r * cell, cell, cell);
      ctx.fillStyle = v > 110 ? "#10131c" : "rgba(220,230,255,0.75)";
      ctx.fillText(String(v), c * cell + cell / 2, r * cell + cell / 2);
    }
  return canvas;
}

export function registerDataPanels(): void {
  registerPanel("warehouse.crates", {
    title: "Crate Shelves — the training set",
    subtitle:
      "3,000 handwritten digits (a balanced sample of MNIST). Click any crate to open it and see what an image really is.",
    render(body, world) {
      let selected = 0;
      const [region, cleanup, refresh] = liveRegion(world.main, (root) => {
        const data = world.data;
        const top = section(
          "Open crate",
          "An image is not a picture to the model — it is a 28×28 grid of numbers (0 = blank paper, 255 = full ink). This is the single most important idea in the whole city.",
        );
        const detail = el("div", "hstack");
        const left = el("div", "vstack");
        left.append(
          digitCanvas(data.trainImages, selected * 784, 6),
          el("div", "caption", `crate #${selected} — label: “${data.trainLabels[selected]}”`),
        );
        detail.append(left, pixelNumberGrid(data.trainImages, selected * 784));
        top.append(detail);
        root.append(top);

        const grid = section("All shelves (first 96 crates)");
        const wrap = el("div", "digit-grid");
        for (let i = 0; i < 96; i++) {
          const cell = el("div", `digit-cell${i === selected ? " digit-cell-sel" : ""}`);
          cell.append(digitCanvas(data.trainImages, i * 784, 2));
          cell.append(el("div", "digit-label", String(data.trainLabels[i])));
          cell.addEventListener("click", () => {
            selected = i;
            refresh();
          });
          wrap.append(cell);
        }
        grid.append(wrap);
        root.append(grid);
      });
      body.append(region);
      return cleanup;
    },
  });

  registerPanel("warehouse.tensor", {
    title: "The Tensor Plaque",
    subtitle: "Shapes, ranks, and why 28×28 becomes 784.",
    render(body, world) {
      const data = world.data;
      const s1 = section(
        "From grid to row",
        "Each image starts as a rank-2 tensor of shape <b>[28, 28]</b>. To feed a linear layer we flatten it into a rank-1 tensor of <b>784</b> values (28 × 28 = 784). Stacking all 3,000 training images gives the training tensor:",
      );
      s1.append(el("div", "bigmath", "x_train.shape = [3000, 784]"));
      const chips = el("div", "chips-row");
      chips.append(
        chip("rank", "2"),
        chip("rows (images)", String(data.nTrain)),
        chip("cols (pixels)", "784"),
        chip("dtype here", "float32 (pixel / 255)"),
      );
      s1.append(chips);
      body.append(s1);

      const s2 = section(
        "One row, sliced",
        "Indexing row <i>i</i> gives image <i>i</i>. The first 28 values of row 0 are its top scanline — mostly 0 (blank paper):",
      );
      const row0 = imageAsFloats(data.trainImages, 0);
      const strip = el("div", "mono small");
      strip.textContent = `x_train[0, 0:28] = [${Array.from(row0.slice(0, 28))
        .map((v) => fmt(v, 2))
        .join(", ")}]`;
      s2.append(strip);
      const mid = el("div", "mono small");
      const r14 = Array.from(row0.slice(14 * 28, 14 * 28 + 28)).map((v) => fmt(v, 2));
      mid.textContent = `x_train[0, 392:420] = [${r14.join(", ")}]  ← scanline 14, through the ink`;
      s2.append(mid);
      s2.append(
        el(
          "p",
          "explain",
          "Everything the city does downstream — matmuls, losses, gradients — is arithmetic on grids like this. There is no other magic ingredient.",
        ),
      );
      body.append(s2);

      const s3 = section("The whole training tensor at a glance", "3,000 rows × 784 columns, one pixel per cell (showing every 4th row):");
      const sampleRows = 256;
      const view = new Float32Array(sampleRows * 784);
      for (let i = 0; i < sampleRows; i++) {
        const src = i * 4;
        for (let j = 0; j < 784; j++) view[i * 784 + j] = data.trainImages[src * 784 + j] / 255;
      }
      s3.append(heatmap(view, sampleRows, 784, { maxWidth: 900, maxHeight: 280 }));
      body.append(s3);
    },
  });

  registerPanel("depot.loader", {
    title: "Shuffle Machine — the DataLoader",
    subtitle: "Every epoch: shuffle the 3,000 indices, then deal them out 64 at a time.",
    render(body, world) {
      const [region, cleanup] = liveRegion(world.main, (root) => {
        const s = world.mlp;
        const loader = s.loader;
        const info = el("div", "chips-row");
        info.append(
          chip("epoch", String(loader.epoch)),
          chip("batch size", String(loader.batchSize)),
          chip("batches / epoch", String(loader.batchesPerEpoch)),
          chip("position in epoch", `${loader.cursor} / ${loader.indices.length}`),
        );
        root.append(info);

        const s1 = section(
          "The shuffled deck",
          "Each pixel-column below is one dataset index, colored by its digit label. SGD needs *random* mini-batches — without shuffling, the model would see waves of similar digits and the gradient would be biased. The bright window is the batch that just shipped.",
        );
        const canvas = document.createElement("canvas");
        const n = loader.indices.length;
        const W = 900, H = 56;
        canvas.width = W * 2;
        canvas.height = H * 2;
        canvas.style.width = `${W}px`;
        canvas.style.height = `${H}px`;
        const ctx = canvas.getContext("2d")!;
        ctx.scale(2, 2);
        for (let i = 0; i < n; i++) {
          const label = world.data.trainLabels[loader.indices[i]];
          ctx.fillStyle = `hsl(${label * 36}, 65%, 55%)`;
          ctx.fillRect((i / n) * W, 8, Math.max(W / n, 0.5), 40);
        }
        const start = loader.cursor - loader.lastBatch.length;
        ctx.strokeStyle = "#ffe44d";
        ctx.lineWidth = 2;
        ctx.strokeRect((start / n) * W, 4, (loader.lastBatch.length / n) * W + 1, 48);
        s1.append(canvas);
        const legend = el("div", "legend-row");
        for (let d = 0; d < 10; d++) {
          const item = el("span", "legend-item", `■ ${d}`);
          (item.style as CSSStyleDeclaration).color = `hsl(${d * 36}, 65%, 55%)`;
          legend.append(item);
        }
        s1.append(legend);
        root.append(s1);

        const s2 = section("The batch that just shipped", "These 64 images (showing 16) are what the Forward Avenue mills are grinding through right now:");
        const strip = el("div", "digit-grid");
        const batch = s.lastBatch;
        for (let i = 0; i < Math.min(batch.length, 16); i++) {
          const cell = el("div", "digit-cell");
          cell.append(digitCanvas(world.data.trainImages, batch[i] * 784, 2));
          cell.append(el("div", "digit-label", String(world.data.trainLabels[batch[i]])));
          strip.append(cell);
        }
        s2.append(strip);
        root.append(s2);
      });
      body.append(region);
      return cleanup;
    },
  });

  registerPanel("depot.batch", {
    title: "Outgoing Batch — x as one tensor",
    subtitle: "The 64 images are stacked into a single [64, 784] tensor so one matmul can process them all at once.",
    render(body, world) {
      const [region, cleanup] = liveRegion(world.main, (root) => {
        const x = world.mlp.nodes["x (batch images)"];
        if (!x) return;
        const s1 = section(
          "x — shape [64, 784]",
          "Every row is one flattened digit. This is why GPUs love deep learning: the whole batch moves through the network as one matrix.",
        );
        s1.append(heatmap(x.data, x.rows, x.cols, { maxWidth: 900, maxHeight: 220 }));
        root.append(s1);
        const y = world.mlp.targets(world.mlp.lastBatch);
        const s2 = section("y — the targets", "Integer class labels, one per row:");
        s2.append(
          el(
            "div",
            "mono small wrap",
            `y = [${Array.from(y).join(", ")}]`,
          ),
        );
        root.append(s2);
      });
      body.append(region);
      return cleanup;
    },
  });
}
