// The Cross-Entropy Foundry: the loss function as a literal assembly line.
// Every station shows the live numbers for one sample of the current batch,
// straight from the recorded computation graph.

import { registerPanel } from "../panel";
import { barChart, chip, digitCanvas, el, flowRow, fmt, numberStrip, section } from "../widgets";
import { liveRegion, picker } from "./common";

const DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

function rowOf(t: { at(r: number, c: number): number; cols: number }, r: number): Float32Array {
  const out = new Float32Array(t.cols);
  for (let c = 0; c < t.cols; c++) out[c] = t.at(r, c);
  return out;
}

export function registerFoundryPanels(): void {
  registerPanel("foundry.line", {
    title: "The Assembly Line — cross-entropy, station by station",
    subtitle:
      "Follow one sample of the live batch through every operation: softmax (shift → exp → normalize) → log → pick the target → negate → average.",
    render(body, world) {
      let sample = 0;
      const [region, cleanup, refresh] = liveRegion(world.main, (root) => {
        const n = world.mlp.nodes;
        const logits = n["logits = a1·W2 + b2"];
        const shifted = n["shifted logits"];
        const expT = n["exp"];
        const sums = n["row sums"];
        const soft = n["softmax"];
        const logp = n["log probs"];
        const picked = n["log p[target]"];
        const nll = n["negative log likelihood"];
        const loss = n["loss"];
        if (!logits || !soft || !logp || !loss || !shifted || !expT || !sums || !picked || !nll) return;
        const batch = world.mlp.lastBatch;
        const targets = world.mlp.targets(batch);
        sample = Math.min(sample, batch.length - 1);
        const t = targets[sample];

        const head = el("div", "hstack");
        head.append(picker("sample on the belt:", batch.length, sample, (v) => { sample = v; refresh(); },
          (i) => `#${i} (a “${targets[i]}”)`));
        const img = el("div", "vstack");
        img.append(digitCanvas(world.data.trainImages, batch[sample] * 784, 3), el("div", "caption", `true label: ${t}`));
        head.append(img);
        root.append(head);

        const lRow = rowOf(logits, sample);
        const shRow = rowOf(shifted, sample);
        const eRow = rowOf(expT, sample);
        const z = sums.at(sample, 0);
        const pRow = rowOf(soft, sample);
        const lpRow = rowOf(logp, sample);

        const s1 = section(
          "Stations 1–3: softmax",
          "The 10 raw logits become 10 probabilities. Subtracting the max changes nothing mathematically (it cancels in the division) but keeps exp from overflowing — a real trick used by every framework.",
        );
        s1.append(
          flowRow([
            { name: "logits (from Mill №2)", body: numberStrip(lRow, t) },
            { name: `− max (${fmt(Math.max(...lRow), 2)})`, body: numberStrip(shRow, t) },
            { name: "exp(·)", body: numberStrip(eRow, t) },
            { name: `÷ sum (${fmt(z, 3)})`, body: numberStrip(pRow, t), accent: true },
          ]),
        );
        s1.append(el("div", "caption", "softmax probabilities (gold = true class):"));
        s1.append(barChart(pRow, { labels: DIGITS, highlight: t, width: 600, height: 130 }));
        root.append(s1);

        const s2 = section(
          "Stations 4–6: negative log likelihood",
          "Take the log of each probability, keep only the true class's entry, and flip the sign. Confident-and-right ⇒ loss near 0; confident-and-wrong ⇒ enormous loss. That asymmetry is the teacher.",
        );
        s2.append(
          flowRow([
            { name: "log(p)", body: numberStrip(lpRow, t) },
            { name: `pick target row (y = ${t})`, body: `<div class="bigmath">log p[${t}] = ${fmt(picked.at(sample, 0))}</div>` },
            { name: "negate", body: `<div class="bigmath">−log p[${t}] = <b>${fmt(nll.at(sample, 0))}</b></div>`, accent: true },
          ]),
        );
        root.append(s2);

        const s3 = section("Station 7: the batch mean", "");
        const chips = el("div", "chips-row");
        chips.append(
          chip(`this sample's loss`, fmt(nll.at(sample, 0), 4)),
          chip(`mean over ${batch.length} samples`, fmt(loss.item(), 4), true),
          chip("this is THE loss", "the single number backprop starts from"),
        );
        s3.append(chips);
        root.append(s3);
      });
      body.append(region);
      return cleanup;
    },
  });

  registerPanel("foundry.ledger", {
    title: "Loss Ledger — who is expensive right now?",
    subtitle: "Per-sample losses for the live batch, worst first. These are the images the model currently finds confusing.",
    render(body, world) {
      const [region, cleanup] = liveRegion(world.main, (root) => {
        const nll = world.mlp.nodes["negative log likelihood"];
        const soft = world.mlp.nodes["softmax"];
        const loss = world.mlp.nodes["loss"];
        if (!nll || !soft || !loss) return;
        const batch = world.mlp.lastBatch;
        const targets = world.mlp.targets(batch);
        const order = Array.from({ length: batch.length }, (_, i) => i).sort(
          (a, b) => nll.at(b, 0) - nll.at(a, 0),
        );
        const chips = el("div", "chips-row");
        chips.append(
          chip("batch mean loss", fmt(loss.item(), 4), true),
          chip("cheapest sample", fmt(nll.at(order[order.length - 1], 0), 4)),
          chip("priciest sample", fmt(nll.at(order[0], 0), 4)),
        );
        root.append(chips);
        const s = section("The ledger (top 12 most expensive)");
        const table = el("table", "num-table");
        table.innerHTML =
          "<thead><tr><th>image</th><th>true</th><th>model's top guess</th><th>p(true class)</th><th>−log p = loss</th></tr></thead>";
        const tb = el("tbody");
        for (const i of order.slice(0, 12)) {
          const tr = el("tr");
          const tdImg = el("td");
          tdImg.append(digitCanvas(world.data.trainImages, batch[i] * 784, 1.4));
          const p = rowOf(soft, i);
          let guess = 0;
          for (let j = 1; j < 10; j++) if (p[j] > p[guess]) guess = j;
          tr.append(tdImg);
          tr.innerHTML += `<td>${targets[i]}</td><td class="${guess === targets[i] ? "pos" : "neg"}">${guess} (${fmt(p[guess] * 100, 1)}%)</td><td>${fmt(p[targets[i]] * 100, 2)}%</td><td>${fmt(nll.at(i, 0), 4)}</td>`;
          tb.append(tr);
        }
        table.append(tb);
        s.append(table);
        s.append(
          el(
            "p",
            "explain",
            "High-loss samples dominate the gradient, so SGD automatically spends its effort on exactly these confusing images. Train for a while and watch this list get cheaper.",
          ),
        );
        root.append(s);
      });
      body.append(region);
      return cleanup;
    },
  });

  registerPanel("foundry.backward", {
    title: "The Return Belt — gradients flowing back through the foundry",
    subtitle:
      "The same stations, traversed in reverse. The foundry's famous output: ∂loss/∂logits = (softmax − one-hot) / batch_size.",
    render(body, world) {
      let sample = 0;
      const [region, cleanup, refresh] = liveRegion(world.main, (root) => {
        const n = world.mlp.nodes;
        const logits = n["logits = a1·W2 + b2"];
        const soft = n["softmax"];
        const logp = n["log probs"];
        if (!logits?.grad || !soft || !logp?.grad) {
          root.append(el("p", "explain", "Take at least one training step first."));
          return;
        }
        const batch = world.mlp.lastBatch;
        const targets = world.mlp.targets(batch);
        sample = Math.min(sample, batch.length - 1);
        const t = targets[sample];
        const B = batch.length;

        root.append(picker("sample:", batch.length, sample, (v) => { sample = v; refresh(); }, (i) => `#${i} (a “${targets[i]}”)`));

        const s1 = section(
          "What autograd recorded",
          "Each station stored how to turn 'gradient w.r.t. my output' into 'gradient w.r.t. my input' (the chain rule). Here are the actual gradients at two key belts for this sample:",
        );
        const gLogp = rowOf({ at: (r, c) => logp.grad![r * 10 + c], cols: 10 }, sample);
        const gLogits = rowOf({ at: (r, c) => logits.grad![r * 10 + c], cols: 10 }, sample);
        s1.append(
          flowRow([
            { name: "∂loss/∂log p  (only the target survives the gather)", body: numberStrip(gLogp, t) },
            { name: "∂loss/∂logits  (after softmax backward)", body: numberStrip(gLogits, t), accent: true },
          ]),
        );
        root.append(s1);

        const s2 = section(
          "The famous identity, checked numerically",
          "Chain the softmax, log, gather, negate and mean derivatives together by hand and everything cancels into one clean line. Compare autograd's numbers to (p − onehot)/B:",
        );
        const table = el("table", "num-table");
        table.innerHTML = `<thead><tr><th>class j</th><th>softmax pⱼ</th><th>one-hot yⱼ</th><th>(pⱼ−yⱼ)/${B}</th><th>autograd ∂loss/∂logitⱼ</th><th></th></tr></thead>`;
        const tb = el("tbody");
        for (let j = 0; j < 10; j++) {
          const p = soft.at(sample, j);
          const y = j === t ? 1 : 0;
          const formula = (p - y) / B;
          const auto = gLogits[j];
          const ok = Math.abs(formula - auto) < 1e-4;
          const tr = el("tr", j === t ? "hl-row" : "");
          tr.innerHTML = `<td>${j}</td><td>${fmt(p)}</td><td>${y}</td><td>${fmt(formula)}</td><td>${fmt(auto)}</td><td class="${ok ? "pos" : "neg"}">${ok ? "✓" : "✗"}</td>`;
          tb.append(tr);
        }
        table.append(tb);
        s2.append(table);
        s2.append(
          el(
            "p",
            "explain",
            "Read the gold row: the true class gets a negative gradient (push that logit UP), every other class gets a positive one (push DOWN), each in proportion to how confidently wrong it is. The whole foundry exists to manufacture this signal.",
          ),
        );
        root.append(s2);
      });
      body.append(region);
      return cleanup;
    },
  });
}
