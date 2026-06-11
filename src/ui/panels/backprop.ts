// Backprop Works: walk the real computation graph in reverse, follow one
// weight's gradient by hand, and verify autograd against finite differences.

import { Tensor } from "../../engine/tensor";
import { registerPanel } from "../panel";
import { el, fmt, heatmap, section, stats } from "../widgets";
import { liveRegion, picker } from "./common";

export function registerBackpropPanels(): void {
  registerPanel("backprop.chain", {
    title: "Chain Rule Walk — the computation graph in reverse",
    subtitle:
      "This is the *actual* graph autograd recorded during the last training step — not a diagram. Click a node to expand its live values and gradients.",
    render(body, world) {
      let expanded: Tensor | null = null;
      const [region, cleanup, refresh] = liveRegion(world.main, (root) => {
        const loss = world.mlp.lastLoss;
        if (!loss) return;
        const nodes = loss.graphNodes().reverse();
        const s = section(
          "From loss back to the leaves",
          "Backward visits nodes in this order. Each one combines incoming gradient with its local derivative and passes it along — that's the whole algorithm.",
        );
        const list = el("div", "graph-list");
        for (const node of nodes) {
          const isParam = node.requiresGrad && node.inputs.length === 0;
          const row = el("div", `graph-node${isParam ? " graph-param" : ""}${node === expanded ? " graph-open" : ""}`);
          const name = node.label || node.op || "(input)";
          const g = node.grad ? stats(node.grad) : null;
          const gmax = g ? Math.max(Math.abs(g.min), Math.abs(g.max)) : 0;
          row.innerHTML = `<span class="graph-name">${isParam ? "🧱 " : ""}${name}</span><span class="graph-op">${node.op || (isParam ? "parameter" : "data")}</span><span class="graph-shape">[${node.shape.join("×")}]</span><span class="graph-grad">${node.grad ? `|grad|max ${fmt(gmax, 5)}` : "no grad"}</span>`;
          row.addEventListener("click", () => {
            expanded = expanded === node ? null : node;
            refresh();
          });
          list.append(row);
          if (node === expanded) {
            const det = el("div", "graph-detail");
            const hrow = el("div", "hstack wrap-row");
            const vb = el("div", "vstack");
            vb.append(el("div", "caption", "values"), heatmap(node.data, node.rows, node.cols, { maxWidth: 380, maxHeight: 160, symmetric: true }));
            hrow.append(vb);
            if (node.grad) {
              const gb = el("div", "vstack");
              gb.append(el("div", "caption", "gradients (∂loss/∂this)"), heatmap(node.grad, node.rows, node.cols, { maxWidth: 380, maxHeight: 160, symmetric: true }));
              hrow.append(gb);
            }
            det.append(hrow);
            if (node.inputs.length)
              det.append(el("div", "caption", `inputs: ${node.inputs.map((i) => i.label || i.op || "leaf").join("  ·  ")}`));
            list.append(det);
          }
        }
        s.append(list);
        root.append(s);
      });
      body.append(region);
      return cleanup;
    },
  });

  registerPanel("backprop.story", {
    title: "One Weight's Story — ∂loss/∂W2[i,j] by hand",
    subtitle:
      "Pick a single weight in Mill №2 and compute its gradient with the chain rule, term by term, then compare with autograd.",
    render(body, world) {
      let wi = 0;
      let wj = 3;
      const [region, cleanup, refresh] = liveRegion(world.main, (root) => {
        const m = world.mlp;
        const a1 = m.nodes["a1 = ReLU(z1)"];
        const soft = m.nodes["softmax"];
        if (!a1 || !soft || !m.w2.grad) {
          root.append(el("p", "explain", "Take at least one training step first."));
          return;
        }
        const batch = m.lastBatch;
        const targets = m.targets(batch);
        const B = batch.length;
        const head = el("div", "hstack");
        head.append(
          picker("hidden feature i =", 64, wi, (v) => { wi = v; refresh(); }),
          picker("digit class j =", 10, wj, (v) => { wj = v; refresh(); }),
        );
        root.append(head);

        const s1 = section(
          "The chain",
          `W2[${wi},${wj}] only touches the loss through logit ${wj} of every sample. For sample b: logits[b,${wj}] = Σᵢ a1[b,i]·W2[i,${wj}] + b2[${wj}], so ∂logits[b,${wj}]/∂W2[${wi},${wj}] = a1[b,${wi}]. The foundry told us ∂loss/∂logits[b,j] = (p−y)/B. Multiply and sum over the batch:`,
        );
        s1.append(
          el(
            "div",
            "bigmath",
            `∂loss/∂W2[${wi},${wj}] = Σ_b  a1[b,${wi}] · (p[b,${wj}] − y[b,${wj}]) / ${B}`,
          ),
        );
        root.append(s1);

        const s2 = section("The terms, with live numbers (first 8 samples)");
        const table = el("table", "num-table");
        table.innerHTML = `<thead><tr><th>sample b</th><th>a1[b,${wi}]</th><th>p[b,${wj}]</th><th>y[b,${wj}]</th><th>term</th></tr></thead>`;
        const tb = el("tbody");
        let total = 0;
        for (let b = 0; b < B; b++) {
          const a = a1.at(b, wi);
          const p = soft.at(b, wj);
          const y = targets[b] === wj ? 1 : 0;
          const term = (a * (p - y)) / B;
          total += term;
          if (b < 8) {
            const tr = el("tr");
            tr.innerHTML = `<td>${b}</td><td>${fmt(a)}</td><td>${fmt(p)}</td><td>${y}</td><td class="${term >= 0 ? "pos" : "neg"}">${fmt(term, 6)}</td>`;
            tb.append(tr);
          }
        }
        const trDots = el("tr", "muted-row");
        trDots.innerHTML = `<td colspan="4">… all ${B} samples summed</td><td><b>${fmt(total, 6)}</b></td>`;
        tb.append(trDots);
        table.append(tb);
        s2.append(table);
        const auto = m.w2.grad[wi * 10 + wj];
        const ok = Math.abs(total - auto) < 1e-4;
        s2.append(
          el(
            "div",
            "bigmath",
            `hand-computed: <b>${fmt(total, 6)}</b> &nbsp;·&nbsp; autograd stored: <b>${fmt(auto, 6)}</b> &nbsp; <span class="${ok ? "pos" : "neg"}">${ok ? "✓ identical" : "✗ mismatch"}</span>`,
          ),
        );
        s2.append(
          el(
            "p",
            "explain",
            "Notice the shape of the story: a weight's gradient is large when (a) its input feature was active AND (b) its output class was confidently wrong. Credit assignment in one sentence.",
          ),
        );
        root.append(s2);
      });
      body.append(region);
      return cleanup;
    },
  });

  registerPanel("backprop.check", {
    title: "Gradient Check Lab — calculus vs. brute force",
    subtitle:
      "The definition of a derivative: nudge one weight by ±ε, re-run the whole forward pass, and see how the loss moves. Autograd should match.",
    render(body, world) {
      const m = world.mlp;
      let paramIdx = 2; // default W2
      let elemIdx = 7;
      const wrap = el("div");
      const controls = el("div", "hstack");
      const result = el("div");

      const run = () => {
        result.innerHTML = "";
        const p = m.params[paramIdx].tensor;
        const name = m.params[paramIdx].name;
        const i = Math.min(elemIdx, p.size - 1);
        const eps = 1e-3;
        // fresh analytic gradient on the last batch (current weights)
        m.opt.zeroGrad();
        const loss0 = m.buildLoss(m.lastBatch, false);
        loss0.backward();
        const analytic = p.grad![i];
        const base = loss0.item();
        m.opt.zeroGrad();
        // numeric: central difference
        const orig = p.data[i];
        p.data[i] = orig + eps;
        const up = m.lossOnLastBatch();
        p.data[i] = orig - eps;
        const down = m.lossOnLastBatch();
        p.data[i] = orig;
        const numeric = (up - down) / (2 * eps);
        const err = Math.abs(numeric - analytic) / Math.max(1e-9, Math.abs(numeric), Math.abs(analytic));

        const s = section(`Probing ${name}[${i}] (current value ${fmt(orig, 5)})`);
        const table = el("table", "num-table");
        table.innerHTML = `<tbody>
          <tr><td>loss with w = ${fmt(orig, 5)}</td><td>${fmt(base, 6)}</td></tr>
          <tr><td>loss with w + ε (ε=${eps})</td><td>${fmt(up, 6)}</td></tr>
          <tr><td>loss with w − ε</td><td>${fmt(down, 6)}</td></tr>
          <tr class="hl-row"><td>numeric slope (up − down) / 2ε</td><td><b>${fmt(numeric, 6)}</b></td></tr>
          <tr class="hl-row"><td>autograd's chain-rule answer</td><td><b>${fmt(analytic, 6)}</b></td></tr>
          <tr><td>relative difference</td><td class="${err < 0.02 ? "pos" : "neg"}">${(err * 100).toFixed(3)}% ${err < 0.02 ? "✓" : ""}</td></tr>
        </tbody>`;
        s.append(table);
        s.append(
          el(
            "p",
            "explain",
            "The brute-force way needs 2 full forward passes per weight — for our 50,890 parameters that's ~100,000 forward passes per step. Backprop gets every gradient in ONE backward pass. That asymmetry is why deep learning is possible at all.",
          ),
        );
        result.append(s);
      };

      controls.append(
        picker("parameter:", m.params.length, paramIdx, (v) => { paramIdx = v; run(); }, (i) => `${m.params[i].name} [${m.params[i].tensor.shape.join("×")}]`),
        picker("element #:", 50, elemIdx, (v) => { elemIdx = v * 37; run(); }, (i) => String(i * 37)),
      );
      wrap.append(controls, result);
      run();
      body.append(wrap);
    },
  });
}
