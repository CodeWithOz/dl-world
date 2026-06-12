// Decision Arboretum: tabular data without gradients. A regression tree is
// grown on the generated rents table by greedy variance reduction; every
// number shown (gains, node means, MAEs, importances) is really computed
// from that data in sim/tabular.ts.

import { registerPanel } from "../panel";
import { barChart, chip, el, fmt, section } from "../widgets";
import { picker } from "./common";
import { getArboretum } from "../../sim/tabular";
import { TAB_DISTRICTS } from "../../sim/datasets";
import { TreeNode, treePath } from "../../sim/tabular";

function fmtFeatVal(feat: number, v: number): string {
  return feat === 0 ? TAB_DISTRICTS[Math.round(v)] : fmt(v, 1);
}

function questionText(featNames: string[], n: TreeNode): string {
  if (n.feat === 0) {
    // district is label-encoded in rent order, so a threshold reads as a set
    const cheap = TAB_DISTRICTS.filter((_, i) => i <= n.thresh);
    return `district ∈ {${cheap.join(", ")}}?`;
  }
  return `${featNames[n.feat]} ≤ ${fmt(n.thresh, 1)}?`;
}

export function registerArborPanels(): void {
  registerPanel("arbor.split", {
    title: "The First Question — how a split is chosen",
    subtitle:
      "A regression tree starts with one node holding all 320 training apartments and asks: which single yes/no question splits them into two groups whose rents vary least? Below is the actual audition, every candidate scored.",
    render(body) {
      const a = getArboretum();
      const { featNames } = a.data;
      const root = a.tree;
      const s1 = section("Before any question");
      const chips = el("div", "chips-row");
      chips.append(
        chip("apartments", String(root.n)),
        chip("mean rent", `${fmt(root.mean, 0)}`),
        chip("MSE around that mean", fmt(root.mse, 0), true),
      );
      s1.append(chips);
      body.append(s1);

      const s2 = section("The audition: best threshold per feature");
      const table = el("table", "num-table");
      table.append(
        el(
          "tr",
          "",
          "<th>question</th><th>weighted MSE after</th><th>error removed (gain)</th>",
        ),
      );
      const best = Math.max(...a.firstSplits.map((c) => c?.gain ?? 0));
      a.firstSplits.forEach((cand, f) => {
        const row = el("tr", cand && cand.gain === best ? "hl-row" : "");
        if (!cand) {
          row.append(el("td", "", featNames[f]), el("td", "", "—"), el("td", "", "no valid split"));
        } else {
          row.append(
            el("td", "", f === 0 ? questionText(featNames, { ...a.tree, feat: 0, thresh: cand.thresh } as TreeNode) : `${featNames[f]} ≤ ${fmt(cand.thresh, 1)}?`),
            el("td", "", fmt(cand.mseAfter, 0)),
            el("td", "", fmt(cand.gain / root.n, 0) + " per row"),
          );
        }
        table.append(row);
      });
      s2.append(table);
      s2.append(
        el(
          "p",
          "explain",
          "The winning row (highlighted) becomes the root of the tree. No gradients, no learning rate — just trying every threshold of every column and keeping the one that explains the most variance. That's the entire algorithm, applied recursively.",
        ),
      );
      body.append(s2);
    },
  });

  registerPanel("arbor.tree", {
    title: "The Grown Tree — walk an apartment to its rent",
    subtitle:
      "The same question-picking, applied recursively 4 levels deep. Pick a real apartment from the held-out set and follow its path; the leaf's mean IS the prediction.",
    render(body) {
      const a = getArboretum();
      const { featNames, rows, rent, validIdx } = a.data;
      let sel = 0;
      const controls = el("div", "controls-row");
      const treeWrap = el("div");
      const render = () => {
        treeWrap.innerHTML = "";
        const rowIdx = validIdx[sel];
        const row = rows[rowIdx];
        const path = new Set(treePath(a.tree, row));
        const sec = section("The chosen apartment");
        const chips = el("div", "chips-row");
        featNames.forEach((fn, f) => chips.append(chip(fn, fmtFeatVal(f, row[f]))));
        chips.append(chip("true rent", `${rent[rowIdx]}`, true));
        sec.append(chips);
        treeWrap.append(sec);
        const renderNode = (n: TreeNode): HTMLElement => {
          const onPath = path.has(n);
          const box = el("div", "tree-node" + (onPath ? " tree-on-path" : ""));
          const head = el(
            "div",
            "tree-q",
            n.feat >= 0
              ? `${questionText(featNames, n)}`
              : `→ predict <b>${fmt(n.mean, 0)}</b>`,
          );
          box.append(head);
          box.append(
            el("div", "caption", `${n.n} apartments · mean ${fmt(n.mean, 0)} · mse ${fmt(n.mse, 0)}`),
          );
          if (n.feat >= 0 && n.left && n.right) {
            const kids = el("div", "tree-kids");
            const l = el("div", "tree-branch");
            l.append(el("div", "caption", "yes ↓"), renderNode(n.left));
            const r = el("div", "tree-branch");
            r.append(el("div", "caption", "no ↓"), renderNode(n.right));
            kids.append(l, r);
            box.append(kids);
          }
          return box;
        };
        treeWrap.append(renderNode(a.tree));
        const leaf = treePath(a.tree, row).at(-1) ?? a.tree;
        treeWrap.append(
          el(
            "div",
            "bigmath",
            `prediction = leaf mean = <b>${fmt(leaf.mean, 0)}</b> &nbsp;·&nbsp; true rent = <b>${rent[rowIdx]}</b> &nbsp;·&nbsp; off by ${fmt(Math.abs(leaf.mean - rent[rowIdx]), 0)}`,
          ),
        );
      };
      controls.append(
        picker("held-out apartment", validIdx.length, sel, (v) => {
          sel = v;
          render();
        }, (i) => {
          const r = rows[validIdx[i]];
          return `#${i + 1}: ${TAB_DISTRICTS[r[0]]}, ${r[1]} m²`;
        }),
      );
      body.append(controls, treeWrap);
      render();
    },
  });

  registerPanel("arbor.forest", {
    title: "Forest Lookout — many imperfect trees beat one",
    subtitle:
      "20 trees, each grown on a different bootstrap sample of the rows. Individually they overfit their own sample; averaged, their errors cancel. Both error numbers below are measured on the same 80 held-out apartments.",
    render(body) {
      const a = getArboretum();
      const s1 = section("Validation error (mean absolute, in rent units)");
      const chips = el("div", "chips-row");
      chips.append(
        chip("single tree MAE", fmt(a.treeMae, 1)),
        chip("forest of 20 MAE", fmt(a.forestMae, 1), true),
        chip(
          "improvement",
          `${fmt((1 - a.forestMae / a.treeMae) * 100, 1)}%`,
        ),
        chip("out-of-bag MAE", fmt(a.oobMae, 1)),
      );
      s1.append(chips);
      s1.append(
        el(
          "p",
          "explain",
          "The out-of-bag number is the forest's party trick: every tree trained on a bootstrap sample, so each training row has some trees that never saw it. Score each row using only those trees and you get a validation error for free, without withholding any data — handy when data is scarce.",
        ),
      );
      body.append(s1);

      const s2 = section(
        "Feature importance",
        "How much total squared error each column removed across every split of every tree (normalized). This is the first thing to look at on any tabular dataset — it tells you what the model actually relies on.",
      );
      s2.append(
        barChart(Float32Array.from(a.importance), {
          labels: a.data.featNames.map((f) => f.split(" ")[0]),
          width: 420,
          height: 150,
        }),
      );
      s2.append(
        el(
          "p",
          "explain",
          "The generator pays 10.5 per m² and the sizes span ~96 m², so size dominates — the forest rediscovered the planted economics without being told. Balcony matters least: a flat +85 is small next to the size range.",
        ),
      );
      body.append(s2);
    },
  });
}
