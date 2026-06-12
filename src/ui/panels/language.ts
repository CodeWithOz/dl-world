// Language Lane: the Tokenizer Mill (text → tokens → numbers, the part of
// NLP that is pure plumbing) and the Sentiment Studio (a live bag-of-words
// classifier whose every weight is inspectable).

import { registerPanel } from "../panel";
import { button, chip, el, fmt, heatmap, section } from "../widgets";
import { liveRegion, picker, trainerControls } from "./common";
import { World } from "../../sim/world";
import { SentimentNet } from "../../sim/scenarios2";
import { numericalize, tokenize } from "../../sim/datasets";

function s(world: World): SentimentNet {
  return world.sent;
}

/** safe token chips — token text goes through textContent, never innerHTML */
function tokenChips(tokens: string[], cls: (i: number) => string): HTMLElement {
  const row = el("div", "token-row");
  tokens.forEach((t, i) => {
    const c = el("span", `token ${cls(i)}`);
    c.textContent = t;
    row.append(c);
  });
  return row;
}

export function registerLanguagePanels(): void {
  registerPanel("tokenmill.split", {
    title: "The Token Splitter — text becomes pieces",
    subtitle:
      "A matmul can't eat a sentence. Step one of every NLP model: cut the text into tokens, and mark structure (xxbos = \"a review starts here\") with special tokens.",
    render(body, world) {
      const sc = s(world);
      let sel = 0;
      const controls = el("div", "controls-row");
      const out = el("div");
      const render = () => {
        out.innerHTML = "";
        const review = sc.td.train[sel];
        const sec1 = section("Raw text, as a human reads it");
        const p = el("p", "bigmath");
        p.textContent = `"${review.text}"`;
        sec1.append(p, el("div", "caption", `labeled ${review.label === 1 ? "positive 👍" : "negative 👎"} in the dataset`));
        out.append(sec1);
        const tokens = tokenize(review.text);
        const sec2 = section(`After the splitter: ${tokens.length} tokens`);
        sec2.append(tokenChips(tokens, (i) => (i === 0 ? "token-special" : "")));
        sec2.append(
          el(
            "p",
            "explain",
            "This mill uses the simplest rule that works: lowercase everything and keep runs of letters. Real tokenizers (subwords, byte-pairs) are smarter about rare words, but the job is identical — and xxbos-style special tokens are exactly how production pipelines mark structure.",
          ),
        );
        out.append(sec2);
      };
      controls.append(
        picker("review", sc.td.train.length, sel, (v) => {
          sel = v;
          render();
        }, (i) => `${i + 1}: ${sc.td.train[i].text.slice(0, 34)}…`),
      );
      body.append(controls, out);
      render();
    },
  });

  registerPanel("tokenmill.vocab", {
    title: "The Vocab Wall — every word the model can know",
    subtitle:
      "The vocabulary is built once, from the training reviews only, sorted by frequency. A word not on this wall simply does not exist for the model — it becomes xxunk.",
    render(body, world) {
      const sc = s(world);
      const { vocab, freq } = sc.td;
      const chips = el("div", "chips-row");
      chips.append(
        chip("vocabulary size", String(vocab.length), true),
        chip("special tokens", "xxunk (id 0), xxbos (id 1)"),
        chip("built from", `${sc.td.train.length} training reviews`),
        chip("min_freq", "2 — words seen once become xxunk"),
      );
      body.append(chips);
      const sec = section("The wall, most frequent first (id in brackets)");
      const wall = el("div", "token-row");
      vocab.slice(0, 60).forEach((w, i) => {
        const t = el("span", `token${i < 2 ? " token-special" : ""}`);
        t.textContent = `${w} [${i}] ×${freq[i]}`;
        wall.append(t);
      });
      sec.append(wall);
      sec.append(
        el(
          "p",
          "explain",
          "Why train-only? If test words got vocabulary slots, the test set would leak into the model's design. And why drop one-off words (a min-frequency cutoff, like production tokenizers use)? A word seen once is a memorization handle, not signal — with a weight of its own it would just store that one review's label. The honest price: at test time some words are unknown — walk to the Numericalizer to see one become xxunk.",
        ),
      );
      body.append(sec);
    },
  });

  registerPanel("tokenmill.numeric", {
    title: "The Numericalizer — tokens become a tensor row",
    subtitle:
      "Each token is looked up on the Vocab Wall and replaced by its id; the ids are then counted into one row of length |vocab|. That row is the x that enters x·w — the exact same shape of computation as a flattened image.",
    render(body, world) {
      const sc = s(world);
      let sel = 0;
      const controls = el("div", "controls-row");
      const out = el("div");
      const render = () => {
        out.innerHTML = "";
        const review = sc.td.test[sel];
        const tokens = tokenize(review.text);
        const ids = numericalize(tokens, sc.td.vocab);
        const sec1 = section("token → id (held-out review, so xxunk can appear)");
        const table = el("table", "num-table");
        table.append(el("tr", "", "<th>token</th><th>id</th><th>note</th>"));
        tokens.forEach((t, i) => {
          const row = el("tr", ids[i] === 0 && t !== "xxunk" ? "hl-row" : "");
          const tdTok = el("td");
          tdTok.textContent = t;
          row.append(
            tdTok,
            el("td", "", String(ids[i])),
            el("td", "", ids[i] === 0 && t !== "xxunk" ? "not on the wall → xxunk" : ""),
          );
          table.append(row);
        });
        sec1.append(table);
        out.append(sec1);
        const V = sc.td.vocab.length;
        const counts = new Float32Array(V);
        for (const id of ids) counts[id] += 1;
        const sec2 = section(`the finished tensor row: [1, ${V}] word counts`);
        sec2.append(heatmap(counts, 1, V, { cellSize: 4, maxWidth: 520 }));
        sec2.append(
          el(
            "div",
            "caption",
            "almost all zeros — language is sparse. The bright cells are this review's words.",
          ),
        );
        out.append(sec2);
      };
      controls.append(
        picker("held-out review", sc.td.test.length, sel, (v) => {
          sel = v;
          render();
        }, (i) => `${i + 1}: ${sc.td.test[i].text.slice(0, 34)}…`),
      );
      body.append(controls, out);
      render();
    },
  });

  registerPanel("sentstudio.bench", {
    title: "Training Bench — logistic regression on words",
    subtitle:
      "One weight per vocabulary word, sigmoid + binary cross-entropy — the same loss as the Multi-Label Workshop, fed word frequencies instead of pixels.",
    render(body, world) {
      const sc = s(world);
      const [controls, cleanCtl] = trainerControls(world.sentiment);
      body.append(controls);
      const [live, cleanLive] = liveRegion(world.sentiment, (root) => {
        const chips = el("div", "chips-row");
        chips.append(
          chip("training reviews", String(sc.td.train.length)),
          chip("held-out reviews", String(sc.td.test.length)),
          chip("parameters", String(sc.td.vocab.length + 1), true),
        );
        root.append(chips);
        root.append(
          el(
            "p",
            "explain",
            "With ~70 training reviews this model can plateau quickly — and that's the honest lesson of small text datasets. Watch the test accuracy on the bench above: it is measured on reviews (and some words) the model never saw.",
          ),
        );
      });
      body.append(live);
      return () => {
        cleanCtl();
        cleanLive();
      };
    },
  });

  registerPanel("sentstudio.reader", {
    title: "The Review Reader — type anything, watch it think",
    subtitle:
      "Your text goes through the real mill (tokenize → numericalize → frequencies) and the live weights. Every word's contribution to the verdict is listed — this model can always explain itself.",
    render(body, world) {
      const sc = s(world);
      let text = sc.td.test[0].text;
      const controls = el("div", "controls-row");
      const input = document.createElement("textarea");
      input.className = "review-input";
      input.rows = 2;
      input.value = text;
      const [live, cleanLive, refresh] = liveRegion(world.sentiment, (root) => {
        if (!text.trim()) return;
        const res = sc.classify(text);
        const verdict = el(
          "div",
          "verdict",
          `${res.p >= 0.5 ? "👍 positive" : "👎 negative"} <span class="verdict-conf">(p(positive) = ${fmt(res.p, 3)})</span>`,
        );
        root.append(verdict);
        const table = el("table", "num-table");
        table.append(el("tr", "", "<th>token</th><th>its weight</th><th></th>"));
        const sorted = res.contribs
          .map((c, i) => ({ ...c, i }))
          .sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib))
          .slice(0, 12);
        for (const ctr of sorted) {
          const row = el("tr");
          const tok = el("td");
          tok.textContent = ctr.known ? ctr.token : `${ctr.token} → xxunk`;
          const val = el("td", ctr.contrib > 0 ? "pos" : ctr.contrib < 0 ? "neg" : "");
          val.textContent = fmt(ctr.contrib, 4);
          row.append(tok, val, el("td", "", ctr.contrib > 0 ? "pushes 👍" : ctr.contrib < 0 ? "pushes 👎" : "neutral"));
          table.append(row);
        }
        root.append(table);
        root.append(
          el(
            "div",
            "caption",
            `logit = bias + Σ contributions = ${fmt(res.logit, 4)} → sigmoid → ${fmt(res.p, 3)}. Unknown words land on xxunk and contribute its (mostly meaningless) weight.`,
          ),
        );
      });
      // the verdict re-reads as you type (debounced a touch), so the panel
      // always reflects exactly what's in the box — no button to remember
      let debounce = 0;
      input.addEventListener("input", () => {
        text = input.value;
        window.clearTimeout(debounce);
        debounce = window.setTimeout(refresh, 150);
      });
      controls.append(
        button("🎲 random held-out review", () => {
          const pick = sc.td.test[Math.floor(Math.random() * sc.td.test.length)];
          text = pick.text;
          input.value = text;
          refresh();
        }),
      );
      body.append(
        input,
        el("div", "caption", "type anything — the verdict updates as you type (and as training moves the weights)"),
        controls,
        live,
      );
      return () => {
        window.clearTimeout(debounce);
        cleanLive();
      };
    },
  });

  registerPanel("sentstudio.weights", {
    title: "Word-Weight Wall — the entire model, readable",
    subtitle:
      "This model has no hidden layers: its opinion of every word is one number. These are the most positive and most negative weights right now, straight from the live tensor.",
    render(body, world) {
      const sc = s(world);
      const [live, cleanLive] = liveRegion(world.sentiment, (root) => {
        const V = sc.td.vocab.length;
        const entries = [...Array(V)].map((_, i) => ({ word: sc.td.vocab[i], w: sc.w.at(i, 0) }));
        entries.sort((a, b) => b.w - a.w);
        const top = entries.slice(0, 12);
        const bottom = entries.slice(-12).reverse();
        const row = el("div", "hstack wrap-row");
        const col = (title: string, list: typeof top, cls: string) => {
          const v = el("div", "vstack");
          v.append(el("div", "caption", title));
          const t = el("table", "num-table");
          for (const e of list) {
            const tr = el("tr");
            const word = el("td");
            word.textContent = e.word;
            const val = el("td", cls);
            val.textContent = fmt(e.w, 3);
            tr.append(word, val);
            t.append(tr);
          }
          v.append(t);
          return v;
        };
        row.append(col("strongest 👍 words", top, "pos"), col("strongest 👎 words", bottom, "neg"));
        root.append(row);
        root.append(
          el(
            "p",
            "explain",
            "Early in training these are noise; after a few epochs words like \"superb\" and \"waste\" take the poles. This transparency is the trade-off of bag-of-words: it cannot know that \"not good\" differs from \"good\" (word order is gone) — which is exactly why the Echo Tower next door processes words in sequence.",
          ),
        );
      }, 500);
      body.append(live);
      return cleanLive;
    },
  });
}
