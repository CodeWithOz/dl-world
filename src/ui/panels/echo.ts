// Echo Tower: a language model from scratch on the human-numbers corpus.
// Three words go in, a hidden state is threaded through them, the fourth
// word comes out. Generation is just running that loop on its own output.

import { registerPanel } from "../panel";
import { barChart, button, chip, el, fmt, flowRow, numberStrip, section } from "../widgets";
import { liveRegion, picker, trainerControls } from "./common";
import { World } from "../../sim/world";
import { RnnLm } from "../../sim/scenarios2";

function r(world: World): RnnLm {
  return world.rnn;
}

export function registerEchoPanels(): void {
  registerPanel("echo.corpus", {
    title: "The Counting Corpus — the model's entire universe",
    subtitle:
      "The numbers 1 to 2000, written out in English, one long stream. Tiny on purpose: small enough to train live, structured enough that 'predict the next word' has real rules to discover.",
    render(body, world) {
      const s = r(world);
      const chips = el("div", "chips-row");
      chips.append(
        chip("tokens in the stream", String(s.hn.tokens.length)),
        chip("distinct words (vocab)", String(s.hn.vocab.length), true),
        chip("training samples (3→1)", String(Math.floor(s.hn.ids.length / s.seqLen) - 1)),
      );
      body.append(chips);
      const s1 = section("The stream itself (a slice)");
      const p = el("p", "bigmath");
      p.textContent = s.hn.tokens.slice(280, 330).join(" ");
      s1.append(p);
      body.append(s1);
      const s2 = section("The whole vocabulary");
      const wall = el("div", "token-row");
      s.hn.vocab.forEach((w, i) => {
        const t = el("span", "token");
        t.textContent = `${w} [${i}]`;
        wall.append(t);
      });
      s2.append(wall);
      body.append(s2);
      const s3 = section("How the stream becomes training pairs");
      const ex = el("table", "num-table");
      ex.append(el("tr", "", "<th>input (3 words)</th><th>target (the 4th)</th>"));
      for (let k = 0; k < 4; k++) {
        const { ctx, target } = s.sampleAt(k + 40);
        const row = el("tr");
        row.append(
          el("td", "", ctx.map((id) => s.vocab[id]).join(" · ")),
          el("td", "", s.vocab[target]),
        );
        ex.append(row);
      }
      s3.append(ex);
      s3.append(
        el(
          "p",
          "explain",
          "The stream is chopped into consecutive, non-overlapping triples; the word after each triple is the label. The last 15% of the stream (the highest numbers) is held out — the model is graded on continuing a part of the count it has never read.",
        ),
      );
      body.append(s3);
    },
  });

  registerPanel("echo.loop", {
    title: "The Loop Room — one hidden state, reused three times",
    subtitle:
      "The defining trick of recurrence: the SAME weights process word 1, word 2 and word 3, accumulating a hidden state as they go. The unrolled loop below is read from the live graph of the most recent training step.",
    render(body, world) {
      const s = r(world);
      const [controls, cleanCtl] = trainerControls(world.echo);
      body.append(controls);
      const [live, cleanLive] = liveRegion(world.echo, (root) => {
        const h1 = s.nodes["hidden state after word 1"];
        const h2 = s.nodes["hidden state after word 2"];
        const h3 = s.nodes["hidden state after word 3"];
        if (!h1 || !h2 || !h3) return;
        const firstWord = (t: number) => {
          const node = s.nodes[`word ${t} (one-hot)`];
          if (!node) return "?";
          for (let j = 0; j < node.cols; j++) if (node.at(0, j) === 1) return s.vocab[j];
          return "?";
        };
        const hview = (h: typeof h1) => {
          const v = el("div", "vstack");
          v.append(numberStrip(h.data.slice(0, 6), -1, 2), el("div", "caption", "h[0..5] of 24"));
          return v;
        };
        root.append(
          el(
            "div",
            "caption",
            `the batch's first sample this step: “${firstWord(1)} ${firstWord(2)} ${firstWord(3)} → ?”`,
          ),
        );
        root.append(
          flowRow([
            { name: `“${firstWord(1)}” → emb`, body: el("div", "caption", "one-hot · E") },
            { name: "h after word 1", body: hview(h1) },
            { name: `+ “${firstWord(2)}” → emb`, body: el("div", "caption", "h + one-hot · E") },
            { name: "h after word 2", body: hview(h2) },
            { name: `+ “${firstWord(3)}” → emb`, body: el("div", "caption", "h + one-hot · E") },
            { name: "h after word 3", body: hview(h3), accent: true },
          ]),
        );
        root.append(
          el(
            "div",
            "bigmath",
            "h ← ReLU((h + emb(word_t)) · <b>Wh</b> + bh) — the same Wh at every t. Then: logits = h · Wo + bo.",
          ),
        );
        root.append(
          el(
            "p",
            "explain",
            "Because Wh is shared across positions, gradients flow back through the loop three times and accumulate — that is backpropagation through time, and it is exactly what the Backprop Works does on the recorded graph, no special case needed. The one-hot · E products in the graph are the proof that an 'embedding layer' is an ordinary matmul.",
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

  registerPanel("echo.next", {
    title: "Next-Word Desk — quiz the model on unseen count",
    subtitle:
      "Every quiz item below comes from the held-out tail of the corpus: numbers the model has never read. The bars are the live softmax over the whole vocabulary.",
    render(body, world) {
      const s = r(world);
      let sel = 0;
      const valid = s.validSamples;
      const controls = el("div", "controls-row");
      const [live, cleanLive, refresh] = liveRegion(world.echo, (root) => {
        const { ctx, target } = s.sampleAt(valid[sel]);
        const probs = s.predictNext(ctx);
        root.append(
          el(
            "div",
            "bigmath",
            `“${ctx.map((id) => s.vocab[id]).join(" ")}” → ? &nbsp; (truth: <b>${s.vocab[target]}</b>)`,
          ),
        );
        const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 8);
        root.append(
          barChart(Float32Array.from(order.map((i) => probs[i])), {
            labels: order.map((i) => s.vocab[i]),
            width: 460,
            height: 150,
            highlight: order.indexOf(target),
          }),
        );
        const chips = el("div", "chips-row");
        const pred = order[0];
        chips.append(
          chip("model's guess", s.vocab[pred], pred === target),
          chip("p(truth)", fmt(probs[target], 3)),
          chip("held-out accuracy", `${(world.echo.lastMetricValue * 100).toFixed(1)}%`, true),
          chip(`baseline: always "${s.baseline().token}"`, `${(s.baseline().acc * 100).toFixed(1)}%`),
        );
        root.append(chips);
        root.append(
          el(
            "p",
            "explain",
            "Some contexts are genuinely ambiguous (after \"hundred\" almost anything can follow), so 100% is impossible — what matters is beating the always-guess-the-separator baseline, and by how much. Train the Loop Room and re-visit.",
          ),
        );
      }, 600);
      controls.append(
        picker("quiz item", Math.min(valid.length, 200), sel, (v) => {
          sel = v;
          refresh();
        }, (i) => {
          const { ctx } = s.sampleAt(valid[i]);
          return ctx.map((id) => s.vocab[id]).join(" ");
        }),
      );
      body.append(controls, live);
      return cleanLive;
    },
  });

  registerPanel("echo.generate", {
    title: "The Counting Machine — let it talk",
    subtitle:
      "Generation is inference in a loop: predict a word, append it, slide the window, repeat. Greedy always takes the top word; sampled rolls dice over the real softmax.",
    render(body, world) {
      const s = r(world);
      const out = el("div");
      let seedNum = 120;
      const run = () => {
        out.innerHTML = "";
        // seed: find "<seedNum> ." in the stream and take the 3 tokens before the next number
        const seedCtx = (() => {
          let count = 0;
          for (let i = 0; i < s.hn.ids.length; i++) {
            if (s.hn.tokens[i] === ".") count++;
            if (count === seedNum) return [...s.hn.ids.subarray(i - 2, i + 1)];
          }
          return [...s.hn.ids.subarray(0, 3)];
        })();
        const seedText = seedCtx.map((id) => s.vocab[id]).join(" ");
        const greedy = s.generate(seedCtx, 28, 0);
        const sampled = s.generate(seedCtx, 28, 1, (Date.now() & 0xffff) | 1);
        const sec1 = section("Greedy (always the most likely word)");
        const p1 = el("p", "bigmath");
        p1.textContent = `${seedText} | ${greedy.join(" ")}`;
        sec1.append(p1);
        out.append(sec1);
        const sec2 = section("Sampled (temperature 1 — roll the dice)");
        const p2 = el("p", "bigmath");
        p2.textContent = `${seedText} | ${sampled.join(" ")}`;
        sec2.append(p2);
        out.append(sec2);
        out.append(
          el(
            "p",
            "explain",
            "Early in training this is word salad; after a few epochs the greedy line usually counts properly for a stretch, and the sampled line shows the model's actual uncertainty. Everything right of the | came out of the network — the seed (left of the |) is the only human text.",
          ),
        );
      };
      const controls = el("div", "controls-row");
      controls.append(
        picker("seed: the count after №", 60, seedNum / 20 - 1, (v) => {
          seedNum = (v + 1) * 20;
          run();
        }, (i) => String((i + 1) * 20)),
        button("🔁 generate again (current weights)", run, "btn-play"),
      );
      body.append(controls, out);
      run();
      return undefined;
    },
  });
}
