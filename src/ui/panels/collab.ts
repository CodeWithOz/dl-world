// Taste Cinema: collaborative filtering. No images anywhere — the dataset
// is a sparse grid of star ratings, and the model is two embedding tables
// plus biases. The planted "genre truth" behind the generator is revealed
// in the Factor Projector so you can see the embeddings rediscover it.

import { registerPanel } from "../panel";
import { chip, el, fmt, heatmap, numberStrip, section } from "../widgets";
import { fmtMetric, liveRegion, picker, trainerControls } from "./common";
import { World } from "../../sim/world";
import { CollabFilter } from "../../sim/scenarios2";

function c(world: World): CollabFilter {
  return world.collab;
}

const GENRE_COLORS = ["#e8807a", "#e88ab0", "#5aa7d9"]; // action, romance, sci-fi

export function registerCollabPanels(): void {
  registerPanel("cinema.ratings", {
    title: "The Ratings Wall — the entire dataset is this grid",
    subtitle:
      "40 movie-goers × 24 (entirely fictional) movies. Each filled cell is a 0.5–5 star rating; the gaps are movies that person never watched. There are no pixels, no captions, no genres — ratings are ALL the model gets.",
    render(body, world) {
      const s = c(world);
      const nU = s.cd.users.length;
      const nM = s.cd.movies.length;
      const grid = new Float32Array(nU * nM);
      let filled = 0;
      for (const r of [...s.cd.train, ...s.cd.test]) {
        grid[r.u * nM + r.m] = r.r;
        filled++;
      }
      const s1 = section("Users (rows) × movies (columns), brightness = stars");
      s1.append(heatmap(grid, nU, nM, { cellSize: 9 }));
      const chips = el("div", "chips-row");
      chips.append(
        chip("ratings observed", String(filled)),
        chip("grid cells", String(nU * nM)),
        chip("held out for testing", String(s.cd.test.length), true),
      );
      s1.append(chips);
      body.append(s1);

      const s2 = section("A few raw rows of the dataset");
      const sample = s.cd.train.slice(0, 6);
      const table = el("table", "num-table");
      table.append(el("tr", "", "<th>who</th><th>watched</th><th>rated</th>"));
      for (const r of sample) {
        const row = el("tr");
        row.append(
          el("td", "", s.cd.users[r.u]),
          el("td", "", s.cd.movies[r.m].title),
          el("td", "", `${"★".repeat(Math.round(r.r))} (${fmt(r.r, 1)})`),
        );
        table.append(row);
      }
      s2.append(table);
      s2.append(
        el(
          "p",
          "explain",
          "The task: predict the empty cells. The trick: invent a small vector of numbers (an embedding) for every user and every movie, and make matching dot products produce matching ratings. Nobody tells the model what the numbers should mean.",
        ),
      );
      body.append(s2);
    },
  });

  registerPanel("cinema.bench", {
    title: "Training Bench — SGD on embeddings",
    subtitle:
      "Exactly the same loop as the digit pipeline — batch, forward, MSE loss, backward, step — but the parameters being nudged are the taste vectors themselves.",
    render(body, world) {
      const s = c(world);
      const [controls, cleanCtl] = trainerControls(world.cinema);
      body.append(controls);
      const [live, cleanLive] = liveRegion(world.cinema, (root) => {
        const chips = el("div", "chips-row");
        chips.append(
          chip("parameters", String(s.params.reduce((a, p) => a + p.tensor.size, 0))),
          chip("test RMSE", fmtMetric(s.metricName, world.cinema.lastMetricValue), true),
        );
        root.append(chips);
        root.append(
          el(
            "p",
            "explain",
            "An RMSE of 0.5 means predictions are off by about half a star on ratings the model never saw. The noise planted in the generator is ~0.25 stars, so there is a real floor — no model can be perfect here.",
          ),
        );
        const wdNode = s.nodes["λ·Σfactors² (weight decay)"];
        if (wdNode)
          root.append(
            el(
              "p",
              "explain",
              `Look closely at the loss graph and you'll find a second ingredient: weight decay, λ·Σ(factors²) = ${fmt(wdNode.item(), 4)} this step. Dot-product models overfit small rating sets easily; adding the squared weights to the loss keeps every factor as small as it can get away with.`,
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

  registerPanel("cinema.dot", {
    title: "Dot-Product Desk — one prediction, every number",
    subtitle:
      "Pick a person and a movie. The predicted rating is just: multiply their two little vectors element-by-element, sum, add both biases, squash into the star range.",
    render(body, world) {
      const s = c(world);
      let u = 0;
      let m = 0;
      const controls = el("div", "controls-row");
      const [live, cleanLive, refresh] = liveRegion(world.cinema, (root) => {
        const d = s.inspect(u, m);
        const row = el("div", "hstack wrap-row");
        const col = (title: string, values: number[]) => {
          const v = el("div", "vstack");
          v.append(el("div", "caption", title), numberStrip(Float32Array.from(values), -1, 3));
          return v;
        };
        row.append(
          col(`${s.cd.users[u]}'s factors`, d.uf),
          el("div", "flow-arrow", "×"),
          col("movie's factors", d.mf),
          el("div", "flow-arrow", "="),
          col("products", d.products),
        );
        root.append(row);
        root.append(
          el(
            "div",
            "bigmath",
            `dot = <b>${fmt(d.dot, 4)}</b> &nbsp;+ user bias <b>${fmt(d.ub, 4)}</b> + movie bias <b>${fmt(d.mb, 4)}</b><br>` +
              `pred = sigmoid(${fmt(d.dot + d.ub + d.mb, 4)}) · 5.5 = <b>${fmt(d.pred, 3)} stars</b>`,
          ),
        );
        const chips = el("div", "chips-row");
        chips.append(chip("model says", `${fmt(d.pred, 2)} ★`, true));
        chips.append(
          d.known !== null
            ? chip(`${s.cd.users[u]} actually rated it`, `${fmt(d.known, 1)} ★`)
            : chip("actual rating", "never watched — this is a true prediction"),
        );
        root.append(chips);
        root.append(
          el(
            "p",
            "explain",
            "The biases learn \"this person is generous\" / \"this movie is broadly liked\", so the factors are free to encode taste. These values update live while the bench trains.",
          ),
        );
      });
      controls.append(
        picker("person", s.cd.users.length, u, (v) => {
          u = v;
          refresh();
        }, (i) => s.cd.users[i]),
        picker("movie", s.cd.movies.length, m, (v) => {
          m = v;
          refresh();
        }, (i) => s.cd.movies[i].title),
      );
      body.append(controls, live);
      return cleanLive;
    },
  });

  registerPanel("cinema.factors", {
    title: "Factor Projector — what did the embeddings discover?",
    subtitle:
      "Each dot is one movie's learned factor vector. The colors are the secret: the generator planted action/romance/sci-fi structure in the ratings, but the model was never told — if the colors separate, it rediscovered genre on its own.",
    render(body, world) {
      const s = c(world);
      let dx = 0;
      let dy = 1;
      const controls = el("div", "controls-row");
      const [live, cleanLive, refresh] = liveRegion(world.cinema, (root) => {
        const W = 440;
        const H = 320;
        const canvas = document.createElement("canvas");
        canvas.width = W * 2;
        canvas.height = H * 2;
        canvas.style.width = `${W}px`;
        canvas.style.height = `${H}px`;
        const ctx = canvas.getContext("2d")!;
        ctx.scale(2, 2);
        const xs = s.cd.movies.map((_, i) => s.M.at(i, dx));
        const ys = s.cd.movies.map((_, i) => s.M.at(i, dy));
        const xmin = Math.min(...xs),
          xmax = Math.max(...xs),
          ymin = Math.min(...ys),
          ymax = Math.max(...ys);
        const px = (v: number) => 14 + ((v - xmin) / (xmax - xmin || 1)) * (W - 28);
        const py = (v: number) => H - 22 - ((v - ymin) / (ymax - ymin || 1)) * (H - 44);
        ctx.strokeStyle = "rgba(255,255,255,0.15)";
        ctx.strokeRect(6, 6, W - 12, H - 12);
        ctx.font = "9.5px ui-monospace, monospace";
        s.cd.movies.forEach((mv, i) => {
          const dominant = mv.genre.indexOf(Math.max(...mv.genre));
          ctx.fillStyle = GENRE_COLORS[dominant];
          ctx.beginPath();
          ctx.arc(px(xs[i]), py(ys[i]), 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = "rgba(255,255,255,0.75)";
          ctx.fillText(mv.title.slice(0, 18), px(xs[i]) + 6, py(ys[i]) + 3);
        });
        root.append(canvas);
        const legend = el("div", "legend-row");
        s.cd.genreNames.forEach((g, i) => {
          legend.append(el("span", "", `<span style="color:${GENRE_COLORS[i]}">●</span> planted ${g}`));
        });
        root.append(legend);
        root.append(
          el(
            "p",
            "explain",
            "Fresh embeddings start as random scatter. Train the bench for a couple of epochs and watch the same-color movies drift together: dot products force movies that the same people like into the same directions. That clustering IS what \"latent factors\" means.",
          ),
        );
        // the book's favorite interpretation: the learned *bias* says how
        // liked a movie is even after taste-matching is accounted for
        const order = s.cd.movies.map((_, i) => i).sort((a, b) => s.Mb.at(b, 0) - s.Mb.at(a, 0));
        const biasRow = el("div", "hstack wrap-row");
        const biasCol = (title: string, list: number[]) => {
          const v = el("div", "vstack");
          v.append(el("div", "caption", title));
          const t = el("table", "num-table");
          for (const i of list) {
            const tr = el("tr");
            tr.append(el("td", "", s.cd.movies[i].title), el("td", "mono", fmt(s.Mb.at(i, 0), 3)));
            t.append(tr);
          }
          v.append(t);
          return v;
        };
        const sec = section(
          "The learned movie biases",
          "A high bias means people rate this movie above what their taste-match predicts — crowd-pleasers. A low bias means even well-matched viewers come away disappointed.",
        );
        biasRow.append(
          biasCol("highest bias", order.slice(0, 4)),
          biasCol("lowest bias", order.slice(-4).reverse()),
        );
        sec.append(biasRow);
        root.append(sec);
      }, 700);
      controls.append(
        picker("x axis: factor", s.k, dx, (v) => {
          dx = v;
          refresh();
        }),
        picker("y axis: factor", s.k, dy, (v) => {
          dy = v;
          refresh();
        }),
      );
      body.append(controls, live);
      return cleanLive;
    },
  });
}
