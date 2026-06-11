// Inference Gallery: draw a digit with the mouse and watch the live forward
// pass — the same computation as training, minus loss and backward.

import { registerPanel } from "../panel";
import { barChart, button, chip, el, heatmap, section } from "../widgets";
import { mulberry32 } from "../../engine/data";

const DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

export function registerGalleryPanels(): void {
  registerPanel("gallery.draw", {
    title: "The Drawing Canvas — run your own inference",
    subtitle:
      "Draw a digit (mouse, hold to paint). Your strokes become a [1, 784] tensor and flow through the exact pipeline from Forward Avenue — using the weights as trained so far.",
    render(body, world) {
      const pixels = new Float32Array(784);
      const SCALE = 10;
      const wrap = el("div", "hstack wrap-row");

      // ------- drawing pad
      const padBox = el("div", "vstack");
      const pad = document.createElement("canvas");
      pad.width = 28 * SCALE;
      pad.height = 28 * SCALE;
      pad.className = "drawpad";
      const pctx = pad.getContext("2d")!;
      const repaint = () => {
        for (let r = 0; r < 28; r++)
          for (let c = 0; c < 28; c++) {
            const v = pixels[r * 28 + c] * 255;
            pctx.fillStyle = `rgb(${v * 0.95 + 12}, ${v * 0.98 + 10}, ${v + 18})`;
            pctx.fillRect(c * SCALE, r * SCALE, SCALE, SCALE);
          }
      };
      let painting = false;
      const paint = (ev: PointerEvent) => {
        const rect = pad.getBoundingClientRect();
        const cx = ((ev.clientX - rect.left) / rect.width) * 28;
        const cy = ((ev.clientY - rect.top) / rect.height) * 28;
        // soft round brush
        for (let r = Math.floor(cy - 2); r <= cy + 2; r++)
          for (let c = Math.floor(cx - 2); c <= cx + 2; c++) {
            if (r < 0 || c < 0 || r >= 28 || c >= 28) continue;
            const d = Math.hypot(c + 0.5 - cx, r + 0.5 - cy);
            const add = Math.max(0, 1.15 - d / 1.7);
            pixels[r * 28 + c] = Math.min(1, pixels[r * 28 + c] + add * 0.8);
          }
        repaint();
        renderForward();
      };
      // pointer events cover mouse, touch and pen; named handlers so the
      // panel teardown can remove the window-level one (it would otherwise
      // leak on every open/close)
      const onPadDown = (e: PointerEvent) => {
        painting = true;
        pad.setPointerCapture?.(e.pointerId);
        paint(e);
        e.preventDefault();
      };
      const onPadMove = (e: PointerEvent) => {
        if (painting) paint(e);
      };
      const onPointerUp = () => {
        painting = false;
      };
      pad.addEventListener("pointerdown", onPadDown);
      pad.addEventListener("pointermove", onPadMove);
      window.addEventListener("pointerup", onPointerUp);
      padBox.append(pad);
      const btnRow = el("div", "controls-row");
      btnRow.append(
        button("🧽 clear", () => { pixels.fill(0); repaint(); renderForward(); }),
        button("🎲 load a test image", () => {
          const i = Math.floor(rand() * world.data.nTest);
          for (let j = 0; j < 784; j++) pixels[j] = world.data.testImages[i * 784 + j] / 255;
          repaint();
          renderForward();
        }),
      );
      const rand = mulberry32(Date.now() & 0xffff);
      padBox.append(btnRow, el("div", "caption", "draw here — thick strokes, centered, work best"));
      wrap.append(padBox);

      // ------- forward pass view
      const fwd = el("div", "vstack forward-view");
      wrap.append(fwd);
      body.append(wrap);

      const renderForward = () => {
        fwd.innerHTML = "";
        const m = world.mlp;
        const r = m.infer(pixels);
        const s1 = el("div", "vstack");
        s1.append(el("div", "caption", "z1 = x·W1 + b1 → a1 = ReLU(z1) — your drawing as 64 hidden features:"));
        s1.append(heatmap(r.a1, 1, 64, { cellSize: 11, symmetric: true }));
        let active = 0;
        for (let i = 0; i < 64; i++) if (r.a1[i] > 0) active++;
        s1.append(el("div", "caption", `${active}/64 features active`));
        s1.append(el("div", "caption", "logits = a1·W2 + b2 — the 10 raw scores:"));
        s1.append(barChart(r.logits, { labels: DIGITS, width: 460, height: 120, highlight: r.pred }));
        s1.append(el("div", "caption", "softmax(logits) — the verdict:"));
        s1.append(barChart(r.probs, { labels: DIGITS, width: 460, height: 120, highlight: r.pred, showValues: false }));
        const conf = r.probs[r.pred] * 100;
        s1.append(el("div", "verdict", `it's a ${r.pred} <span class="verdict-conf">(${conf.toFixed(1)}% sure)</span>`));
        fwd.append(s1);
      };
      repaint();
      renderForward();

      body.append(
        el(
          "p",
          "explain",
          "Note what is NOT here: no loss, no targets, no gradients, no weight updates. Inference = the forward pass alone. If the main pipeline is still training while you draw, you are literally watching the model get better at reading your handwriting between strokes — redraw to re-run.",
        ),
      );
      // re-run inference as training changes the weights
      const off = world.main.on("step", () => {
        // throttled implicitly by training speed; cheap single-sample forward
        renderForward();
      });
      return () => {
        off();
        window.removeEventListener("pointerup", onPointerUp);
      };
    },
  });

  registerPanel("gallery.about", {
    title: "On Frozen Weights — training vs. inference",
    subtitle: "The same forward pass, two different lives.",
    render(body, world) {
      const m = world.mlp;
      const { correct } = m.predictTestAll();
      const s1 = section("What changes when training stops?");
      s1.append(
        el(
          "p",
          "explain",
          "Nothing in the forward math. During training, each forward pass is followed by loss → backward → update, and the weights shift under your feet. At deployment ('inference'), the weights are simply frozen — the final state of W1, b1, W2, b2 — and the forward pass runs alone, as in the drawing canvas next door.",
        ),
      );
      const chips = el("div", "chips-row");
      chips.append(
        chip("this model's frozen-right-now accuracy", `${((correct / world.data.nTest) * 100).toFixed(1)}%`, true),
        chip("steps trained so far", String(m.step)),
        chip("parameters that would ship", "50,890"),
      );
      s1.append(chips);
      body.append(s1);
      const s2 = section("Why this matters");
      s2.append(
        el(
          "p",
          "explain",
          "Everything expensive about deep learning — the foundry, Backprop Works, the Optimizer Depot — exists only at training time. The product that ships is just the Data → Mills → Springs → Mills path plus a softmax: a few matrix multiplies. Pause the main pipeline (HUD, top) and the city's machines go quiet; the gallery keeps working forever on the frozen weights.",
        ),
      );
      body.append(s2);
    },
  });
}
