// The always-visible HUD: main-pipeline training controls along the top,
// movement hints at the bottom.

import { World } from "../sim/world";
import { el, fmt } from "./widgets";

export function mountHud(world: World): void {
  const hud = el("div", "hud");
  const title = el("div", "hud-title", "🏙 <b>DL WORLD</b>");
  hud.append(title);

  // ︎ forces text presentation — iOS would otherwise render ▶/⏭/⏸ as emoji
  const play = el("button", "hud-btn hud-play", "▶︎ train");
  play.addEventListener("click", () => world.main.toggle());
  const step = el("button", "hud-btn", "⏭︎ step");
  step.addEventListener("click", () => world.main.stepOnce());
  hud.append(play, step);

  const speed = el("label", "hud-speed", "speed ");
  const speedInput = document.createElement("input");
  speedInput.type = "range";
  speedInput.min = "1";
  speedInput.max = "60";
  speedInput.value = String(world.main.speed);
  speedInput.addEventListener("input", () => {
    world.main.speed = parseInt(speedInput.value);
  });
  speed.append(speedInput);
  hud.append(speed);

  const stats = el("div", "hud-stats");
  hud.append(stats);
  document.body.append(hud);

  const hint = el(
    "div",
    "hud-hint",
    "WASD / arrows — walk · <b>E</b> — enter / inspect · <b>Esc</b> — back · Shift — run",
  );
  document.body.append(hint);

  const update = () => {
    const s = world.mlp;
    stats.innerHTML =
      `<span class="hud-stat">epoch <b>${s.epoch}</b></span>` +
      `<span class="hud-stat">step <b>${s.step}</b></span>` +
      `<span class="hud-stat">loss <b>${fmt(world.main.lastLossValue, 3)}</b></span>` +
      `<span class="hud-stat">test acc <b>${(world.main.lastMetricValue * 100).toFixed(1)}%</b></span>` +
      `<span class="hud-stat">lr <b>${fmt(s.opt.lr, 3)}</b></span>`;
  };
  world.main.on("state", () => {
    play.textContent = world.main.running ? "⏸︎ pause" : "▶︎ train";
    play.classList.toggle("hud-running", world.main.running);
  });
  update();
  window.setInterval(update, 300);
}
