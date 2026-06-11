// On-screen controls for touch devices: a D-pad, an inspect button, a back
// button, and a run/walk toggle — semi-transparent so the city stays visible.
// They write into the same key-state the keyboard uses.

import { Game } from "../world/game";
import { el } from "./widgets";

export function mountTouchControls(game: Game): void {
  const isTouch =
    window.matchMedia("(pointer: coarse)").matches ||
    navigator.maxTouchPoints > 0 ||
    "ontouchstart" in window ||
    new URLSearchParams(location.search).has("touch"); // dev/testing override
  if (!isTouch) return;

  const layer = el("div", "touch-layer");

  // ---- D-pad (writes w/a/s/d into the game's key set)
  const dpad = el("div", "touch-dpad");
  const dirs: [string, string, string][] = [
    ["▲", "w", "touch-up"],
    ["◀", "a", "touch-left"],
    ["▶", "d", "touch-right"],
    ["▼", "s", "touch-down"],
  ];
  for (const [glyph, key, cls] of dirs) {
    const b = el("button", `touch-btn ${cls}`, glyph);
    const press = (e: Event) => {
      game.keys.add(key);
      e.preventDefault();
    };
    const release = () => game.keys.delete(key);
    b.addEventListener("pointerdown", press);
    b.addEventListener("pointerup", release);
    b.addEventListener("pointercancel", release);
    b.addEventListener("pointerleave", release);
    b.addEventListener("contextmenu", (e) => e.preventDefault());
    dpad.append(b);
  }
  layer.append(dpad);

  // ---- action cluster
  const cluster = el("div", "touch-actions");
  const act = el("button", "touch-btn touch-action", "E");
  act.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "e" }));
  });
  const back = el("button", "touch-btn touch-small", "⟵ back");
  back.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
  });
  // run/walk toggle: the touch equivalent of holding Shift
  const run = el("button", "touch-btn touch-small", "🚶 walk");
  let running = false;
  run.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    running = !running;
    if (running) game.keys.add("shift");
    else game.keys.delete("shift");
    run.textContent = running ? "🏃 run" : "🚶 walk";
    run.classList.toggle("touch-on", running);
  });
  // blur/visibility handlers clear all keys — restore the run state after
  const restoreRun = () => {
    if (running) game.keys.add("shift");
  };
  window.addEventListener("focus", restoreRun);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) restoreRun();
  });
  cluster.append(act, run, back);
  layer.append(cluster);

  document.body.append(layer);
}
