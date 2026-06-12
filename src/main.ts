import "./style.css";
import { loadMnist } from "./engine/data";
import { World } from "./sim/world";
import { Game } from "./world/game";
import { mountHud } from "./ui/hud";
import { mountTouchControls } from "./ui/touch";
import { registerAllPanels } from "./ui/panels/index";
import { closePanel, openPanel } from "./ui/panel";
import { el } from "./ui/widgets";

declare global {
  interface Window {
    /** debugging/automation handle (e.g. teleporting while developing) */
    dlworld?: {
      game: Game;
      world: World;
      openPanel: (id: string) => void;
      closePanel: () => void;
    };
  }
}

function welcome(onStart: () => void): void {
  const overlay = el("div", "welcome");
  overlay.innerHTML = `
    <div class="welcome-card">
      <h1>🏙 Welcome to DL WORLD</h1>
      <p class="gate-sub">the city inside the black box</p>
      <p>A real two-layer neural network is training on real MNIST digits in your
      browser right now — and this city <i>is</i> that computation. Every building
      opens up one piece of it, all the way down to single multiply-adds.</p>
      <p><b>For the guided route, follow the numbered signs ① → ㉑</b> — they trace
      the full story: data → forward pass → loss → backprop → optimizer → metrics,
      then the historical baselines, learning-rate tuning, other kinds of targets,
      and inference, where you draw digits for the trained model to read.
      Then cross the river south into <b>the Frontier</b>, where the data stops being
      images: training tricks, movie-taste embeddings, decision-tree forests, text
      pipelines and a tiny language model — all training live too.
      Or ignore the numbers and wander — every door is open.</p>
      <p><b>Keyboard:</b> WASD / arrows to walk · <b>E</b> to enter &amp; inspect ·
      <b>Esc</b> to go back · Shift to run.<br>
      <b>Touch:</b> use the on-screen pad and buttons.</p>
      <p>Press <b>▶&#xFE0E; train</b> in the top bar and the whole city comes alive.
      Every number on every panel is live — nothing is canned.</p>
      <button class="btn btn-play welcome-btn">step into the city →</button>
    </div>`;
  overlay.querySelector("button")!.addEventListener("click", () => {
    overlay.remove();
    onStart();
  });
  document.body.append(overlay);
}

function showScreen(title: string, message: string): HTMLElement {
  const screen = el("div", "gate");
  const card = el("div", "gate-card");
  card.append(el("h1", "", title));
  const p = el("p");
  p.textContent = message; // textContent: error strings must never render as HTML
  card.append(p);
  screen.append(card);
  document.body.append(screen);
  return screen;
}

async function boot(): Promise<void> {
  const loading = showScreen("DL WORLD", "unpacking 3,600 handwritten digits…");
  const data = await loadMnist();
  loading.remove();

  const world = new World(data);
  registerAllPanels();
  const canvas = document.createElement("canvas");
  canvas.id = "game";
  document.body.append(canvas);
  const prompt = el("div", "prompt");
  const blurb = el("div", "blurb");
  document.body.append(prompt, blurb);
  const game = new Game(canvas, world, prompt, blurb);
  mountHud(world);
  mountTouchControls(game);
  window.dlworld = {
    game,
    world,
    openPanel: (id: string) => openPanel(id, world),
    closePanel,
  };
  welcome(() => {
    world.main.setRunning(true);
  });
}

boot().catch((err) => {
  showScreen("DL WORLD", `failed to load: ${err}`);
});
