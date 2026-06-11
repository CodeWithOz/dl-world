import "./style.css";
import { loadMnist } from "./engine/data";
import { World } from "./sim/world";
import { Game } from "./world/game";
import { mountHud } from "./ui/hud";
import { registerAllPanels } from "./ui/panels/index";
import { closePanel, openPanel } from "./ui/panel";
import { el } from "./ui/widgets";

function isDesktop(): boolean {
  // Gate on viewport size, not pointer type — touch laptops are fine.
  // ?nogate skips the check (useful for development/automation).
  if (new URLSearchParams(location.search).has("nogate")) return true;
  return window.innerWidth >= 980 && window.innerHeight >= 560;
}

function mobileGate(): void {
  const gate = el("div", "gate");
  gate.innerHTML = `
    <div class="gate-card">
      <div class="gate-skyline">🏭🏛🏗🗼🏡</div>
      <h1>DL WORLD</h1>
      <p class="gate-sub">a walkable city inside a neural network</p>
      <p>The city gates are closed to travelers on small screens.<br>
      DL World needs a desktop browser — a real keyboard to stroll
      Forward Avenue, and a wide viewport to read the tensors flowing
      through the Cross-Entropy Foundry.</p>
      <p class="gate-foot">Come back on a laptop or desktop. The mills will be running. ⚙️</p>
    </div>`;
  document.body.append(gate);
}

function welcome(onStart: () => void): void {
  const overlay = el("div", "welcome");
  overlay.innerHTML = `
    <div class="welcome-card">
      <h1>🏙 Welcome to DL WORLD</h1>
      <p class="gate-sub">the city inside the black box</p>
      <p>A real two-layer neural network is training on real MNIST digits in your
      browser right now — and this city <i>is</i> that computation. Every building
      maps to a piece of the fastai (ch. 4–6) story:</p>
      <ul>
        <li><b>Data Quarter</b> → images as tensors, the shuffling DataLoader</li>
        <li><b>Forward Avenue</b> → the matmul mills and the ReLU spring</li>
        <li><b>Loss District</b> → cross-entropy as an assembly line</li>
        <li><b>Gradient Row</b> → backprop, the chain rule, and the SGD update</li>
        <li><b>South side</b> → ch.4's 3-vs-7 learner &amp; pixel baseline, ch.5's LR finder,
            ch.6's multi-label &amp; regression, and a gallery where you draw digits
            for the trained model to read</li>
      </ul>
      <p>Walk with <b>WASD</b>, enter buildings and inspect machines with <b>E</b>,
      leave with <b>Esc</b>. Press <b>▶ train</b> in the top bar and the whole city
      comes alive. Every number on every panel is live — nothing is canned.</p>
      <button class="btn btn-play welcome-btn">step into the city →</button>
    </div>`;
  overlay.querySelector("button")!.addEventListener("click", () => {
    overlay.remove();
    onStart();
  });
  document.body.append(overlay);
}

async function boot(): Promise<void> {
  if (!isDesktop()) {
    mobileGate();
    return;
  }
  const loading = el("div", "gate");
  loading.innerHTML = `<div class="gate-card"><h1>DL WORLD</h1><p>unpacking 3,600 handwritten digits…</p></div>`;
  document.body.append(loading);
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
  // debugging/automation handle (e.g. teleporting while developing)
  (window as unknown as { dlworld: object }).dlworld = {
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
  const e = el("div", "gate");
  e.innerHTML = `<div class="gate-card"><h1>DL WORLD</h1><p>failed to load: ${err}</p></div>`;
  document.body.append(e);
});
