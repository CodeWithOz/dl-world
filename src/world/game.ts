// The game loop: input, scenes (city / interior), camera, prompts, and the
// bridge into the panel system.

import { Avatar } from "./avatar";
import { City, MAP_H, MAP_W, TILE } from "./city";
import { Interior } from "./interior";
import { BuildingDef, doorTile } from "./buildings";
import { World } from "../sim/world";
import { closePanel, hasPanel, isPanelOpen, openPanel } from "../ui/panel";

type Scene =
  | { kind: "city" }
  | { kind: "interior"; interior: Interior; returnTo: { x: number; y: number } };

export class Game {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  world: World;
  city = new City();
  avatar = new Avatar(31.5 * TILE, 20.5 * TILE);
  scene: Scene = { kind: "city" };
  keys = new Set<string>();
  promptEl: HTMLElement;
  blurbEl: HTMLElement;
  time = 0;
  private last = performance.now();
  /** disable movement while a panel is open */
  get inputLocked(): boolean {
    return isPanelOpen();
  }

  constructor(canvas: HTMLCanvasElement, world: World, promptEl: HTMLElement, blurbEl: HTMLElement) {
    this.canvas = canvas;
    this.world = world;
    this.promptEl = promptEl;
    this.blurbEl = blurbEl;
    this.ctx = canvas.getContext("2d")!;
    this.resize();
    window.addEventListener("resize", () => this.resize());
    window.addEventListener("keydown", (e) => this.onKeyDown(e));
    window.addEventListener("keyup", (e) => this.keys.delete(e.key.toLowerCase()));
    requestAnimationFrame(() => this.frame());
  }

  private resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = window.innerWidth * dpr;
    this.canvas.height = window.innerHeight * dpr;
    this.canvas.style.width = `${window.innerWidth}px`;
    this.canvas.style.height = `${window.innerHeight}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  private onKeyDown(e: KeyboardEvent): void {
    const k = e.key.toLowerCase();
    if (k === "escape") {
      if (isPanelOpen()) closePanel();
      else if (this.scene.kind === "interior") this.exitBuilding();
      return;
    }
    if (isPanelOpen()) return; // panels own the keyboard
    this.keys.add(k);
    if (["arrowup", "arrowdown", "arrowleft", "arrowright", " "].includes(k)) e.preventDefault();
    if (k === "e" || k === "enter") this.interact();
  }

  private interact(): void {
    const tx = Math.floor(this.avatar.x / TILE);
    const ty = Math.floor(this.avatar.y / TILE);
    if (this.scene.kind === "city") {
      const b = this.city.doorAt(tx, ty);
      if (b) this.enterBuilding(b);
    } else {
      const s = this.scene.interior.stationNear(this.avatar.x, this.avatar.y);
      if (s && hasPanel(s.id)) {
        openPanel(s.id, this.world);
        return;
      }
      if (this.scene.interior.atExit(this.avatar.x, this.avatar.y)) this.exitBuilding();
    }
  }

  enterBuilding(b: BuildingDef): void {
    const d = doorTile(b);
    const interior = new Interior(b);
    this.scene = {
      kind: "interior",
      interior,
      returnTo: { x: (d.x + 0.5) * TILE, y: (d.y + 1.6) * TILE },
    };
    const sp = interior.spawnPoint();
    this.avatar.x = sp.x;
    this.avatar.y = sp.y;
    this.avatar.dir = "up";
  }

  exitBuilding(): void {
    if (this.scene.kind !== "interior") return;
    const r = this.scene.returnTo;
    this.scene = { kind: "city" };
    this.avatar.x = r.x;
    this.avatar.y = r.y;
    this.avatar.dir = "down";
  }

  private inputVector(): { dx: number; dy: number; run: boolean } {
    let dx = 0;
    let dy = 0;
    if (this.keys.has("w") || this.keys.has("arrowup")) dy -= 1;
    if (this.keys.has("s") || this.keys.has("arrowdown")) dy += 1;
    if (this.keys.has("a") || this.keys.has("arrowleft")) dx -= 1;
    if (this.keys.has("d") || this.keys.has("arrowright")) dx += 1;
    return { dx, dy, run: this.keys.has("shift") };
  }

  private frame(): void {
    const now = performance.now();
    const dt = Math.min((now - this.last) / 1000, 0.1);
    this.last = now;
    this.time += dt;
    this.world.tick(dt);

    if (!this.inputLocked) {
      const input = this.inputVector();
      if (this.scene.kind === "city")
        this.avatar.update(dt, input, (x, y) => this.city.isSolid(x, y), TILE);
      else {
        const interior = this.scene.interior;
        this.avatar.update(dt, input, (x, y) => interior.isSolid(x, y), TILE);
        if (interior.atExit(this.avatar.x, this.avatar.y) && this.avatar.dir === "down" && input.dy > 0)
          this.exitBuilding();
      }
    } else {
      this.keys.clear();
      this.avatar.moving = false;
    }

    this.render();
    this.updatePrompt();
    requestAnimationFrame(() => this.frame());
  }

  private render(): void {
    const ctx = this.ctx;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    ctx.fillStyle = "#274a2a";
    ctx.fillRect(0, 0, vw, vh);

    if (this.scene.kind === "city") {
      const camX = Math.max(0, Math.min(this.avatar.x - vw / 2, MAP_W * TILE - vw));
      const camY = Math.max(0, Math.min(this.avatar.y - vh / 2, MAP_H * TILE - vh));
      ctx.save();
      ctx.translate(-camX, -camY);
      this.city.draw(ctx, { x: camX, y: camY, w: vw, h: vh }, this.time, this.world);
      this.avatar.draw(ctx, this.time);
      ctx.restore();
    } else {
      const interior = this.scene.interior;
      // dim backdrop
      ctx.fillStyle = "#171a22";
      ctx.fillRect(0, 0, vw, vh);
      const roomW = interior.w * TILE;
      const roomH = interior.h * TILE;
      const ox = (vw - roomW) / 2;
      const oy = (vh - roomH) / 2;
      ctx.save();
      ctx.translate(ox, oy);
      const near = interior.stationNear(this.avatar.x, this.avatar.y);
      interior.draw(ctx, this.time, this.world, near?.id ?? null);
      this.avatar.draw(ctx, this.time);
      ctx.restore();
    }
  }

  private updatePrompt(): void {
    let prompt = "";
    let blurb = "";
    if (!isPanelOpen()) {
      const tx = Math.floor(this.avatar.x / TILE);
      const ty = Math.floor(this.avatar.y / TILE);
      if (this.scene.kind === "city") {
        const b = this.city.doorAt(tx, ty);
        if (b) {
          prompt = `Press E — enter ${b.name}`;
          blurb = b.blurb;
        }
      } else {
        const s = this.scene.interior.stationNear(this.avatar.x, this.avatar.y);
        if (s) prompt = `Press E — inspect ${s.name}`;
        else if (this.scene.interior.atExit(this.avatar.x, this.avatar.y))
          prompt = "Press E (or walk down) — back to the city";
        blurb = this.scene.interior.building.blurb;
      }
    }
    if (this.promptEl.textContent !== prompt) this.promptEl.textContent = prompt;
    this.promptEl.style.opacity = prompt ? "1" : "0";
    if (this.blurbEl.textContent !== blurb) this.blurbEl.textContent = blurb;
    this.blurbEl.style.opacity = blurb ? "1" : "0";
  }
}
