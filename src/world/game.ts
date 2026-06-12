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
  // spawn beside the tour kiosk so the "ride the express to ①" hint is
  // the first thing a new player sees
  avatar = new Avatar(30.5 * TILE, 22.5 * TILE);
  scene: Scene = { kind: "city" };
  keys = new Set<string>();
  promptEl: HTMLElement;
  blurbEl: HTMLElement;
  time = 0;
  private last = performance.now();
  /** the tour express: a short scripted flight from the kiosk to stop ① */
  private travel: { pts: { x: number; y: number }[]; lens: number[]; total: number; t: number; dur: number } | null = null;
  private trail: { x: number; y: number; age: number }[] = [];
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
    // keyup events are missed while unfocused/hidden — don't let keys stick
    window.addEventListener("blur", () => this.keys.clear());
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) this.keys.clear();
    });
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
    if (this.travel) return;
    const tx = Math.floor(this.avatar.x / TILE);
    const ty = Math.floor(this.avatar.y / TILE);
    if (this.scene.kind === "city") {
      if (this.city.kioskNear(tx, ty)) {
        this.startTourExpress();
        return;
      }
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

  /** the kiosk ride: glide up the plaza and west along the road to the
   *  Dataset Warehouse — the same route the tour itself takes, in reverse */
  startTourExpress(): void {
    if (this.scene.kind !== "city" || this.travel) return;
    const pts = [
      { x: this.avatar.x, y: this.avatar.y },
      { x: 28.5 * TILE, y: 21.5 * TILE },
      { x: 28.5 * TILE, y: 14.5 * TILE },
      { x: 10.5 * TILE, y: 14.5 * TILE },
      { x: 10.5 * TILE, y: 13.55 * TILE },
    ];
    const lens: number[] = [];
    let total = 0;
    for (let i = 1; i < pts.length; i++) {
      const L = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
      lens.push(L);
      total += L;
    }
    this.travel = { pts, lens, total, t: 0, dur: Math.max(1.4, total / 420) };
  }

  private updateTravel(dt: number): void {
    const tr = this.travel!;
    tr.t += dt;
    const raw = Math.min(tr.t / tr.dur, 1);
    const u = raw * raw * (3 - 2 * raw); // ease in/out
    let dist = u * tr.total;
    let i = 0;
    while (i < tr.lens.length - 1 && dist > tr.lens[i]) {
      dist -= tr.lens[i];
      i++;
    }
    const a = tr.pts[i];
    const b = tr.pts[i + 1];
    const f = tr.lens[i] === 0 ? 0 : Math.min(dist / tr.lens[i], 1);
    this.avatar.x = a.x + (b.x - a.x) * f;
    this.avatar.y = a.y + (b.y - a.y) * f;
    this.avatar.moving = true;
    this.avatar.dir =
      Math.abs(b.x - a.x) > Math.abs(b.y - a.y) ? (b.x > a.x ? "right" : "left") : b.y > a.y ? "down" : "up";
    this.trail.push({ x: this.avatar.x, y: this.avatar.y, age: 0 });
    if (raw >= 1) {
      this.travel = null;
      this.avatar.moving = false;
      this.avatar.dir = "up";
    }
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

    // fade the express trail even after arrival
    for (const p of this.trail) p.age += dt;
    this.trail = this.trail.filter((p) => p.age < 0.7);

    if (this.travel) {
      this.updateTravel(dt);
    } else if (!this.inputLocked) {
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
      // don't clear keys here: keyup still fires globally, and wiping the set
      // would also kill the touch run-toggle's held "shift"
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
      // the express ride's sparkle trail
      for (const p of this.trail) {
        const a = 0.5 * (1 - p.age / 0.7);
        const g = ctx.createRadialGradient(p.x, p.y, 1, p.x, p.y, 10);
        g.addColorStop(0, `rgba(255, 235, 130, ${a})`);
        g.addColorStop(1, "rgba(255, 235, 130, 0)");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
        ctx.fill();
      }
      this.avatar.draw(ctx, this.time);
      ctx.restore();
    } else {
      const interior = this.scene.interior;
      // dim backdrop
      ctx.fillStyle = "#171a22";
      ctx.fillRect(0, 0, vw, vh);
      const roomW = interior.w * TILE;
      const roomH = interior.h * TILE;
      // center the room when it fits; otherwise clamp-follow the avatar,
      // the same camera behavior as the street view (small screens)
      const ox =
        roomW <= vw
          ? (vw - roomW) / 2
          : -Math.max(0, Math.min(this.avatar.x - vw / 2, roomW - vw));
      const oy =
        roomH <= vh
          ? (vh - roomH) / 2
          : -Math.max(0, Math.min(this.avatar.y - vh / 2, roomH - vh));
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
    if (!isPanelOpen() && !this.travel) {
      const tx = Math.floor(this.avatar.x / TILE);
      const ty = Math.floor(this.avatar.y / TILE);
      if (this.scene.kind === "city") {
        const b = this.city.doorAt(tx, ty);
        if (b) {
          prompt = `Press E — enter ${b.name}`;
          blurb = b.blurb;
        } else if (this.city.kioskNear(tx, ty)) {
          prompt = "Press E — ride the express to tour stop ①";
          blurb =
            "The guided tour starts at the Dataset Warehouse in the north-west. Stops ① → ㉑ trace the whole story: data → forward → loss → backward → step → metrics — then across the river, where the data stops being images.";
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
