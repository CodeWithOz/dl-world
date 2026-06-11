// The city: terrain, collision, and rendering. Everything is drawn
// procedurally — no image assets.

import { BUILDINGS, BuildingDef, doorTile } from "./buildings";
import { World } from "../sim/world";
import { mulberry32 } from "../engine/data";

export const TILE = 32;
export const MAP_W = 66;
export const MAP_H = 38;

const enum T {
  Grass = 0,
  Road = 1,
  Plaza = 2,
  Water = 3,
  Tree = 4,
  Building = 5,
  Door = 6,
  Fountain = 7,
  Flower = 8,
}

export class City {
  tiles = new Uint8Array(MAP_W * MAP_H);
  /** building occupying each tile (index into BUILDINGS) or -1 */
  owner = new Int16Array(MAP_W * MAP_H).fill(-1);

  constructor() {
    this.generate();
  }

  private set(x: number, y: number, t: T, owner = -1): void {
    if (x < 0 || y < 0 || x >= MAP_W || y >= MAP_H) return;
    this.tiles[y * MAP_W + x] = t;
    this.owner[y * MAP_W + x] = owner;
  }

  get(x: number, y: number): T {
    if (x < 0 || y < 0 || x >= MAP_W || y >= MAP_H) return T.Tree;
    return this.tiles[y * MAP_W + x] as T;
  }

  private rect(x0: number, y0: number, w: number, h: number, t: T, owner = -1): void {
    for (let y = y0; y < y0 + h; y++)
      for (let x = x0; x < x0 + w; x++) this.set(x, y, t, owner);
  }

  private generate(): void {
    // roads
    this.rect(2, 13, MAP_W - 4, 2, T.Road);
    this.rect(2, 22, MAP_W - 4, 2, T.Road);
    this.rect(2, 31, MAP_W - 4, 2, T.Road);
    this.rect(2, 13, 2, 20, T.Road);
    this.rect(MAP_W - 4, 13, 2, 20, T.Road);
    this.rect(30, 13, 2, 20, T.Road);
    // plaza + fountain
    this.rect(26, 16, 11, 6, T.Plaza);
    this.rect(31, 17, 2, 2, T.Fountain);
    // pond (bottom left)
    this.rect(13, 34, 8, 3, T.Water);
    this.rect(14, 33, 6, 1, T.Water);
    // buildings (door punched into bottom wall)
    BUILDINGS.forEach((b, i) => {
      this.rect(b.x, b.y, b.w, b.h, T.Building, i);
      const d = doorTile(b);
      this.set(d.x, d.y, T.Door, i);
      // short path from door to the road below
      let y = d.y + 1;
      while (y < MAP_H && this.get(d.x, y) === T.Grass) {
        this.set(d.x, y, T.Road);
        y++;
      }
    });
    // border trees + scattered decor
    const rand = mulberry32(5150);
    for (let x = 0; x < MAP_W; x++) {
      this.set(x, 0, T.Tree);
      this.set(x, 1, T.Tree);
      this.set(x, MAP_H - 1, T.Tree);
      if (rand() < 0.7) this.set(x, MAP_H - 2, T.Tree);
    }
    for (let y = 0; y < MAP_H; y++) {
      this.set(0, y, T.Tree);
      this.set(1, y, T.Tree);
      this.set(MAP_W - 1, y, T.Tree);
      if (rand() < 0.7) this.set(MAP_W - 2, y, T.Tree);
    }
    for (let i = 0; i < 240; i++) {
      const x = 2 + Math.floor(rand() * (MAP_W - 4));
      const y = 2 + Math.floor(rand() * (MAP_H - 4));
      if (this.get(x, y) !== T.Grass) continue;
      // keep walking corridors near roads clear
      this.set(x, y, rand() < 0.35 ? T.Tree : T.Flower);
    }
  }

  isSolid(x: number, y: number): boolean {
    const t = this.get(x, y);
    return t === T.Tree || t === T.Water || t === T.Building || t === T.Fountain;
  }

  /** building whose door is at/adjacent to this tile, for the enter prompt */
  doorAt(x: number, y: number): BuildingDef | null {
    for (const [dx, dy] of [[0, 0], [0, -1], [1, 0], [-1, 0], [0, 1]]) {
      const t = this.get(x + dx, y + dy);
      if (t === T.Door) {
        const idx = this.owner[(y + dy) * MAP_W + (x + dx)];
        if (idx >= 0) return BUILDINGS[idx];
      }
    }
    return null;
  }

  // ------------------------------------------------------------- drawing ---

  draw(
    ctx: CanvasRenderingContext2D,
    cam: { x: number; y: number; w: number; h: number },
    time: number,
    world: World,
  ): void {
    const x0 = Math.max(0, Math.floor(cam.x / TILE) - 1);
    const y0 = Math.max(0, Math.floor(cam.y / TILE) - 1);
    const x1 = Math.min(MAP_W, Math.ceil((cam.x + cam.w) / TILE) + 1);
    const y1 = Math.min(MAP_H, Math.ceil((cam.y + cam.h) / TILE) + 1);

    for (let y = y0; y < y1; y++)
      for (let x = x0; x < x1; x++) this.drawTile(ctx, x, y, time);

    // buildings drawn as whole structures on top of their tiles
    for (const b of BUILDINGS) {
      if (b.x + b.w < x0 || b.x > x1 || b.y + b.h < y0 || b.y > y1) continue;
      this.drawBuilding(ctx, b, time, world);
    }

    this.drawTourArrows(ctx);
    this.drawDistrictLabels(ctx);
    this.drawLoopPulses(ctx, time, world);
  }

  private drawTile(ctx: CanvasRenderingContext2D, x: number, y: number, time: number): void {
    const t = this.get(x, y);
    const px = x * TILE;
    const py = y * TILE;
    // grass base everywhere
    const checker = (x + y) % 2 === 0;
    ctx.fillStyle = checker ? "#79b75c" : "#74b157";
    ctx.fillRect(px, py, TILE, TILE);
    switch (t) {
      case T.Road: {
        ctx.fillStyle = "#b8af9e";
        ctx.fillRect(px, py, TILE, TILE);
        ctx.fillStyle = "rgba(0,0,0,0.05)";
        if ((x * 7 + y * 13) % 5 === 0) ctx.fillRect(px + 6, py + 6, 4, 4);
        break;
      }
      case T.Plaza: {
        ctx.fillStyle = checker ? "#d6cdb8" : "#cfc5af";
        ctx.fillRect(px, py, TILE, TILE);
        break;
      }
      case T.Water: {
        ctx.fillStyle = "#5aa7d9";
        ctx.fillRect(px, py, TILE, TILE);
        ctx.fillStyle = "rgba(255,255,255,0.25)";
        const ph = Math.sin(time * 2 + x * 1.7 + y * 2.3) * 3;
        ctx.fillRect(px + 4, py + 12 + ph, 12, 2);
        break;
      }
      case T.Tree: {
        ctx.fillStyle = "#6b4a2f";
        ctx.fillRect(px + 13, py + 18, 6, 10);
        ctx.fillStyle = "#3e8a3e";
        ctx.beginPath();
        ctx.arc(px + 16, py + 12, 11, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "rgba(255,255,255,0.12)";
        ctx.beginPath();
        ctx.arc(px + 13, py + 9, 5, 0, Math.PI * 2);
        ctx.fill();
        break;
      }
      case T.Flower: {
        ctx.fillStyle = ["#e8c84d", "#e88ab0", "#9ad1e8"][(x * 31 + y * 17) % 3];
        ctx.beginPath();
        ctx.arc(px + 10 + ((x * 13) % 12), py + 10 + ((y * 11) % 12), 3, 0, Math.PI * 2);
        ctx.fill();
        break;
      }
      case T.Fountain: {
        ctx.fillStyle = "#9fb6c9";
        ctx.fillRect(px, py, TILE, TILE);
        ctx.fillStyle = "#5aa7d9";
        ctx.fillRect(px + 3, py + 3, TILE - 6, TILE - 6);
        const ph = (time * 40) % 16;
        ctx.fillStyle = "rgba(255,255,255,0.7)";
        ctx.beginPath();
        ctx.arc(px + TILE / 2, py + TILE / 2 - ph / 3, 2.5, 0, Math.PI * 2);
        ctx.fill();
        break;
      }
      default:
        // occasional grass tufts
        if ((x * 29 + y * 37) % 11 === 0) {
          ctx.fillStyle = "rgba(40,90,30,0.35)";
          ctx.fillRect(px + 8, py + 14, 2, 5);
          ctx.fillRect(px + 12, py + 12, 2, 7);
          ctx.fillRect(px + 16, py + 15, 2, 4);
        }
    }
  }

  private drawBuilding(
    ctx: CanvasRenderingContext2D,
    b: BuildingDef,
    time: number,
    world: World,
  ): void {
    const px = b.x * TILE;
    const py = b.y * TILE;
    const w = b.w * TILE;
    const h = b.h * TILE;
    const roofH = Math.floor(h * 0.32);
    const running =
      b.trainer !== null && world[b.trainer as "main"].running;

    // shadow
    ctx.fillStyle = "rgba(0,0,0,0.15)";
    ctx.fillRect(px + 4, py + h - 4, w, 8);
    // walls
    ctx.fillStyle = b.color;
    ctx.fillRect(px, py + roofH, w, h - roofH);
    ctx.fillStyle = "rgba(0,0,0,0.08)";
    ctx.fillRect(px, py + roofH, 6, h - roofH);
    ctx.fillRect(px + w - 6, py + roofH, 6, h - roofH);
    // roof
    ctx.fillStyle = b.roof;
    ctx.fillRect(px - 3, py, w + 6, roofH);
    ctx.fillStyle = "rgba(255,255,255,0.12)";
    ctx.fillRect(px - 3, py, w + 6, 6);
    // windows (glow when the building's trainer is running)
    const winGlow = running ? 0.65 + 0.35 * Math.sin(time * 5) : 0;
    const cols = Math.max(2, Math.floor(b.w / 2));
    for (let i = 0; i < cols; i++) {
      const wx = px + 14 + (i * (w - 28)) / Math.max(cols - 1, 1);
      const wy = py + roofH + 12;
      ctx.fillStyle = running
        ? `rgba(255, 220, 120, ${0.55 + winGlow * 0.45})`
        : "rgba(40, 50, 70, 0.75)";
      ctx.fillRect(wx - 6, wy, 12, 14);
      ctx.strokeStyle = "rgba(0,0,0,0.3)";
      ctx.strokeRect(wx - 6, wy, 12, 14);
    }
    // door
    const d = doorTile(b);
    const dx = d.x * TILE;
    const dy = d.y * TILE;
    ctx.fillStyle = "#3a3026";
    ctx.fillRect(dx + 4, dy + 2, TILE - 8, TILE - 2);
    ctx.fillStyle = "rgba(255, 226, 130, 0.9)";
    ctx.fillRect(dx + 8, dy + 8, TILE - 16, 6);
    // sign, with the guided-tour stop number as a gold badge
    const label = `${b.icon} ${b.name}`;
    ctx.font = "bold 12px 'Trebuchet MS', sans-serif";
    const badgeR = 8;
    const textW = ctx.measureText(label).width;
    const tw = textW + badgeR * 2 + 22;
    const sx = px + w / 2 - tw / 2;
    const sy = py + roofH - 9;
    ctx.fillStyle = "rgba(34, 28, 20, 0.88)";
    this.roundedRect(ctx, sx, sy, tw, 18, 5);
    ctx.fill();
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#ffd34d";
    ctx.beginPath();
    ctx.arc(sx + badgeR + 5, sy + 9, badgeR, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#2a2417";
    ctx.font = "bold 11px 'Trebuchet MS', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(String(b.tour), sx + badgeR + 5, sy + 9.5);
    ctx.font = "bold 12px 'Trebuchet MS', sans-serif";
    ctx.fillStyle = "#f5e9cf";
    ctx.textAlign = "left";
    ctx.fillText(label, sx + badgeR * 2 + 14, sy + 9.5);
    ctx.textAlign = "center";
    // smokestack puffs for the foundry while training
    if (b.id === "foundry" && running) {
      ctx.fillStyle = "rgba(240,240,240,0.5)";
      for (let i = 0; i < 3; i++) {
        const ph = ((time * 22 + i * 26) % 80);
        ctx.beginPath();
        ctx.arc(px + w - 22, py - 6 - ph, 5 + ph / 12, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  private roundedRect(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    r: number,
  ): void {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  private drawDistrictLabels(ctx: CanvasRenderingContext2D): void {
    // each label floats on the grass strip above its own building group, so
    // it never collides with the signs (which sit at the rooflines below)
    ctx.font = "bold 13px 'Trebuchet MS', sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = "rgba(30, 60, 25, 0.5)";
    const labels: [string, number, number][] = [
      ["— DATA QUARTER —", 14.5, 6.3],
      ["— FORWARD AVENUE —", 37, 6.3],
      ["— LOSS DISTRICT —", 55.5, 6.3],
      ["— CIVIC CENTER —", 9.5, 15.3],
      ["— GRADIENT ROW —", 47.5, 15.3],
      ["— FIRST STEPS QUARTER —", 14, 24.3],
      ["— TUNING HEIGHTS —", 26.5, 24.3],
      ["— SIDE QUEST YARDS —", 40, 24.3],
      ["— DEPLOYMENT DOCK —", 54, 24.3],
    ];
    for (const [text, tx, ty] of labels) ctx.fillText(text, tx * TILE, ty * TILE);
    // plaza title + tour hint
    ctx.font = "bold 22px 'Trebuchet MS', sans-serif";
    ctx.fillStyle = "rgba(90, 70, 30, 0.5)";
    ctx.fillText("⭐ DL WORLD ⭐", 31.5 * TILE, 16.8 * TILE);
    ctx.font = "bold 12px 'Trebuchet MS', sans-serif";
    ctx.fillStyle = "rgba(90, 70, 30, 0.65)";
    ctx.fillText("guided tour: follow the numbered signs ① → ⑮", 31.5 * TILE, 19.8 * TILE);
  }

  /** subtle chevrons painted on the roads, tracing the tour route */
  private drawTourArrows(ctx: CanvasRenderingContext2D): void {
    ctx.fillStyle = "rgba(80, 60, 30, 0.32)";
    const chevron = (x: number, y: number, dx: number, dy: number) => {
      const s = 5.5;
      ctx.beginPath();
      ctx.moveTo(x + dx * s, y + dy * s);
      ctx.lineTo(x - dx * s + dy * s, y - dy * s + dx * s);
      ctx.lineTo(x - dx * s - dy * s, y - dy * s - dx * s);
      ctx.closePath();
      ctx.fill();
    };
    // stops 1→6: east along the north road
    for (let x = 6; x <= 59; x += 3) chevron(x * TILE, 14 * TILE, 1, 0);
    // down the east edge to Gradient Row
    for (let y = 15.5; y <= 21; y += 2.5) chevron(63 * TILE, y * TILE, 0, 1);
    // stops 7→9: west along the mid road
    for (let x = 60; x >= 6; x -= 3) chevron(x * TILE, 23 * TILE, -1, 0);
    // down the west edge to the south row
    for (let y = 24.5; y <= 30; y += 2.5) chevron(3 * TILE, y * TILE, 0, 1);
    // stops 10→15: east along the south road
    for (let x = 5; x <= 58; x += 3) chevron(x * TILE, 32 * TILE, 1, 0);
  }

  /** glowing dots that travel the training loop while the main model trains */
  private drawLoopPulses(ctx: CanvasRenderingContext2D, time: number, world: World): void {
    if (!world.main.running) return;
    // waypoints through the loop, in tile coords (door fronts, on roads)
    const path: [number, number][] = [
      [10, 14], [19.5, 14], [28.5, 14], [37, 14], [45.5, 14], [55.5, 14],
      [62.5, 14], [62.5, 23], [52, 23], [42.5, 23], [31, 23], [31, 14], [10, 14],
    ];
    const segLens: number[] = [];
    let total = 0;
    for (let i = 1; i < path.length; i++) {
      const L = Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
      segLens.push(L);
      total += L;
    }
    const N = 7;
    for (let k = 0; k < N; k++) {
      let dist = ((time * 3.2 + (k * total) / N) % total);
      let i = 0;
      while (dist > segLens[i]) {
        dist -= segLens[i];
        i++;
      }
      const t = dist / segLens[i];
      const x = (path[i][0] + (path[i + 1][0] - path[i][0]) * t) * TILE;
      const y = (path[i][1] + (path[i + 1][1] - path[i][1]) * t) * TILE + TILE / 2;
      const g = ctx.createRadialGradient(x, y, 1, x, y, 9);
      g.addColorStop(0, "rgba(255, 235, 130, 0.95)");
      g.addColorStop(1, "rgba(255, 235, 130, 0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
