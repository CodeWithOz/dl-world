// The city: terrain, collision, and rendering. Everything is drawn
// procedurally — no image assets.

import { BUILDINGS, BuildingDef, doorTile } from "./buildings";
import { World } from "../sim/world";
import { mulberry32 } from "../engine/data";

export const TILE = 32;
export const MAP_W = 66;
export const MAP_H = 65;

/**
 * District lawn signs: dark placards on posts, planted on the grass strip
 * above each building group. Coordinates are tile-centers; `generate()`
 * keeps trees/flowers out of each sign's footprint so the text stays
 * readable against the board (not the greenery).
 */
interface SignDef {
  text: string;
  x: number;
  y: number;
  big?: boolean;
}

const DISTRICT_SIGNS: SignDef[] = [
  // old town
  { text: "DATA QUARTER", x: 14.5, y: 5.1 },
  { text: "FORWARD AVENUE", x: 37, y: 5.1 },
  { text: "LOSS DISTRICT", x: 55.5, y: 5.1 },
  { text: "CIVIC CENTER", x: 9.5, y: 16.4 },
  { text: "GRADIENT ROW", x: 47.5, y: 16.4 },
  { text: "FIRST STEPS QUARTER", x: 14, y: 27.4 },
  { text: "TUNING HEIGHTS", x: 26.5, y: 27.4 },
  { text: "SIDE QUEST YARDS", x: 40, y: 27.4 },
  { text: "DEPLOYMENT DOCK", x: 54, y: 27.4 },
  // the frontier (beyond the river: words, tables & taste). The two edge
  // bridges get their own crossing markers — on a phone the central sign
  // is easy to miss entirely.
  { text: "THE FRONTIER — BEYOND IMAGES", x: 31.5, y: 37.6, big: true },
  { text: "↓ THE FRONTIER", x: 7.5, y: 37.7 },
  { text: "↓ THE FRONTIER", x: 56.5, y: 37.7 },
  { text: "TABLE GROVE", x: 10, y: 42.5 },
  { text: "TASTE QUARTER", x: 26, y: 42.5 },
  { text: "REFINEMENT ROW", x: 46, y: 42.5 },
  { text: "LANGUAGE LANE", x: 19, y: 53.5 },
  { text: "SEQUENCE SUMMIT", x: 47.5, y: 53.5 },
];

/** rough tile footprint of a sign (board + posts), for decor clearing */
function signZone(s: SignDef): { x0: number; x1: number; y0: number; y1: number } {
  const halfTiles = Math.ceil(((s.text.length * (s.big ? 9 : 7.5)) / 2 + 20) / TILE);
  return {
    x0: Math.floor(s.x - halfTiles),
    x1: Math.ceil(s.x + halfTiles),
    y0: Math.floor(s.y - 0.7),
    y1: Math.floor(s.y + 0.7),
  };
}

const enum T {
  Grass = 0,
  Road = 1,
  Plaza = 2,
  Water = 3,
  Tree = 4,
  Building = 5,
  Door = 6,
  /** the plaza monument's footprint (solid; drawn by drawMonument) */
  Monument = 7,
  Flower = 8,
  /** the tour kiosk (solid; interactable — express ride to stop ①) */
  Kiosk = 9,
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
    // horizontal roads — old town (3) and the frontier (2)
    this.rect(2, 13, MAP_W - 4, 2, T.Road);
    this.rect(2, 24, MAP_W - 4, 2, T.Road);
    this.rect(2, 35, MAP_W - 4, 2, T.Road);
    this.rect(2, 50, MAP_W - 4, 2, T.Road);
    this.rect(2, 61, MAP_W - 4, 2, T.Road);
    // vertical roads run the whole city, old town through the frontier —
    // keep building footprints clear of all three columns
    this.rect(2, 13, 2, 50, T.Road);
    this.rect(MAP_W - 4, 13, 2, 50, T.Road);
    this.rect(30, 13, 2, 50, T.Road);
    // plaza, the network monument, and the tour kiosk
    this.rect(26, 18, 11, 6, T.Plaza);
    this.rect(30, 19, 4, 2, T.Monument);
    this.set(29, 22, T.Kiosk);
    // the river that separates old town (images) from the frontier
    // (words, tables & taste) — crossable on the three road bridges
    this.rect(2, 39, MAP_W - 4, 2, T.Water);
    this.rect(2, 39, 2, 2, T.Road);
    this.rect(30, 39, 2, 2, T.Road);
    this.rect(MAP_W - 4, 39, 2, 2, T.Road);
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
    const zones = DISTRICT_SIGNS.map(signZone);
    const inSignZone = (x: number, y: number) =>
      zones.some((z) => x >= z.x0 && x <= z.x1 && y >= z.y0 && y <= z.y1);
    for (let i = 0; i < 380; i++) {
      const x = 2 + Math.floor(rand() * (MAP_W - 4));
      const y = 2 + Math.floor(rand() * (MAP_H - 4));
      if (this.get(x, y) !== T.Grass) continue;
      // keep walking corridors near roads — and every sign's footprint — clear
      if (inSignZone(x, y)) continue;
      this.set(x, y, rand() < 0.35 ? T.Tree : T.Flower);
    }
  }

  isSolid(x: number, y: number): boolean {
    const t = this.get(x, y);
    return (
      t === T.Tree || t === T.Water || t === T.Building || t === T.Monument || t === T.Kiosk
    );
  }

  /** is the tour kiosk on/next to this tile? (for the express prompt) */
  kioskNear(x: number, y: number): boolean {
    return Math.abs(x - 29) <= 1 && Math.abs(y - 22) <= 1;
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
    this.drawMonument(ctx, time, world);
    this.drawKiosk(ctx, time);
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
      case T.Monument:
      case T.Kiosk: {
        // plaza floor under the structures; drawMonument/drawKiosk paint
        // the actual objects after the buildings pass
        ctx.fillStyle = checker ? "#d6cdb8" : "#cfc5af";
        ctx.fillRect(px, py, TILE, TILE);
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

  /** lawn-sign placards: bright text on a dark board so district names stay
   *  readable against grass and trees */
  private drawDistrictLabels(ctx: CanvasRenderingContext2D): void {
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const s of DISTRICT_SIGNS) {
      ctx.font = `bold ${s.big ? 15 : 12.5}px 'Trebuchet MS', sans-serif`;
      const tw = ctx.measureText(s.text).width;
      const w = tw + 26;
      const h = s.big ? 28 : 22;
      const cx = s.x * TILE;
      const cy = s.y * TILE;
      const x = cx - w / 2;
      const y = cy - h / 2;
      // wooden posts
      ctx.fillStyle = "#6b4a2f";
      ctx.fillRect(x + 8, y + h - 2, 5, 13);
      ctx.fillRect(x + w - 13, y + h - 2, 5, 13);
      // drop shadow, then the board
      ctx.fillStyle = "rgba(0, 0, 0, 0.28)";
      this.roundedRect(ctx, x + 2, y + 3, w, h, 6);
      ctx.fill();
      ctx.fillStyle = s.big ? "#27345e" : "#1d3357";
      this.roundedRect(ctx, x, y, w, h, 6);
      ctx.fill();
      ctx.strokeStyle = "rgba(255, 211, 77, 0.85)";
      ctx.lineWidth = 1.5;
      this.roundedRect(ctx, x + 2, y + 2, w - 4, h - 4, 4);
      ctx.stroke();
      ctx.fillStyle = "#ffe44d";
      ctx.fillText(s.text, cx, cy + 1);
    }
    ctx.textBaseline = "alphabetic";
  }

  /** the plaza centerpiece: a little neural network on a plinth. The three
   *  layers mirror the city's real pipeline (input → hidden → output) and
   *  the edges light up while the main model trains. */
  private drawMonument(ctx: CanvasRenderingContext2D, time: number, world: World): void {
    const x0 = 30 * TILE;
    const y0 = 19 * TILE;
    const w = 4 * TILE;
    const running = world.main.running;
    // plinth with the city's name
    ctx.fillStyle = "rgba(0,0,0,0.18)";
    ctx.fillRect(x0 + 4, y0 + TILE * 1.55, w - 4, 10);
    ctx.fillStyle = "#a8a193";
    this.roundedRect(ctx, x0 + 2, y0 + TILE * 0.95, w - 4, TILE * 0.95, 5);
    ctx.fill();
    ctx.fillStyle = "#1d3357";
    this.roundedRect(ctx, x0 + 8, y0 + TILE * 1.12, w - 16, 20, 4);
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 211, 77, 0.85)";
    ctx.lineWidth = 1.5;
    this.roundedRect(ctx, x0 + 10, y0 + TILE * 1.12 + 2, w - 20, 16, 3);
    ctx.stroke();
    ctx.font = "bold 13px 'Trebuchet MS', sans-serif";
    ctx.fillStyle = "#ffe44d";
    ctx.textAlign = "center";
    ctx.fillText("⭐ DL WORLD ⭐", x0 + w / 2, y0 + TILE * 1.12 + 14);
    // the network: 3 → 4 → 3 nodes above the plinth
    const colX = [x0 + 22, x0 + w / 2, x0 + w - 22];
    const colN = [3, 4, 3];
    const top = y0 - 26;
    const bot = y0 + TILE * 0.85;
    const nodeY = (col: number, i: number) =>
      top + ((i + 1) / (colN[col] + 1)) * (bot - top);
    ctx.lineWidth = 1;
    for (let c = 0; c < 2; c++)
      for (let i = 0; i < colN[c]; i++)
        for (let j = 0; j < colN[c + 1]; j++) {
          ctx.strokeStyle = "rgba(245, 233, 207, 0.3)";
          ctx.beginPath();
          ctx.moveTo(colX[c], nodeY(c, i));
          ctx.lineTo(colX[c + 1], nodeY(c + 1, j));
          ctx.stroke();
          if (running) {
            // activations travel the edges while the main pipeline trains
            const u = (time * 0.55 + i * 0.23 + j * 0.41 + c * 0.5) % 1;
            const px = colX[c] + (colX[c + 1] - colX[c]) * u;
            const py = nodeY(c, i) + (nodeY(c + 1, j) - nodeY(c, i)) * u;
            ctx.fillStyle = "rgba(255, 228, 77, 0.85)";
            ctx.beginPath();
            ctx.arc(px, py, 1.6, 0, Math.PI * 2);
            ctx.fill();
          }
        }
    const colors = ["#8fd1c8", "#ffd34d", "#e8907a"];
    for (let c = 0; c < 3; c++)
      for (let i = 0; i < colN[c]; i++) {
        const glow = running ? 0.75 + 0.25 * Math.sin(time * 3 + c + i) : 0.85;
        ctx.fillStyle = colors[c];
        ctx.globalAlpha = glow;
        ctx.beginPath();
        ctx.arc(colX[c], nodeY(c, i), 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "rgba(0,0,0,0.35)";
        ctx.stroke();
      }
  }

  /** the tour kiosk: a signpost with the ① disc — stand next to it and
   *  press E to ride the express to the first stop */
  private drawKiosk(ctx: CanvasRenderingContext2D, time: number): void {
    const cx = 29.5 * TILE;
    const cy = 22 * TILE;
    // post
    ctx.fillStyle = "#6b4a2f";
    ctx.fillRect(cx - 3, cy - 4, 6, TILE + 2);
    // the ① disc, gently bobbing so it reads as interactable
    const bob = Math.sin(time * 2.2) * 2;
    ctx.fillStyle = "rgba(0,0,0,0.22)";
    ctx.beginPath();
    ctx.ellipse(cx, cy + TILE - 3, 9, 3, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#ffd34d";
    ctx.beginPath();
    ctx.arc(cx, cy - 12 + bob, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#2a2417";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = "#2a2417";
    ctx.font = "bold 12px 'Trebuchet MS', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("1", cx, cy - 11.5 + bob);
    // little board: where the tour begins
    ctx.font = "bold 10.5px 'Trebuchet MS', sans-serif";
    const label = "tour start — press E";
    const tw = ctx.measureText(label).width + 14;
    ctx.fillStyle = "#1d3357";
    this.roundedRect(ctx, cx - tw / 2, cy + 6, tw, 15, 4);
    ctx.fill();
    ctx.fillStyle = "#ffe44d";
    ctx.fillText(label, cx, cy + 14);
    ctx.textBaseline = "alphabetic";
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
    for (let y = 15.5; y <= 23; y += 2.5) chevron(63 * TILE, y * TILE, 0, 1);
    // stops 7→9: west along the mid road
    for (let x = 60; x >= 6; x -= 3) chevron(x * TILE, 25 * TILE, -1, 0);
    // down the west edge to the south row
    for (let y = 26.5; y <= 34; y += 2.5) chevron(3 * TILE, y * TILE, 0, 1);
    // stops 10→15: east along the south road, all the way to the
    // T-junction so the route visibly continues past the last old-town stop
    for (let x = 5; x <= 61; x += 3) chevron(x * TILE, 36 * TILE, 1, 0);
    // over the east bridge into the frontier
    for (let y = 37.5; y <= 49; y += 2.5) chevron(63 * TILE, y * TILE, 0, 1);
    // stops 16→18: west along the frontier's first road
    for (let x = 60; x >= 6; x -= 3) chevron(x * TILE, 51 * TILE, -1, 0);
    // down the west edge to Language Lane
    for (let y = 52.5; y <= 60; y += 2.5) chevron(3 * TILE, y * TILE, 0, 1);
    // stops 19→21: east along the frontier's south road — the route ends
    // at Echo Tower's door, so the arrows stop there too
    for (let x = 5; x <= 46; x += 3) chevron(x * TILE, 62 * TILE, 1, 0);
  }

  /** the current mini-batch riding the training loop: little chips carrying
   *  the batch's actual digits circulate data → mills → foundry → backprop →
   *  optimizer and back, while the main model trains */
  private drawLoopPulses(ctx: CanvasRenderingContext2D, time: number, world: World): void {
    if (!world.main.running) return;
    // waypoints through the loop, in tile coords (door fronts, on roads)
    const path: [number, number][] = [
      [10, 14], [19.5, 14], [28.5, 14], [37, 14], [45.5, 14], [55.5, 14],
      [62.5, 14], [62.5, 24], [52, 24], [42.5, 24], [31, 24], [31, 14], [10, 14],
    ];
    const segLens: number[] = [];
    let total = 0;
    for (let i = 1; i < path.length; i++) {
      const L = Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
      segLens.push(L);
      total += L;
    }
    const N = 7;
    const batch = world.mlp.lastBatch;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
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
      // a chip carrying one real digit from the batch currently in the loop
      const digit = batch.length > 0 ? world.data.trainLabels[batch[k % batch.length]] : null;
      const g = ctx.createRadialGradient(x, y, 2, x, y, 13);
      g.addColorStop(0, "rgba(255, 235, 130, 0.55)");
      g.addColorStop(1, "rgba(255, 235, 130, 0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, 13, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#1b2030";
      this.roundedRect(ctx, x - 7, y - 8.5, 14, 17, 3);
      ctx.fill();
      ctx.strokeStyle = "rgba(255, 211, 77, 0.9)";
      ctx.lineWidth = 1;
      this.roundedRect(ctx, x - 7, y - 8.5, 14, 17, 3);
      ctx.stroke();
      if (digit !== null) {
        ctx.fillStyle = "#ffe44d";
        ctx.font = "bold 11px ui-monospace, monospace";
        ctx.fillText(String(digit), x, y + 0.5);
      }
    }
    ctx.textBaseline = "alphabetic";
  }
}
