// Inside a building: a cozy machine hall. Stations are pedestals the player
// inspects (E) to open the real inspection panel for that concept.

import { BuildingDef, StationDef } from "./buildings";
import { TILE } from "./city";
import { World } from "../sim/world";

export class Interior {
  building: BuildingDef;

  constructor(building: BuildingDef) {
    this.building = building;
  }

  get w(): number {
    return this.building.interior.w;
  }
  get h(): number {
    return this.building.interior.h;
  }

  /** exit door tile (bottom center) */
  get exit(): { x: number; y: number } {
    return { x: Math.floor(this.w / 2), y: this.h - 1 };
  }

  spawnPoint(): { x: number; y: number } {
    return { x: (this.exit.x + 0.5) * TILE, y: (this.h - 1.6) * TILE };
  }

  isSolid(tx: number, ty: number): boolean {
    if (tx < 1 || ty < 1 || tx >= this.w - 1 || ty >= this.h - 1)
      return !(ty === this.h - 1 && tx === this.exit.x); // walls except exit
    for (const s of this.building.interior.stations)
      if (tx >= s.x - 1 && tx <= s.x + 1 && ty === s.y) return true;
    return false;
  }

  /** station near the given world-pixel position, if any */
  stationNear(px: number, py: number): StationDef | null {
    for (const s of this.building.interior.stations) {
      const sx = (s.x + 0.5) * TILE;
      const sy = (s.y + 0.5) * TILE;
      if (Math.hypot(px - sx, py - (sy + TILE)) < TILE * 1.7) return s;
    }
    return null;
  }

  atExit(px: number, py: number): boolean {
    const ex = (this.exit.x + 0.5) * TILE;
    const ey = (this.exit.y + 0.5) * TILE;
    return Math.hypot(px - ex, py - ey) < TILE * 1.1;
  }

  draw(ctx: CanvasRenderingContext2D, time: number, world: World, nearId: string | null): void {
    const b = this.building;
    const running = b.trainer !== null && world[b.trainer as "main"].running;
    // floor
    for (let y = 0; y < this.h; y++)
      for (let x = 0; x < this.w; x++) {
        const px = x * TILE;
        const py = y * TILE;
        const isWall = x === 0 || y === 0 || x === this.w - 1 || y === this.h - 1;
        if (isWall && !(y === this.h - 1 && x === this.exit.x)) {
          ctx.fillStyle = b.roof;
          ctx.fillRect(px, py, TILE, TILE);
          ctx.fillStyle = "rgba(0,0,0,0.25)";
          ctx.fillRect(px, py + TILE - 6, TILE, 6);
        } else {
          ctx.fillStyle = (x + y) % 2 === 0 ? "#cdbfa3" : "#c4b699";
          ctx.fillRect(px, py, TILE, TILE);
        }
      }
    // exit mat
    const e = this.exit;
    ctx.fillStyle = "#8a4a3a";
    ctx.fillRect(e.x * TILE + 3, (e.y - 0) * TILE + 2, TILE - 6, TILE - 4);
    ctx.fillStyle = "#f5e9cf";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("EXIT", (e.x + 0.5) * TILE, (e.y + 0.55) * TILE);

    // big rug with the building blurb hint
    ctx.fillStyle = "rgba(120, 90, 60, 0.18)";
    ctx.fillRect(2 * TILE, (this.h - 3.4) * TILE, (this.w - 4) * TILE, TILE * 1.6);

    // stations
    for (const s of this.building.interior.stations) {
      const px = s.x * TILE;
      const py = s.y * TILE;
      const near = nearId === s.id;
      // machine body spanning 3 tiles
      ctx.fillStyle = near ? "#5d6f96" : "#4c5a7a";
      ctx.beginPath();
      ctx.roundRect(px - TILE + 4, py - 10, TILE * 3 - 8, TILE + 8, 6);
      ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,0.1)";
      ctx.fillRect(px - TILE + 4, py - 10, TILE * 3 - 8, 6);
      // screen
      const glow = running ? 0.6 + 0.4 * Math.sin(time * 4 + s.x) : 0.35;
      ctx.fillStyle = `rgba(120, 230, 215, ${glow})`;
      ctx.fillRect(px - TILE + 12, py - 2, TILE * 3 - 24, 12);
      // icon
      ctx.font = "20px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(s.icon, px + TILE / 2, py - 16);
      // label
      ctx.font = `bold 11px 'Trebuchet MS', sans-serif`;
      ctx.fillStyle = near ? "#ffe44d" : "#f5e9cf";
      const label = s.name;
      const tw = ctx.measureText(label).width + 10;
      ctx.fillStyle = "rgba(34,28,20,0.85)";
      ctx.beginPath();
      ctx.roundRect(px + TILE / 2 - tw / 2, py + TILE + 2, tw, 16, 4);
      ctx.fill();
      ctx.fillStyle = near ? "#ffe44d" : "#f5e9cf";
      ctx.fillText(label, px + TILE / 2, py + TILE + 13.5);
    }

    // building title inside
    ctx.font = "bold 16px 'Trebuchet MS', sans-serif";
    ctx.fillStyle = "rgba(60, 45, 25, 0.65)";
    ctx.textAlign = "center";
    ctx.fillText(`${b.icon} ${b.name}`, (this.w / 2) * TILE, TILE * 0.7);
  }
}
