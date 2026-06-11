// The player character: a little researcher with a hard hat, drawn
// procedurally, moving in world pixels with corner-based collision.

export class Avatar {
  x: number;
  y: number;
  dir: "up" | "down" | "left" | "right" = "down";
  moving = false;
  walkPhase = 0;

  constructor(x: number, y: number) {
    this.x = x;
    this.y = y;
  }

  /** half-extent of the collision box, px */
  static readonly R = 9;

  update(
    dt: number,
    input: { dx: number; dy: number; run: boolean },
    isSolid: (tileX: number, tileY: number) => boolean,
    tileSize: number,
  ): void {
    const speed = input.run ? 230 : 145;
    let { dx, dy } = input;
    const len = Math.hypot(dx, dy);
    this.moving = len > 0;
    if (!this.moving) return;
    dx /= len;
    dy /= len;
    if (Math.abs(dx) > Math.abs(dy)) this.dir = dx > 0 ? "right" : "left";
    else this.dir = dy > 0 ? "down" : "up";
    this.walkPhase += dt * speed * 0.09;

    const tryMove = (nx: number, ny: number): boolean => {
      const r = Avatar.R;
      for (const [cx, cy] of [
        [nx - r, ny - r], [nx + r, ny - r], [nx - r, ny + r], [nx + r, ny + r],
      ]) {
        if (isSolid(Math.floor(cx / tileSize), Math.floor(cy / tileSize))) return false;
      }
      this.x = nx;
      this.y = ny;
      return true;
    };
    const stepX = dx * speed * dt;
    const stepY = dy * speed * dt;
    // axis-separated so we can slide along walls
    if (!tryMove(this.x + stepX, this.y + stepY)) {
      tryMove(this.x + stepX, this.y);
      tryMove(this.x, this.y + stepY);
    }
  }

  draw(ctx: CanvasRenderingContext2D, time: number): void {
    const { x, y } = this;
    const bob = this.moving ? Math.sin(this.walkPhase) * 2 : Math.sin(time * 2) * 0.7;
    // shadow
    ctx.fillStyle = "rgba(0,0,0,0.25)";
    ctx.beginPath();
    ctx.ellipse(x, y + 12, 9, 4, 0, 0, Math.PI * 2);
    ctx.fill();
    // legs
    const legSwing = this.moving ? Math.sin(this.walkPhase) * 4 : 0;
    ctx.fillStyle = "#3b4a6b";
    ctx.fillRect(x - 6, y + 2 + bob * 0.3, 5, 10 + legSwing * 0.5);
    ctx.fillRect(x + 1, y + 2 + bob * 0.3, 5, 10 - legSwing * 0.5);
    // body (lab coat)
    ctx.fillStyle = "#e9e4d8";
    ctx.beginPath();
    ctx.roundRect(x - 8, y - 8 + bob, 16, 14, 4);
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.2)";
    ctx.stroke();
    // head — deliberately a playful mint (matches the city's accent color),
    // not any real human skin tone
    ctx.fillStyle = "#7fd8c6";
    ctx.beginPath();
    ctx.arc(x, y - 14 + bob, 7.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(20, 60, 55, 0.35)";
    ctx.stroke();
    // hard hat
    ctx.fillStyle = "#f2c33d";
    ctx.beginPath();
    ctx.arc(x, y - 16 + bob, 7.5, Math.PI, 0);
    ctx.fill();
    ctx.fillRect(x - 9, y - 16.5 + bob, 18, 3);
    // face direction: eyes
    ctx.fillStyle = "#2a2a33";
    const ex = this.dir === "left" ? -3.5 : this.dir === "right" ? 3.5 : 0;
    if (this.dir !== "up") {
      ctx.beginPath();
      ctx.arc(x - 2.5 + ex, y - 13.5 + bob, 1.3, 0, Math.PI * 2);
      ctx.arc(x + 2.5 + ex, y - 13.5 + bob, 1.3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
