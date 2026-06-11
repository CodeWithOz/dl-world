// Global simulation state: one trainer per scenario. The main pipeline
// (mlp10) is what the central districts visualize; the ch.4/ch.6 buildings
// own their smaller scenarios.

import { MnistData } from "../engine/data";
import { Linear37, Mlp10, MultiLabel, Regression } from "./scenarios";
import { Trainer } from "./trainer";
import { LrFinderResult } from "./lrfinder";

export class World {
  data: MnistData;
  /** the city's central pipeline: 2-layer digit classifier */
  main: Trainer;
  cottage: Trainer;
  workshop: Trainer;
  studio: Trainer;
  lrResult: LrFinderResult | null = null;

  constructor(data: MnistData) {
    this.data = data;
    this.main = new Trainer(new Mlp10(data));
    this.cottage = new Trainer(new Linear37(data));
    this.workshop = new Trainer(new MultiLabel(data));
    this.studio = new Trainer(new Regression(data));
    // one warm-up step each so every panel has a real graph to show
    this.main.ensureWarm();
    this.cottage.ensureWarm();
    this.workshop.ensureWarm();
    this.studio.ensureWarm();
  }

  get mlp(): Mlp10 {
    return this.main.scenario as Mlp10;
  }
  get lin37(): Linear37 {
    return this.cottage.scenario as Linear37;
  }
  get multi(): MultiLabel {
    return this.workshop.scenario as MultiLabel;
  }
  get reg(): Regression {
    return this.studio.scenario as Regression;
  }

  trainers(): Trainer[] {
    return [this.main, this.cottage, this.workshop, this.studio];
  }

  tick(dt: number): void {
    for (const t of this.trainers()) t.tick(dt);
  }
}
