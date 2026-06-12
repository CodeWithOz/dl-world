// Global simulation state: one trainer per scenario. The main pipeline
// (mlp10) is what the central districts visualize; the ch.4/ch.6 buildings
// own their smaller scenarios.

import { MnistData } from "../engine/data";
import { Linear37, Mlp10, MultiLabel, Regression } from "./scenarios";
import { CollabFilter, Refinery, RnnLm, SentimentNet } from "./scenarios2";
import { Trainer } from "./trainer";
import { LrFinderResult } from "./lrfinder";

export class World {
  data: MnistData;
  /** the city's central pipeline: 2-layer digit classifier */
  main: Trainer;
  cottage: Trainer;
  workshop: Trainer;
  studio: Trainer;
  // the frontier (beyond images): tricks, ratings, text, sequences
  refinery: Trainer;
  cinema: Trainer;
  sentiment: Trainer;
  echo: Trainer;
  lrResult: LrFinderResult | null = null;

  constructor(data: MnistData) {
    this.data = data;
    this.main = new Trainer(new Mlp10(data));
    this.cottage = new Trainer(new Linear37(data));
    this.workshop = new Trainer(new MultiLabel(data));
    this.studio = new Trainer(new Regression(data));
    this.refinery = new Trainer(new Refinery(data));
    this.cinema = new Trainer(new CollabFilter(data));
    this.sentiment = new Trainer(new SentimentNet(data));
    this.echo = new Trainer(new RnnLm(data));
    // one warm-up step each so every panel has a real graph to show
    for (const t of this.trainers()) t.ensureWarm();
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
  get refine(): Refinery {
    return this.refinery.scenario as Refinery;
  }
  get collab(): CollabFilter {
    return this.cinema.scenario as CollabFilter;
  }
  get sent(): SentimentNet {
    return this.sentiment.scenario as SentimentNet;
  }
  get rnn(): RnnLm {
    return this.echo.scenario as RnnLm;
  }

  trainers(): Trainer[] {
    return [
      this.main,
      this.cottage,
      this.workshop,
      this.studio,
      this.refinery,
      this.cinema,
      this.sentiment,
      this.echo,
    ];
  }

  tick(dt: number): void {
    for (const t of this.trainers()) t.tick(dt);
  }
}
