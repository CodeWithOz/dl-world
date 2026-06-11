// Drives a Scenario's training loop in real time. The HUD and every panel
// subscribe to its events; the city's ambient animations key off `running`.

import { Scenario } from "./scenarios";

export type TrainerEvent = "step" | "epoch" | "state";

export class Trainer {
  scenario: Scenario;
  running = false;
  /** training steps per second while running */
  speed = 8;
  private listeners: Record<TrainerEvent, Set<() => void>> = {
    step: new Set(),
    epoch: new Set(),
    state: new Set(),
  };
  private acc = 0;
  private lastEpochSeen = 0;
  lastLossValue = NaN;
  lastMetricValue = NaN;

  constructor(scenario: Scenario) {
    this.scenario = scenario;
  }

  on(ev: TrainerEvent, fn: () => void): () => void {
    this.listeners[ev].add(fn);
    return () => this.listeners[ev].delete(fn);
  }

  private emit(ev: TrainerEvent): void {
    for (const fn of this.listeners[ev]) fn();
  }

  setRunning(r: boolean): void {
    if (this.running === r) return;
    this.running = r;
    this.acc = 0;
    this.emit("state");
  }

  toggle(): void {
    this.setRunning(!this.running);
  }

  /** one manual training step */
  stepOnce(): void {
    this.doStep();
  }

  private doStep(): void {
    this.lastLossValue = this.scenario.trainStep();
    if (this.scenario.epoch !== this.lastEpochSeen) {
      this.lastEpochSeen = this.scenario.epoch;
      this.lastMetricValue = this.scenario.recordMetric();
      this.emit("epoch");
    }
    this.emit("step");
  }

  /** called every animation frame with dt in seconds */
  tick(dt: number): void {
    if (!this.running) return;
    this.acc += dt * this.speed;
    // never block a frame with more than a handful of steps
    let budget = Math.min(Math.floor(this.acc), 12);
    this.acc -= Math.floor(this.acc);
    while (budget-- > 0) this.doStep();
  }

  /** make sure at least one step has happened so panels have a graph to show */
  ensureWarm(): void {
    if (this.scenario.step === 0) {
      this.doStep();
      this.lastMetricValue = this.scenario.recordMetric();
    }
  }
}
