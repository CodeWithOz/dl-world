// The inspection panel: a large overlay that opens when the player
// interacts with a station. Content comes from the panel registry; each
// renderer returns a cleanup function (used to unsubscribe from trainers).

import { World } from "../sim/world";
import { el } from "./widgets";

export type PanelRenderer = (
  body: HTMLElement,
  world: World,
) => (() => void) | void;

export interface PanelDef {
  title: string;
  subtitle?: string;
  render: PanelRenderer;
}

const registry = new Map<string, PanelDef>();

export function registerPanel(id: string, def: PanelDef): void {
  registry.set(id, def);
}

export function hasPanel(id: string): boolean {
  return registry.has(id);
}

let overlay: HTMLElement | null = null;
let cleanup: (() => void) | null = null;
let onCloseCb: (() => void) | null = null;

export function isPanelOpen(): boolean {
  return overlay !== null;
}

export function closePanel(): void {
  cleanup?.();
  cleanup = null;
  overlay?.remove();
  overlay = null;
  onCloseCb?.();
  onCloseCb = null;
}

export function openPanel(id: string, world: World, onClose?: () => void): void {
  const def = registry.get(id);
  if (!def) return;
  closePanel();
  onCloseCb = onClose ?? null;
  overlay = el("div", "panel-overlay");
  const panel = el("div", "panel");
  const header = el("div", "panel-header");
  const titles = el("div");
  titles.append(el("h2", "", def.title));
  if (def.subtitle) titles.append(el("div", "panel-subtitle", def.subtitle));
  header.append(titles);
  const close = el("button", "panel-close", "✕");
  close.addEventListener("click", closePanel);
  header.append(close);
  const body = el("div", "panel-body");
  panel.append(header, body);
  overlay.append(panel);
  overlay.addEventListener("mousedown", (e) => {
    if (e.target === overlay) closePanel();
  });
  document.body.append(overlay);
  const c = def.render(body, world);
  if (c) cleanup = c;
}
