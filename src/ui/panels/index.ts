import { registerDataPanels } from "./data";
import { registerMillPanels } from "./mills";
import { registerSpringPanels } from "./springs";
import { registerFoundryPanels } from "./foundry";
import { registerBackpropPanels } from "./backprop";
import { registerOptimPanels } from "./optim";
import { registerObservatoryPanels } from "./observatory";
import { registerCh4Panels } from "./ch4";
import { registerTowerPanels } from "./tower";
import { registerCh6Panels } from "./ch6";
import { registerGalleryPanels } from "./gallery";
import { registerRefineryPanels } from "./refinery";
import { registerCollabPanels } from "./collab";
import { registerArborPanels } from "./arbor";
import { registerLanguagePanels } from "./language";
import { registerEchoPanels } from "./echo";

export function registerAllPanels(): void {
  registerDataPanels();
  registerMillPanels();
  registerSpringPanels();
  registerFoundryPanels();
  registerBackpropPanels();
  registerOptimPanels();
  registerObservatoryPanels();
  registerCh4Panels();
  registerTowerPanels();
  registerCh6Panels();
  registerGalleryPanels();
  // the frontier
  registerRefineryPanels();
  registerCollabPanels();
  registerArborPanels();
  registerLanguagePanels();
  registerEchoPanels();
}
