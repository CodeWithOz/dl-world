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
}
