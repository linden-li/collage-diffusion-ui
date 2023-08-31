import { generateId } from "../utils/utils";

const LAYER_KEY_LENGTH = 20;

export type LayerKey = string;
export function generateLayerKey() {
  return generateId(LAYER_KEY_LENGTH);
}

type Transform = {
  position: {
    x: number;
    y: number;
  };
  scale: number;
  rotation: number;
};

type Layer = {
  id: number; // Keeps track of the index within the layer
  key: string; // Random string
  originalImgUrl: string; // URL of the original image that the user uploaded
  currentImgUrl: string; // URL of the image with the mask applied
  textPrompt: string;
  transform: Transform;
  polygon: {
    x: number;
    y: number;
  }[]; // Array of coordinates, in image coordinate space, which defines a segmentation mask
  opacity: number;
  cacStrength: number;
  negativeStrength: number;
  cannyStrength: number;
  noiseStrength: number;
};
 
export { Layer };
