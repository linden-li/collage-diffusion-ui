// Type for submitting a layer in the request. Certain fields are omitted

import { Layer } from "./layer";
import { imageCoordToScaledImageCoord } from "../utils/transforms";

// so this is the same type as the Layer type on the backend
type RequestLayer = {
  id: number;
  key: string;
  originalImgUrl: string;
  textPrompt: string;
  cacStrength: number;
  negativeStrength: number;
  cannyStrength: number;
  noiseStrength: number;
  transform: {
    position: {
      x: number;
      y: number;
    };
    scale: number;
    rotation: number;
  };
  polygon: {
    x: number;
    y: number;
  }[];
};

export function layerToRequestLayer(
  layer: Layer,
  canvasWidth: number,
  canvasHeight: number,
) {
  let newLayer: RequestLayer = {
    id: layer.id,
    key: layer.key,
    originalImgUrl: layer.originalImgUrl,
    textPrompt: layer.textPrompt,
    cacStrength: layer.cacStrength,
    cannyStrength: layer.cannyStrength,
    negativeStrength: layer.negativeStrength,
    noiseStrength: layer.noiseStrength,
    transform: {
      position: layer.transform.position,
      scale: layer.transform.scale,
      rotation: layer.transform.rotation || 0,
    },
    polygon: layer.polygon,
  };
  return newLayer;
}

export function convertLayersToRequestLayers(
  layers: Layer[],
  canvasWidth: number,
  canvasHeight: number,
) {
  // Read transform from canvas
  let requestLayers: RequestLayer[] = [];
  for (var i = 0; i < layers.length; i++) {
    const newLayer = layerToRequestLayer(
      layers[i],
      canvasWidth,
      canvasHeight,
    );
    requestLayers.push(newLayer);
  }
  return requestLayers;
}

async function segmentLayer(requestLayer: RequestLayer) {
  if (requestLayer.polygon.length === 0) return;
  const maskedCanvas = document.createElement("canvas");
  const ctx = maskedCanvas.getContext("2d");
  if (!ctx) return;

  let img = new Image();
  img.crossOrigin = "anonymous";  
  img.src = requestLayer.originalImgUrl;
  await new Promise((resolve) => (img.onload = resolve));
  maskedCanvas.width = img.width;
  maskedCanvas.height = img.height;

  ctx.drawImage(img, 0, 0);
  const imgData = ctx.getImageData(0, 0, img.width, img.height);

  // Create mask with 0s, divide by 4 because each pixel has 4 values
  const mask = new Uint8ClampedArray(imgData.data.length / 4);

  const polygonPath = new Path2D();
  for (let i = 0; i < requestLayer.polygon.length; i++) {
    const scaledImageCoord = imageCoordToScaledImageCoord(
      requestLayer.polygon[i].x,
      requestLayer.polygon[i].y,
      img.width,
      img.height
    );
    if (i === 0) {
      polygonPath.moveTo(scaledImageCoord.x, scaledImageCoord.y);
    } else {
      polygonPath.lineTo(scaledImageCoord.x, scaledImageCoord.y);
    }
  }
  polygonPath.closePath();

  // Fill mask with 1s for points that intersect the polygon
  for (let i = 0; i < imgData.data.length; i += 4) {
    const x = (i / 4) % img.width;
    const y = Math.floor(i / 4 / img.width);
    if (ctx.isPointInPath(polygonPath, x, y)) {
      mask[i / 4] = 1;
    }
  }

  // Compute a tight bounding box for the mask
  let minX = img.width;
  let maxX = 0;
  let minY = img.height;
  let maxY = 0;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 1) {
      const x = i % img.width;
      const y = Math.floor(i / img.width);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }

  // Apply mask to image data
  for (let i = 0; i < imgData.data.length; i += 4) {
    if (mask[i / 4] === 0) {
      imgData.data[i + 3] = 0;
    }
  }

  // Crop image data to bounding box
  const croppedImgData = ctx.createImageData(maxX - minX, maxY - minY);
  for (let i = 0; i < croppedImgData.data.length; i += 4) {
      const x = (i / 4) % croppedImgData.width;
      const y = Math.floor((i / 4) / croppedImgData.width);
      const imgDataIndex = ((y + minY) * img.width + (x + minX)) * 4;
      croppedImgData.data[i] = imgData.data[imgDataIndex];
      croppedImgData.data[i + 1] = imgData.data[imgDataIndex + 1];
      croppedImgData.data[i + 2] = imgData.data[imgDataIndex + 2];
      croppedImgData.data[i + 3] = imgData.data[imgDataIndex + 3];
  }

  // Create a new cropped canvas
  const croppedCanvas = document.createElement('canvas');
  croppedCanvas.width = croppedImgData.width;
  croppedCanvas.height = croppedImgData.height;
  const croppedCtx = croppedCanvas.getContext('2d');
  if (!croppedCtx) return;
  croppedCtx.putImageData(croppedImgData, 0, 0);
  ctx.putImageData(croppedImgData, 0, 0);

  return croppedCanvas.toDataURL();
}

export async function requestLayerToLayer(
  requestLayer: RequestLayer,
  canvasWidth: number,
  canvasHeight: number
) {
  let currentImgUrl = await segmentLayer(requestLayer);
  currentImgUrl = currentImgUrl || requestLayer.originalImgUrl;
  let newLayer: Layer = {
    id: requestLayer.id,
    key: requestLayer.key,
    originalImgUrl: requestLayer.originalImgUrl,
    currentImgUrl: currentImgUrl!,
    textPrompt: requestLayer.textPrompt,
    cacStrength: requestLayer.cacStrength,
    negativeStrength: requestLayer.negativeStrength,
    cannyStrength: requestLayer.cannyStrength,
    noiseStrength: requestLayer.noiseStrength,
    transform: requestLayer.transform,
    polygon: requestLayer.polygon,
    opacity: 1,
  };
  return newLayer;
}

export async function convertRequestLayersToLayers(
  requestLayers: RequestLayer[],
  canvasWidth: number,
  canvasHeight: number
) {
  let layers: Layer[] = [];
  for (var i = 0; i < requestLayers.length; i++) {
    const newLayer = await requestLayerToLayer(
      requestLayers[i],
      canvasWidth,
      canvasHeight
    );
    layers.push(newLayer);
  }
  return layers;
}

export { RequestLayer };
