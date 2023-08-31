import { createCanvas, loadImage, Image, Canvas } from 'canvas';
import { Layer } from '../types/layer';

async function computeVisibleMasks(layers: Layer[], width: number, height: number): Promise<boolean[][][]> {
  // Create a temporary canvas for drawing images
  const tempCanvas = createCanvas(width, height);
  const tempCtx = tempCanvas.getContext('2d');

  // Load images for all layers
  const layerImages: Image[] = [];
  for (const layer of layers) {
    const img = await loadImage(layer.currentImgUrl, { crossOrigin: 'anonymous' });
    layerImages.push(img);
  }

  // Compute binary masks for each layer based on the alpha channel
  let masks: boolean[][][] = [];
  for (let i = 0; i < layers.length; i++) {
    const mask = new Array(height);
    for (let y = 0; y < height; y++) {
      mask[y] = new Array(width).fill(false);
    }
    masks.push(mask);
  }


  for (let i = 0; i < layers.length; i++) {
    // Clear the temporary canvas
    tempCtx.clearRect(0, 0, width, height);

    // Draw the layer's image on the temporary canvas
    // tempCtx.drawImage(layerImages[i], 0, 0, width, height);
    const [imgWidth, imgHeight] = [layerImages[i].width, layerImages[i].height]
    tempCtx.drawImage(
      layerImages[i], 
      layers[i].transform.position.x, 
      layers[i].transform.position.y, 
      layers[i].transform.scale * imgWidth,
      layers[i].transform.scale * imgHeight,
    )

    // Get the image data from the temporary canvas
    const imageData = tempCtx.getImageData(0, 0, width, height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const index = (y * width + x) * 4;
        const alpha = imageData.data[index + 3];

        // If the alpha channel value is greater than 0, mark the pixel as visible
        if (alpha > 0) {
          masks[i][y][x] = true;

          // Make the pixel invisible for all layers below the current one
          for (let j = 0; j < i; j++) {
            masks[j][y][x] = false;
          }
        }
      }
    }
  }

  return masks;
}

function renderMasks(masks: boolean[][][], width: number, height: number): Canvas {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  const colors = [
    'rgba(255, 0, 0, 0.5)',
    'rgba(0, 255, 0, 0.5)',
    'rgba(0, 0, 255, 0.5)',
    'rgba(255, 255, 0, 0.5)',
    'rgba(255, 0, 255, 0.5)',
    'rgba(0, 255, 255, 0.5)',
  ];

  for (let i = 0; i < masks.length; i++) {
    const mask = masks[i];
    ctx.fillStyle = colors[i % colors.length];

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (mask[y][x]) {
          ctx.fillRect(x, y, 1, 1);
        }
      }
    }
  }

  return canvas;
}

function updateMaskCanvas(
  canvas: HTMLCanvasElement,
  masks: boolean[][][],
  opacities: number[],
  maskWidth: number,
  maskHeight: number,
  imageWidth: number,
  imageHeight: number
) {
  const ctx = canvas.getContext('2d')!;

  // Clear the canvas
  ctx.clearRect(0, 0, imageWidth, imageHeight);

  // Calculate the scale factors
  const scaleX = imageWidth / maskWidth;
  const scaleY = imageHeight / maskHeight;

  // Draw the masks with adjustable opacity and scaling
  for (let i = 0; i < masks.length; i++) {
    const mask = masks[i];
    const opacity = opacities[i];
    if (opacity <= 0.2) continue;

    ctx.fillStyle = `rgba(255, 0, 0, ${opacity})`;

    for (let y = 0; y < maskHeight; y++) {
      for (let x = 0; x < maskWidth; x++) {
        if (mask[y][x]) {
          // Scale the mask to fit the image size
          ctx.fillRect(x * scaleX, y * scaleY, scaleX, scaleY);
        }
      }
    }
  }
}

export { computeVisibleMasks, renderMasks, updateMaskCanvas };