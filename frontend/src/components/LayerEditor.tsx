// LayerEditor.tsx
// Takes in a layer and allows the user to select points via clicking.
// The clicking generates a polygon, which defines a segmentation mask where
// all pixels inside the polygon are part of the mask.

import React, { useEffect, useState } from "react";
import { Layer } from "../types/layer";
import {
  imageCoordToScaledImageCoord,
  scaledCanvasCoordToImageCoord,
  imageCoordToScaledCanvasCoord,
} from "../utils/transforms";
import {
  Box,
  Button,
  Center,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Flex,
} from "@chakra-ui/react";

type LayerEditorProps = {
  layer: Layer | undefined;
  rerender: number;
  setRerender: (rerender: number) => void;
  canvasHeight: number;
  canvasWidth: number;
  onClose: () => void;
};

function drawCheckeredBackground(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  nRows: number,
  nCols: number
) {
  width /= nCols; // width of a block
  height /= nRows; // height of a block
  // Change black color to gray
  ctx.fillStyle = "rgba(100,100,100,0.3)";

  for (var i = 0; i < nRows; ++i) {
    for (var j = 0, col = nCols / 2; j < col; ++j) {
      ctx.rect(2 * j * width + (i % 2 ? 0 : width), i * height, width, height);
    }
  }

  ctx.fill();
}

function clearCanvas(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawCheckeredBackground(ctx, canvas.width, canvas.height, 30, 30);
}

export default function LayerEditor({
  layer,
  rerender,
  setRerender,
  canvasHeight,
  canvasWidth,
  onClose,
}: LayerEditorProps) {
  const [clicks, setClicks] = useState<{ x: number; y: number }[]>([]);
  const img = new Image();
  img.crossOrigin = "Anonymous";
  const [polygonDone, setPolygonDone] = useState(false);
  const canvasAspectRatio = canvasWidth / canvasHeight;
  const [segmentedUrl, setSegmentedUrl] = useState<string | undefined>(
    undefined
  );

  // Render layer on canvas on mount
  useEffect(() => {
    const canvas = document.getElementById("layerCanvas") as HTMLCanvasElement;
    if (!canvas) {
      return;
    }
    clearCanvas(canvas);
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    if (!layer) return;
    img.src = layer.originalImgUrl;
    // Fit image to canvas
    img.onload = () => {
      const scale = Math.min(
        canvas.width / img.width,
        canvas.height / img.height
      );
      ctx.drawImage(img, 0, 0, img.width * scale, img.height * scale);
    };
  }, []);

  function drawPolygon() {
    const canvas = document.getElementById("layerCanvas") as HTMLCanvasElement;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (clicks.length === 0) return;
    if (!polygonDone) ctx.fillStyle = "rgba(100,100,100,0.3)";
    ctx.strokeStyle = "#df4b26";
    ctx.lineWidth = 1;
    ctx.beginPath();

    const imageAspectRatio = img.width / img.height;
    const firstClick = imageCoordToScaledCanvasCoord(
      clicks[0].x,
      clicks[0].y,
      imageAspectRatio,
      canvasAspectRatio,
      canvasWidth,
      canvasHeight
    );
    ctx.moveTo(firstClick.x, firstClick.y);
    for (let i = 1; i < clicks.length; i++) {
      const scaledCanvasCoord = imageCoordToScaledCanvasCoord(
        clicks[i].x,
        clicks[i].y,
        imageAspectRatio,
        canvasAspectRatio,
        canvasWidth,
        canvasHeight
      );
      ctx.lineTo(scaledCanvasCoord.x, scaledCanvasCoord.y);
    }
    if (polygonDone) {
      ctx.closePath();
      ctx.fill();
    }
    ctx.stroke();
  }

  function drawPoints() {
    const canvas = document.getElementById("layerCanvas") as HTMLCanvasElement;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.strokeStyle = "#df4b26";
    ctx.lineJoin = "round";
    ctx.lineWidth = 5;
    const imageAspectRatio = img.width / img.height;

    for (let i = 0; i < clicks.length; i++) {
      // Color the first point a different color
      if (i === 0) {
        ctx.strokeStyle = "#ff037d";
      } else {
        ctx.strokeStyle = "#df4b26";
      }
      ctx.beginPath();
      const scaledCanvasCoord = imageCoordToScaledCanvasCoord(
        clicks[i].x,
        clicks[i].y,
        imageAspectRatio,
        canvasAspectRatio,
        canvasWidth,
        canvasHeight
      );
      ctx.arc(
        scaledCanvasCoord.x,
        scaledCanvasCoord.y,
        3,
        0,
        2 * Math.PI,
        false
      );
      ctx.fill();
      ctx.stroke();
    }
  }

  function handleClick(event: React.MouseEvent<HTMLCanvasElement, MouseEvent>) {
    const canvas = document.getElementById("layerCanvas") as HTMLCanvasElement;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();
    const scaledCanvasX = event.clientX - rect.left;
    const scaledCanvasY = event.clientY - rect.top;
    // Store polygon in image space
    let scaledFirstClick = { x: 0, y: 0 };
    const imageAspectRatio = img.width / img.height;
    if (clicks.length > 0) {
      scaledFirstClick = imageCoordToScaledCanvasCoord(
        clicks[0].x,
        clicks[0].y,
        imageAspectRatio,
        canvasAspectRatio,
        canvasWidth,
        canvasHeight
      );
    }

    // Polygon is complete if click is within 10 pixels of first click
    if (
      clicks.length > 0 &&
      Math.abs(scaledFirstClick.x - scaledCanvasX) < 10 &&
      Math.abs(scaledFirstClick.y - scaledCanvasY) < 10
    ) {
      // Make fill darker
      ctx.fillStyle = "rgba(100,100,100,0.8)";
      updateImage();
      return;
    }

    const imageCoord = scaledCanvasCoordToImageCoord(
      scaledCanvasX,
      scaledCanvasY,
      imageAspectRatio,
      canvasAspectRatio,
      canvasWidth,
      canvasHeight
    );
    const newClicks = [...clicks, imageCoord];
    setClicks(newClicks);

    if (layer) {
      layer.polygon = newClicks;
    }
  }

  function updateImage() {
    if (!layer) return;
    const maskedCanvas = document.createElement("canvas");
    const ctx = maskedCanvas.getContext("2d");
    if (!ctx) return;
    maskedCanvas.width = img.width;
    maskedCanvas.height = img.height;

    ctx.drawImage(img, 0, 0);
    const imgData = ctx.getImageData(0, 0, img.width, img.height);

    // Create mask with 0s, divide by 4 because each pixel has 4 values
    const mask = new Uint8ClampedArray(imgData.data.length / 4);

    const polygonPath = new Path2D();
    for (let i = 0; i < clicks.length; i++) {
      const scaledImageCoord = imageCoordToScaledImageCoord(
        clicks[i].x,
        clicks[i].y,
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
      const y = Math.floor(i / 4 / croppedImgData.width);
      const imgDataIndex = ((y + minY) * img.width + (x + minX)) * 4;
      croppedImgData.data[i] = imgData.data[imgDataIndex];
      croppedImgData.data[i + 1] = imgData.data[imgDataIndex + 1];
      croppedImgData.data[i + 2] = imgData.data[imgDataIndex + 2];
      croppedImgData.data[i + 3] = imgData.data[imgDataIndex + 3];
    }

    // Create a new cropped canvas
    const croppedCanvas = document.createElement("canvas");
    croppedCanvas.width = croppedImgData.width;
    croppedCanvas.height = croppedImgData.height;
    const croppedCtx = croppedCanvas.getContext("2d");
    if (!croppedCtx) return;
    croppedCtx.putImageData(croppedImgData, 0, 0);

    // // Clear canvas and draw image with mask
    clearCanvas(maskedCanvas);
    ctx.putImageData(croppedImgData, 0, 0);

    setTimeout(() => {
      const newImgUrl = croppedCanvas.toDataURL();
      setSegmentedUrl(newImgUrl);
      setPolygonDone(true);
    }, 100);
  }

  useEffect(() => {
    // Redraw polygon and points on clicks change
    const canvas = document.getElementById("layerCanvas") as HTMLCanvasElement;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas and redraw image
    clearCanvas(canvas);
    if (!layer) return;
    img.src = layer.originalImgUrl;
    // Fit image to canvas
    const scale = Math.min(
      canvas.width / img.width,
      canvas.height / img.height
    );
    ctx.drawImage(img, 0, 0, img.width * scale, img.height * scale);

    // if (!polygonDone) {
    drawPoints();
    drawPolygon();
    // }
  }, [clicks, polygonDone]);

  return (
    <>
      <ModalHeader>Segment layer</ModalHeader>
      <ModalBody>
        Draw a polygon around the object you would like to select. Each click
        adds a new point to the polygon. Click the first point you selected to
        finalize the mask.
        <Box pt={5}>
          <Center>
            <canvas
              id="layerCanvas"
              width={canvasWidth}
              height={canvasHeight}
              onClick={handleClick}
              style={{
                border: "2px solid #000000",
              }}
            />
          </Center>
        </Box>
      </ModalBody>
      <ModalFooter>
        <Flex>
          {segmentedUrl && layer && (
            <Button
              onClick={() => {
                layer.currentImgUrl = segmentedUrl!;
                setRerender(rerender + 1);
                onClose();
              }}
              mx={2}
            >
              Save mask
            </Button>
          )}
          {clicks.length > 0 && (
            <Button
              onClick={() => {
                setClicks([]);
                setSegmentedUrl(undefined);
                setPolygonDone(false);
              }}
              mx={2}
            >
              Clear mask
            </Button>
          )}
          <Button ml={2} onClick={onClose}>
            Close
          </Button>
        </Flex>
      </ModalFooter>
    </>
  );
}
