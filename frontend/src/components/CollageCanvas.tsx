import React, { useEffect, useState } from "react";
import { fabric } from "fabric";
import { Layer } from "../types/layer";
import { scaledCanvasCoordToCanvasCoord } from "../utils/transforms";

type CollageCanvasProps = {
  layers: Layer[];
  rerender: number;
  setRerender: (rerender: number) => void;
  canvasWidth: number;
  canvasHeight: number;
};

export default function CollageCanvas({
  layers,
  rerender,
  setRerender,
  canvasWidth,
  canvasHeight,
}: CollageCanvasProps) {
  fabric.Object.prototype.set({
    transparentCorners: false,
    borderColor: "#0062ff",
    cornerColor: "#ffffff",
  });
  const [canvas, setCanvas] = useState<fabric.Canvas | null>(null);

  // Setup canvas
  useEffect(() => {
    setCanvas(
      () =>
        new fabric.Canvas("canvas", {
          height: canvasHeight,
          width: canvasWidth,
          preserveObjectStacking: true,
          backgroundColor: "#969696",
        })
    );
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Change canvas size
  useEffect(() => {
    if (canvas) {
      canvas.setHeight(canvasHeight);
      canvas.setWidth(canvasWidth);
    }
  }, [canvasHeight, canvasWidth]);

  // On object modification, update the transform of the layer
  useEffect(() => {
    if (canvas) {
      canvas.on("object:modified", (e) => {
        const activeObjects = canvas.getActiveObjects();
        if (activeObjects.length > 0) {
          for (let i = 0; i < activeObjects.length; i++) {
            const obj = activeObjects[i];
            let objX = obj.left || 0;
            let objY = obj.top || 0;
            let objScale = obj.scaleX || 0;
            // Handle object in a group
            if (obj && obj.group) {
              let group = obj.group;
              let groupLeft = group.left || 0;
              let groupTop = group.top || 0;
              let groupWidth = group.width || 0;
              let groupHeight = group.height || 0;
              let groupScale = group.scaleX || 1;
              objX += groupLeft + groupWidth / 2;
              objY += groupTop + groupHeight / 2;
              objScale *= groupScale;
            }

            let layer = layers.find(
              (layer) => layer.key.toString() === obj.name
            );
            if (layer) {
              const canvasCoord = scaledCanvasCoordToCanvasCoord(
                objX,
                objY,
                canvasWidth,
                canvasHeight
              );
              layer.transform.position.x = canvasCoord.x;
              layer.transform.position.y = canvasCoord.y;
              layer.transform.scale = (objScale || 0) / canvasWidth;
              layer.transform.rotation = obj.angle || 0;
              layers[layer.id] = layer;
            }
          }
          setRerender(0);
        }
      });
    }
  }, [canvas]);

  useEffect(() => {
    if (canvas && rerender !== 0) {
      canvas.remove(...canvas.getObjects());

      Promise.all(
        layers.map(
          (layer) =>
            new Promise<{ img: fabric.Image; layerId: number } | null>(
              (resolve) => {
                if (layer.opacity > 0) {
                  fabric.Image.fromURL(layer.currentImgUrl, (img) => {
                    img.set({
                      originX: "left",
                      originY: "top",
                      left: layer.transform.position.x * canvasWidth,
                      top: layer.transform.position.y * canvasWidth,
                      scaleX: layer.transform.scale * canvasWidth,
                      scaleY: layer.transform.scale * canvasWidth,
                      name: layer.key.toString(),
                    });
                    img.setControlsVisibility({
                      mtr: false, // rotation
                      ml: false,
                      mt: false,
                      mr: false,
                      mb: false,
                    });
                    resolve({ img, layerId: layer.id });
                  });
                } else {
                  resolve(null);
                }
              }
            )
        )
      ).then((imagesAndIds) => {
        imagesAndIds
          .filter(
            (item): item is { img: fabric.Image; layerId: number } =>
              item !== null
          )
          .forEach(({ img, layerId }) => {
            canvas.add(img);
            img.moveTo(layerId);
          });
      });
    }
  }, [canvas, rerender, canvasWidth, canvasHeight]);

  return (
    <div>
      <canvas
        id="canvas"
        style={{
          border: "1px solid black",
          boxShadow: "rgba(0, 0, 0, 0.15) 1.95px 1.95px 2.6px",
        }}
      />
    </div>
  );
}
