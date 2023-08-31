import React, { useState } from "react";
import {
  Box,
  Input,
  Spacer,
  IconButton,
  Image,
  HStack,
  Tooltip,
} from "@chakra-ui/react";
import {
  AiFillCaretUp,
  AiFillCaretDown,
  AiFillEye,
  AiOutlineClose,
  AiOutlineCopy,
} from "react-icons/ai";
import { FiEdit } from "react-icons/fi";
import { generateId } from "../utils/utils";

import { Layer, LayerKey } from "../types/layer";

type LayerRowProps = {
  layer: Layer;
  layers: Layer[];
  rerender: number;
  setRerender: (rerender: number) => void;
  setRerenderTokens: (rerender: boolean) => void;
  setDeletedLayers: (key: LayerKey) => void;
  selectedLayer: Layer | null;
  setSelectedLayer: (layer: Layer | null) => void;
};

export default function LayerRow({
  layer,
  layers,
  rerender,
  setRerender,
  setRerenderTokens,
  setDeletedLayers,
  selectedLayer,
  setSelectedLayer,
}: LayerRowProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [layerTextPrompt, setLayerTextPrompt] = useState(layer.textPrompt);

  function sendBackward(layerIdx: number) {
    // Change the order of the layers. The order of the layers
    // determines the z-index of the layers.
    if (layerIdx === 0) return;
    const layer = layers[layerIdx];
    const swapLayer = layers[layerIdx - 1];
    if (swapLayer) {
      const tempId = layer.id;
      layer.id = swapLayer.id;
      swapLayer.id = tempId;
      layers[layerIdx] = swapLayer;
      layers[layerIdx - 1] = layer;
      setRerender(rerender + 1);
    }
  }

  function bringForward(layerIdx: number) {
    // Change the order of the layers. The order of the layers
    // determines the z-index of the layers.
    if (layerIdx === layers.length - 1) return;
    const layer = layers[layerIdx];
    const swapLayer = layers[layerIdx + 1];
    if (swapLayer) {
      const tempId = layer.id;
      layer.id = swapLayer.id;
      swapLayer.id = tempId;
      layers[layerIdx] = swapLayer;
      layers[layerIdx + 1] = layer;
      setRerender(rerender + 1);
    }
  }

  return (
    <Box bg={selectedLayer === layer ? "#c5c5c5" : "#eeeeee"} py={1}>
      <HStack spacing={1.5} alignItems={"center"}>
        <Tooltip
          hasArrow
          label="Toggle the visibility of layer"
          bg="gray.300"
          color="black"
          openDelay={750}
        >
          <IconButton
            icon={layer.opacity === 0 ? undefined : <AiFillEye />}
            aria-label={`layer${layer.id}_visibility`}
            id={`layer-visibility-${layer.id}`}
            size="sm"
            onClick={() => {
              layers[layer.id].opacity = layers[layer.id].opacity === 0 ? 1 : 0;
              setRerender(rerender + 1);
            }}
            variant="outline"
          />
        </Tooltip>
        <Image src={layer.currentImgUrl} alt="layer_preview" boxSize="40px" />
        <Spacer />
        <Tooltip
          hasArrow
          label="Add a text prompt that describes this layer"
          bg="gray.300"
          color="black"
          openDelay={750}
        >
          <Input
            key={layer.id}
            placeholder="Add prompt..."
            value={layer.textPrompt}
            size="sm"
            onChange={(event) => {
              setLayerTextPrompt(event.target.value);
              layer.textPrompt = event.target.value;
              setRerenderTokens(true);
            }}
            onMouseEnter={() => {
              setIsHovered(true);
            }}
            onMouseLeave={() => {
              setIsHovered(false);
            }}
          />
        </Tooltip>
        <Spacer />
        <HStack spacing={0.5}>
          <Tooltip
            hasArrow
            label="Edit layer-specific parameters, including the mask and the diffusion parameters"
            bg="gray.300"
            color="black"
            openDelay={750}
          >
            <IconButton
              icon={<FiEdit />}
              aria-label={`layer${layer.id}_edit`}
              id={`layer-edit-${layer.id}`}
              size="sm"
              onClick={() => {
                setSelectedLayer(layer);
              }}
            />
          </Tooltip>
          <Spacer />
          <Tooltip
            hasArrow
            label="Bring layer forward"
            bg="gray.300"
            color="black"
            openDelay={750}
          >
            <IconButton
              icon={<AiFillCaretUp />}
              aria-label={`layer${layer.id}_up`}
              id={`layer-up-${layer.id}`}
              size="sm"
              onClick={() => bringForward(layer.id)}
            />
          </Tooltip>
          <Spacer />
          <Tooltip
            hasArrow
            label="Bring layer backward"
            bg="gray.300"
            color="black"
            openDelay={750}
          >
            <IconButton
              icon={<AiFillCaretDown />}
              aria-label={`layer${layer.id}_down`}
              id={`layer-down-${layer.id}`}
              size="sm"
              onClick={() => sendBackward(layer.id)}
            />
          </Tooltip>
          <Spacer />
          <Tooltip
            hasArrow
            label="Delete layer"
            bg="gray.300"
            color="black"
            openDelay={750}
          >
            <IconButton
              icon={<AiOutlineClose />}
              aria-label={`layer${layer.id}_close`}
              id={`layer-delete-${layer.id}`}
              size="sm"
              onClick={() => {
                if (
                  window.confirm("Are you sure you want to delete this layer?")
                ) {
                  setDeletedLayers(layer.key);
                  if (selectedLayer?.id === layer.id) setSelectedLayer(null);

                  // Cleanup layers
                  const newLayers = layers.filter(
                    (currLayer) => currLayer.id !== layer.id
                  );
                  layers.length = 0;
                  layers.push(...newLayers);
                  // Update the layer IDs
                  for (let i = 0; i < layers.length; i++) {
                    layers[i].id = i;
                  }
                  setRerender(rerender + 1);
                  setRerenderTokens(true);
                }
              }}
            />
          </Tooltip>
          <Spacer />
          <Tooltip
            hasArrow
            label="Duplicate layer"
            bg="gray.300"
            color="black"
            openDelay={750}
          >
            <IconButton
              icon={<AiOutlineCopy />}
              aria-label={`layer${layer.id}_dup`}
              id={`layer-dup-${layer.id}`}
              size="sm"
              onClick={() => {
                const newLayerId = layers.length;
                // New Layer: deep copy of the layer
                const newLayer = structuredClone(layer);
                newLayer.id = newLayerId;
                newLayer.key = generateId(20);
                layers.push(newLayer);
                setRerender(rerender + 1);
              }}
            />
          </Tooltip>
        </HStack>
      </HStack>
    </Box>
  );
}
