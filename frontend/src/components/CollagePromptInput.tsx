// Code adapted from https://github.com/trananhtuat/react-draggable-list/blob/main/src/components/list/DraggableList.jsx

import { HStack, Flex, Input } from "@chakra-ui/react";
import React, { useEffect, useRef, useState } from "react";
import { Layer, LayerKey } from "../types/layer";
import { generateId } from "../utils/utils";

import { Token, TokenData } from "./Token";

type CollagePromptInputProps = {
  layers: Layer[];
  data: TokenData[];
  deletedLayers: LayerKey[];
  rerenderTokens: boolean;
  layerToIndexMap: Map<LayerKey, number>;
  setData: (data: TokenData[]) => void;
  setLayerToIndexMap: (map: Map<LayerKey, number>) => void;
  setPrompt: (prompt: string) => void;
  setRerenderTokens: (rerender: boolean) => void;
  setDeletedLayers: (newSet: LayerKey[]) => void;
  tokenIdLength: number;
};

export function CollagePromptInput({
  layers,
  data,
  deletedLayers,
  rerenderTokens,
  layerToIndexMap,
  setData,
  setLayerToIndexMap,
  setPrompt,
  setRerenderTokens,
  setDeletedLayers,
  tokenIdLength,
}: CollagePromptInputProps) {
  const containerRef = useRef<HTMLInputElement>(null);
  const [dragStartIndex, setdragStartIndex] = useState<number>(-1);

  // Current
  let list: TokenData[] = [...data];
  let map = new Map(layerToIndexMap);

  function createEmptyToken() {
    return {
      data: undefined,
      layerKey: undefined,
      id: generateId(tokenIdLength),
    };
  }

  function addNewLayer(newLayer: Layer) {
    list.push({
      data: newLayer.textPrompt,
      layerKey: newLayer.key,
      id: generateId(tokenIdLength),
    });
    list.push(createEmptyToken());
    map.set(newLayer.key, list.length - 2);

    setData(list);
    setLayerToIndexMap(map);
  }

  function updateDataTextPrompts(layers: Layer[]) {
    let shouldRerender = false;

    // TODO do this more intelligently
    // If layers change (forcing rerender), update data
    // Need to keep map consistent in case of text change and/or deletion of layer
    layers.forEach((layer: Layer) => {
      const idx = map.get(layer.key)!;
      if (list[idx]!.data !== layer.textPrompt) {
        list[idx]!.data = layer.textPrompt;
        shouldRerender = true;
      }
    });

    if (shouldRerender) {
      setData(list);
    }
  }

  function deleteAndMerge(index: number) {
    if (list.length === 0) {
      return;
    }
    if (list[index - 1] === undefined || list[index + 1] === undefined) {
      return;
    }
    let s1 = list[index - 1].data ? list[index - 1]!.data : "";
    let s2 = list[index + 1].data ? list[index + 1]!.data : "";

    list[index - 1] = {
      data: `${s1} ${s2}`,
      layerKey: undefined,
      id: list[index + 1]!.id,
    };
    list.splice(index, 2);

    setData(list);
  }

  // get index of draged item
  function onDragStart(index: number) {
    setdragStartIndex(index);
  }

  // update list when item dropped
  function onDrop(dropIndex: number) {
    if (dropIndex < dragStartIndex) {
      dropIndex = Math.max(1, dropIndex % 2 === 0 ? dropIndex + 1 : dropIndex);
    } else {
      dropIndex = Math.min(
        data.length - 2,
        dropIndex % 2 === 0 ? dropIndex - 1 : dropIndex
      );
    }

    // update map
    map.set(data[dragStartIndex]!.layerKey!, dropIndex);
    map.set(data[dropIndex]!.layerKey!, dragStartIndex);

    // swap items
    let list = [...data];
    let tmp = list[dragStartIndex];
    list[dragStartIndex] = list[dropIndex];
    list[dropIndex] = tmp;
    setData(list);

    setLayerToIndexMap(map);
  }

  // Handle deleted layers
  useEffect(() => {
    if (deletedLayers.length > 0) {
      deletedLayers.forEach((key: LayerKey, i: number) => {
        const index = layerToIndexMap.get(key);
        deleteAndMerge(index!);
        map.delete(key);
        // If we delete an element, need to update map indices past the deleted
        // index (shift to the left)
        layerToIndexMap.forEach((idx: number, key: LayerKey) => {
          if (idx > index!) {
            map.set(key, idx - 2);
          }
        });
      });

      setDeletedLayers([]);
      setLayerToIndexMap(map);
    }
  }, [deletedLayers]);

  // Combine tokens into prompt
  useEffect(() => {
    let prompt = "";
    list.forEach((token: TokenData) => {
      if (token.data !== undefined) {
        // Bold text if it's a layer
        if (token.layerKey !== undefined) {
          prompt += `${token.data} `;
        } else {
          prompt += `${token.data} `;
        }
      }
    });
    setPrompt(prompt);
  }, [data]);

  // Handle all layer changes
  useEffect(() => {
    if (layers.length * 2 > data.length) {
      const newLayer = layers[layers.length - 1];
      addNewLayer(newLayer);
    } else {
      updateDataTextPrompts(layers);
    }
    setRerenderTokens(false);
  }, [rerenderTokens]);

  return (
    <Flex
      overflow="auto"
      id="input_with_tokens"
      ref={containerRef}
      borderRadius="25px"
      borderColor="#000000"
      borderWidth="1px"
      style={{
        position: "relative",
        width: "100%",
        height: "auto",
      }}
      flexWrap="wrap"
    >
      {data.map((item, index) => {
        return (
          <Token
            id={item.id}
            index={index}
            type={index % 2 === 0 ? Input : undefined}
            draggable={index % 2 === 0 ? false : true}
            onDragStart={(index: number) => onDragStart(index)}
            onDrop={(index: number) => onDrop(index)}
            value={item.data}
            layerKey={item.layerKey}
            setToken={(token: TokenData) => {
              let newData = [...data];
              newData[index] = token;
              setData(newData);
            }}
            maxW={80 / (data.length / 2 - 1)} // 80 / num_tokens ... idk if we need this
          />
        );
      })}
    </Flex>
  );
}
