import React, { useEffect } from "react";
import { Box, Divider, Text, VStack, StackDivider } from "@chakra-ui/react";
import { Layer, LayerKey } from "../types/layer";
import LayerRow from "./LayerRow";
import { JoyrideState } from "./CollageEditor";

type LayerSidebarProps = {
  layers: Layer[];
  run: boolean;
  setJoyrideState: (state: Partial<JoyrideState>) => void;
  rerender: number;
  setRerender: (rerender: number) => void;
  setRerenderTokens: (rerender: boolean) => void;
  setDeletedLayers: (key: LayerKey) => void;
  selectedLayer: Layer | null;
  setSelectedLayer: (layer: Layer | null) => void;
};

export default function LayerSidebar({
  layers,
  run,
  setJoyrideState,
  rerender,
  setRerender,
  setRerenderTokens,
  setDeletedLayers,
  selectedLayer,
  setSelectedLayer,
}: LayerSidebarProps) {
  useEffect(() => {
    if (run && layers.length === 1) {
      setTimeout(() => {
        setJoyrideState({
          stepIndex: 4,
        });
      }, 500);
    }
  }, [layers.length]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Box>
      <Text>Layers ({layers.length})</Text>
      <Divider borderColor="gray.600" />

      <VStack
        align={"stretch"}
        divider={<StackDivider borderColor="gray.400" />}
        maxHeight="300px"
        overflowY="auto"
      >
        {layers
          .slice()
          .reverse()
          .map((layer, index) => (
            <Box id={`layer-control-${index}`}>
              <LayerRow
                key={layer.id}
                layer={layer}
                layers={layers}
                rerender={rerender}
                setRerender={setRerender}
                setRerenderTokens={setRerenderTokens}
                setDeletedLayers={setDeletedLayers}
                selectedLayer={selectedLayer}
                setSelectedLayer={setSelectedLayer}
              />
            </Box>
          ))}
      </VStack>
      {layers.length > 0 && <Divider borderColor="gray.600" />}
    </Box>
  );
}
