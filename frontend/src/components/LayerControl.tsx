import React, { useState } from "react";
import { Layer } from "../types/layer";
import {
  Box,
  VStack,
  Slider,
  SliderTrack,
  Text,
  SliderFilledTrack,
  Modal,
  ModalOverlay,
  ModalContent,
  SliderThumb,
  Tooltip,
  Flex,
  useDisclosure,
  Button,
  Image,
} from "@chakra-ui/react";
import { FiPlus, FiMinus } from "react-icons/fi";
import LayerEditor from "./LayerEditor";

type LayerControlProps = {
  layer: Layer;
  rerender: number;
  setRerender: (rerender: number) => void;
};

export default function LayerControl({
  layer,
  rerender,
  setRerender,
}: LayerControlProps) {
  const [noiseStrength, setNoiseStrength] = useState(layer.noiseStrength);
  const [cacStrength, setCacStrength] = useState(layer.cacStrength);
  const [negativeStrength, setNegativeStrength] = useState(
    layer.negativeStrength
  );
  const [layerCannyStrength, setLayerCannyStrength] = useState(
    layer.cannyStrength
  );

  const { isOpen, onOpen, onClose } = useDisclosure();

  return (
    <>
      <Modal isOpen={isOpen} onClose={onClose} size="4xl">
        <ModalOverlay />
        <ModalContent>
          <LayerEditor
            layer={layer}
            rerender={rerender}
            setRerender={setRerender}
            canvasHeight={512}
            canvasWidth={512}
            onClose={onClose}
          />
        </ModalContent>
      </Modal>

      <Flex direction="column" id="layer-parameter-editor" my={2}>
        <Flex
          backgroundColor="#c5c5c5"
          padding="8px"
          marginBottom="4"
          justifyContent="center"
          alignItems="center"
        >
          <Image
            src={layer.currentImgUrl}
            alt="layer_preview"
            boxSize="40px"
            marginRight="4"
          />
          <Text>{layer.textPrompt}</Text>
        </Flex>

        <Flex justifyContent="space-around" alignItems="center">
          <Button id="layer-mask-button" onClick={onOpen}>
            Edit Layer Mask
          </Button>

          <VStack align="stretch" spacing="2">
            <Text fontSize="sm">Noise strength</Text>
            <Slider
              aria-label={`layer${layer.id}_noise`}
              id={`layer-noise-slider-${layer.id}`}
              colorScheme="blackAlpha"
              value={layer.noiseStrength}
              min={0.2}
              max={1}
              step={0.1}
              onChange={(value) => {
                setNoiseStrength(value);
                layer.noiseStrength = value;
              }}
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <SliderThumb />
            </Slider>

            <Text fontSize="sm">Shape strength</Text>
            <Slider
              aria-label={`layer${layer.id}_shapeStrength`}
              id={`layer-shape-slider-${layer.id}`}
              colorScheme="blackAlpha"
              value={layer.cannyStrength || 0.0}
              defaultValue={0.0}
              min={0}
              max={1}
              step={0.1}
              onChange={(value) => {
                setLayerCannyStrength(value);
                layer.cannyStrength = value;
              }}
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <Tooltip
                hasArrow
                label="Change the ControlNet canny edge strength. A greater value will preserve the shape more."
                bg="gray.300"
                color="black"
                openDelay={750}
              >
                <SliderThumb />
              </Tooltip>
            </Slider>
            <Text fontSize="sm">Negative attention strength</Text>
            <Slider
              aria-label={`layer${layer.id}_negCac`}
              id={`layer-negCac-slider-${layer.id}`}
              colorScheme="red"
              value={layer.negativeStrength || 1.0}
              defaultValue={1.0}
              min={0}
              max={1}
              step={0.1}
              onChange={(value) => {
                setNegativeStrength(value);
                layer.negativeStrength = value;
              }}
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <Tooltip
                hasArrow
                label="Change the negative cross attention strength. A higher value the object will not appear where it isn't supposed to."
                bg="gray.300"
                color="black"
                openDelay={750}
              >
                <SliderThumb>
                  <Box color="red" as={FiMinus} />
                </SliderThumb>
              </Tooltip>
            </Slider>
            <Text fontSize="sm">Positive attention strength</Text>
            <Slider
              aria-label={`layer${layer.id}_cac`}
              id={`layer-cac-slider-${layer.id}`}
              colorScheme="whatsapp"
              value={layer.cacStrength || 0.0}
              defaultValue={0.0}
              min={0}
              max={1}
              step={0.1}
              onChange={(value) => {
                setCacStrength(value);
                layer.cacStrength = value;
              }}
            >
              <SliderTrack>
                <SliderFilledTrack />
              </SliderTrack>
              <Tooltip
                hasArrow
                label="Change the positive attention strength. A higher value that the object will appear where it's supposed to."
                bg="gray.300"
                color="black"
                openDelay={750}
              >
                <SliderThumb>
                  <Box color="whatsapp" as={FiPlus} />
                </SliderThumb>
              </Tooltip>
            </Slider>
          </VStack>
        </Flex>
      </Flex>
    </>
  );
}
