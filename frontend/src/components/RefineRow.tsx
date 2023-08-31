import React from "react";
import { Layer} from "../types/layer";
import { 
  HStack,
  Image, 
  Box,
  Text,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
} from "@chakra-ui/react";
import { BsSoundwave } from "react-icons/bs";

type RefineRowProps = {
  layer: Layer;
  noiseStrengths: number[];
  setNoiseStrengths: (noiseStrengths: number[]) => void;
}

export default function RefineRow({
  layer,
  noiseStrengths,
  setNoiseStrengths,
}: RefineRowProps) {
  return (
    <>
      <HStack alignItems="center" spacing="5">
        <Image src={layer.currentImgUrl} alt="layer_preview" boxSize="80px" />
        <Text width="25%" mx={2}>{layer.textPrompt}</Text>
        <Slider
          aria-label={`layer${layer.id}_cac`}
          colorScheme="blackAlpha"
          width="40%"
          defaultValue={noiseStrengths[layer.id]}
          mx={2}
          min={0.2}
          max={1}
          step={0.1}
          onChange={(value) => {
            let newNoiseStrengths = [...noiseStrengths];
            newNoiseStrengths[layer.id] = value;
            setNoiseStrengths(newNoiseStrengths);
          }}
        >
          <SliderTrack>
            <SliderFilledTrack />
          </SliderTrack>
          <SliderThumb boxSize={6}>
            <Box color='blackAlpha' as={BsSoundwave} />
          </SliderThumb>
        </Slider>
      </HStack>
    </>
  )
}