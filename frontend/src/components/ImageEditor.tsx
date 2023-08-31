import { useState, useEffect, useRef } from 'react';
import { Box, HStack, Flex, Spacer, Text, Divider, Image } from '@chakra-ui/react';
import RefineRow from "./RefineRow";
import { ParameterInput } from "./ParameterInput";
import DiffusionParameters from "./DiffusionParameters";
import RefineDispatcher from "./RefineDispatcher";
import { Layer } from '../types/layer';
import { JobSet } from './JobDispatcher';
import { ImageGallery } from './ImageGallery';
import { updateMaskCanvas } from '../utils/mask';

type ImageEditorProps = {
  imageSrc: string;
  layers: Layer[];
  masks: boolean[][][];
  prompt: string;
  webserverAddress: string;
  imageId: string;
  canvasHeight: number;
  canvasWidth: number;
  addTab: (props: ImageEditorProps) => void;
};

function ImageEditor({
  imageSrc,
  layers,
  masks,
  prompt,
  webserverAddress,
  imageId,
  canvasHeight,
  canvasWidth,
  addTab,
}: ImageEditorProps) {
  const [numFrames, setNumFrames] = useState<number>(5);
  const [numInferenceSteps, setNumInferenceSteps] = useState<number>(30);
  const [guidanceScale, setGuidanceScale] = useState<number>(7);
  const [seed, setSeed] = useState<number>(42);
  const [noiseStrengths, setNoiseStrengths] = useState<number[]>(
    Array(layers.length).fill(0.2)
  );
  const [history, setHistory] = useState<JobSet[]>([]);

  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    if (maskCanvasRef.current && imageDimensions) {
      const { width, height } = imageDimensions;
      maskCanvasRef.current.width = width;
      maskCanvasRef.current.height = height;
      const maskWidth = masks[0][0].length;
      const maskHeight = masks[0].length;
      updateMaskCanvas(maskCanvasRef.current, masks, noiseStrengths, maskWidth, maskHeight, width, height);
    }
  }, [imageDimensions, masks, noiseStrengths]);

  function appendToHistory(jobSet: JobSet) {
    setHistory([jobSet, ...history]);
  }
  
  function handleParamInput(paramName: string, paramValue: string | number) {
    if (paramName === "numFrames") setNumFrames(paramValue as number);
    else if (paramName === "numInferenceSteps")
      setNumInferenceSteps(paramValue as number);
    else if (paramName === "guidanceScale")
      setGuidanceScale(paramValue as number);
    else if (paramName === "seed") setSeed(paramValue as number);
    else console.log(`Invalid parameter name: ${paramName}`);
  }

  function extractDiffusionParams() {
    return new DiffusionParameters(
      prompt,
      numInferenceSteps,
      guidanceScale,
      seed,
    );
  }

  function renderImageGallery(shouldRender: boolean) {
    if (shouldRender) {
      const jobSet = history[0];
      return (
        <ImageGallery
          key={jobSet[0].jobId}
          webserverAddress={webserverAddress}
          jobSet={jobSet}
          layers={layers}
          canvasHeight={canvasHeight}
          canvasWidth={canvasWidth}
        />
      );
    } else {
      return null;
    }
  }

  return (
    <>
      <Box mx={8} my={5} >
        <Flex alignItems={"top"}>
          <Box width="60%" position="relative" display="inline-block">
            <Image
              src={imageSrc}
              alt="Diffusion result"
              onLoad={(e) => {
                const img = e.target as HTMLImageElement;
                setImageDimensions({ width: img.width, height: img.height });
              }}
            />
            <canvas
              ref={maskCanvasRef}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                pointerEvents: 'none',
              }}
            />
          </Box>
          <Flex 
            width="40%"
            flexDirection={'column'}
          >
            <Box>
              <Text>Layers ({layers.length})</Text>
              <Divider />
              {layers
                .slice()
                .reverse()
                .map((layer) => (
                  <RefineRow
                    layer={layer}
                    noiseStrengths={noiseStrengths}
                    setNoiseStrengths={setNoiseStrengths}
                  />
                ))}
            </Box>
            <Spacer />
            <HStack>
            <ParameterInput
              htmlSize={5}
              paramName="numFrames"
              type="number"
              size="sm"
              description="Number of images: "
              value={numFrames}
              placeholder={numFrames}
              onChange={handleParamInput}
            />
            <ParameterInput
              paramName="numInferenceSteps"
              htmlSize={5}
              size="sm"
              type="number"
              description="Inference steps: "
              value={numInferenceSteps}
              placeholder={numInferenceSteps}
              onChange={handleParamInput}
            />
            <ParameterInput
              paramName="guidanceScale"
              htmlSize={5}
              size="sm"
              type="number"
              description="Guidance scale: "
              value={guidanceScale}
              placeholder={guidanceScale}
              onChange={handleParamInput}
            />
            <ParameterInput
              paramName="seed"
              size="sm"
              type="number"
              description="Initial seed: "
              value={seed}
              placeholder={seed}
              onChange={handleParamInput}
            />
            </HStack>
            <RefineDispatcher
              webserverAddress={webserverAddress}
              appendToHistory={appendToHistory}
              diffusionParams={extractDiffusionParams()}
              collageSrc={imageId} // TODO: change to google cloud storage eventually
              layers={layers}
              noiseStrengths={noiseStrengths}
              numFrames={numFrames}
              canvasHeight={canvasHeight}
              canvasWidth={canvasWidth}
            />
          </Flex>
        </Flex>
        {renderImageGallery(history.length > 0)}
      </Box>
    </>
      
  )
}

export { ImageEditor, ImageEditorProps}