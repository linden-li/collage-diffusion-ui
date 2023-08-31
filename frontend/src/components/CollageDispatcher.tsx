import React, { useState } from "react";
import DiffusionParameters from "./DiffusionParameters";
import {
  Button,
  Box,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
} from "@chakra-ui/react";
import { Job, JobSet } from "./JobDispatcher";
import { Layer } from "../types/layer";
import { b64ToUrl, generateId } from "../utils/utils";
import { convertLayersToRequestLayers, RequestLayer } from "../types/requests";

const MAX_FRAMES = 100;
const MAX_STEPS = 100;

type CollageDispatcherProps = {
  webserverAddress: string;
  appendToHistory: (jobSet: JobSet) => void;
  diffusionParams: DiffusionParameters;
  layers: Layer[];
  needsAutoAdjust: boolean;
  tuneDiffusionParams: () => Promise<void>;
  setRerender: (rerender: number) => void;
  rerender: number;
  numFrames: number;
  canvasHeight: number;
  canvasWidth: number;
  killCurrentJobs: () => void;
};

export default function CollageDispatcher({
  webserverAddress,
  appendToHistory,
  diffusionParams,
  layers,
  needsAutoAdjust,
  tuneDiffusionParams,
  setRerender,
  rerender,
  numFrames,
  canvasHeight,
  canvasWidth,
  killCurrentJobs,
}: CollageDispatcherProps) {
  const [disabled, setDisabled] = useState<boolean>(false);
  const [badParams, setBadParams] = useState<boolean>(false);
  const [badCollage, setBadCollage] = useState<boolean>(false);

  async function startJob() {
    setDisabled(true);
    let timeout = new Promise((resolve) =>
      setTimeout(() => {
        resolve(null);
      }, 5000)
    );

    // Kill current/oustanding jobs
    killCurrentJobs();

    // Prevent user from overloading the system
    if (
      numFrames > MAX_FRAMES ||
      diffusionParams.numInferenceSteps > MAX_STEPS
    ) {
      setBadParams(true);
      await timeout;
      setDisabled(false);
      return;
    } else {
      setBadParams(false);
    }

    // Read transform from canvas
    let requestLayers: RequestLayer[] = convertLayersToRequestLayers(
      layers,
      canvasWidth,
      canvasHeight
    );

    const initialSeed = Number(diffusionParams.seed);
    var seeds: number[] = [];
    for (var i = 0; i < numFrames; i++) {
      seeds.push(initialSeed + i);
    }
    for (let i = 0; i < requestLayers.length; i++) {
      if (requestLayers[i].textPrompt === "") {
        setBadCollage(true);
        await timeout;
        setDisabled(false);
        setBadCollage(false);
        return;
      }
    }

    if (diffusionParams.prompt === "") {
      setBadCollage(true);
      await timeout;
      setDisabled(false);
      setBadCollage(false);
      return;
    }

    // Check that each request layer text prompt is a substring
    // of the collage prompt
    for (let i = 0; i < requestLayers.length; i++) {
      if (!diffusionParams.prompt.includes(requestLayers[i].textPrompt)) {
        setBadCollage(true);
        await timeout;
        setDisabled(false);
        setBadCollage(false);
        return;
      }
    }

    if (needsAutoAdjust) {
      await tuneDiffusionParams();
      setRerender(rerender + 1);
    }

    var localDiffusionParamsArr: DiffusionParameters[] = [];
    for (var i = 0; i < numFrames; i++) {
      var localDiffusionParams = structuredClone(diffusionParams);
      localDiffusionParams.seed = seeds[i];
      // Add a space to the end of the prompt bc of string matching bug, TODO: FIX!!!
      localDiffusionParams.prompt = diffusionParams.prompt + " ";
      localDiffusionParamsArr.push(localDiffusionParams);
    }

    const request = {
      method: "POST",
      headers: {
        accept: "application/json",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        layers: requestLayers,
        numInferenceSteps: localDiffusionParamsArr[0].numInferenceSteps,
        guidanceScale: localDiffusionParamsArr[0].guidanceScale,
        seed: localDiffusionParamsArr[0].seed,
        prompt: localDiffusionParamsArr[0].prompt,
        numImages: numFrames,
      }),
    };

    const response: Promise<string[]> = fetch(
      `${webserverAddress}/together_inference`,
      request
    )
      .then((response) => {
        if (response.ok) {
          return response.json();
        } else {
          throw new Error("Network response was not ok.");
        }
      })
      .then((response) => {
        // Response is an array, so get the base64 image from each element
        return response.map((element: any) => {
          return b64ToUrl(element.output.image.image_base64);
        });
      });

    const imgUrls = await response;

    var newJobSet: JobSet = [];
    for (var i = 0; i < imgUrls.length; i++) {
      newJobSet.push(
        new Job(generateId(10), localDiffusionParamsArr[i], imgUrls[i])
      );
    }
    appendToHistory(newJobSet);

    await timeout;
    setDisabled(false);
  }

  return (
    <Box>
      <Button onClick={startJob} isLoading={disabled} borderRadius="full">
        Generate images
      </Button>
      {badParams && (
        <Alert status="error">
          <AlertIcon />
          <AlertTitle>Bad parameters!</AlertTitle>
          <AlertDescription>
            Number of images and number of inference steps must be less than or
            equal to {MAX_FRAMES} and {MAX_STEPS}, respectively.
          </AlertDescription>
        </Alert>
      )}
      {badCollage && (
        <Alert status="error" my={2}>
          <AlertIcon />
          <AlertTitle>Please fix your collage parameters!</AlertTitle>
          <AlertDescription>
            You must enter a prompt for each layer and include a global text
            prompt explaining the whole collage. Each layer's text prompt needs
            to be included in the collage prompt.
          </AlertDescription>
        </Alert>
      )}
    </Box>
  );
}
