/* Deprecated component. For use with textual inversion, which is no longer supported */
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
import { fetchWithRetryAndTimeout } from "../utils/utils";
import { convertLayersToRequestLayers, RequestLayer } from "../types/requests";

const MAX_FRAMES = 100;
const MAX_STEPS = 100;

type RefineDispatcherProps = {
  webserverAddress: string;
  appendToHistory: (jobSet: JobSet) => void;
  diffusionParams: DiffusionParameters;
  collageSrc: string;
  layers: Layer[];
  noiseStrengths: number[];
  numFrames: number;
  canvasHeight: number;
  canvasWidth: number;
};

export default function RefineDispatcher({
  webserverAddress,
  appendToHistory,
  diffusionParams,
  collageSrc,
  layers,
  noiseStrengths,
  numFrames,
  canvasHeight,
  canvasWidth,
}: RefineDispatcherProps) {
  const [disabled, setDisabled] = useState<boolean>(false);
  const [badParams, setBadParams] = useState<boolean>(false);

  async function startJob() {
    setDisabled(true);
    let timeout = new Promise((resolve) =>
      setTimeout(() => {
        resolve(null);
      }, 5000)
    );

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

    for (var i = 0; i < requestLayers.length; i++) {
      requestLayers[i].noiseStrength = noiseStrengths[i];
    }

    const initialSeed = Number(diffusionParams.seed);
    var seeds: number[] = [];
    for (var i = 0; i < numFrames; i++) {
      seeds.push(initialSeed + i);
    }

    var localDiffusionParamsArr: DiffusionParameters[] = [];
    for (var i = 0; i < numFrames; i++) {
      var localDiffusionParams = structuredClone(diffusionParams);
      localDiffusionParams.seed = seeds[i];
      // Add a space to the end of the prompt bc of string matching bug, TODO: FIX!!!
      localDiffusionParams.prompt = diffusionParams.prompt + " ";
      localDiffusionParamsArr.push(localDiffusionParams);
    }

    // Hit sample API endpoint, which will return a list of paths to images
    Promise.allSettled(
      Array.from({ length: numFrames }, (x, i) => i).map((i) => {
        const request = {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            collageSrc: collageSrc,
            layers: requestLayers,
            ...localDiffusionParamsArr[i],
          }),
        };

        return fetchWithRetryAndTimeout(
          `${webserverAddress}/collage_edit`,
          request,
          1500,
          10
        )
          .then((response) => {
            if (response.ok) {
              return response.json();
            } else {
              throw new Error("Network response was not ok.");
            }
          })
          .then((response) => {
            return response.job_id;
          });
      })
    )
      .then((newJobIdsPromise) => {
        return newJobIdsPromise.map((promise) => {
          if (promise.status === "fulfilled") {
            return promise.value;
          } else {
            return null;
          }
        });
      })
      .then((newJobIds) => {
        var newJobSet: JobSet = [];
        for (var i = 0; i < newJobIds.length; i++) {
          newJobSet.push(new Job(newJobIds[i], localDiffusionParamsArr[i]));
        }
        appendToHistory(newJobSet);
      });

    await timeout;
    setDisabled(false);
  }

  return (
    <Box>
      <Button id="refine" onClick={startJob} disabled={disabled}>
        Refine collage
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
    </Box>
  );
}
