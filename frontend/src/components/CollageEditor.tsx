import React, { useEffect, useState, useRef } from "react";
import {
  Box,
  Divider,
  Text,
  Flex,
  HStack,
  Button,
  Checkbox,
  Tooltip,
} from "@chakra-ui/react";
import { JobSet } from "./JobDispatcher";
import useResizeObserver from "../hooks/useResizeObserver";
import { useLocation } from "react-router-dom";
import LayerGallery from "./LayerAdd";
import LayerSidebar from "./LayerSidebar";
import CollageCanvas from "./CollageCanvas";
import Header from "./Header";
import DiffusionParameters from "./DiffusionParameters";
// import CollageDispatcher from "./CollageDispatcher";
import CollageDispatcher from "./ray-components/CollageDispatcher";
import LayerControl from "./LayerControl";
// import { ImageGallery } from "./ImageGallery";
import { ImageGallery } from "./ray-components/ImageGallery";
import { ParameterInput } from "./ParameterInput";
import {
  fetchWithRetryAndTimeout,
  fetchWithTimeout,
  generateId,
  killJob,
  updateUrlSearchParams,
} from "../utils/utils";
import { Layer, LayerKey } from "../types/layer";
import {
  convertLayersToRequestLayers,
  convertRequestLayersToLayers,
  RequestLayer,
} from "../types/requests";
import { CollagePromptInput } from "./CollagePromptInput";
import { TokenData } from "./Token";
import config from "../config.json";
import Joyride, {
  ACTIONS,
  CallBackProps,
  EVENTS,
  STATUS,
  Step,
} from "react-joyride";
import { useMount, useSetState } from "react-use";

/* How often to push updated collage state to the webserver */
const collagePushInterval = 5000;
const tokenIdLength = 10;

const collageIdLength = 10;
let layers: Layer[] = [];

export interface Collage {
  collagePrompt: string;
  numInferenceSteps: number;
  guidanceScale: number;
  seed: number;
  numFrames: number;
  layers: RequestLayer[];
  inputTokenData: TokenData[];
  layerToIndexMap: [string, number][];
}

export interface JoyrideState {
  run: boolean; // Whether to run the joyride
  modalOpen: boolean; // Whether the add layer modal is open
  stepIndex: number;
  steps: Step[];
}

export function CollageEditor() {
  /* If the user has selected a data collage, this will be populated
   * with the selected collage. Otherwise, it will be undefined.
   * The location object is used for state passing from the LandingPage parent
   * component, which contains an optional collage. */
  const location = useLocation();
  const [collage, setCollage] = useState<Collage | undefined>(
    location.state?.collage
  );
  /* Used for internal debugging */
  const debug = false;

  const webserverAddress = config.frontend.webserverAddress;
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const { width: canvasSize } = useResizeObserver(canvasRef);

  /* Keep track of whether data was fetched */
  const hasFetchedData = useRef(false);

  /* The collage id is either retrieved query params, which is used to fetch from the
   * server, or a new random one is generated. */
  const url = new URL(window.location.href);
  const result = url.searchParams.get("collageId");
  const initialCollageId =
    result === null ? generateId(collageIdLength) : result;
  const cloneCollageOnStart = result === null;
  let initialPrompt = "";

  const [rerender, setRerender] = useState<number>(0); // 0 means no rerender
  const [rerenderTokens, setRerenderTokens] = useState<boolean>(false); // 0 means no rerender
  const [deletedLayers, setDeletedLayers] = useState<LayerKey[]>([]);
  const [layerToIndexMap, setLayerToIndexMap] = useState<Map<LayerKey, number>>(
    new Map()
  );
  const [inputTokenData, setInputTokenData] = useState<TokenData[]>([
    { data: undefined, layerKey: undefined, id: generateId(tokenIdLength) },
  ]);

  /* Diffusion specific parameters for dispatch */
  const [selectedLayer, setSelectedLayer] = useState<Layer | null>(null);
  const [collageId, setCollageId] = useState<string>(initialCollageId);
  const [prompt, setPrompt] = useState<string>(initialPrompt);
  const [numFrames, setNumFrames] = useState<number>(5);
  const [numInferenceSteps, setNumInferenceSteps] = useState<number>(50);
  const [guidanceScale, setGuidanceScale] = useState<number>(7);
  const [seed, setSeed] = useState<number>(42);
  const [history, setHistory] = useState<JobSet[]>([]);
  const [needsAutoAdjust, setNeedsAutoAdjust] = useState<boolean>(
    collage !== undefined ? false : true
  );

  /* Joyride specific state */
  const [{ run, modalOpen, stepIndex, steps }, setJoyrideState] =
    useSetState<JoyrideState>({
      run: false,
      modalOpen: false,
      stepIndex: 0,
      steps: [],
    });

  const layerAddButtonRef = useRef<HTMLButtonElement>(null);
  const galleryModalRef = useRef<HTMLDivElement>(null);

  useMount(() => {
    setJoyrideState({
      run: run,
      steps: [
        {
          content:
            "Welcome to Collage Diffusion! You've just created a collage. If you look in the URL above, you'll see that your collage was assigned a unique collage ID. You can send this URL to anyone - and they'll be able to see and replicate your collage!",
          target: "#collage-editor",
          disableBeacon: true,
          disableOverlayClose: true,
          hideCloseButton: true,
          disableOverlay: true,
          styles: {
            options: {
              arrowColor: "transparent",
            },
          },
        },
        {
          content:
            "To begin creating a collage, we need to add a layer. Click this button to add a new layer.",
          target: "#layer-add-button",
          styles: {
            options: {
              zIndex: 10000,
            },
            buttonNext: {
              display: "none",
            },
          },
          disableOverlayClose: true,
        },
        {
          content:
            'You can add a layer in one of two ways. You can either upload your own images (png or jpeg) by clicking the "Choose File" button below or selecting one of our chosen layers. As with any diffusion model, human and animals tend not to work well.',
          target: "#gallery-modal",
          placement: "bottom",
          styles: {
            options: {
              zIndex: 10000,
              arrowColor: "transparent",
            },
          },
        },
        {
          content:
            'Let\'s add a layer from the gallery. Click the image of the tree stump. When you click it, it will prompt you to add a text description of the layer. Type "tree stump" and click "Add Layer".',
          target: "#gallery-image-0",
          styles: {
            options: {
              zIndex: 1300,
            },
            buttonNext: {
              display: "none",
            },
          },
          disableOverlay: true,
        },
        {
          content:
            "Great! You've just added your first layer. There are several per-layer controls at your disposal.",
          target: "#layer-control-0",
          // Fix the position
          styles: {
            options: {
              zIndex: 10000,
              arrowColor: "transparent",
            },
          },
        },
        {
          content: "You can move a layer forward by clicking this button.",
          target: "#layer-up-0",
        },
        {
          content: "You can move a layer backward by clicking this button.",
          target: "#layer-down-0",
        },
        {
          content:
            "You can delete a layer by clicking this button. This operation can't be undone, so be careful!",
          target: "#layer-delete-0",
        },
        {
          content: "You can duplicate a layer by clicking this button.",
          target: "#layer-dup-0",
        },
        {
          content: "You can choose to hide a layer by clicking this button.",
          target: "#layer-visibility-0",
        },
        {
          content:
            "You can edit a layer by clicking this button. Try clicking it now, and see what layer controls appear.",
          target: "#layer-edit-0",
          styles: {
            buttonNext: {
              display: "none",
            },
          },
        },
        {
          content:
            "This is the editor used to edit layer-specific parameters in Collage Diffusion. You'll notice that once you click the button, the tree stump layer in the list of layers is now highlighted in dark grey. There are several parameters that can be edited for the selected layer.",
          target: "#layer-parameter-editor",
          placement: "left",
        },
        {
          content:
            "Click this button to edit the layer mask. This is used to eliminate any background pixels if you would like to cut out the image. In our case, we don't need to edit the mask since the tree is already cut out!",
          target: "#layer-mask-button",
        },
        {
          content:
            "This slider contains the noise strength. A low value for noise strength will leave the resulting layer largely unchanged. While a very low noise strength may preserve the layer's original appearance, it gives the image generation model less room to harmonize the layer into the overall image. A high value will allow the model to change the layer more drastically, but will also make the layer lose its original appearance.",
          target: "#slider-track-layer-noise-slider-0",
        },
        {
          content:
            "This slider regulates the layer's 'shape strength'. A high value of shape strength will tell the model to maintain the edges of the layer as closely as possible. Similar to noise strength, a higher value of shape strength might better preserve the layer's apperance better, but you might want to set a moderate value for this slider if you want to give the model room to harmonize the layer's edges with the overall image.",
          target: "#slider-track-layer-shape-slider-0",
        },
        {
          content:
            "This is the negative attention strength slider. A high value of negative attention strength tells to model to not place the layer anywhere outside of where the layer is in the collage. This value should be high if a layer has contaminated other regions of the image and low to encourage additional harmonization (e.g. for a sky or grass layer).",
          target: "#slider-track-layer-negCac-slider-0",
        },
        {
          content:
            "This is the positive attention strength slider. A high value for positive attention strength will make the object more likely to appear in the specified area. A value for positive attention strength that is too high tends to make the layer appear less realistic. This slider is useful for smaller or foreground objects.",
          target: "#slider-track-layer-cac-slider-0",
        },
        {
          content:
            "These are draggable tokens to use with the global text prompt. Collage Diffusion requires that each individual layer's textual description be included in the overall text prompt that describes the entire image. You can fill in the surrounding elements with '...' in them; these are editable input areas that you can use to include information about the prompt.",
          target: "#token-1",
        },
        {
          content:
            "Once you've added a couple of more layers, you're ready to submit your job. Below are the typical parameters you'd see in a diffusion job. When you're ready, hit the \"Generate Images\" button to submit your job!",
          target: "#diffusion-parameters",
        },
      ],
    });
  });

  /* Joyride specific method. When there's a layer selected, we want to
   * give the user a tour of CD layer controls */
  useEffect(() => {
    if (selectedLayer !== null) {
      setJoyrideState({ stepIndex: 11 });
    }
  }, [selectedLayer]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleJoyrideCallback = (data: CallBackProps) => {
    const { action, index, status, type } = data;

    if (([STATUS.FINISHED, STATUS.SKIPPED] as string[]).includes(status)) {
      // Need to set our running state to false, so we can restart if we click start again.
      setJoyrideState({ run: false, stepIndex: 0 });
    } else if (
      ([EVENTS.STEP_AFTER, EVENTS.TARGET_NOT_FOUND] as string[]).includes(type)
    ) {
      const nextStepIndex = index + (action === ACTIONS.PREV ? -1 : 1);
      if ((modalOpen && stepIndex === 0) || stepIndex >= 0) {
        setJoyrideState({ stepIndex: nextStepIndex });
      }
    }
  };

  useEffect(() => {
    /* Fetches collage state */
    if (canvasSize === 0 || collage || hasFetchedData.current) return;

    updateUrlSearchParams("collageId", collageId);
    const request = {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    };
    fetchWithRetryAndTimeout(
      `${webserverAddress}/get_collage_state/${initialCollageId}`,
      request,
      1000,
      5
    )
      .then((response) => {
        if (response.ok) {
          return response.json();
        } else {
          throw new Error("Network response was not ok.");
        }
      })
      .then(async (response) => {
        // Update layers with fetched layers
        const fetchedLayers = await convertRequestLayersToLayers(
          response["layers"],
          canvasSize,
          canvasSize
        );

        if (fetchedLayers && fetchedLayers.length > 0) {
          layers.length = 0;
          layers.push(...fetchedLayers);
        }

        setPrompt(response["collage_prompt"]);
        setNumFrames(response["num_frames"]);
        setNumInferenceSteps(response["num_inference_steps"]);
        setGuidanceScale(response["guidance_scale"]);
        setSeed(response["seed"]);

        let rawInputTokenData = Array.from(response["input_token_data"]);
        let processedInputTokenData = rawInputTokenData.map((value: any) => {
          return {
            data: value.data === null ? undefined : value.data,
            layerKey: value.layerKey === null ? undefined : value.layerKey,
            id: value.id,
          };
        });

        setInputTokenData(processedInputTokenData);
        setLayerToIndexMap(
          new Map(Object.entries(response["layer_to_index_map"]))
        );
        if (cloneCollageOnStart) {
          const newCollageId = generateId(collageIdLength);
          setCollageId(newCollageId);
          updateUrlSearchParams("collageId", newCollageId);
        }
        setRerender(rerender + 1);
        hasFetchedData.current = true;
      });
  }, [canvasSize]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const fetchLayers = async () => {
      if (canvasSize === 0 || hasFetchedData.current) return;
      if (!collage) return;

      const fetchedLayers = await convertRequestLayersToLayers(
        collage["layers"],
        canvasSize,
        canvasSize
      );

      if (fetchedLayers && fetchedLayers.length > 0) {
        layers.length = 0;
        layers.push(...fetchedLayers);
      }

      let rawInputTokenData = Array.from(collage.inputTokenData);
      let processedInputTokenData = rawInputTokenData.map((value: any) => {
        return {
          data: value.data === null ? undefined : value.data,
          layerKey: value.layerKey === null ? undefined : value.layerKey,
          id: value.id,
        };
      });

      setInputTokenData(processedInputTokenData);
      setLayerToIndexMap(new Map<string, number>(collage.layerToIndexMap));

      const newCollageId = generateId(collageIdLength);
      setCollageId(newCollageId);
      updateUrlSearchParams("collageId", newCollageId);

      setPrompt(collage.collagePrompt);
      setNumFrames(collage.numFrames);
      setNumInferenceSteps(collage.numInferenceSteps);
      setGuidanceScale(collage.guidanceScale);
      setSeed(collage.seed);
      setRerender(rerender + 1);
    };
    fetchLayers();
  }, [canvasSize]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleCheckboxChange(e: React.ChangeEvent<HTMLInputElement>) {
    setNeedsAutoAdjust(e.target.checked);
  }

  function appendToHistory(jobSet: JobSet) {
    setHistory([jobSet, ...history]);
  }

  function extractDiffusionParams() {
    return new DiffusionParameters(
      prompt,
      numInferenceSteps,
      guidanceScale,
      seed
    );
  }

  function killCurrentJobs() {
    const jobSet = history[0];
    if (jobSet !== undefined) {
      jobSet.forEach((job) => {
        killJob(job.jobId, webserverAddress);
      });
    }
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
          canvasHeight={canvasSize}
          canvasWidth={canvasSize}
        />
      );
    } else {
      return null;
    }
  }

  function getImageSizeFromDataURL(
    dataURL: string
  ): Promise<{ width: number; height: number }> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.src = dataURL;
      img.onload = () => {
        resolve({ width: img.width, height: img.height });
      };
      img.onerror = (error) => {
        reject(error);
      };
    });
  }

  async function tuneDiffusionParams() {
    for (let idx = 0; idx < layers.length; idx++) {
      let layer: Layer = layers[idx];
      const imageSize = await getImageSizeFromDataURL(layer.currentImgUrl);
      const sizeRatio = Math.max(
        imageSize.width * layer.transform.scale,
        imageSize.height * layer.transform.scale
      );

      // Adjust the sizeRatio to be between 0 and 1 (smaller object to larger object)
      const adjustedSizeRatio = Math.min(Math.max(sizeRatio, 0), 1);

      layer.cacStrength = 0.2 + 0.4 * (1 - adjustedSizeRatio);
      layer.negativeStrength = 0.9 - 0.5 * adjustedSizeRatio;
      layer.noiseStrength =
        idx === layers.length - 1
          ? Math.max(0.4, 0.4 - 0.1 * adjustedSizeRatio)
          : 0.5 + 0.2 * adjustedSizeRatio;
      layer.cannyStrength = 0.3 + 0.6 * (1 - adjustedSizeRatio);
    }
  }

  function pushCollageState() {
    // Don't push any state if layers is empty
    if (layers.length === 0) {
      return;
    }

    // Read transform from canvas
    let requestLayers: RequestLayer[] = convertLayersToRequestLayers(
      layers,
      canvasSize,
      canvasSize
    );

    const request = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        collage_id: collageId,
        collage_prompt: prompt,
        num_inference_steps: numInferenceSteps,
        guidance_scale: guidanceScale,
        seed: seed,
        num_frames: numFrames,
        layers: requestLayers,
        input_token_data: inputTokenData,
        layer_to_index_map: Array.from(layerToIndexMap.entries()),
      }),
    };

    fetchWithTimeout(
      `${webserverAddress}/push_collage_state`,
      request,
      collagePushInterval / 2
    )
      .then((response) => {
        if (response.ok) {
          return response.json();
        } else {
          throw new Error("Network response was not ok.");
        }
      })
      .then((response) => {
        return response.collage_id;
      });
  }

  /* Sets timer interval to periodically propagate canvas information
   * to the webserver */
  useEffect(() => {
    const interval = setInterval(pushCollageState, collagePushInterval);
    return () => clearInterval(interval);
  });

  function handleParamInput(paramName: string, paramValue: string | number) {
    if (paramName === "prompt") setPrompt(paramValue as string);
    else if (paramName === "numFrames") setNumFrames(paramValue as number);
    else if (paramName === "numInferenceSteps")
      setNumInferenceSteps(paramValue as number);
    else if (paramName === "guidanceScale")
      setGuidanceScale(paramValue as number);
    else if (paramName === "seed") setSeed(paramValue as number);
    else console.log(`Invalid parameter name: ${paramName}`);
  }

  return (
    <>
      <Joyride
        callback={handleJoyrideCallback}
        run={run}
        spotlightClicks={true}
        continuous={true}
        stepIndex={stepIndex}
        steps={steps}
        hideBackButton={true}
      />
      <Header setJoyrideState={setJoyrideState} run={run} />
      <Box id="collage-editor" mx={8} my={5}>
        <Flex>
          <Box verticalAlign={"top"} ref={canvasRef} width="50%">
            <CollageCanvas
              key={collageId}
              layers={layers}
              rerender={rerender}
              setRerender={setRerender}
              canvasHeight={canvasSize}
              canvasWidth={canvasSize}
            />
          </Box>

          {/* Right Side */}
          <Flex ml={10} direction="column" width="50%">
            <Box width="100%">
              <Flex mb={4} justifyContent="space-between">
                <LayerGallery
                  layers={layers}
                  layerAddButtonRef={layerAddButtonRef}
                  galleryModalRef={galleryModalRef}
                  setJoyrideState={setJoyrideState}
                  rerender={rerender}
                  setRerender={setRerender}
                  setRerenderTokens={setRerenderTokens}
                  webserverAddress={webserverAddress}
                  canvasSize={canvasSize}
                />
                <Box>
                  <Button
                    size="sm"
                    onClick={() => {
                      if (
                        window.confirm(
                          "Are you sure you want to create a new collage?"
                        )
                      ) {
                        const collageId = generateId(collageIdLength);
                        // Clear layers
                        setDeletedLayers(
                          layers.map((layer: Layer) => {
                            return layer.key;
                          })
                        );
                        setSelectedLayer(null);
                        setInputTokenData([
                          {
                            data: undefined,
                            layerKey: undefined,
                            id: generateId(tokenIdLength),
                          },
                        ]);
                        layers = [];
                        setCollage(undefined);
                        setNeedsAutoAdjust(true);
                        setCollageId(collageId);
                        setRerender(rerender + 1);
                        updateUrlSearchParams("collageId", collageId);
                      }
                    }}
                    mr={4}
                  >
                    New Collage
                  </Button>
                  <Button
                    size="sm"
                    onClick={() => {
                      const collageId = generateId(collageIdLength);
                      setCollageId(collageId);
                      updateUrlSearchParams("collageId", collageId);
                    }}
                    variant="secondary"
                  >
                    Duplicate Collage
                  </Button>
                </Box>
              </Flex>
            </Box>

            <Divider borderColor="gray.600" />
            <Flex flexGrow={1} direction="column">
              <Box
                flex={1}
                style={
                  selectedLayer === null
                    ? {
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        height: "100%",
                      }
                    : {}
                }
              >
                {selectedLayer === null ? (
                  <Text>
                    {" "}
                    No layer selected. Click a layer's edit button to see layer
                    parameters.{" "}
                  </Text>
                ) : (
                  <LayerControl
                    layer={selectedLayer}
                    rerender={rerender}
                    setRerender={setRerender}
                  />
                )}
              </Box>
              <Box flex={1}>
                <LayerSidebar
                  layers={layers}
                  run={run}
                  setJoyrideState={setJoyrideState}
                  rerender={rerender}
                  setRerender={setRerender}
                  setRerenderTokens={setRerenderTokens}
                  setDeletedLayers={(key: LayerKey) => {
                    let newlyDeletedLayers = [...deletedLayers];
                    newlyDeletedLayers.push(key);
                    setDeletedLayers(newlyDeletedLayers);
                  }}
                  selectedLayer={selectedLayer}
                  setSelectedLayer={setSelectedLayer}
                />
              </Box>
            </Flex>
            {/* Collage params */}
            <Box verticalAlign={"top"} width="95%">
              <HStack id="diffusion-parameters" spacing="42px">
                <ParameterInput
                  htmlSize={5}
                  paramName="numFrames"
                  type="number"
                  size="sm"
                  description="Total images:"
                  value={numFrames}
                  placeholder={numFrames}
                  onChange={handleParamInput}
                />
                <ParameterInput
                  paramName="numInferenceSteps"
                  htmlSize={5}
                  size="sm"
                  type="number"
                  description="Inference steps:"
                  value={numInferenceSteps}
                  placeholder={numInferenceSteps}
                  onChange={handleParamInput}
                />
                <ParameterInput
                  paramName="guidanceScale"
                  htmlSize={5}
                  size="sm"
                  type="number"
                  description="Guidance scale:"
                  value={guidanceScale}
                  placeholder={guidanceScale}
                  onChange={handleParamInput}
                />
                <ParameterInput
                  paramName="seed"
                  size="sm"
                  type="number"
                  description="Initial seed:"
                  value={seed}
                  placeholder={seed}
                  onChange={handleParamInput}
                />
              </HStack>
              <Flex my={2} justifyContent={"space-between"}>
                {debug && (
                  <Button
                    mr={3}
                    onClick={() => {
                      console.log(
                        JSON.stringify({
                          collagePrompt: prompt,
                          numInferenceSteps: numInferenceSteps,
                          guidanceScale: guidanceScale,
                          seed: seed,
                          numFrames: numFrames,
                          layers: convertLayersToRequestLayers(
                            layers,
                            canvasSize,
                            canvasSize
                          ),
                          inputTokenData: inputTokenData,
                          layerToIndexMap: Array.from(
                            layerToIndexMap.entries()
                          ),
                        })
                      );
                    }}
                  >
                    Log layers
                  </Button>
                )}
                <Tooltip
                  hasArrow
                  label="Check this box to automatically adjust the sliders. If you have made manual adjustments, uncheck this box to prevent them from being overwritten."
                  bg="gray.300"
                  color="black"
                  openDelay={500}
                >
                  <Box>
                    <Checkbox
                      isChecked={needsAutoAdjust}
                      onChange={handleCheckboxChange}
                    >
                      Auto-adjust layer parameters
                    </Checkbox>
                  </Box>
                </Tooltip>
                <CollageDispatcher
                  webserverAddress={webserverAddress}
                  appendToHistory={appendToHistory}
                  layers={layers}
                  needsAutoAdjust={needsAutoAdjust}
                  tuneDiffusionParams={tuneDiffusionParams}
                  setRerender={setRerender}
                  rerender={rerender}
                  diffusionParams={extractDiffusionParams()}
                  numFrames={numFrames}
                  canvasHeight={canvasSize}
                  canvasWidth={canvasSize}
                  killCurrentJobs={killCurrentJobs}
                />
              </Flex>
            </Box>
          </Flex>
        </Flex>
        <Text fontSize="md" mt={10} mb={4}>
          {prompt}
        </Text>
        <CollagePromptInput
          layers={layers}
          data={inputTokenData}
          deletedLayers={deletedLayers}
          rerenderTokens={rerenderTokens}
          layerToIndexMap={layerToIndexMap}
          setData={setInputTokenData}
          setLayerToIndexMap={setLayerToIndexMap}
          setPrompt={setPrompt}
          setRerenderTokens={setRerenderTokens}
          setDeletedLayers={setDeletedLayers}
          tokenIdLength={tokenIdLength}
        />

        {renderImageGallery(history.length > 0)}
      </Box>
    </>
  );
}
