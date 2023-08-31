import React, { Component, ReactElement } from "react";
import {
  SimpleGrid,
  Box,
  Text,
  Stack,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
} from "@chakra-ui/react";
import { FiPlus, FiMinus } from "react-icons/fi";
import { JobSet, Job, JobId } from "../JobDispatcher";
import DiffusionImage from "../DiffusionImage";
import { Layer } from "../../types/layer";

const MAX_EMPTY_POLLS = 100;

type ImageContainerProps = {
  setSelectedIdx: () => void;
  selected: boolean;
  isModalOpen: boolean;
  toggleModal: () => void;
  webserverAddress: string;
  imgUrl: string;
  key: string;
  job: Job;
  latestStep: number;
  layers: Layer[];
  canvasHeight: number;
  canvasWidth: number;
};

type ImageContainerState = {
  imgStep: number;
  showBorder: boolean;
  mouseover: boolean;
  directive: string;
};

class ImageContainer extends Component<
  ImageContainerProps,
  ImageContainerState
> {
  constructor(props: ImageContainerProps) {
    super(props);
    this.state = {
      imgStep: 0,
      showBorder: false,
      mouseover: false,
      directive: "none",
    };
  }

  submitDirective(directive: string) {
    var newDiffusionParams = structuredClone(this.props.job.diffusionParams);

    // Preparing directive request
    const directive_request = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        job_id: this.props.job.jobId,
        directive: directive,
      }),
    };

    return fetch(`${this.props.webserverAddress}/directive`, directive_request)
      .then((response) => {
        if (response.ok) {
          return response.json();
        } else {
          throw new Error("Network response was not ok.");
        }
      })
      .then((response) => {
        // Directive specific behaviors
        if (directive === "finish" && this.state.directive !== "finish") {
          this.setState({
            directive: "finish",
          });

          // Preparing generate request
          const generate_request = {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(newDiffusionParams),
          };

          // Retrieve job
          fetch(`${this.props.webserverAddress}/generate`, generate_request)
            .then((response) => {
              if (response.ok) {
                return response.json();
              } else {
                this.setState({
                  directive: "none",
                });
                throw new Error("Network response was not ok.");
              }
            })
            .then((response) => {
              this.setState({
                directive: "finish",
              });
              return response.job_id;
            });
        } else if (directive === "cancel") {
          this.setState({
            directive: "cancel",
          });
        }
      });
  }

  render() {
    return (
      <div className="img-container" key={this.props.job.jobId}>
        <DiffusionImage
          selected={this.props.selected}
          setSelectedIdx={this.props.setSelectedIdx}
          isModalOpen={this.props.isModalOpen}
          toggleModal={this.props.toggleModal}
          imageSrc={this.props.imgUrl}
          id={this.props.job.jobId}
        />
      </div>
    );
  }
}

type ImageGalleryProps = {
  key: string;
  webserverAddress: string;
  jobSet: JobSet;
  layers: Layer[];
  canvasHeight: number;
  canvasWidth: number;
};

type ImageGalleryState = {
  numCols: number;
  pollStats: number[];
  pollIntervalId?: number | undefined | NodeJS.Timer; // TODO figure out wtf this is...
  stillPolling: Map<JobId, number>;
  imgSrcs: Map<JobId, string>;
  numEmptyPolls: number;
  selectedIdx: number;
  isModalOpen: boolean;
};

class ImageGallery extends Component<ImageGalleryProps, ImageGalleryState> {
  constructor(props: ImageGalleryProps) {
    super(props);

    var pollingMap = new Map<JobId, number>();
    var imgSrcsMap = new Map<JobId, string>();
    this.props.jobSet.forEach((job) => pollingMap.set(job.jobId, -1));
    this.props.jobSet.forEach((job) =>
      imgSrcsMap.set(job.jobId, "https://via.placeholder.com/512")
    );

    this.state = {
      numCols: 4,
      pollStats: new Array(this.props.jobSet.length).fill(0),
      pollIntervalId: undefined,
      stillPolling: pollingMap,
      imgSrcs: imgSrcsMap,
      numEmptyPolls: 0,
      selectedIdx: 0,
      isModalOpen: false,
    };

    this.pollImages = this.pollImages.bind(this);
    this.handleKeyDown = this.handleKeyDown.bind(this);
    this.toggleModal = this.toggleModal.bind(this);
    this.setSelectedIdx = this.setSelectedIdx.bind(this);
  }

  componentDidMount() {
    const pollIntervalId = setInterval(
      this.pollImages,
      1000 // hardcoded poll interval for now...
    );
    window.addEventListener("keydown", this.handleKeyDown);
    this.setState({ pollIntervalId: pollIntervalId });
  }

  componentWillUnmount() {
    window.removeEventListener("keydown", this.handleKeyDown);
    clearInterval(this.state.pollIntervalId);
  }

  handleKeyDown = (event: KeyboardEvent) => {
    if (this.state.isModalOpen) {
      if (event.key === "ArrowLeft") {
        this.setState((prevState) => ({
          selectedIdx:
            (prevState.selectedIdx - 1 + this.props.jobSet.length) %
            this.props.jobSet.length,
        }));
      } else if (event.key === "ArrowRight") {
        this.setState((prevState) => ({
          selectedIdx: (prevState.selectedIdx + 1) % this.props.jobSet.length,
        }));
      }
    }
  };

  pollImages() {
    // Guard because on startup this will be empty!
    if (this.props.jobSet) {
      // Clearing poll interval
      if (this.state.stillPolling.size === 0) {
        clearInterval(this.state.pollIntervalId);
      }

      // numInferenceSteps should be homogenous accross the batch!
      const numInferenceSteps =
        this.props.jobSet[0].diffusionParams.numInferenceSteps;
      const jobIdsToPoll = Array.from(this.state.stillPolling.keys());
      const pollRequest = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          job_ids: jobIdsToPoll,
        }),
      };

      return fetch(`${this.props.webserverAddress}/poll`, pollRequest)
        .then((response) => {
          if (response.ok) {
            return response.json();
          } else {
            throw new Error("Network response was not ok.");
          }
        })
        .then((response) => {
          const jobIds: JobId[] = response["job_ids"];
          const latestSteps: number[] = response["latest_steps"];
          const updatedJobs: Map<JobId, number> = new Map();
          var updatedImgSrcs: Map<JobId, string> = new Map(this.state.imgSrcs);
          var updatedStillPolling: Map<JobId, number> = new Map(
            this.state.stillPolling
          );

          for (var i = 0; i < jobIds.length; i++) {
            // Telemetry
            const numNewSteps =
              latestSteps[i] - this.state.stillPolling.get(jobIds[i])!;

            if (numNewSteps > 0) {
              updatedJobs.set(jobIds[i], latestSteps[i]);
            }

            // Check if any jobs are done polling
            if (latestSteps[i] >= numInferenceSteps) {
              updatedStillPolling.delete(jobIds[i]);
            }
          }

          // (async) fetch new images for updated jobs
          var promises: Promise<string>[] = [];
          var updatedJobIds = Array.from(updatedJobs.keys());
          updatedJobs.forEach((latestStep, jobId) => {
            promises.push(this.fetchImageUrl(jobId, latestStep));
          });

          // Once we've fetched all images, set state
          Promise.all(promises)
            .then((imgUrls) => {
              for (var i = 0; i < imgUrls.length; i++) {
                updatedImgSrcs.set(updatedJobIds[i], imgUrls[i]);
              }
            })
            .then(() => {
              var newNumEmptyPolls = this.state.numEmptyPolls;
              if (updatedJobIds.length === 0) {
                newNumEmptyPolls += 1;
              } else {
                newNumEmptyPolls = 0;
              }

              if (newNumEmptyPolls > MAX_EMPTY_POLLS) {
                clearInterval(this.state.pollIntervalId);
              }

              this.setState({
                imgSrcs: updatedImgSrcs,
                stillPolling: updatedStillPolling,
                numEmptyPolls: newNumEmptyPolls,
              });
            });
        });
    }
  }

  fetchImageUrl(jobId: string, step: number) {
    // Construct request
    const request = {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    };

    return fetch(
      `${this.props.webserverAddress}/dreams/${jobId}/${step}.png`,
      request
    )
      .then((response) => {
        if (response.ok) {
          return response.blob();
        } else {
          throw new Error("Network response was not ok.");
        }
      })
      .then((blob) => {
        return URL.createObjectURL(blob);
      });
  }

  toggleModal = () => {
    this.setState((prevState) => ({
      isModalOpen: !prevState.isModalOpen,
    }));
  };

  setSelectedIdx = (idx: number) => {
    this.setState({
      selectedIdx: idx,
    });
  };

  render() {
    var gallery: ReactElement[] = [];
    // Loop through all distinct images and display them in the gallery
    for (let i = 0; i < this.props.jobSet.length; i++) {
      const currentJob = this.props.jobSet[i];
      var imgContainer = (
        <ImageContainer
          selected={i === this.state.selectedIdx}
          setSelectedIdx={() => this.setSelectedIdx(i)}
          toggleModal={this.toggleModal}
          isModalOpen={this.state.isModalOpen}
          webserverAddress={this.props.webserverAddress}
          imgUrl={this.state.imgSrcs.get(currentJob.jobId)!}
          key={`${currentJob.jobId}`}
          job={currentJob}
          latestStep={this.state.stillPolling.get(currentJob.jobId)!}
          layers={this.props.layers}
          canvasHeight={this.props.canvasHeight}
          canvasWidth={this.props.canvasWidth}
        />
      );
      gallery.push(imgContainer);
    }

    // Parse Job object corresponding to current image gallery
    const prompt = this.props.jobSet[0].diffusionParams.prompt;

    // Generate URL to replicate query
    var requestParams = Object(this.props.jobSet[0].diffusionParams);
    requestParams.numFrames = this.props.jobSet.length;

    return (
      <Box pt={10}>
        <Text fontSize="lg">{prompt}</Text>
        <Text fontSize="10px" color="gray.600">
          {window.location.href}
        </Text>
        <Stack mt={10} direction="row">
          {/* <Text color="white">{`Thumbnail size: ${this.state.numCols}`}</Text> */}
          <FiMinus />
          <Slider
            key={this.props.jobSet[0].jobId}
            id="slider"
            min={1}
            max={20}
            colorScheme="blackAlpha"
            width="20%"
            value={21 - this.state.numCols}
            onChange={(v) => {
              this.setState({ numCols: 21 - v });
            }}
          >
            <SliderTrack>
              <SliderFilledTrack />
            </SliderTrack>
            <SliderThumb />
          </Slider>
          <FiPlus />
        </Stack>
        <SimpleGrid mt={5} columns={this.state.numCols} spacing={5}>
          {gallery}
        </SimpleGrid>
      </Box>
    );
  }
}

export { ImageGallery };
