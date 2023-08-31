import React, { Component } from "react";
import DiffusionParameters from "./DiffusionParameters";
import {
  Button,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
} from "@chakra-ui/react";

const MAX_FRAMES = 100;
const MAX_STEPS = 100;

class Job {
  constructor(
    public jobId: string,
    public diffusionParams: DiffusionParameters,
    public imgUrl?: string
  ) {}
}

type JobSet = Job[];
type JobId = string;

export { Job, JobSet, JobId };

type JobDispatcherProps = {
  webserverAddress: string;
  appendToHistory: (jobSet: JobSet) => void;
  diffusionParams: DiffusionParameters;
  numFrames: number;
  selected: JobSet;
  clearSelected: () => void;
};

type JobDispatcherState = {
  disabled: boolean;
  badParams: boolean;
};

class JobDispatcher extends Component<JobDispatcherProps, JobDispatcherState> {
  constructor(props: JobDispatcherProps) {
    super(props);
    this.state = {
      disabled: false,
      badParams: false,
    };
    this.startJob = this.startJob.bind(this);
    this.setDisabledState = this.setDisabledState.bind(this);
  }

  setDisabledState(disabled: boolean) {
    this.setState({ disabled: disabled });
  }

  // Callback that is ran every time the user clicks the "Start Job button"
  async startJob(generate_similar: boolean) {
    // FIXME: race when user clicks start job multiple times

    // Display loading indicator
    this.setDisabledState(true);
    let timeout = new Promise((resolve) =>
      setTimeout(() => {
        resolve(null);
      }, 5000)
    );

    // Prevent user from overloading the system
    if (
      this.props.numFrames > MAX_FRAMES ||
      this.props.diffusionParams.numInferenceSteps > MAX_STEPS
    ) {
      this.setState({ badParams: true });
      await timeout;
      this.setDisabledState(false);
      return;
    } else {
      this.setState({ badParams: false });
    }

    // Create rng seeds for images
    // TODO actually use a prng
    const initialSeed = Number(this.props.diffusionParams.seed);
    var seeds: number[] = [];
    for (let i = 0; i < this.props.numFrames; i++) {
      seeds.push(initialSeed + i);
    }

    var localDiffusionParamsArr: DiffusionParameters[] = [];
    for (let i = 0; i < this.props.numFrames; i++) {
      var localDiffusionParams = structuredClone(this.props.diffusionParams);
      localDiffusionParams.seed = seeds[i];
      localDiffusionParamsArr.push(localDiffusionParams);
    }

    // Hit sample API endpoint, which will return a list of paths to images
    Promise.allSettled(
      Array.from({ length: this.props.numFrames }, (x, i) => i).map((i) => {
        // Preparing request
        var requestBody: any = { ...localDiffusionParamsArr[i] };
        if (generate_similar) {
          requestBody["originalJobId"] = this.props.selected[0].jobId;
          this.props.clearSelected();
        }
        const request = {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        };

        if (!generate_similar) {
          return fetch(`${this.props.webserverAddress}/generate`, request)
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
        } else {
          return fetch(
            `${this.props.webserverAddress}/generate_similar`,
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
              return response.job_id;
            });
        }
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
        this.props.appendToHistory(newJobSet);
      });

    await timeout;
    this.setDisabledState(false);
  }

  render() {
    return (
      <div>
        <Button
          id="start_job"
          onClick={() => {
            this.startJob(false);
          }}
          mx={4}
          disabled={this.state.disabled}
        >
          Start Job
        </Button>
        {this.props.selected.length > 0 && (
          <Button
            id="generate_similar"
            onClick={() => {
              this.startJob(true);
            }}
            disabled={this.state.disabled}
          >
            Generate similar
          </Button>
        )}
        {this.state.badParams && (
          <Alert status="error">
            <AlertIcon />
            <AlertTitle>Bad parameters!</AlertTitle>
            <AlertDescription>
              Number of images and number of inference steps must be less than
              or equal to {MAX_FRAMES} and {MAX_STEPS}, respectively.
            </AlertDescription>
          </Alert>
        )}
      </div>
    );
  }
}

export { JobDispatcher };
