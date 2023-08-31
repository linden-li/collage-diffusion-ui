class DiffusionParameters {
  constructor(
    public prompt: string,
    public numInferenceSteps: number,
    public guidanceScale: number,
    public seed: number
  ) {}
}

export default DiffusionParameters;
