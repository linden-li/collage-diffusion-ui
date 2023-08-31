import React, { useState, useEffect, ReactElement } from "react";
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
import { JobSet, Job } from "./JobDispatcher";
import DiffusionImage from "./DiffusionImage";
import { Layer } from "../types/layer";

type ImageContainerProps = {
  setSelectedIdx: () => void;
  selected: boolean;
  isModalOpen: boolean;
  toggleModal: () => void;
  webserverAddress: string;
  imgUrl: string;
  key: string;
  job: Job;
  layers: Layer[];
  canvasHeight: number;
  canvasWidth: number;
};

function ImageContainer({
  setSelectedIdx,
  selected,
  isModalOpen,
  toggleModal,
  webserverAddress,
  imgUrl,
  job,
  layers,
  canvasHeight,
  canvasWidth,
}: ImageContainerProps) {
  const [imgSrc, setImgSrc] = useState("https://via.placeholder.com/512");

  useEffect(() => {
    setImgSrc(imgUrl);
  }, [imgUrl]);

  return (
    <div className="img-container" key={job.jobId}>
      <DiffusionImage
        selected={selected}
        setSelectedIdx={setSelectedIdx}
        isModalOpen={isModalOpen}
        toggleModal={toggleModal}
        imageSrc={imgSrc}
        id={job.jobId}
      />
    </div>
  );
}

type ImageGalleryProps = {
  webserverAddress: string;
  jobSet: JobSet;
  layers: Layer[];
  canvasHeight: number;
  canvasWidth: number;
};

const ImageGallery: React.FC<ImageGalleryProps> = ({
  webserverAddress,
  jobSet,
  layers,
  canvasHeight,
  canvasWidth,
}) => {
  const [numCols, setNumCols] = useState(4);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (isModalOpen) {
        if (event.key === "ArrowLeft") {
          setSelectedIdx(
            (prevSelectedIdx) =>
              (prevSelectedIdx - 1 + jobSet.length) % jobSet.length
          );
        } else if (event.key === "ArrowRight") {
          setSelectedIdx(
            (prevSelectedIdx) => (prevSelectedIdx + 1) % jobSet.length
          );
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isModalOpen, jobSet.length]);

  const toggleModal = () => {
    setIsModalOpen((prevState) => !prevState);
  };

  const gallery: ReactElement[] = [];
  for (let i = 0; i < jobSet.length; i++) {
    const currentJob = jobSet[i];
    const imgContainer = (
      <ImageContainer
        selected={i === selectedIdx}
        setSelectedIdx={() => setSelectedIdx(i)}
        toggleModal={toggleModal}
        isModalOpen={isModalOpen}
        webserverAddress={webserverAddress}
        imgUrl={jobSet[i].imgUrl!}
        key={`${currentJob.jobId}`}
        job={currentJob}
        layers={layers}
        canvasHeight={canvasHeight}
        canvasWidth={canvasWidth}
      />
    );
    gallery.push(imgContainer);
  }

  const prompt = jobSet[0].diffusionParams.prompt;

  const requestParams = Object(jobSet[0].diffusionParams);
  requestParams.numFrames = jobSet.length;
  return (
    <Box pt={10}>
      <Text fontSize="lg">{prompt}</Text>
      <Text fontSize="10px" color="gray.600">
        {window.location.href}
      </Text>
      <Stack mt={10} direction="row">
        <FiMinus />
        <Slider
          key={jobSet[0].jobId}
          id="slider"
          min={1}
          max={20}
          colorScheme="blackAlpha"
          width="20%"
          value={21 - numCols}
          onChange={(v) => {
            setNumCols(21 - v);
          }}
        >
          <SliderTrack>
            <SliderFilledTrack />
          </SliderTrack>
          <SliderThumb />
        </Slider>
        <FiPlus />
      </Stack>
      <SimpleGrid mt={5} columns={numCols} spacing={5}>
        {gallery}
      </SimpleGrid>
    </Box>
  );
};

export { ImageGallery };
