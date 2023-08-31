import React, { useState, useEffect, useRef } from "react";
import {
  Box,
  Button,
  Center,
  SimpleGrid,
  Heading,
  Link,
  Text,
} from "@chakra-ui/react";
import CollagePreview from "./CollagePreview";
import { Collage } from "./CollageEditor";
import { useNavigate } from "react-router-dom";
import Joyride, { Step } from "react-joyride";

interface LandingPageProps {
  collages: {
    collage: Collage;
    collageImage: string;
    generatedImage: string;
  }[];
}

function LandingPage({ collages }: LandingPageProps) {
  const [run, setRun] = useState(false);
  const headingRef = useRef<HTMLDivElement>(null);
  const newCollageRef = useRef<HTMLDivElement>(null);
  const steps: Step[] = [
    {
      target: headingRef.current!,
      content: "Welcome to Collage Diffusion!",
    },
    {
      target: newCollageRef.current!,
      content: "Click here to create a new collage from scratch.",
    },
  ];

  /* Start the tour on component mount. */
  useEffect(() => {
    setRun(true);
  }, []);

  const navigate = useNavigate();
  return (
    <>
      <Joyride
        steps={steps}
        spotlightClicks={true}
        run={run}
        continuous={true}
      />
      <Box py={10} mx={16}>
        <Center>
          <Heading ref={headingRef} size="lg" mb={8}>
            Collage Diffusion
          </Heading>
        </Center>
        <Box mb={8}>
          <Text>
            Collage Diffusion allows you to go beyond typing in text prompts
            when making AI-generated images. Put together whatever layers you
            want just like in Photoshop—without worrying about the details like
            lighting and perspective—and let the AI give you a well-harmonized
            image. Understand more by reading our{" "}
            <Link color="blue.500" href="https://arxiv.org/abs/2303.00262">
              paper.
            </Link>{" "}
          </Text>
          <br></br>
          <Text>
            Tuning the parameters to get desirable images requires some
            knowledge of the parameters, so we've made a{" "}
            <Link
              href="https://www.youtube.com/watch?v=bg8FatdUUdU"
              color="blue.500"
            >
              tutorial
            </Link>{" "}
            to help you get started.
          </Text>
          <br></br>
          <Text>
            Click one of the examples below to get started or start a new
            collage.
          </Text>
        </Box>
        <Center>
          <Box mb={4} ref={newCollageRef}>
            <Button borderRadius="full" onClick={() => navigate("/edit")}>
              Start a new collage from scratch
            </Button>
          </Box>
        </Center>
        <SimpleGrid m={2} columns={2} spacing={8}>
          {collages.map((collage) => (
            <Box w="100%" h="auto">
              <CollagePreview
                collage={collage.collage}
                collageImage={collage.collageImage}
                generatedImage={collage.generatedImage}
                collagePrompt={collage.collage.collagePrompt}
              />
            </Box>
          ))}
        </SimpleGrid>
      </Box>
    </>
  );
}

export default LandingPage;
