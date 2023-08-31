import React from "react";
import { Collage } from "./CollageEditor";
import { Flex, AspectRatio, Image, Box, Text } from "@chakra-ui/react";
import { useNavigate } from "react-router-dom";
import { FiArrowRight } from "react-icons/fi";

interface CollagePreviewProps {
  collage: Collage;
  collageImage: string;
  generatedImage: string;
  collagePrompt: string;
}

export default function CollagePreview({
  collage,
  collageImage,
  generatedImage,
  collagePrompt,
}: CollagePreviewProps) {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate("/edit", { state: { collage } });
  };

  return (
    <Flex
      position="relative" // Ensures child elements can be positioned relative to this Flex
      direction="row"
      align="center"
      justify="center"
      height="100%" // Ensure the component takes up the full height of its container
      onClick={handleClick}
      _hover={{
        "::before": {
          content: '""',
          display: "block",
          position: "absolute",
          top: 0,
          right: 0,
          bottom: 0,
          left: 0,
          backgroundColor: "rgba(0, 0, 0, 0.7)",
          zIndex: 1,
        },
        cursor: "pointer",
      }} // Blackout effect
      role="group"
      transition="backgroundColor 0.2s"
    >
      <Box
        position="absolute"
        zIndex={2} // Ensure it's above the blackout layer
        width="100%"
        textAlign="center"
        color="white"
        opacity={0}
        _groupHover={{ opacity: 1 }} // Show on hover
        display="flex"
        justifyContent="center"
        alignItems="center"
        height="100%"
        px={3}
      >
        <Text>{collagePrompt}</Text>
      </Box>

      <AspectRatio ratio={1} width="50%">
        <Image src={collageImage} alt="Collage Image" objectFit="cover" />
      </AspectRatio>
      <Box px={2} fontSize="2xl">
        {" "}
        🪄
      </Box>

      <AspectRatio ratio={1} width="50%">
        <Image src={generatedImage} alt="Generated Image" objectFit="cover" />
      </AspectRatio>
    </Flex>
  );
}
