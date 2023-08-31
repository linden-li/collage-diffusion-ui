import React from "react";
import {
  Modal,
  ModalOverlay,
  ModalContent,
  useDisclosure,
  Image,
  IconButton,
  Flex,
} from "@chakra-ui/react";
import { DownloadIcon } from "@chakra-ui/icons";
import { saveAs } from "file-saver";

type DiffusionImageProps = {
  selected: boolean;
  setSelectedIdx: () => void;
  toggleModal: () => void;
  isModalOpen: boolean;
  imageSrc: string;
  id: string;
};

const downloadImage = async (url: string) => {
  const response = await fetch(url);
  const blob = await response.blob();
  saveAs(blob, "image.png");
};

export default function DiffusionImage({
  selected,
  setSelectedIdx,
  isModalOpen,
  toggleModal,
  imageSrc,
  id,
}: DiffusionImageProps) {
  const { isOpen, onOpen, onClose } = useDisclosure(); // eslint-disable-line

  return (
    <>
      <Image
        src={imageSrc}
        fallbackSrc="https://via.placeholder.com/512"
        id={id}
        alt="Diffusion result"
        width="100%"
        onClick={() => {
          toggleModal();
          setSelectedIdx();
          onOpen();
        }}
      />
      <Modal
        isOpen={selected && isModalOpen}
        size={"xl"}
        onClose={() => {
          toggleModal();
          onClose();
        }}
      >
        <ModalOverlay />
        <ModalContent width={"80%"}>
          <Flex
            position="relative"
            border="1px solid black"
            boxShadow="rgba(0, 0, 0, 0.3) 0px 19px 38px, rgba(0, 0, 0, 0.22) 0px 15px 12px;"
          >
            <Image
              src={imageSrc}
              fallbackSrc="https://via.placeholder.com/512"
              alt="Diffusion result"
              width="100%"
              height="100%"
              flex={2}
            />
            <IconButton
              icon={<DownloadIcon />}
              position="absolute"
              top="5px"
              right="5px"
              onClick={() => downloadImage(imageSrc)}
              aria-label="Download image"
            />
          </Flex>
        </ModalContent>
      </Modal>
    </>
  );
}
