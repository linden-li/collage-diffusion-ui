import React from "react";
import { Flex, Link, Switch } from "@chakra-ui/react";
import { useNavigate } from "react-router-dom";
import { JoyrideState } from "./CollageEditor";

type HeaderProps = {
  setJoyrideState: (state: Partial<JoyrideState>) => void;
  run: boolean;
};

export default function Header({ setJoyrideState, run }: HeaderProps) {
  const navigate = useNavigate();
  return (
    <Flex
      bg="#c5c5c5"
      w="100%"
      mb={4}
      px={8}
      py={3}
      justifyContent="space-between"
    >
      <Link
        fontFamily="Lato"
        fontSize="xl"
        as="b"
        onClick={() => navigate("/")}
      >
        Collage Diffusion
      </Link>
      <Switch
        colorScheme="green"
        size="sm"
        isChecked={run}
        onChange={() => {
          setJoyrideState({ run: !run });
        }}
      >
        {run ? "Stop" : "Start"} Tutorial
      </Switch>
    </Flex>
  );
}
