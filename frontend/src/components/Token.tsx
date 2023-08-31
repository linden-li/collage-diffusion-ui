// Code adapted from: https://github.com/trananhtuat/react-draggable-list/blob/main/src/components/list/DraggableListItem.jsx

import React, { useRef, DragEvent } from "react";

import { Box, Input, Text } from "@chakra-ui/react";
import { LayerKey } from "../types/layer";

export type TokenData = {
  data: string | undefined;
  layerKey: LayerKey | undefined;
  id: string;
};

type TokenProps = {
  id: string;
  index: number;
  type: any;
  draggable: boolean;
  onDragStart: ((index: number) => void) | undefined;
  onDrop: (index: number) => void;
  value: string | undefined;
  layerKey: LayerKey | undefined;
  setToken: (token: TokenData) => void;
  maxW: number;
  //   renderFunction: () => any;
};

export function Token(props: TokenProps) {
  const itemRef = useRef<HTMLInputElement>(null);

  const onDragStart = (e: DragEvent<HTMLDivElement>) => {
    // remove default drag ghost
    e.dataTransfer.effectAllowed = "move";
    // e.dataTransfer.setDragImage(e.target, 50000, 50000);

    // custom drag ghost
    // let ghostNode = e.target.cloneNode(true);

    // ghostNode.style.position = "absolute";

    // // show ghost add mouse pointer position
    // ghostNode.style.top = e.pageY - e.target.offsetHeight / 2 + "px";
    // ghostNode.style.left = e.pageX - e.target.offsetWidth / 2 + "px";

    // // add width height to ghost node
    // ghostNode.style.height = e.target.offsetHeight + "px";
    // ghostNode.style.width = e.target.offsetWidth + "px";

    // // add some style
    // ghostNode.style.opacity = "0.8";
    // ghostNode.style.pointerEvents = "none";

    // // add id
    // ghostNode.id = "ghostNode";

    // document.body.prepend(ghostNode);

    // identify selected item
    if (itemRef.current) {
      itemRef.current.classList.add("dragstart");
    }

    if (props.onDragStart) {
      props.onDragStart(props.index);
    }
  };

  // event when dragging
  const onDrag = () => {
    // move ghost node with mouse
    // if (e) {
    //     let ghostNode = document.querySelector("#ghostNode");
    //     ghostNode.style.top = e.pageY - e.target.offsetHeight / 2 + "px";
    //     ghostNode.style.left = e.pageX - e.target.offsetWidth / 2 + "px";
    // }
  };

  // event when drag end
  const onDragEnd = () => {
    // remove ghost node
    // document.querySelector("#ghostNode").remove();
    // remove selected item style
    if (itemRef.current) {
      itemRef.current.classList.remove("dragstart");
    }
  };

  // event when drag over item
  const onDragEnter = () => {
    if (itemRef.current) {
      itemRef.current.classList.add("dragover");
    }
  };

  // event when drag leave item
  const onDragLeave = () => {
    if (itemRef.current) {
      itemRef.current.classList.remove("dragover");
    }
  };
  // add event for item can drop
  const onDragOver = (e: any) => e.preventDefault();

  // event when drop
  const onDrop = () => {
    if (itemRef.current) {
      itemRef.current.classList.remove("dragover");
      props.onDrop(props.index);
    }
  };

  return (
    <Box
      id={`token-${props.index}`}
      key={props.id}
      as={props.type}
      h="50px"
      maxW={props.type === Input ? "100px" : `${props.maxW}%`}
      p="2"
      borderRadius="lg"
      white-space="normal"
      borderWidth={props.type === Input ? 0 : "1px"}
      backgroundColor={props.type === Input ? undefined : "#bdbdbd"}
      ref={itemRef}
      className="collage-prompt-input__item"
      draggable={props.draggable}
      onDragStart={onDragStart}
      onDrag={onDrag}
      onDragEnd={onDragEnd}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
      value={props.value}
      placeholder={"..."}
      display="inline-flex"
      onChange={(e: any) => {
        props.setToken({
          data: e.target.value,
          layerKey: props.layerKey,
          id: props.id,
        });
      }}
    >
      {props.type === Input ? undefined : (
        <Text fontSize="xs" as="em" noOfLines={1}>
          {props.value}
        </Text>
      )}
    </Box>
  );
}
