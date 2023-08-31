import React, { Component } from "react";
import { Text, Input, Flex } from "@chakra-ui/react";

type ParameterInputProps = {
  paramName: string;
  type: string;
  description?: string;
  value: string | number;
  placeholder: string | number;
  onChange: (paramName: string, paramValue: string) => void;
  htmlSize?: number;
  size?: string;
};

class ParameterInput extends Component<ParameterInputProps> {
  constructor(props: ParameterInputProps) {
    super(props);
    this.handleInput = this.handleInput.bind(this);
  }

  // Trigger callback passed down from parent to update parent state
  handleInput(e: { target: { value: string } }) {
    this.props.onChange(this.props.paramName, e.target.value);
  }

  render() {
    const placeholder =
      typeof this.props.placeholder === "string"
        ? this.props.placeholder
        : this.props.placeholder.toString();

    return (
      <Flex my={5} flexDirection={"column"}>
        <Text fontSize="sm">
          {this.props.description}
        </Text>
        <Input
          type={this.props.type}
          id={this.props.paramName}
          value={this.props.value}
          placeholder={placeholder}
          onChange={this.handleInput}
          htmlSize={this.props.htmlSize}
          size={this.props.size}
        />
      </Flex>
    );
  }
}

export { ParameterInput };
