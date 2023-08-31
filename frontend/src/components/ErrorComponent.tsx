import React from "react";
import { Box, Text } from "@chakra-ui/react";

interface ErrorComponentProps {
  error?: any;
}

const ErrorComponent: React.FC<ErrorComponentProps> = ({ error }) => {
  return (
    <Box p={4}>
      <Text>
        Oops! An error occurred. You can go back to the homepage to start again.
      </Text>
      {error && (
        <details style={{ whiteSpace: "pre-wrap" }}>{error.message}</details>
      )}
    </Box>
  );
};

export default ErrorComponent;
