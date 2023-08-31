import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import LandingPage from "./components/Landing";
import { CollageEditor } from "./components/CollageEditor";
import { ChakraProvider } from "@chakra-ui/react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { extendTheme } from "@chakra-ui/react";
import collages from "./collage-examples.json";
import { Collage } from "./components/CollageEditor";
import ErrorComponent from "./components/ErrorComponent";
import "typeface-lato";

window.global = window;

const theme = extendTheme({
  fonts: {
    heading: "Lato",
  },
  shadows: {
    custom: "0 0 20px 5px rgba(0, 0, 0, 0.7)", // Customize the shadow color and blur here
  },
  components: {
    Button: {
      baseStyle: {
        // Set the base background color to #757575
        backgroundColor: "#bdbdbd",
        fontWeight: "semibold",
      },
      variants: {
        base: {},
        secondary: {
          // Set the secondary background color to #bdbdbd
          backgroundColor: "#C5C5C5",
        },
        tertiary: {
          // Set the tertiary background color to #e5e7eb
          backgroundColor: "eeeeee",
        },
      },
      defaultProps: {
        // Set the default variant to 'base'
        variant: "base",
      },
    },
    Divider: {
      baseStyle: {
        borderColor: "#757575",
      },
    },
    ModalContent: {
      defaultProps: {
        bg: "#424242",
      },
    },
    Input: {
      variants: {
        flushed: {
          field: {
            borderBottom: "2px solid",
            borderBottomColor: "#bdbdbd",
            _hover: {
              borderColor: "black",
            },
            _focus: {
              borderColor: "black",
              boxShadow: "none",
            },
          },
        },
      },
      defaultProps: {
        variant: "flushed",
      },
    },
  },
});

const defaultCollages = collages as {
  collage: Collage;
  collageImage: string;
  generatedImage: string;
}[];

const router = createBrowserRouter([
  {
    path: "/",
    element: <LandingPage collages={defaultCollages} />,
    errorElement: <ErrorComponent />,
  },
  {
    path: "/edit",
    element: <CollageEditor />,
    errorElement: <ErrorComponent />,
  },
]);

const root = ReactDOM.createRoot(document.getElementById("root")!);
root.render(
  <ChakraProvider theme={theme}>
    <RouterProvider router={router} />
  </ChakraProvider>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
// reportWebVitals();
