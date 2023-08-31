import pRetry from "p-retry";
import { JobId } from "../components/JobDispatcher";

// dec2hex :: Integer -> String
// i.e. 0-255 -> '00'-'ff'
function dec2hex(dec: number) {
  return dec.toString(16).padStart(2, "0");
}

// generateId :: Integer -> String
export function generateId(len: number) {
  var arr = new Uint8Array((len || 40) / 2);
  window.crypto.getRandomValues(arr);
  return Array.from(arr, dec2hex).join("");
}

export function parseUrlParamValue(
  urlParams: URLSearchParams,
  paramName: string,
  defaultVal: string
) {
  const result = urlParams.get(paramName);
  if (result != null) {
    return result;
  } else {
    return defaultVal;
  }
}

export function updateUrlSearchParams(key: string, value: string) {
  var url = new URL(window.location.href);
  var search_params = url.searchParams;

  search_params.set(key, value);
  url.search = search_params.toString();

  var new_url = url.toString();
  window.history.pushState(null, "", new_url);
}

export async function fetchWithTimeout(
  resource: string,
  options: any,
  timeout: number = 3000
) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  const response = await fetch(resource, {
    ...options,
    signal: controller.signal,
  });
  clearTimeout(id);

  // Abort retrying if the resource doesn't exist
  if (response.status === 404) {
    throw new pRetry.AbortError(response.statusText);
  }

  return response;
}

export async function fetchWithRetryAndTimeout(
  url: string,
  options: any,
  timeout: number = 3000,
  numRetries: number = 5
) {
  return await pRetry(
    () => {
      return fetchWithTimeout(url, options, timeout);
    },
    { retries: numRetries }
  );
}

export function killJob(jobId: JobId, webserverAddress: string) {
  // Preparing directive request
  const directive_request = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      job_id: jobId,
      directive: "cancel",
    }),
  };

  console.log(`Sending kill request for ${jobId}`);

  return fetch(`${webserverAddress}/directive`, directive_request)
    .then((response) => {
      if (response.ok) {
        return response.json();
      } else {
        throw new Error("Network response was not ok.");
      }
    })
    .then((response) => {
      console.log(
        "Received /directive POST response: " + JSON.stringify(response)
      );
    });
}

// https://stackoverflow.com/questions/16245767/creating-a-blob-from-a-base64-string-in-javascript
function b64toBlob(b64Data: string, contentType = "", sliceSize = 512) {
  const byteCharacters = atob(b64Data);
  const byteArrays = [];

  for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
    const slice = byteCharacters.slice(offset, offset + sliceSize);

    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }

    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }

  const blob = new Blob(byteArrays, { type: contentType });
  return blob;
}

export function b64ToUrl(b64Data: string, contentType = "", sliceSize = 512) {
  return URL.createObjectURL(b64toBlob(b64Data, contentType, sliceSize));
}
