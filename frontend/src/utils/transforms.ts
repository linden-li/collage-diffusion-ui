function canvasCoordToImageCoord(
    x: number, y: number, imageAspectRatio: number, canvasAspectRatio: number
) {
    // Canvas coord is a pixel between (0, 0) and (1, 1) on the canvas
    // Want a pixel between (0, 0) and (1, 1) relative to the image
    if (imageAspectRatio > canvasAspectRatio) {
        const imageX = x;
        const imageY = y * imageAspectRatio / canvasAspectRatio;
        return { x: imageX, y: imageY };
    } else {
        const imageX = x * canvasAspectRatio / imageAspectRatio;
        const imageY = y;
        return { x: imageX, y: imageY };
    }
}

function imageCoordToCanvasCoord(
    x: number, y: number, imageAspectRatio: number, canvasAspectRatio: number
) {
    // Image coord is a pixel between (0, 0) and (1, 1) on the image
    // Want a pixel between (0, 0) and (1, 1) relative to the canvas
    if (imageAspectRatio > canvasAspectRatio) {
        const canvasX = x;
        const canvasY = y * canvasAspectRatio / imageAspectRatio;
        return { x: canvasX, y: canvasY };
    } else {
        const canvasX = x * imageAspectRatio / canvasAspectRatio;
        const canvasY = y;
        return { x: canvasX, y: canvasY };
    }
}

function imageCoordToScaledImageCoord(
    x: number, y: number, imageWidth: number, imageHeight: number
) {
    const scaledX = x * imageWidth;
    const scaledY = y * imageHeight;
    return { x: scaledX, y: scaledY };
}

function scaledImageCoordToImageCoord(
    x: number, y: number, imageWidth: number, imageHeight: number
) {
    const imageX = x / imageWidth;
    const imageY = y / imageHeight;
    return { x: imageX, y: imageY };
}

function scaledCanvasCoordToImageCoord(
    x: number, y: number, imageAspectRatio: number, canvasAspectRatio: number,
    canvasWidth: number, canvasHeight: number
) {
    // Canvas coord is a pixel between (0, 0) and (canvasWidth, canvasHeight) on the canvas
    // Want a pixel between (0, 0) and (1, 1) relative to the image
    if (imageAspectRatio > canvasAspectRatio) {
        const imageX = x / canvasWidth;
        const imageY = y / (canvasHeight * canvasAspectRatio / imageAspectRatio)
        return { x: imageX, y: imageY };
    } else {
        const imageX = x / (canvasWidth * imageAspectRatio / canvasAspectRatio);
        const imageY = y / canvasHeight;
        return { x: imageX, y: imageY };
    }
}

function imageCoordToScaledCanvasCoord(
    x: number, y: number, imageAspectRatio: number, canvasAspectRatio: number,
    canvasWidth: number, canvasHeight: number
) {
    // Image coord is a pixel between (0, 0) and (1, 1) on the image
    // Want a pixel between (0, 0) and (canvasWidth, canvasHeight) relative to the canvas
    if (imageAspectRatio > canvasAspectRatio) {
        const canvasX = x * canvasWidth;
        const canvasY = y * (canvasHeight * canvasAspectRatio / imageAspectRatio);
        return { x: canvasX, y: canvasY };
    } else {
        const canvasX = x * (canvasWidth * imageAspectRatio / canvasAspectRatio);
        const canvasY = y * canvasHeight;
        return { x: canvasX, y: canvasY };
    }
}

function canvasCoordToScaledCanvasCoord(
    x: number, y: number, canvasWidth: number, canvasHeight: number
) {
    const scaledX = x * canvasWidth;
    const scaledY = y * canvasHeight;
    return { x: scaledX, y: scaledY };
}

function scaledCanvasCoordToCanvasCoord(
    x: number, y: number, canvasWidth: number, canvasHeight: number
) {
    const canvasX = x / canvasWidth;
    const canvasY = y / canvasHeight;
    return { x: canvasX, y: canvasY };
}

export {
    canvasCoordToImageCoord,
    imageCoordToCanvasCoord,
    imageCoordToScaledImageCoord,
    scaledImageCoordToImageCoord,
    scaledCanvasCoordToImageCoord,
    imageCoordToScaledCanvasCoord,
    canvasCoordToScaledCanvasCoord,
    scaledCanvasCoordToCanvasCoord,
};