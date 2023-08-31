import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
from urllib.request import urlopen
import os

IMAGE_DIR_NAME = "dreams"
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, IMAGE_DIR_NAME)

class ImageLayer:
    def __init__(
        self,
        rgba: Image,
        pos: tuple,
        image_str: str,
        noise_strength: float,
        scale: float = 1.0,
        rotation: float = 0.0,
        ftc=None,
    ):
        self.rgba = rgba
        self.pos = pos
        self.noise_strength = noise_strength
        self.scale = scale
        self.rotation = rotation
        self.image_str = image_str  # text prompt
        self.ftc = ftc  # Optional

        # Apply the scale and rotation
        self.apply_transform()

    def apply_transform(self):
        """
        Apply the scale and rotation to the image
        """
        self.rgba = self.rgba.resize(
            (
                max(1, int(self.rgba.size[0] * self.scale)),
                max(1, int(self.rgba.size[1] * self.scale)),
            )
        )
        self.rgba = self.rgba.rotate(self.rotation)

    def get_mask(self):
        return np.array(self.rgba)[:, :, 3] / 255

    def get_mask(image):
        return np.array(image)[:, :, 3] / 255

    # Revisit this: axis order or format is probably wrong...
    """
    def get_bbox(self):
        mask = self.get_mask()
        min_x = (mask > 0).argmax(axis=0).min()
        max_x = mask.shape[0] - np.flip(mask>0, axis=0).argmax(axis=0).max()
        min_y = (mask > 0).argmax(axis=1).min()
        max_y = mask.shape[1] - np.flip(mask > 0, axis=1).argmax(axis=1).max()
        return [min_x, min_y, max_x, max_y]
    """

    def get_bbox(self):
        img = self.get_mask() > 0.5
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, cmin, rmax, cmax

    def get_bbox_from_mask(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, cmin, rmax, cmax

    def get_box_mask(img):
        print(img.shape)
        init_mask = np.zeros(img.shape, dtype=float)
        bbox = ImageLayer.get_bbox_from_mask(img)
        init_mask[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] = 1.0
        return init_mask

    def get_init_mask(self, bbox=False):
        if bbox:
            init_mask = np.zeros(self.rgba.size, dtype=float)
            bbox = self.get_bbox()
            init_mask[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] = 1.0
        else:
            init_mask = self.get_mask()
        return init_mask

    def get_pyramid(init_mask, res_list, device):
        # Now we have a float mask for all the regions we want to emphasize
        torch_mask = torch.tensor(init_mask).unsqueeze(0)
        return_dict = {}
        for res in res_list:
            return_dict[res] = transforms.Resize(size=res)(torch_mask).to(device)
        return return_dict

    def bottom_visible(bottom, top):
        return bottom * (1.0 - top)

    def add_layers(layer_list, background=None):
        # Return the composite image, and also return a mask per image
        # First back-to-front composite for the image
        if background is None:
            composite_image = Image.new("RGB", (512, 512), (150, 150, 150))
        else:
            # Load the background image from a URL string
            # TODO: get rid of hardcoded paths and use google cloud storage
            path = os.path.join(IMAGE_DIR, background)
            # Get the first png file in the directory
            if os.path.isdir(path):
                background_image = sorted(
                    [
                        f
                        for f in os.listdir(path)
                        if os.path.isfile(os.path.join(path, f))
                        and f.endswith(".png")
                    ]
                )[0]
                background_image = os.path.join(path, background_image)
            composite_image = Image.open(background_image).convert("RGB")

        num_layers = len(layer_list)

        # Now composite front-to-back to help compute the per-layer masks
        ftb_composites = []  # [top_composite.copy()]
        for i in range(1, num_layers):
            new_composite = Image.new("RGBA", (512, 512), (150, 150, 150, 0))
            for j in range(i, num_layers):
                new_composite.paste(
                    layer_list[j].rgba, layer_list[j].pos, layer_list[j].rgba
                )
            ftb_composites.append(new_composite)
        ftb_composites.append(Image.new("RGBA", (512, 512), (150, 150, 150, 0)))
        mask_layers = []
        for i in range(0, num_layers):
            bottom_mask = Image.new("RGBA", (512, 512), (150, 150, 150, 0))
            bottom_mask.paste(layer_list[i].rgba, layer_list[i].pos, layer_list[i].rgba)
            bottom_mask = ImageLayer.get_mask(bottom_mask)
            top_mask = ImageLayer.get_mask(ftb_composites[i])
            mask_layers.append(ImageLayer.bottom_visible(bottom_mask, top_mask))

        for i in range(0, num_layers):
            if background:
                if layer_list[i].noise_strength > 0.2:
                    # Create an empty image, size 512 by 512
                    empty_image = Image.new("RGBA", (512, 512), (150, 150, 150, 0))
                    # Paste the layer on top of the empty image
                    empty_image.paste(
                        layer_list[i].rgba, layer_list[i].pos, layer_list[i].rgba
                    )
                    # Get the mask for the layer
                    layer_mask = mask_layers[i]
                    # apply the mask 
                    empty_np = np.array(empty_image)
                    empty_np[:, :, 3] = layer_mask * 255
                    empty_image = Image.fromarray(empty_np)

                    # Paste the layer on top of the composite image, at 
                    composite_image.paste(
                        empty_image, (0, 0), empty_image
                    )

            else:
                composite_image.paste(
                    layer_list[i].rgba, layer_list[i].pos, layer_list[i].rgba
                )
        # Return
        composite_image.save("test.png")
        return composite_image, mask_layers
