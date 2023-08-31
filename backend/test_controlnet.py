from pipeline_controlnet import StableDiffusionControlNetPipeline, Control
# from real_cn import StableDiffusionControlNetPipeline
from diffusers.models import ControlNetModel
from diffusers import EulerAncestralDiscreteScheduler
import torch
import os

cuda_visible_devices = int(os.environ["CUDA_VISIBLE_DEVICES"])
print(f"Running on rank {int(cuda_visible_devices)}")
device = torch.device('cuda')

# controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
# controlnet_pose = controlnet_pose.to(device)
controlnet_canny = controlnet_canny.to(device)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=[controlnet_canny], 
).to(device)
print(pipe.device)
controls = [Control.CANNY]

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config,
)

layers = [
	{
		"id": 0,
		"key": "16b99e6a4003d2796869",
		"originalImgUrl": "https://storage.googleapis.com/collage-bucket/2044212b55c84f949ca95a9f43a8d37b",
		"textPrompt": "iceberg",
		"cacStrength": 0,
		"negativeStrength": 1,
		"noiseStrength": 0.8,
		"transform": {
			"position": {
				"x": -0.07081802527616662,
				"y": -0.010978285870585585
			},
			"scale": 0.0008108549688892926,
			"rotation": 0
		},
		"polygon": []
	}, {
		"id": 1,
		"key": "e0e8bc2096816f7655db",
		"originalImgUrl": "https://storage.googleapis.com/collage-bucket/642aae4a09059cc744686f716eb81dc9",
		"textPrompt": "deer",
		"cacStrength": 0,
		"negativeStrength": 1,
		"noiseStrength": 0.8,
		"transform": {
			"position": {
				"x": 0.1469149146357053,
				"y": -0.004172332438431937
			},
			"scale": 0.000401930817536968,
			"rotation": 0
		},
		"polygon": [{
			"x": 0.35778879154624277,
			"y": 0.5537109375
		}, {
			"x": 0.3333013457369942,
			"y": 0.5966796875
		}, {
			"x": 0.3333013457369942,
			"y": 0.6611328125
		}, {
			"x": 0.35778879154624277,
			"y": 0.7392578125
		}, {
			"x": 0.39315954660404623,
			"y": 0.7763671875
		}, {
			"x": 0.4203678197254335,
			"y": 0.8544921875
		}, {
			"x": 0.4475760928468208,
			"y": 0.9794921875
		}, {
			"x": 0.4965509844653179,
			"y": 0.9833984375
		}, {
			"x": 0.5074342937138728,
			"y": 0.8857421875
		}, {
			"x": 0.515596775650289,
			"y": 0.8291015625
		}, {
			"x": 0.545525876083815,
			"y": 0.9775390625
		}, {
			"x": 0.5836174584537572,
			"y": 0.9658203125
		}, {
			"x": 0.6217090408236994,
			"y": 0.8330078125
		}, {
			"x": 0.6570797958815029,
			"y": 0.7353515625
		}, {
			"x": 0.6842880690028902,
			"y": 0.6728515625
		}, {
			"x": 0.7087755148121387,
			"y": 0.5947265625
		}, {
			"x": 0.6652422778179191,
			"y": 0.5361328125
		}, {
			"x": 0.5999424223265896,
			"y": 0.4931640625
		}, {
			"x": 0.5700133218930635,
			"y": 0.4716796875
		}, {
			"x": 0.5618508399566474,
			"y": 0.4287109375
		}, {
			"x": 0.5618508399566474,
			"y": 0.4013671875
		}, {
			"x": 0.5727341492052023,
			"y": 0.3759765625
		}, {
			"x": 0.6217090408236994,
			"y": 0.3427734375
		}, {
			"x": 0.6870088963150289,
			"y": 0.3173828125
		}, {
			"x": 0.716937996748555,
			"y": 0.3154296875
		}, {
			"x": 0.7387046152456648,
			"y": 0.3525390625
		}, {
			"x": 0.7577504064306358,
			"y": 0.3408203125
		}, {
			"x": 0.7876795068641619,
			"y": 0.3486328125
		}, {
			"x": 0.7659128883670521,
			"y": 0.3076171875
		}, {
			"x": 0.7441462698699421,
			"y": 0.2763671875
		}, {
			"x": 0.735983787933526,
			"y": 0.2353515625
		}, {
			"x": 0.7223796513728323,
			"y": 0.2021484375
		}, {
			"x": 0.7468670971820809,
			"y": 0.1865234375
		}, {
			"x": 0.7767961976156069,
			"y": 0.1787109375
		}, {
			"x": 0.7278213059971098,
			"y": 0.1591796875
		}, {
			"x": 0.7414254425578035,
			"y": 0.1259765625
		}, {
			"x": 0.6897297236271677,
			"y": 0.0947265625
		}, {
			"x": 0.6108257315751445,
			"y": 0.0634765625
		}, {
			"x": 0.5237592575867052,
			"y": 0.0458984375
		}, {
			"x": 0.5101551210260116,
			"y": 0.0693359375
		}, {
			"x": 0.5564091853323699,
			"y": 0.0947265625
		}, {
			"x": 0.48022602059248554,
			"y": 0.0751953125
		}, {
			"x": 0.46934271134393063,
			"y": 0.1044921875
		}, {
			"x": 0.5128759483381503,
			"y": 0.2373046875
		}, {
			"x": 0.5019926390895953,
			"y": 0.2880859375
		}, {
			"x": 0.48022602059248554,
			"y": 0.3525390625
		}, {
			"x": 0.41492616510115604,
			"y": 0.3564453125
		}, {
			"x": 0.38499706466763006,
			"y": 0.3056640625
		}, {
			"x": 0.39315954660404623,
			"y": 0.2255859375
		}, {
			"x": 0.39588037391618497,
			"y": 0.1572265625
		}, {
			"x": 0.41492616510115604,
			"y": 0.1123046875
		}, {
			"x": 0.43125112897398843,
			"y": 0.0732421875
		}, {
			"x": 0.3686721007947977,
			"y": 0.0947265625
		}, {
			"x": 0.3169763818641618,
			"y": 0.1240234375
		}, {
			"x": 0.3169763818641618,
			"y": 0.0869140625
		}, {
			"x": 0.3496263096098266,
			"y": 0.0654296875
		}, {
			"x": 0.3224180364884393,
			"y": 0.0361328125
		}, {
			"x": 0.2625598356213873,
			"y": 0.0615234375
		}, {
			"x": 0.205422462066474,
			"y": 0.1181640625
		}, {
			"x": 0.1972599801300578,
			"y": 0.1533203125
		}, {
			"x": 0.12651847001445088,
			"y": 0.1533203125
		}, {
			"x": 0.13740177926300579,
			"y": 0.1806640625
		}, {
			"x": 0.1673308796965318,
			"y": 0.1904296875
		}, {
			"x": 0.18093501625722544,
			"y": 0.2158203125
		}, {
			"x": 0.1673308796965318,
			"y": 0.2783203125
		}, {
			"x": 0.16461005238439305,
			"y": 0.3408203125
		}, {
			"x": 0.19181832550578035,
			"y": 0.3623046875
		}, {
			"x": 0.2081432893786127,
			"y": 0.3935546875
		}, {
			"x": 0.2271890805635838,
			"y": 0.4287109375
		}, {
			"x": 0.3006514179913295,
			"y": 0.4521484375
		}, {
			"x": 0.34690548229768786,
			"y": 0.4560546875
		}]
	}]
# [ground, mountains, house, house house]

cac_strengths = [0.0, 0.0]
cac_negative_strengths = [0.8, 0.8]
noise_strengths = [0.8, 0.8]

canny_strengths = [0.5, 0.5]

prompt = "deer on an iceberg"
prompt += ", best quality, high quality"
(
    composite_image,
    mask_layers,
    attention_mod,
    collage_prompt,
) = pipe.preprocess_layers(
    layers,
    cac_strengths,
    cac_negative_strengths,
    prompt,
)

img2img_strength = max(noise_strengths)
composite_image.save('composite_test.png')

# pose_mask = pipe.generate_control_mask(mask_layers, pose_strengths)
canny_mask = pipe.generate_control_mask(mask_layers, canny_strengths)

image = pipe.collage(
    prompt=collage_prompt,
    negative_prompt='worst quality, low quality, collage',
    image=composite_image, # canny image
    num_inference_steps=50,
    mask_image=pipe.generate_mask(mask_layers, img2img_strength, noise_strengths),
    strength=img2img_strength,  # img2img strength
    guidance_scale=7,
    attention_mod=attention_mod,
    controlnet_conditioning_scale=[canny_mask],
	controls=controls,
).images[0]

os.makedirs("output_images", exist_ok=True)
image.save(os.path.join("output_images", f"{prompt}_{cuda_visible_devices}.png"))