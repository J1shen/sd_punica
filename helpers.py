import math
import os
from typing import List, Union
import time

import numpy as np
import torch
from einops import rearrange, repeat
from imwatermark import WatermarkEncoder
from omegaconf import ListConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torch import autocast
from torchvision import transforms
from torchvision.utils import make_grid

from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.modules.diffusionmodules.sampling import (
    DPMPP2MSampler,
    DPMPP2SAncestralSampler,
    EulerAncestralSampler,
    EulerEDMSampler,
    HeunEDMSampler,
    LinearMultistepSampler,
)
from sgm.util import append_dims, instantiate_from_config

lowvram_mode = True

def init_model(version_dict, load_ckpt=True, load_filter=True, verbose=True):
    state = dict()
    config = version_dict["config"]
    ckpt = version_dict["ckpt"]
    config = OmegaConf.load(config)

    model = instantiate_from_config(config.model)
    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        if ckpt.endswith("ckpt"):
            pl_sd = torch.load(ckpt, map_location="cpu")
            if "global_step" in pl_sd:
                global_step = pl_sd["global_step"]
                print(f"loaded ckpt from global step {global_step}")
                print(f"Global Step: {pl_sd['global_step']}")
            sd = pl_sd["state_dict"]
        elif ckpt.endswith("safetensors"):
            sd = load_safetensors(ckpt)
        else:
            raise NotImplementedError

        m, u = model.load_state_dict(sd, strict=False)

        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

    if lowvram_mode:
        model.model.half()
    else:
        model.cuda()
    
    model.eval()
    state["msg"] = None
    state["model"] = model        
    state["ckpt"] = ckpt if load_ckpt else None
    state["config"] = config
    if load_filter:
        state["filter"] = DeepFloydDataFiltering(verbose=False)
    return state

class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: torch.Tensor):
        """
        Adds a predefined watermark to the input image

        Args:
            image: ([N,] B, C, H, W) in range [0, 1]

        Returns:
            same as input but watermarked
        """
        # watermarking libary expects input as cv2 BGR format
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        image_np = rearrange(
            (255 * image).detach().cpu(), "n b c h w -> (n b) h w c"
        ).numpy()[:, :, :, ::-1]
        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        image = torch.from_numpy(
            rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)
        ).to(image.device)
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        return image


# A fixed 48-bit message that was choosen at random
# WATERMARK_MESSAGE = 0xB3EC907BB19E
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
embed_watemark = WatermarkEmbedder(WATERMARK_BITS)

def load_model(model):
    model.cuda()

def set_lowvram_mode(mode):
    global lowvram_mode
    lowvram_mode = mode


def initial_model_load(model):
    global lowvram_mode
    if lowvram_mode:
        model.model.half()
    else:
        model.cuda()
    return model


def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        torch.cuda.empty_cache()

def init_embedder_options(keys, init_dict, prompt=None, negative_prompt=None):
    # Hardcoded demo settings; might undergo some changes in the future
    value_dict = {}
    for key in keys:
        if key == "txt":
            if prompt is None:
                prompt = "A professional photograph of an astronaut riding a pig"
            if negative_prompt is None:
                negative_prompt = ''

            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = negative_prompt

        if key == "original_size_as_tuple":
            orig_width = init_dict["orig_width"]
            orig_height = init_dict["orig_height"]

            value_dict["orig_width"] = orig_width
            value_dict["orig_height"] = orig_height

        if key == "crop_coords_top_left":
            crop_coord_top = 0
            crop_coord_left = 0

            value_dict["crop_coords_top"] = crop_coord_top
            value_dict["crop_coords_left"] = crop_coord_left

        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = 6.0
            value_dict["negative_aesthetic_score"] = 2.5

        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]

    return value_dict


def perform_save_locally(save_path, samples):
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    samples = embed_watemark(samples)
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        Image.fromarray(sample.astype(np.uint8)).save(
            os.path.join(save_path, f"{base_count:09}.png")
        )
        base_count += 1


class Img2ImgDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas
    params:
        strength: float between 0.0 and 1.0. 1.0 means full sampling (all sigmas are returned)
    """

    def __init__(self, discretization, strength: float = 1.0):
        self.discretization = discretization
        self.strength = strength
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)
        #print(f"sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        sigmas = sigmas[: max(int(self.strength * len(sigmas)), 1)]
        #print("prune index:", max(int(self.strength * len(sigmas)), 1))
        sigmas = torch.flip(sigmas, (0,))
        #print(f"sigmas after pruning: ", sigmas)
        return sigmas


class Txt2NoisyDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas
    params:
        strength: float between 0.0 and 1.0. 0.0 means full sampling (all sigmas are returned)
    """

    def __init__(self, discretization, strength: float = 0.0, original_steps=None):
        self.discretization = discretization
        self.strength = strength
        self.original_steps = original_steps
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)
        #print(f"sigmas after discretization, before pruning img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        if self.original_steps is None:
            steps = len(sigmas)
        else:
            steps = self.original_steps + 1
        prune_index = max(min(int(self.strength * steps) - 1, steps - 1), 0)
        sigmas = sigmas[prune_index:]
        #print("prune index:", prune_index)
        sigmas = torch.flip(sigmas, (0,))
        #print(f"sigmas after pruning: ", sigmas)
        return sigmas


def get_guider(key):
    guiders = [
            "VanillaCFG",
            "IdentityGuider",
        ]
    guider = guiders[0]

    if guider == "IdentityGuider":
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif guider == "VanillaCFG":
        scale = 5.0 #cfg-scale

        thresholder = 'None' #Thresholder

        if thresholder == "None":
            dyn_thresh_config = {
                "target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"
            }
        else:
            raise NotImplementedError

        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {"scale": scale, "dyn_thresh_config": dyn_thresh_config},
        }
    else:
        raise NotImplementedError
    return guider_config


def init_sampling(
    key=1,
    img2img_strength=1.0,
    stage2strength=None,
    steps = 40,
    sampler = "EulerEDMSampler",
    discretization = "LegacyDDPMDiscretization",
):
    discretization_config = get_discretization(discretization, key=key)

    guider_config = get_guider(key=key)

    sampler = get_sampler(sampler, steps, discretization_config, guider_config, key=key)
    if img2img_strength < 1.0:
        print(
            f"Wrapping {sampler.__class__.__name__} with Img2ImgDiscretizationWrapper"
        )
        sampler.discretization = Img2ImgDiscretizationWrapper(
            sampler.discretization, strength=img2img_strength
        )
    if stage2strength is not None:
        sampler.discretization = Txt2NoisyDiscretizationWrapper(
            sampler.discretization, strength=stage2strength, original_steps=steps
        )
    return sampler


def get_discretization(discretization, key=1):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        }
    elif discretization == "EDMDiscretization":
        sigma_min = 0.0292  # 0.0292
        sigma_max = 14.6146  # 14.6146
        rho = 3.0
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
            },
        }

    return discretization_config


def get_sampler(sampler_name, steps, discretization_config, guider_config, key=1):
    if sampler_name == "EulerEDMSampler" or sampler_name == "HeunEDMSampler":
        s_churn = 0
        s_tmin = 0
        s_tmax = 999.0
        s_noise = 1.0

        if sampler_name == "EulerEDMSampler":
            sampler = EulerEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "HeunEDMSampler":
            sampler = HeunEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
    elif (
        sampler_name == "EulerAncestralSampler"
        or sampler_name == "DPMPP2SAncestralSampler"
    ):
        s_noise = 1.0
        eta = 1.0

        if sampler_name == "EulerAncestralSampler":
            sampler = EulerAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "DPMPP2SAncestralSampler":
            sampler = DPMPP2SAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
    elif sampler_name == "DPMPP2MSampler":
        sampler = DPMPP2MSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name == "LinearMultistepSampler":
        order = 4
        sampler = LinearMultistepSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            order=order,
            verbose=True,
        )
    else:
        raise ValueError(f"unknown sampler {sampler_name}!")

    return sampler


def get_interactive_image(key=None) -> Image.Image:
    image = key
    if image is not None:
        image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image


def load_img(display=True, key=None):
    image = get_interactive_image(key=key)
    if image is None:
        return None
    if display:
        print(image)
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )
    img = transform(image)[None, ...]
    print(f"input min/max/mean: {img.min():.3f}/{img.max():.3f}/{img.mean():.3f}")
    return img


def get_init_img(batch_size=1, key=None):
    init_image = load_img(key=key).cuda()
    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
    return init_image


def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: List = [],
    batch2model_input: List = [],
    return_latents=False,
    filter=None,
):
    print("Sampling")
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                t = time.time()
                num_samples = [num_samples]

                load_model(model.conditioner)
                if isinstance(value_dict['prompt'],str):
                    batch, batch_uc = get_batch(model.conditioner,value_dict,num_samples,)
                else:
                    batch, batch_uc = get_batch_m(model.conditioner,value_dict,num_samples,)

                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )
                unload_model(model.conditioner)

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                        )

                additional_model_inputs = {}
                for k in batch2model_input:
                    additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")
                print('Condition Time: ', time.time()-t)
                t = time.time()

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                load_model(model.denoiser)
                load_model(model.model)
                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                unload_model(model.model)
                unload_model(model.denoiser)
                print('Sampling Time: ', time.time()-t)
                t = time.time()
                load_model(model.first_stage_model)
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                unload_model(model.first_stage_model)

                if filter is not None:
                    samples = filter(samples)

                grid = torch.stack([samples])
                grid = rearrange(grid, "n b c h w -> (n h) (b w) c")
                print('Decoding Time: ', time.time()-t)
                if return_latents:
                    return samples, samples_z
                return samples

def get_batch(conditioner, value_dict, N: Union[List, ListConfig], device="cuda"):
    # Hardcoded demo setups; might undergo some changes in the future
    keys = list(set([x.input_key for x in conditioner.embedders]))
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        else:
            batch[key] = value_dict[key]

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def get_batch_m(conditioner, value_dict, N: Union[List, ListConfig], device="cuda"):
    # Hardcoded demo setups; might undergo some changes in the future
    keys = list(set([x.input_key for x in conditioner.embedders]))
    batch = {}
    batch_uc = {}
    num_input = N[0]//len(value_dict["prompt"])

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat(value_dict["prompt"], repeats=num_input)
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat(value_dict["negative_prompt"], repeats=num_input)
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        else:
            batch[key] = value_dict[key]

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


@torch.no_grad()
def do_img2img(
    img,
    model,
    sampler,
    value_dict,
    num_samples,
    force_uc_zero_embeddings=[],
    additional_kwargs={},
    offset_noise_level: int = 0.0,
    return_latents=False,
    skip_encode=False,
    filter=None,
    add_noise=True,
):
    print("Sampling")
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                load_model(model.conditioner)
                batch, batch_uc = get_batch(
                    model.conditioner,
                    value_dict,
                    [num_samples],
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )
                unload_model(model.conditioner)
                for k in c:
                    c[k], uc[k] = map(lambda y: y[k][:num_samples].to("cuda"), (c, uc))

                for k in additional_kwargs:
                    c[k] = uc[k] = additional_kwargs[k]
                if skip_encode:
                    z = img
                else:
                    load_model(model.first_stage_model)
                    z = model.encode_first_stage(img)
                    unload_model(model.first_stage_model)

                noise = torch.randn_like(z)

                sigmas = sampler.discretization(sampler.num_steps).cuda()
                sigma = sigmas[0]

                print(f"all sigmas: {sigmas}")
                print(f"noising sigma: {sigma}")
                if offset_noise_level > 0.0:
                    noise = noise + offset_noise_level * append_dims(
                        torch.randn(z.shape[0], device=z.device), z.ndim
                    )
                if add_noise:
                    noised_z = z + noise * append_dims(sigma, z.ndim).cuda()
                    noised_z = noised_z / torch.sqrt(
                        1.0 + sigmas[0] ** 2.0
                    )  # Note: hardcoded to DDPM-like scaling. need to generalize later.
                else:
                    noised_z = z / torch.sqrt(1.0 + sigmas[0] ** 2.0)

                def denoiser(x, sigma, c):
                    return model.denoiser(model.model, x, sigma, c)

                load_model(model.denoiser)
                load_model(model.model)
                samples_z = sampler(denoiser, noised_z, cond=c, uc=uc)
                unload_model(model.model)
                unload_model(model.denoiser)

                load_model(model.first_stage_model)
                samples_x = model.decode_first_stage(samples_z)
                unload_model(model.first_stage_model)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                if filter is not None:
                    samples = filter(samples)

                grid = embed_watemark(torch.stack([samples]))
                grid = rearrange(grid, "n b c h w -> (n h) (b w) c")
                if return_latents:
                    return samples, samples_z
                return samples
