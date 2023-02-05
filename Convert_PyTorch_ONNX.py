#
# Convert PyTorch Stable Diffusion models to ONNX format, which then can be converted to OpenVINO IR format using Model Optimizer tool
#
# Based on OpenVINO toolkit notebook:
# https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/225-stable-diffusion-text-to-image/225-stable-diffusion-text-to-image.ipynb
#

# Resolution
res_v = 512
res_h = 512

opset = 16

from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path
import torch
import numpy as np

# Load the pre-trained weights of all components of the model.
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, low_cpu_mem_usage=True)


UNET_ONNX_PATH = Path('unet/unet.onnx')
UNET_OV_PATH = UNET_ONNX_PATH.parents[1] / 'unet.xml'

@torch.no_grad()
def convert_unet_onnx(pipe:StableDiffusionPipeline, onnx_path:Path):
    """
    Convert Unet model to ONNX, then IR format.
    Function accepts pipeline, prepares example inputs for ONNX conversion via torch.export,
    Parameters:
        pipe (StableDiffusionPipeline): Stable Diffusion pipeline
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    if not onnx_path.exists():
        # prepare inputs
        text = 'a photo of an astronaut riding a horse on mars'
        text_encoder = pipe.text_encoder
        input_ids = pipe.tokenizer(
            text,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        with torch.no_grad():
            text_encoder_output = text_encoder(input_ids)
        latents_shape = (2, 4, res_v // 8, res_h // 8)
        latents = torch.randn(latents_shape)
        t = torch.from_numpy(np.array(1, dtype=float))

        # model size > 2Gb, it will be represented as onnx with external data files, we will store it in separated directory for avoid a lot of files in current directory
        onnx_path.parent.mkdir(exist_ok=True, parents=True)

        max_length = input_ids.shape[-1]

        # we plan to use unet with classificator free guidence, in this cace conditionaly generated text embeddings should be concatenated with uncoditional
        uncond_input = pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]
        encoder_hidden_state = torch.cat([uncond_embeddings, text_encoder_output[0]])
        encoder_hidden_state = torch.cat([encoder_hidden_state, encoder_hidden_state], axis=1)

        # to make sure that model works
        pipe.unet(latents, t, encoder_hidden_state)[0]

        with torch.no_grad():
            torch.onnx.export(
                pipe.unet,
                (latents, t, encoder_hidden_state), str(onnx_path),
                input_names  = ['sample', 'timestep', 'encoder_hidden_state'],
                output_names = ['out_sample'],
                dynamic_axes = {
                    "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                    "encoder_hidden_state": {0: "batch", 1: "sequence"},
                },
                opset_version = opset
            )
        print('Unet successfully converted to ONNX')

if not UNET_OV_PATH.exists():
    convert_unet_onnx(pipe, UNET_ONNX_PATH)
    print(f"mo --input_model {UNET_ONNX_PATH} --compress_to_fp16")
    print('Unet successfully converted to IR')


TEXT_ENCODER_ONNX_PATH = Path('text_encoder.onnx')
TEXT_ENCODER_OV_PATH = TEXT_ENCODER_ONNX_PATH.with_suffix('.xml')

@torch.no_grad()
def convert_encoder_onnx(pipe: StableDiffusionPipeline, onnx_path:Path):
    """
    Convert Text Encoder model to ONNX.
    Function accepts pipeline, prepares example inputs for ONNX conversion via torch.export,
    Parameters:
        pipe (StableDiffusionPipeline): Stable Diffusion pipeline
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    if not onnx_path.exists():
        text = 'a photo of an astronaut riding a horse on mars'
        text_encoder = pipe.text_encoder
        input_ids = pipe.tokenizer(
            text,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # switch model to inference mode
        text_encoder.eval()

        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            # infer model, just to make sure that it works
            text_encoder(input_ids)
            # export model to ONNX format
            torch.onnx.export(
                text_encoder,  # model instance
                input_ids,  # inputs for model tracing
                onnx_path,  # output file for saving result
                input_names  = ['input_ids'],  # model input name for onnx representation
                output_names = ['last_hidden_state', 'pooler_out'],  # model output names for onnx representation
                dynamic_axes = {'input_ids': {0: 'batch', 1: 'sequence'}},
                opset_version = opset  # onnx opset version for export
            )
        print('Text Encoder successfully converted to ONNX')

if not TEXT_ENCODER_OV_PATH.exists():
    convert_encoder_onnx(pipe, TEXT_ENCODER_ONNX_PATH)
    print(f"mo --input_model {TEXT_ENCODER_ONNX_PATH} --compress_to_fp16")
    print('Text Encoder successfully converted to IR')


@torch.no_grad()
def convert_vae_decoder_onnx(pipe:StableDiffusionPipeline, onnx_path:Path):
    """
    Convert VAE model to ONNX, then IR format.
    Function accepts pipeline, creates wrapper class for export only necessary for inference part,
    prepares example inputs for ONNX conversion via torch.export,
    Parameters:
        pipe (StableDiffusionPipeline): Stable Diffusion pipeline
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    if not onnx_path.exists():
        vae_decoder = VAEDecoderWrapper(pipe.vae)
        text = 'a photo of an astronaut riding a horse on mars'
        text_encoder = pipe.text_encoder
        input_ids = pipe.tokenizer(
            text,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        with torch.no_grad():
            text_encoder_output = text_encoder(input_ids)
        latents_shape = (2, 4, res_v // 8, res_h // 8)
        latents = torch.randn(latents_shape)
        t = torch.from_numpy(np.array(1, dtype=float))
        max_length = input_ids.shape[-1]
        uncond_input = pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]
        encoder_hidden_state = torch.cat([uncond_embeddings, text_encoder_output[0]])
        output_latents = pipe.unet(latents, t, encoder_hidden_state)[0]
        latents_uncond, latents_text = output_latents[0].unsqueeze(0), output_latents[1].unsqueeze(0)
        latents = latents_uncond + 7.5 * (latents_text - latents_uncond)

        vae_decoder.eval()
        with torch.no_grad():
            torch.onnx.export(
                vae_decoder, latents, onnx_path, input_names=['latents'], output_names=['sample'],
                dynamic_axes={"latents": {0: "batch", 1: "channels", 2: "height", 3: "width"}},
                opset_version = opset  # onnx opset version for export
            )
        print('VAE decoder successfully converted to ONNX')

VAE_ONNX_PATH = Path('vae_decoder.onnx')
VAE_OV_PATH = VAE_ONNX_PATH.with_suffix('.xml')

if not VAE_OV_PATH.exists():
    convert_vae_decoder_onnx(pipe, VAE_ONNX_PATH)
    print(f"mo --input_model {VAE_ONNX_PATH} --compress_to_fp16")
    print('VAE successfully converted to IR')


@torch.no_grad()
def convert_vae_encoder_onnx(pipe:StableDiffusionPipeline, onnx_path:Path):
    """
    Convert VAE model to ONNX, then IR format.
    Function accepts pipeline, creates wrapper class for export only necessary for inference part,
    prepares example inputs for ONNX conversion via torch.export,
    Parameters:
        pipe (StableDiffusionPipeline): Stable Diffusion pipeline
        onnx_path (Path): File for storing onnx model
    Returns:
        None
    """

    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, sample):
            latent = self.vae.encode(sample)[0].sample()        #, return_dict
            return latent


    if not onnx_path.exists():
        vae_encoder = VAEEncoderWrapper(pipe.vae)
        text = 'a photo of an astronaut riding a horse on mars'
        text_encoder = pipe.text_encoder
        input_ids = pipe.tokenizer(
            text,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        with torch.no_grad():
            text_encoder_output = text_encoder(input_ids)
        image_shape = (1, 3, res_v, res_h)
        image = torch.randn(image_shape)
        t = torch.from_numpy(np.array(1, dtype=float))
        max_length = input_ids.shape[-1]
        uncond_input = pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]
        encoder_hidden_state = torch.cat([uncond_embeddings, text_encoder_output[0]])

        vae_encoder.eval()
        with torch.no_grad():
            torch.onnx.export(
                vae_encoder, (image,), onnx_path, input_names=['init_image'], output_names=['sample'],
                dynamic_axes={"init_image": {0: "batch", 1: "channels", 2: "height", 3: "width"}},
                opset_version = opset  # onnx opset version for export
            )
        print('VAE encoder successfully converted to ONNX')

VAEE_ONNX_PATH = Path('vae_encoder.onnx')
VAEE_OV_PATH = VAEE_ONNX_PATH.with_suffix('.xml')

if not VAEE_OV_PATH.exists():
    convert_vae_encoder_onnx(pipe, VAEE_ONNX_PATH)
    print(f"mo --input_model {VAEE_ONNX_PATH} --compress_to_fp16")
    print('VAE successfully converted to IR')


del pipe
