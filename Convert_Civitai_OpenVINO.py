res_v = 768
res_h = res_v
print(f'{res_h}x{res_v}')
opset = 19

import torch
import numpy as np
from pathlib import Path
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

from openvino.tools import mo
from openvino.runtime import serialize

model_unet = 'artiusv15_v1.safetensors'                             # The file was downloaded from https://civitai.com/models/47691/artiusv1-5
pipe1 = StableDiffusionPipeline.from_single_file(model_unet)

model_id = "runwayml/stable-diffusion-v1-5"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")    # Load custom VAE
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, unet=pipe1.unet, low_cpu_mem_usage=True, safety_checker=None, feature_extractor=None, requires_safety_checker=False, local_files_only=False, vae=vae).to("cpu")
pipe.enable_attention_slicing(slice_size='max')

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
            text_encoder_output = text_encoder(input_ids.to('cpu'))
        latents_shape = (2, 4, res_v // 8, res_h // 8)
        t = torch.from_numpy(np.array(1, dtype=float))
        latents = torch.randn(latents_shape)

        # model size > 2Gb, it will be represented as onnx with external data files, we will store it in separated directory for avoid a lot of files in current directory
        onnx_path.parent.mkdir(exist_ok=True, parents=True)

        max_length = input_ids.shape[-1]

        # we plan to use unet with classificator free guidence, in this cace conditionaly generated text embeddings should be concatenated with uncoditional
        uncond_input = pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to('cpu'))[0]
        encoder_hidden_state = torch.cat([uncond_embeddings, text_encoder_output[0]])

        # to make sure that model works
        pipe.unet(latents, t, encoder_hidden_state)[0]

        with torch.no_grad():
            torch.onnx.export(
                pipe.unet,
                (latents, t, encoder_hidden_state), str(onnx_path),
                input_names = ['sample', 'timestep', 'encoder_hidden_state'],
                output_names = ['out_sample'],
                dynamic_axes = {
                    "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                    "encoder_hidden_state": {0: "batch", 1: "sequence"},
                },
                opset_version = opset,
                export_params=True, 
                do_constant_folding=True,
            )
        print('Unet successfully converted to ONNX')

if not UNET_OV_PATH.exists():
    convert_unet_onnx(pipe, UNET_ONNX_PATH)
    model = mo.convert_model(UNET_ONNX_PATH, compress_to_fp16=True)
    serialize(model, xml_path=UNET_OV_PATH)
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
        ).input_ids.to("cpu")

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
                input_names=['input_ids'],  # model input name for onnx representation
                output_names=['last_hidden_state', 'pooler_out'],  # model output names for onnx representation
                dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}},
                opset_version = opset  # onnx opset version for export
            )
        print('Text Encoder successfully converted to ONNX')

if not TEXT_ENCODER_OV_PATH.exists():
    convert_encoder_onnx(pipe, TEXT_ENCODER_ONNX_PATH)
    model = mo.convert_model(TEXT_ENCODER_ONNX_PATH, compress_to_fp16=True)
    serialize(model, xml_path=TEXT_ENCODER_OV_PATH)
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
        ).input_ids.to("cpu")
        with torch.no_grad():
            text_encoder_output = text_encoder(input_ids)
        latents_shape = (2, 4, res_v // 8, res_h // 8)
        latents = torch.randn(latents_shape).to("cpu")
        t = torch.FloatTensor(1).to("cpu")
        max_length = input_ids.shape[-1]
        with torch.no_grad():
            uncond_input = pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt").to("cpu")
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
    model = mo.convert_model(VAE_ONNX_PATH, compress_to_fp16=True)
    serialize(model, xml_path=VAE_OV_PATH)
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
            latent = self.vae.encode(sample)[0].sample()
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
        ).input_ids.to("cpu")
        with torch.no_grad():
            text_encoder_output = text_encoder(input_ids)
        image_shape = (1, 3, res_v, res_h)
        image = torch.randn(image_shape).to("cpu")
        t = torch.FloatTensor(1).to("cpu")
        max_length = input_ids.shape[-1]
        with torch.no_grad():
            uncond_input = pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt").to("cpu")
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
    model = mo.convert_model(VAEE_ONNX_PATH, compress_to_fp16=True)
    serialize(model, xml_path=VAEE_OV_PATH)
    print('VAE successfully converted to IR')


del pipe
