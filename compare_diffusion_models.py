from modal import Image as ModalImage, App, gpu, Secret, Mount
from collections import namedtuple
import os, sys

checkbin_app_key = "compare_diffusion_models"
test_prompts_path = "woman_lying_in_grass_prompts.json"

ModelConfig = namedtuple(
    'ModelConfig', 
    [
        'description', # This is your name for the model, it can be anything
        'base_model_id', # This is the model id, it should be the path of the model you trained on top of
        'pipeline_type', # This is the type of model you trained on top of
    ]
)

# Update this with the models you want to test!
models_to_test = [
    ModelConfig(
        description="Stable Diffusion 3",
        base_model_id="stabilityai/stable-diffusion-3-medium-diffusers",
        pipeline_type="sd3",
    ),
    ModelConfig(
        description="Stable Diffusion 3.5",
        base_model_id="stabilityai/stable-diffusion-3.5-large",
        pipeline_type="sd3",
    ),
    ModelConfig(
        description="Flux.1.Dev",
        base_model_id="black-forest-labs/FLUX.1-dev",
        pipeline_type="flux",
    ),
]

compare_diffusion_models_image = (
    ModalImage.from_registry(
        "nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        [
            "git", 
            "libgl1",
            "libglib2.0-0",
            "libsm6",
            "libxext6", 
            "ffmpeg",
        ]
    )
    .pip_install(
        [
            "transformers",
            "diffusers",
            "torch",
            "numpy",
            "huggingface_hub",
            "accelerate",
            "sentencepiece",
            "peft",
            "opencv-python",
            "boto3",
            "google-cloud-storage",
            "azure-storage-blob",
            "azure-storage-file-datalake",
            "pydantic",
            "tinydb",
        ]
    )
    # TODO everything after "opencv-python" are needed for the local version of checkbin, but can be removed when we move to the pip packages
)

app = App("compare-diffusion-models", image=compare_diffusion_models_image)

def run_inference(checkbins, checkin_name, model_id, pipeline_type):
    import torch
    from diffusers import FluxPipeline, StableDiffusionPipeline, StableDiffusion3Pipeline, StableDiffusionXLPipeline

    if pipeline_type == "flux":
        print(f"Loading Flux model from {model_id}")
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    elif pipeline_type == "sd":
        print(f"Loading Stable Diffusion model from {model_id}")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    elif pipeline_type == "sd3":
        print(f"Loading Stable Diffusion 3 model from {model_id}")
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    elif pipeline_type == "sdxl":
        print(f"Loading Stable Diffusion XL model from {model_id}")
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    else:
        raise ValueError(f"Invalid pipeline type: {pipeline_type}")
    
    for checkbin in checkbins:
        negative_prompt = checkbin.get_input_data('negative_prompt')
        prompt = checkbin.get_input_data('prompt')

        checkbin.checkin(checkin_name)
        if pipeline_type == "flux":
            image = pipe(
                prompt=prompt
            ).images[0]
        else:
            image = pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt,
            ).images[0]

        image.save("inference_output.png")
        checkbin.upload_file(
            "inference_output",
            "inference_output.png", 
            "image"
        )
        
@app.function(
    gpu=gpu.A100(size="80GB"),
    timeout=86400,
    image=compare_diffusion_models_image,
    secrets=[Secret.from_name("checkbin-secret"), Secret.from_name("huggingface-secret")],
    mounts=[Mount.from_local_dir("./checkbin-python", remote_path="/root/checkbin-python"),
            Mount.from_local_dir("./inputs", remote_path="/root/inputs")]
)
def compare_diffusion_models():
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

    # TODO migrate to use the python package instead of local checkbin-python
    sys.path.insert(0, 'checkbin-python/src')
    import checkbin

    checkbin.authenticate(token=os.environ["CHECKBIN_TOKEN"])
    checkbin_app = checkbin.App(app_key=checkbin_app_key, mode="remote")

    with checkbin_app.start_run(json_file=f'/root/inputs/{test_prompts_path}') as bin_generator:
        # Convert the generator to a list so we can iterate over it multiple times
        bins = list(bin_generator)
        for config in models_to_test:
            run_inference(bins, config.description, config.base_model_id, config.pipeline_type)
        for checkbin in bins:
            checkbin.submit()
