# Checkbin ✅🗑️ - Compare Models Demo

This demo shows how to use Checkbin ✅🗑️ to evaluate the performance of different diffusion models against a list of prompts!

![view-run](https://syntheticco.blob.core.windows.net/model-compare-demo/diffusion_model_compare_scrolling_grid.gif)

## Step 1 - Tools

To run this demo code, you'll need auth tokens (and accounts) from the following services:
- **[Modal](www.modal.com)** - We use Modal to run the training and inference script on cloud GPUs. You can get a Modal token by signing up [here](https://modal.com/signup).
- **[HuggingFace](www.huggingface.com)** - We use HuggingFace to download models for fine-tuning. We also upload the fine-tuned models to HuggingFace. You can get a HuggingFace token by signing up [here](https://huggingface.co/join).
- **[Checkbin](www.checkbin.dev)** - We use Checkbin ✅🗑️ to compare the results of different models. You can get a Checkbin token by signing up [here](www.checkbin.dev/signup).

Once you have tokens for each service, you'll need to add them as environment variables. Since we're running these scripts on Modal, you should add the tokens as [Modal Secrets](https://modal.com/secrets).


## Step 2 - Set up you Checkbin App

If you haven't already, you'll need to create an App on Checkbin ✅🗑️. You can do this from the main [Checkbin Dash](https://app.checkbin.dev/dashboard/apps). I named mine "Compare Diffusion Models", which gives an app key of 'compare_diffusion_models'. Replace "checkbin_app_key" variable in `compare_diffusion_models.py` with the unique key for your app: 

```
# Replace with your app key, after creating your app!
checkbin_app_key = "compare_diffusion_models" 
```

## Step 3 - Select your Prompts 📜

After setting up your Checkbin app, the next step is choosing the prompts you want to test. For this demo, I've chosen a number of variations of the now infamous ["woman laying in grass"](https://www.reddit.com/r/StableDiffusion/comments/1de85nc/why_is_sd3_so_bad_at_generating_girls_lying_on/) prompt! My prompts are included in this directory in the `inputs/woman_lying_in_grass_prompts.json`. You should create a new JSON file with the prompts you'd like to test. The file has this structure:

```
[
    {
        "prompt": "A woman lying in grass",
        "negative_prompt": "overcast sky"
    },
    {
        "prompt": "A man lying on a beach",
        "negative_prompt": "crowded"
    },
    ... // More prompts!
]
```

After creating your list of prompts, replace the test_prompts_path variable at the top of `compare_diffusion_models.py`. Your JSON file should be in the 'inputs' directory.  

```
# Replace with your JSON file, which needs to be in the inputs directory!
test_prompts_path = "woman_lying_in_grass_prompts.json" 
```

## Step 4 - Select your Models 🕵️‍♂️

The next step is to choose which models you want to compare! The script uses a namedtuple called "ModelConfig" to store the information about the models to be tested: 

```
models_to_test = [
    ModelConfig(
        description="Stable Diffusion 3",
        base_model_id="stabilityai/stable-diffusion-3-medium-diffusers",
        pipeline_type="sd3",
    ),
    ...
```

You should overwrite this array with the models that you'd like to test!

- **description** (ex. "Stable Diffusion 3") - a plain english name of the model you're testing. This will appear in the header of the Checkbin columns. 
- **base_model_id** (ex. "stabilityai/stable-diffusion-3-medium-diffusers") - the path of the model you're testing on HuggingFace. 
- **pipeline_type** (ex. "sd3") - the type of diffusion pipeline from the [diffusers library](https://github.com/huggingface/diffusers). The script supports 4 options, "flux", "sd", "sd3", and "sdxl". 

## Step 5 - Run the script

Once you've created your app, customized your prompts, and selected your models, you're ready to run the app! Start it with the command:

```
modal run compare_diffusion_models.py
```

The script will print a Checkbin run id:

```
Checkbin: started run b9cc49a4-3007-40c4-a2d9-0d73dff0f811 with 94 tests
```

When the run has completed, you can load in the [Checkbin Grid](https://app.checkbin.dev/grid) and view your results!

![view-run](https://syntheticco.blob.core.windows.net/model-compare-demo/diffusion_model_compare_scrolling_grid.gif)

## Acknowledgments
This project wouldn't be possible without HuggingFace's [diffusers library](https://github.com/huggingface/diffusers) or Modal's infrastructure. A big thanks to the HuggingFace and Modal teams for their excellent contributions!

