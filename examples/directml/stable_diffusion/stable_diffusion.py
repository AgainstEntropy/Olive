# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import os
import shutil
import threading
import tkinter as tk
import tkinter.ttk as ttk
import warnings
from pathlib import Path
from tkinter import filedialog as fd

import config
import numpy as np
import onnxruntime as ort
import torch
from diffusers import DiffusionPipeline, OnnxRuntimeModel, OnnxStableDiffusionPipeline, StableDiffusionPipeline
from lora_weights_conversion import convert_kohya_lora_to_diffusers
from lora_weights_renamer import LoraWeightsRenamer
from onnx import load_model, save_model
from onnxruntime.transformers.onnx_model import OnnxModel
from packaging import version
from PIL import Image, ImageTk
from safetensors.numpy import load_file
from user_script import get_base_model_name

from olive.model import ONNXModel
from olive.workflows import run as olive_run


def read_lora_weights(lora_weights_filename):
    if lora_weights_filename.endswith(".bin"):
        lora_weights = torch.load(lora_weights_filename, map_location="cpu", weights_only=True)
    elif lora_weights_filename.endswith(".safetensors"):
        lora_weights = load_file(lora_weights_filename)

    alpha = 1.0

    if all((k.startswith("lora_te_") or k.startswith("lora_unet_")) for k in lora_weights.keys()):
        lora_weights, alpha = convert_kohya_lora_to_diffusers(lora_weights)

    rank = lora_weights[
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.to_k_lora.down.weight"
    ].shape[0]

    return lora_weights, alpha, rank


def run_inference_loop(
    pipeline,
    prompt,
    negative_prompt,
    num_images,
    batch_size,
    image_size,
    num_inference_steps,
    image_callback=None,
    step_callback=None,
):
    images_saved = 0

    def update_steps(step, timestep, latents):
        if step_callback:
            step_callback((images_saved // batch_size) * num_inference_steps + step)

    while images_saved < num_images:
        print(f"\nInference Batch Start (batch size = {batch_size}).")
        result = pipeline(
            [prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size,
            num_inference_steps=num_inference_steps,
            callback=update_steps if step_callback else None,
            height=image_size,
            width=image_size,
        )
        passed_safety_checker = 0

        for image_index in range(batch_size):
            if result.nsfw_content_detected is None or not result.nsfw_content_detected[image_index]:
                passed_safety_checker += 1
                if images_saved < num_images:
                    output_path = f"result_{images_saved}.png"
                    result.images[image_index].save(output_path)
                    if image_callback:
                        image_callback(images_saved, output_path)
                    images_saved += 1
                    print(f"Generated {output_path}")

        print(f"Inference Batch End ({passed_safety_checker}/{batch_size} images passed the safety checker).")


def run_inference_gui(
    pipeline,
    prompt,
    negative_prompt,
    num_images,
    batch_size,
    image_size,
    num_inference_steps,
):
    def update_progress_bar(total_steps_completed):
        progress_bar["value"] = total_steps_completed

    lora_weights_filename = None

    def image_completed(index, path):
        img = Image.open(path)
        photo = ImageTk.PhotoImage(img)
        gui_images[index].config(image=photo)
        gui_images[index].image = photo
        if index == num_images - 1:
            generate_button["state"] = "normal"

    def on_generate_click():
        generate_button["state"] = "disabled"
        progress_bar["value"] = 0
        threading.Thread(
            target=run_inference_loop,
            args=(
                pipeline,
                prompt_textbox.get(),
                negative_prompt_textbox.get(),
                num_images,
                batch_size,
                image_size,
                num_inference_steps,
                image_completed,
                update_progress_bar,
            ),
        ).start()

    def on_lora_weights_click():
        nonlocal lora_weights_filename
        lora_weights_filename = fd.askopenfilename()

    if num_images > 9:
        print("WARNING: interactive UI only supports displaying up to 9 images")
        num_images = 9

    image_rows = 1 + (num_images - 1) // 3
    image_cols = 2 if num_images == 4 else min(num_images, 3)
    min_batches_required = 1 + (num_images - 1) // batch_size

    bar_height = 10
    button_width = 80
    textbox_height = 30
    label_width = 100
    padding = 2
    window_width = image_cols * image_size + (image_cols + 1) * padding + 100
    window_height = image_rows * image_size + (image_rows + 1) * padding + bar_height + textbox_height * 2

    window = tk.Tk()
    window.title("Stable Diffusion")
    window.resizable(width=False, height=False)
    window.geometry(f"{window_width}x{window_height}")

    gui_images = []
    for row in range(image_rows):
        for col in range(image_cols):
            label = tk.Label(window, width=image_size, height=image_size, background="black")
            gui_images.append(label)
            label.place(x=col * image_size, y=row * image_size)

    y = image_rows * image_size + (image_rows + 1) * padding

    progress_bar = ttk.Progressbar(window, value=0, maximum=num_inference_steps * min_batches_required)
    progress_bar.place(x=0, y=y, height=bar_height, width=window_width)

    y += bar_height

    prompt_label = tk.Label(window, text="Prompt")
    prompt_label.place(x=0, y=y)

    prompt_textbox = tk.Entry(window)
    prompt_textbox.insert(tk.END, prompt)
    prompt_textbox.place(x=label_width, y=y, width=window_width - button_width * 2 - label_width, height=textbox_height)

    generate_button = tk.Button(window, text="Generate", command=on_generate_click)
    generate_button.place(x=window_width - button_width * 2, y=y, width=button_width, height=textbox_height * 2)

    generate_button = tk.Button(window, text="LoRA Weights", command=on_lora_weights_click)
    generate_button.place(x=window_width - button_width, y=y, width=button_width, height=textbox_height * 2)

    y += textbox_height

    negative_prompt_label = tk.Label(window, text="Negative Prompt")
    negative_prompt_label.place(x=0, y=y)

    negative_prompt_textbox = tk.Entry(window)
    negative_prompt_textbox.insert(tk.END, negative_prompt)
    negative_prompt_textbox.place(
        x=label_width, y=y, width=window_width - button_width * 2 - label_width, height=textbox_height
    )

    window.mainloop()


def run_inference(
    optimized_model_dir,
    prompt,
    negative_prompt,
    lora_weights_path,
    lora_scale,
    num_images,
    batch_size,
    image_size,
    num_inference_steps,
    static_dims,
    interactive,
):
    ort.set_default_logger_severity(3)

    print("Loading models into ORT session...")
    sess_options = ort.SessionOptions()
    sess_options.enable_mem_pattern = False

    if lora_weights_path is not None:
        lora_weights, alpha, rank = read_lora_weights(lora_weights_path)
        sess_options.add_initializer("lora_network_alpha_per_rank", np.array(alpha / rank, dtype=np.float16))

        for weight_name, weight_value in lora_weights.items():
            if "lora" not in weight_name:
                continue

            if isinstance(weight_value, torch.Tensor):
                weight_value = weight_value.numpy()

            new_values = np.transpose(weight_value.astype(np.float16))

            if weight_name.startswith("unet."):
                sess_options.add_initializer(weight_name, new_values)
            elif weight_name.startswith("text_encoder."):
                sess_options.add_initializer(weight_name, new_values)

        sess_options.add_initializer("lora_scale", np.array(lora_scale, dtype=np.float16))

    if static_dims:
        # Not necessary, but helps DML EP further optimize runtime performance.
        # batch_size is doubled for sample & hidden state because of classifier free guidance:
        # https://github.com/huggingface/diffusers/blob/46c52f9b9607e6ecb29c782c052aea313e6487b7/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L672
        sess_options.add_free_dimension_override_by_name("unet_sample_batch", batch_size * 2)
        sess_options.add_free_dimension_override_by_name("unet_sample_channels", 4)
        sess_options.add_free_dimension_override_by_name("unet_sample_height", image_size // 8)
        sess_options.add_free_dimension_override_by_name("unet_sample_width", image_size // 8)
        sess_options.add_free_dimension_override_by_name("unet_time_batch", 1)
        sess_options.add_free_dimension_override_by_name("unet_hidden_batch", batch_size * 2)
        sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)

    pipeline = OnnxStableDiffusionPipeline.from_pretrained(
        optimized_model_dir, provider="DmlExecutionProvider", sess_options=sess_options
    )

    if interactive:
        run_inference_gui(
            pipeline,
            prompt,
            negative_prompt,
            num_images,
            batch_size,
            image_size,
            num_inference_steps,
        )
    else:
        run_inference_loop(
            pipeline,
            prompt,
            negative_prompt,
            num_images,
            batch_size,
            image_size,
            num_inference_steps,
        )


def optimize(
    model_id: str,
    unoptimized_model_dir: Path,
    optimized_model_dir: Path,
):
    from google.protobuf import __version__ as protobuf_version

    # protobuf 4.x aborts with OOM when optimizing unet
    if version.parse(protobuf_version) > version.parse("3.20.3"):
        print("This script requires protobuf 3.20.3. Please ensure your package version matches requirements.txt.")
        exit(1)

    ort.set_default_logger_severity(4)
    script_dir = Path(__file__).resolve().parent

    # Clean up previously optimized models, if any.
    shutil.rmtree(script_dir / "footprints", ignore_errors=True)
    shutil.rmtree(unoptimized_model_dir, ignore_errors=True)
    shutil.rmtree(optimized_model_dir, ignore_errors=True)

    # The model_id and base_model_id are identical when optimizing a standard stable diffusion model like
    # runwayml/stable-diffusion-v1-5. These variables are only different when optimizing a LoRA variant.
    base_model_id = get_base_model_name(model_id)

    # Load the entire PyTorch pipeline to ensure all models and their configurations are downloaded and cached.
    # This avoids an issue where the non-ONNX components (tokenizer, scheduler, and feature extractor) are not
    # automatically cached correctly if individual models are fetched one at a time.
    print("Download stable diffusion PyTorch pipeline...")
    if model_id.endswith(".safetensors"):
        pipeline = StableDiffusionPipeline.from_single_file(base_model_id, torch_dtype=torch.float32)
    else:
        print("Download stable diffusion PyTorch pipeline...")
        pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32)

    model_info = dict()

    submodel_names = ["vae_encoder", "vae_decoder", "unet"]

    has_text_encoder = getattr(pipeline, "text_encoder", None) is not None
    has_text_encoder_2 = getattr(pipeline, "text_encoder_2", None) is not None
    has_safety_checker = getattr(pipeline, "safety_checker", None) is not None

    if has_text_encoder:
        submodel_names.append("text_encoder")

    if has_text_encoder_2:
        submodel_names.append("text_encoder_2")

    if has_safety_checker:
        submodel_names.append("safety_checker")

    for submodel_name in submodel_names:
        print(f"\nOptimizing {submodel_name}")

        olive_config = None
        with open(script_dir / f"config_{submodel_name}.json", "r") as fin:
            olive_config = json.load(fin)

        if submodel_name in ("unet", "text_encoder"):
            olive_config["input_model"]["config"]["model_path"] = model_id
        else:
            # Only the unet & text encoder are affected by LoRA, so it's better to use the base model ID for
            # other models: the Olive cache is based on the JSON config, and two LoRA variants with the same
            # base model ID should be able to reuse previously optimized copies.
            olive_config["input_model"]["config"]["model_path"] = base_model_id

        olive_run(olive_config)

        footprints_file_path = (
            Path(__file__).resolve().parent / "footprints" / f"{submodel_name}_gpu-dml_footprints.json"
        )
        with footprints_file_path.open("r") as footprint_file:
            footprints = json.load(footprint_file)

            conversion_footprint = None
            optimizer_footprint = None
            for _, footprint in footprints.items():
                if footprint["from_pass"] == "OnnxConversion":
                    conversion_footprint = footprint
                elif footprint["from_pass"] == "OrtTransformersOptimization":
                    optimizer_footprint = footprint

            assert conversion_footprint and optimizer_footprint

            unoptimized_olive_model = ONNXModel(**conversion_footprint["model_config"]["config"])
            optimized_olive_model = ONNXModel(**optimizer_footprint["model_config"]["config"])

            # LoRA support is still in the experimentation phase, so we do this post-processing over here
            if (
                config.lora_weights_strategy == "inserted"
                and submodel_name == "unet"
                or submodel_name == "text_encoder"
            ):
                base_model = OnnxModel(load_model(optimized_olive_model.model_path))
                lora_weights_renamer = LoraWeightsRenamer(base_model, submodel_name)
                lora_weights_renamer.apply()
                save_model(base_model.model, optimized_olive_model.model_path)

            model_info[submodel_name] = {
                "unoptimized": {
                    "path": Path(unoptimized_olive_model.model_path),
                },
                "optimized": {
                    "path": Path(optimized_olive_model.model_path),
                },
            }

            print(f"Unoptimized Model : {model_info[submodel_name]['unoptimized']['path']}")
            print(f"Optimized Model   : {model_info[submodel_name]['optimized']['path']}")

    # Save the unoptimized models in a directory structure that the diffusers library can load and run.
    # This is optional, and the optimized models can be used directly in a custom pipeline if desired.
    print("\nCreating ONNX pipeline...")

    if has_safety_checker:
        safety_checker = OnnxRuntimeModel.from_pretrained(model_info["safety_checker"]["unoptimized"]["path"].parent)
    else:
        safety_checker = None

    # TODO: Enable inference with XL models when available
    # XL model inference is not supported yet because the diffusers library doesn't have an ONNX pipeline for it
    if has_text_encoder:
        onnx_pipeline = OnnxStableDiffusionPipeline(
            vae_encoder=OnnxRuntimeModel.from_pretrained(model_info["vae_encoder"]["unoptimized"]["path"].parent),
            vae_decoder=OnnxRuntimeModel.from_pretrained(model_info["vae_decoder"]["unoptimized"]["path"].parent),
            text_encoder=OnnxRuntimeModel.from_pretrained(model_info["text_encoder"]["unoptimized"]["path"].parent),
            tokenizer=pipeline.tokenizer,
            unet=OnnxRuntimeModel.from_pretrained(model_info["unet"]["unoptimized"]["path"].parent),
            scheduler=pipeline.scheduler,
            safety_checker=safety_checker,
            feature_extractor=pipeline.feature_extractor,
            requires_safety_checker=True,
        )

        print("Saving unoptimized models...")
        onnx_pipeline.save_pretrained(unoptimized_model_dir)

        # Create a copy of the unoptimized model directory, then overwrite with optimized models from the olive cache.
        print("Copying optimized models...")
        shutil.copytree(unoptimized_model_dir, optimized_model_dir, ignore=shutil.ignore_patterns("weights.pb"))
        for submodel_name in submodel_names:
            src_path = model_info[submodel_name]["optimized"]["path"]
            dst_path = optimized_model_dir / submodel_name / "model.onnx"
            shutil.copyfile(src_path, dst_path)
    else:
        # For XL models, just copy the optimized models for now without the whole structure
        print("Copying optimized models...")
        for submodel_name in submodel_names:
            src_path = model_info[submodel_name]["optimized"]["path"]
            dst_path = optimized_model_dir / submodel_name / "model.onnx"
            os.makedirs(optimized_model_dir / submodel_name, exist_ok=True)
            shutil.copyfile(src_path, dst_path)

    print(f"The optimized pipeline is located here: {optimized_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5", type=str)
    parser.add_argument("--interactive", action="store_true", help="Run with a GUI")
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--clean_cache", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--test_unoptimized", action="store_true", help="Use unoptimized model for inference")
    parser.add_argument(
        "--prompt",
        default=(
            "castle surrounded by water and nature, village, volumetric lighting, photorealistic, "
            "detailed and intricate, fantasy, epic cinematic shot, mountains, 8k ultra hd"
        ),
        type=str,
    )
    parser.add_argument(
        "--negative_prompt",
        default="",
        type=str,
    )
    parser.add_argument(
        "--lora_weights",
        default=None,
        type=str,
        help="Path to the .bin or .safetensors file containing the LoRA weights to add to the model.",
    )
    parser.add_argument(
        "--lora_weights_strategy",
        choices=["inserted", "folded"],
        default=None,
        type=str,
        help="Strategy to use when adding the LoRA weights. `folded` means that the weights will be completely "
        "merged with the original model's weights. Doing so can improve performance but makes it impossible "
        "to change the weights after the model has been optimized. `inserted` means that the weights will be "
        "inserted in the model as initializers. This strategy has worse performance than `folded` due to "
        "the additional GEMM operations, but adds the flexibility of being able to update the LoRA weights in the "
        "at session creation, independently of the default weights. Note that graph optimizations could be leveraged "
        "in onnxruntime to fold the inserted weights at runtime and therefore make the `inserted` option's performance "
        "similar to the `baked` option.",
    )
    parser.add_argument(
        "--lora_scale",
        default=1.0,
        type=float,
    )
    parser.add_argument("--num_images", default=1, type=int, help="Number of images to generate")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of images to generate per batch")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="Number of steps in diffusion process")
    parser.add_argument(
        "--static_dims",
        action="store_true",
        help="DEPRECATED (now enabled by default). Use --dynamic_dims to disable static_dims.",
    )
    parser.add_argument("--dynamic_dims", action="store_true", help="Disable static shape optimization")
    args = parser.parse_args()

    if args.static_dims:
        print(
            "WARNING: the --static_dims option is deprecated, and static shape optimization is enabled by default. "
            "Use --dynamic_dims to disable static shape optimization."
        )

    model_to_image_size = {
        "CompVis/stable-diffusion-v1-4": 512,
        "runwayml/stable-diffusion-v1-5": 512,
        "sayakpaul/sd-model-finetuned-lora-t4": 512,
        "stabilityai/stable-diffusion-2": 768,
        "stabilityai/stable-diffusion-2-base": 768,
        "stabilityai/stable-diffusion-2-1": 768,
        "stabilityai/stable-diffusion-2-1-base": 768,
        "stabilityai/stable-diffusion-xl-refiner-0.9": 1024,
        "rhendz/niji-lora": 512,
        "sayakpaul/lora-trained": 512,
    }

    if args.model_id not in list(model_to_image_size.keys()):
        print(
            f"WARNING: {args.model_id} is not an officially supported model for this example and may not work as "
            + "expected."
        )

    if version.parse(ort.__version__) < version.parse("1.15.0"):
        print("This script requires onnxruntime-directml 1.15.0 or newer")
        exit(1)

    script_dir = Path(__file__).resolve().parent

    if args.model_id.endswith(".safetensors"):
        suffix = Path(args.model_id).stem
        unoptimized_model_dir = script_dir / "models" / "unoptimized" / suffix
        optimized_model_dir = script_dir / "models" / "optimized" / suffix
    else:
        unoptimized_model_dir = script_dir / "models" / "unoptimized" / args.model_id
        optimized_model_dir = script_dir / "models" / "optimized" / args.model_id

    if args.clean_cache:
        shutil.rmtree(script_dir / "cache", ignore_errors=True)

    config.image_size = model_to_image_size.get(args.model_id, 512)
    config.lora_weights_file = args.lora_weights
    config.lora_weights_strategy = args.lora_weights_strategy

    if args.optimize or not optimized_model_dir.exists():
        # TODO: clean up warning filter (mostly during conversion from torch to ONNX)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimize(args.model_id, unoptimized_model_dir, optimized_model_dir)

    xl_models = [
        "stabilityai/stable-diffusion-xl-refiner-0.9",
    ]

    is_xl_model = args.model_id in xl_models

    # TODO: Enable inference with XL models when available
    # XL model inference is not supported yet because the diffusers library doesn't have an ONNX pipeline for it
    if not args.optimize and not is_xl_model:
        model_dir = unoptimized_model_dir if args.test_unoptimized else optimized_model_dir
        use_static_dims = not args.dynamic_dims

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_inference(
                model_dir,
                args.prompt,
                args.negative_prompt,
                args.lora_weights,
                args.lora_scale,
                args.num_images,
                args.batch_size,
                config.image_size,
                args.num_inference_steps,
                use_static_dims,
                args.interactive,
            )
