#!/usr/bin/env python3
"""Shared helpers for Wan 2.1 text-to-video and image-to-video generation."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

DEFAULT_HF_CACHE = "/mnt/data0/parth/hf_models_cache"
DEFAULT_T2V_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_I2V_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
DEFAULT_I2V_MODEL_PATH = (
    "/mnt/data0/parth/hf_models_cache/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers"
)
DEFAULT_T2V_MODEL_PATH = (
    "/mnt/data0/parth/hf_models_cache/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
)

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_FRAMES = 81
DEFAULT_FPS = 16
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_INFERENCE_STEPS = 50
DEFAULT_SEED = 42


def configure_hf_cache(cache_dir: str | Path) -> Path:
    cache_path = Path(cache_dir).expanduser().resolve()
    os.environ.setdefault("HF_HOME", str(cache_path))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_path))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path))
    return cache_path


def resolve_hf_hub_cache_dir(model_dir: Path) -> Path:
    """Resolve a diffusers dir or HF hub models--* cache folder to a snapshot path."""
    model_dir = model_dir.expanduser().resolve()
    if (model_dir / "model_index.json").exists():
        return model_dir

    snapshots_dir = model_dir / "snapshots"
    if snapshots_dir.is_dir():
        candidates = sorted(
            child
            for child in snapshots_dir.iterdir()
            if child.is_dir() and (child / "model_index.json").exists()
        )
        if candidates:
            return candidates[-1]

    raise FileNotFoundError(
        f"Could not find model_index.json under {model_dir} or its snapshots/"
    )


def resolve_model_source(model_id: str, model_path: str | None, hf_cache: Path) -> str:
    if model_path:
        resolved = resolve_hf_hub_cache_dir(Path(model_path))
        print(f"Using local model path: {resolved}")
        return str(resolved)

    configure_hf_cache(hf_cache)
    print(f"Using Hugging Face model id from cache: {model_id}")
    return model_id


def slugify(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug[:max_length] or "video"


def build_prompt(prompt: str) -> str:
    return prompt.strip()


def resolve_torch_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")
    return resolved


def load_wan_t2v_pipeline(model_source: str, device):
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    vae = AutoencoderKLWan.from_pretrained(
        model_source,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanPipeline.from_pretrained(
        model_source,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    if device.type == "cuda":
        pipe.to(str(device))
    else:
        pipe.enable_model_cpu_offload()
    return pipe


def load_wan_i2v_pipeline(model_source: str, device):
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    vae = AutoencoderKLWan.from_pretrained(
        model_source,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_source,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    if device.type == "cuda":
        pipe.to(str(device))
    else:
        pipe.enable_model_cpu_offload()
    return pipe


def extract_frame_from_video(video_path: Path, time_seconds: float = 0.0):
    from PIL import Image

    try:
        import cv2
    except ImportError:
        cv2 = None

    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            if time_seconds > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(
                    f"Could not read frame at {time_seconds:.3f}s from {video_path}"
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        finally:
            cap.release()

    import imageio.v3 as iio

    frame_idx = 0
    if time_seconds > 0:
        meta = iio.immeta(video_path, plugin="pyav")
        fps = float(meta.get("fps") or 24.0)
        frame_idx = max(0, int(time_seconds * fps))

    frame = iio.imread(video_path, index=frame_idx, plugin="pyav")
    return Image.fromarray(frame)


def load_reference_image(image_path: Path | None, video_path: Path | None, frame_time: float):
    from diffusers.utils import load_image

    if image_path is not None:
        return load_image(str(image_path))

    if video_path is None:
        raise ValueError("Provide either --image or --video.")

    print(f"Extracting reference frame at {frame_time:.3f}s from {video_path}")
    return extract_frame_from_video(video_path, frame_time)


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt describing the video to generate.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for generation.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename stem (without extension). Defaults to a slug from the prompt.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"Directory for generated videos (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_T2V_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_T2V_MODEL_ID}).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local diffusers model directory. Overrides --model-id.",
    )
    parser.add_argument(
        "--hf-cache",
        default=DEFAULT_HF_CACHE,
        help=f"Hugging Face cache directory (default: {DEFAULT_HF_CACHE}).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda). Use 'auto' to pick cuda/cpu.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Force model CPU offload even when CUDA is available.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=DEFAULT_INFERENCE_STEPS,
        help="Number of denoising steps.",
    )


def require_generation_input(args: argparse.Namespace) -> None:
    if args.captions_root:
        return
    if not args.prompt:
        raise SystemExit("--prompt is required unless --captions-root is set.")
