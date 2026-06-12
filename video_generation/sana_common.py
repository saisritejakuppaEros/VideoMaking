#!/usr/bin/env python3
"""Shared helpers for SANA-Video text-to-video and image-to-video generation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "output"

DEFAULT_MODEL_ID = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
DEFAULT_NEGATIVE_PROMPT = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
    "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
    "frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience."
)

DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_FRAMES = 81
DEFAULT_FPS = 16
DEFAULT_GUIDANCE_SCALE = 6.0
DEFAULT_INFERENCE_STEPS = 50
DEFAULT_MOTION_SCORE = 30
DEFAULT_SEED = 42


def model_local_dir(model_id: str, models_dir: Path) -> Path:
    """Map a Hugging Face repo id to a local models/ subdirectory."""
    safe_name = model_id.split("/")[-1] if "/" in model_id else model_id
    return models_dir / safe_name


def ensure_model(model_id: str, models_dir: Path) -> Path:
    """Download SANA-Video weights into models/ if not already present."""
    from huggingface_hub import snapshot_download

    local_dir = model_local_dir(model_id, models_dir)
    if (local_dir / "model_index.json").exists():
        print(f"Using cached model: {local_dir}")
        return local_dir

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} -> {local_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return local_dir


def slugify(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug[:max_length] or "video"


def build_prompt(prompt: str, motion_score: int) -> str:
    motion_prompt = f" motion score: {motion_score}."
    if motion_prompt.strip().lower() in prompt.lower():
        return prompt
    return prompt + motion_prompt


def resolve_torch_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")
    return resolved


def extract_frame_from_video(video_path: Path, time_seconds: float = 0.0):
    """Extract a single RGB frame from a video at the given timestamp."""
    import cv2
    from PIL import Image

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


def load_reference_image(image_path: Path | None, video_path: Path | None, frame_time: float):
    from diffusers.utils import load_image

    if image_path is not None:
        return load_image(str(image_path))

    if video_path is None:
        raise ValueError("Provide either --image or --video.")

    print(f"Extracting reference frame at {frame_time:.3f}s from {video_path}")
    return extract_frame_from_video(video_path, frame_time)


def add_generation_args(parser):
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download the model into models/ and exit.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt describing the video to generate (not required with --download-only).",
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
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--models-dir",
        default=str(MODELS_DIR),
        help=f"Directory to store downloaded models (default: {MODELS_DIR}).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda). Use 'auto' to pick cuda/cpu.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--motion-score",
        type=int,
        default=DEFAULT_MOTION_SCORE,
        help="Motion score appended to the prompt.",
    )
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
    if args.download_only or args.captions_root:
        return
    if not args.prompt:
        raise SystemExit(
            "--prompt is required unless --captions-root or --download-only is set."
        )
