#!/usr/bin/env python3
"""Generate a video from a reference image and text prompt using Wan 2.1 (image-to-video)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "video_generation"))

from caption_jobs import CaptionJob, add_caption_batch_args, collect_caption_jobs
from tqdm import tqdm
from wan_common import (
    DEFAULT_I2V_MODEL_PATH,
    add_generation_args,
    build_prompt,
    extract_frame_from_video,
    load_reference_image,
    load_wan_i2v_pipeline,
    require_generation_input,
    resolve_model_source,
    resolve_torch_device,
    slugify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate video from a reference image and text prompt with Wan 2.1 I2V. "
            "Provide --image for a still frame, --video to extract a frame, "
            "or --captions-root to batch from scene captions."
        ),
    )
    add_generation_args(parser)
    add_caption_batch_args(parser)
    parser.set_defaults(model_path=DEFAULT_I2V_MODEL_PATH)

    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a reference image (PNG/JPG).",
    )
    source.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to the original video; a frame is extracted as the reference image.",
    )
    parser.add_argument(
        "--frame-time",
        type=float,
        default=0.0,
        help="Timestamp in seconds when extracting a frame from --video (default: 0.0).",
    )
    parser.add_argument(
        "--save-reference",
        action="store_true",
        help="Save the extracted reference frame next to the output video.",
    )
    return parser.parse_args()


def output_path_for_job(output_dir: Path, job: CaptionJob) -> Path:
    video_dir = output_dir / job.video_title
    return video_dir / f"{job.output_stem}_wan_i2v.mp4"


def generate_video(pipe, args: argparse.Namespace, image, prompt: str, seed: int):
    import torch

    generator_device = "cuda" if resolve_torch_device(args.device).type == "cuda" else "cpu"
    return pipe(
        image=image,
        prompt=build_prompt(prompt),
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device=generator_device).manual_seed(seed),
    ).frames[0]


def run_single(args: argparse.Namespace, model_source: str, output_dir: Path) -> dict:
    from diffusers.utils import export_to_video

    if not args.image and not args.video:
        raise SystemExit("Provide --image, --video, or --captions-root.")

    image_path = Path(args.image).expanduser().resolve() if args.image else None
    video_path = Path(args.video).expanduser().resolve() if args.video else None

    if image_path is not None and not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    if video_path is not None and not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    device = resolve_torch_device(args.device)
    prompt = build_prompt(args.prompt)
    image = load_reference_image(image_path, video_path, args.frame_time)

    print(f"Loading Wan 2.1 I2V pipeline from {model_source}")
    pipe = load_wan_i2v_pipeline(model_source, device)
    if args.cpu_offload and device.type == "cuda":
        pipe.enable_model_cpu_offload()

    video = generate_video(pipe, args, image, prompt, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_name or slugify(args.prompt)
    output_path = output_dir / f"{stem}_wan_i2v.mp4"
    export_to_video(video, str(output_path), fps=args.fps)
    print(f"Saved video to {output_path}")

    if args.save_reference and video_path is not None:
        ref_path = output_path.with_name(f"{stem}_reference.jpg")
        image.save(ref_path)
        print(f"Saved reference frame to {ref_path}")

    return {"generated": 1, "skipped": 0, "failed": 0, "outputs": [str(output_path)]}


def run_batch_from_captions(
    args: argparse.Namespace, model_source: str, output_dir: Path
) -> dict:
    from diffusers.utils import export_to_video

    captions_root = Path(args.captions_root).expanduser().resolve()
    scene_ids = set(args.scene_ids) if args.scene_ids else None
    jobs = collect_caption_jobs(
        captions_root,
        video_title=args.video_title,
        scene_ids=scene_ids,
    )

    pending: list[CaptionJob] = []
    skipped = 0
    for job in jobs:
        target = output_path_for_job(output_dir, job)
        if args.skip_existing and target.is_file():
            skipped += 1
            continue
        pending.append(job)

    print(f"Found {len(jobs)} captioned scene(s); generating {len(pending)}, skipping {skipped}.")
    if not pending:
        return {"generated": 0, "skipped": skipped, "failed": 0, "outputs": []}

    device = resolve_torch_device(args.device)
    print(f"Loading Wan 2.1 I2V pipeline from {model_source}")
    pipe = load_wan_i2v_pipeline(model_source, device)
    if args.cpu_offload and device.type == "cuda":
        pipe.enable_model_cpu_offload()

    generated = 0
    failed = 0
    outputs: list[str] = []
    results: list[dict] = []

    for job in tqdm(pending, desc="Wan I2V from captions", unit="scene"):
        target = output_path_for_job(output_dir, job)
        target.parent.mkdir(parents=True, exist_ok=True)
        seed = args.seed + job.scene_id

        try:
            if job.reference_video is None:
                raise ValueError(f"Scene {job.scene_id} has no reference video.")
            image = extract_frame_from_video(job.reference_video, job.frame_time)
            if args.save_reference:
                image.save(target.with_name(f"{job.output_stem}_reference.jpg"))

            video = generate_video(pipe, args, image, job.prompt, seed)
            export_to_video(video, str(target), fps=args.fps)
            generated += 1
            outputs.append(str(target))
            results.append(
                {
                    "video_title": job.video_title,
                    "scene_id": job.scene_id,
                    "prompt": job.prompt,
                    "reference_video": str(job.reference_video),
                    "output_path": str(target),
                    "status": "success",
                }
            )
        except Exception as exc:
            failed += 1
            results.append(
                {
                    "video_title": job.video_title,
                    "scene_id": job.scene_id,
                    "prompt": job.prompt,
                    "output_path": str(target),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"Failed {job.video_title} scene {job.scene_id}: {exc}")

    manifest = {
        "mode": "wan_i2v",
        "model_source": model_source,
        "captions_root": str(captions_root),
        "output_dir": str(output_dir),
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    manifest_path = output_dir / "wan_i2v_generation_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"Wrote manifest to {manifest_path}")

    return {
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "outputs": outputs,
    }


def main() -> int:
    args = parse_args()
    require_generation_input(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    hf_cache = Path(args.hf_cache).expanduser().resolve()
    model_source = resolve_model_source(args.model_id, args.model_path, hf_cache)

    if args.captions_root:
        summary = run_batch_from_captions(args, model_source, output_dir)
    else:
        summary = run_single(args, model_source, output_dir)

    print(
        "Done. "
        f"generated={summary['generated']} "
        f"skipped={summary['skipped']} "
        f"failed={summary['failed']}"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
