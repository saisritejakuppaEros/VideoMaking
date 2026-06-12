#!/usr/bin/env python3
"""Generate a video from a reference image and text prompt using SANA-Video (image-to-video)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from caption_jobs import CaptionJob, add_caption_batch_args, collect_caption_jobs
from sana_common import (
    add_generation_args,
    build_prompt,
    ensure_model,
    extract_frame_from_video,
    load_reference_image,
    require_generation_input,
    resolve_torch_device,
    slugify,
)
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate video from a reference image and text prompt with SANA-Video. "
            "Provide --image for a still frame, --video to extract a frame, "
            "or --captions-root to batch from scene captions."
        ),
    )
    add_generation_args(parser)
    add_caption_batch_args(parser)

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
    return video_dir / f"{job.output_stem}_i2v.mp4"


def run_single(args: argparse.Namespace, model_path: Path, output_dir: Path) -> dict:
    import torch
    from diffusers import SanaImageToVideoPipeline
    from diffusers.utils import export_to_video

    if not args.image and not args.video:
        raise SystemExit("Provide --image, --video, --captions-root, or --download-only.")

    image_path = Path(args.image).expanduser().resolve() if args.image else None
    video_path = Path(args.video).expanduser().resolve() if args.video else None

    if image_path is not None and not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    if video_path is not None and not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    device = resolve_torch_device(args.device)
    prompt = build_prompt(args.prompt, args.motion_score)
    image = load_reference_image(image_path, video_path, args.frame_time)

    print(f"Loading SANA-Video I2V pipeline from {model_path}")
    pipe = SanaImageToVideoPipeline.from_pretrained(str(model_path))
    pipe.transformer.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(torch.float32)
    pipe.to(str(device))

    generator_device = "cuda" if device.type == "cuda" else "cpu"
    result = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        frames=args.frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device=generator_device).manual_seed(args.seed),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_name or slugify(args.prompt)
    output_path = output_dir / f"{stem}_i2v.mp4"
    export_to_video(result.frames[0], str(output_path), fps=args.fps)

    if args.save_reference and video_path is not None:
        ref_path = output_dir / f"{stem}_reference.jpg"
        image.save(ref_path)
        print(f"Saved reference frame to {ref_path}")

    print(f"Saved video to {output_path}")
    return {"generated": 1, "skipped": 0, "failed": 0, "outputs": [str(output_path)]}


def run_batch_from_captions(
    args: argparse.Namespace, model_path: Path, output_dir: Path
) -> dict:
    import torch
    from diffusers import SanaImageToVideoPipeline
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
        if job.reference_video is None:
            skipped += 1
            continue
        target = output_path_for_job(output_dir, job)
        if args.skip_existing and target.is_file():
            skipped += 1
            continue
        pending.append(job)

    print(f"Found {len(jobs)} captioned scene(s); generating {len(pending)}, skipping {skipped}.")
    if not pending:
        return {"generated": 0, "skipped": skipped, "failed": 0, "outputs": []}

    device = resolve_torch_device(args.device)
    print(f"Loading SANA-Video I2V pipeline from {model_path}")
    pipe = SanaImageToVideoPipeline.from_pretrained(str(model_path))
    pipe.transformer.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(torch.float32)
    pipe.to(str(device))

    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generated = 0
    failed = 0
    outputs: list[str] = []
    results: list[dict] = []

    for job in tqdm(pending, desc="I2V from captions", unit="scene"):
        target = output_path_for_job(output_dir, job)
        target.parent.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(job.prompt, args.motion_score)
        seed = args.seed + job.scene_id

        try:
            image = extract_frame_from_video(job.reference_video, job.frame_time)
            if args.save_reference:
                image.save(target.with_name(f"{job.output_stem}_reference.jpg"))

            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                frames=args.frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=torch.Generator(device=generator_device).manual_seed(seed),
            )
            export_to_video(result.frames[0], str(target), fps=args.fps)
            generated += 1
            outputs.append(str(target))
            results.append(
                {
                    "video_title": job.video_title,
                    "scene_id": job.scene_id,
                    "prompt": job.prompt,
                    "reference_video": str(job.reference_video),
                    "frame_time": job.frame_time,
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
                    "reference_video": str(job.reference_video),
                    "frame_time": job.frame_time,
                    "output_path": str(target),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"Failed {job.video_title} scene {job.scene_id}: {exc}")

    manifest = {
        "mode": "i2v",
        "captions_root": str(captions_root),
        "output_dir": str(output_dir),
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    manifest_path = output_dir / "i2v_generation_manifest.json"
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
    models_dir = Path(args.models_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = ensure_model(args.model_id, models_dir)

    if args.download_only:
        print(f"Model ready at {model_path}")
        return 0

    if args.captions_root:
        summary = run_batch_from_captions(args, model_path, output_dir)
    else:
        summary = run_single(args, model_path, output_dir)

    print(
        "Done. "
        f"generated={summary['generated']} "
        f"skipped={summary['skipped']} "
        f"failed={summary['failed']}"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
