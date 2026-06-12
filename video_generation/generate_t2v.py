#!/usr/bin/env python3
"""Generate a video from a text prompt using SANA-Video (text-to-video)."""

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
    require_generation_input,
    resolve_torch_device,
    slugify,
)
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate video from a text prompt with SANA-Video.",
    )
    add_generation_args(parser)
    add_caption_batch_args(parser)
    return parser.parse_args()


def output_path_for_job(output_dir: Path, job: CaptionJob) -> Path:
    video_dir = output_dir / job.video_title
    return video_dir / f"{job.output_stem}_t2v.mp4"


def run_single(args: argparse.Namespace, model_path: Path, output_dir: Path) -> dict:
    import torch
    from diffusers import SanaVideoPipeline
    from diffusers.utils import export_to_video

    device = resolve_torch_device(args.device)
    prompt = build_prompt(args.prompt, args.motion_score)

    print(f"Loading SANA-Video T2V pipeline from {model_path}")
    pipe = SanaVideoPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
    )
    pipe.vae.to(torch.float32)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.to(str(device))

    print("Generating video...")
    print(f"  Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(
        f"  Size: {args.width}x{args.height}, frames={args.frames}, "
        f"steps={args.num_inference_steps}, guidance={args.guidance_scale}, seed={args.seed}"
    )

    generator_device = "cuda" if device.type == "cuda" else "cpu"
    result = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        frames=args.frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device=generator_device).manual_seed(args.seed),
    )
    video = result.frames[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_name or slugify(args.prompt)
    output_path = output_dir / f"{stem}_t2v.mp4"
    export_to_video(video, str(output_path), fps=args.fps)
    print(f"Saved video to {output_path}")
    return {"generated": 1, "skipped": 0, "failed": 0, "outputs": [str(output_path)]}


def run_batch_from_captions(
    args: argparse.Namespace, model_path: Path, output_dir: Path
) -> dict:
    import torch
    from diffusers import SanaVideoPipeline
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
    print(f"Loading SANA-Video T2V pipeline from {model_path}")
    pipe = SanaVideoPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
    )
    pipe.vae.to(torch.float32)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.to(str(device))

    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generated = 0
    failed = 0
    outputs: list[str] = []
    results: list[dict] = []

    for job in tqdm(pending, desc="T2V from captions", unit="scene"):
        target = output_path_for_job(output_dir, job)
        target.parent.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(job.prompt, args.motion_score)
        seed = args.seed + job.scene_id

        try:
            result = pipe(
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
        "mode": "t2v",
        "captions_root": str(captions_root),
        "output_dir": str(output_dir),
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    manifest_path = output_dir / "t2v_generation_manifest.json"
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
