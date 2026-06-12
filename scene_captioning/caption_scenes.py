#!/usr/bin/env python3
"""Caption scene clips with Qwen3-VL loaded locally from models/."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

# torchvision 0.26+ removed io.read_video; prefer decord for qwen-vl-utils.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "captions"
MODELS_DIR = SCRIPT_DIR / "models"

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MIN_SCENE_CLIP_BYTES = 1024
MAX_FALLBACK_FRAMES = 16

TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:\.(\d+))?$")

STORYBOARD_PROMPT = """You are a professional storyboard artist.

Analyze this video scene and return ONLY valid JSON with these keys:
{
  "scene_summary": "",
  "shot_type": "",
  "camera": "",
  "characters": "",
  "action": "",
  "objects": "",
  "environment": "",
  "visual_style": "",
  "emotion": ""
}

Include people, actions, objects, camera movement, and environment details."""

DETAILED_PROMPT = """Describe this video scene in detail.

Include:
- people
- actions
- objects
- camera movement
- environment

Be concise but specific."""


def parse_time(value: str | None) -> float | None:
    if value is None:
        return None

    if re.fullmatch(r"\d+(\.\d+)?", value):
        return float(value)

    match = TIME_RE.match(value.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid time format: {value!r}. Use seconds or HH:MM:SS."
        )

    hours, minutes, seconds, millis = match.groups()
    total = int(minutes) * 60 + int(seconds)
    if hours:
        total += int(hours) * 3600
    if millis:
        total += float(f"0.{millis}")
    return float(total)


def load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def scene_in_range(
    scene: dict,
    start_seconds: float | None,
    end_seconds: float | None,
) -> bool:
    scene_start = scene["start_seconds"]
    scene_end = scene["end_seconds"]

    if start_seconds is not None and scene_end <= start_seconds:
        return False
    if end_seconds is not None and scene_start >= end_seconds:
        return False
    return True


def select_scenes(
    manifest: dict,
    *,
    start_seconds: float | None,
    end_seconds: float | None,
    scene_ids: set[int] | None,
) -> list[dict]:
    scenes = manifest.get("scenes", [])
    selected = []
    for scene in scenes:
        if scene_ids and scene["scene_id"] not in scene_ids:
            continue
        if not scene_in_range(scene, start_seconds, end_seconds):
            continue
        if not scene.get("clip_path"):
            print(
                f"Skipping scene {scene['scene_id']}: missing clip_path. "
                "Run scene_cut.py without --no-split first."
            )
            continue
        clip_path = Path(scene["clip_path"])
        if not clip_path.exists():
            print(
                f"Skipping scene {scene['scene_id']}: "
                f"missing clip file {scene['clip_path']}"
            )
            continue
        try:
            clip_size = clip_path.stat().st_size
        except OSError as exc:
            print(f"Skipping scene {scene['scene_id']}: cannot read clip ({exc})")
            continue
        if clip_size < MIN_SCENE_CLIP_BYTES:
            print(
                f"Skipping scene {scene['scene_id']}: "
                f"clip too small to decode ({clip_size} bytes)"
            )
            continue
        selected.append(scene)
    return selected


def model_local_dir(model_id: str, models_dir: Path) -> Path:
    return models_dir / model_id.split("/")[-1]


def ensure_model(model_id: str, models_dir: Path) -> Path:
    """Download Qwen3-VL weights into models/ if not already present."""
    from huggingface_hub import snapshot_download

    local_dir = model_local_dir(model_id, models_dir)
    if (local_dir / "config.json").exists():
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


def resolve_torch_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build, e.g.:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        )
    return resolved


def get_model_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def load_qwen3_vl(
    model_path: Path,
    *,
    device: str,
    attn_implementation: str | None,
):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_device = resolve_torch_device(device)
    model_kwargs: dict = {}
    if torch_device.type == "cuda":
        model_kwargs["dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = torch.float32
        model_kwargs["device_map"] = "cpu"

    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    print(f"Loading model from {model_path}")
    if torch_device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(torch_device.index or 0)}")

    model = AutoModelForImageTextToText.from_pretrained(str(model_path), **model_kwargs)
    processor = AutoProcessor.from_pretrained(str(model_path))
    model_device = get_model_device(model)
    print(
        f"Model ready on {model_device} "
        f"(torch {torch.__version__}, cuda={torch.cuda.is_available()})"
    )
    return model, processor


def build_video_messages(
    clip_path: Path,
    prompt: str,
    video_fps: float | None,
    *,
    frame_urls: list[str] | None = None,
) -> list[dict]:
    if frame_urls is not None:
        video_content: dict = {
            "type": "video",
            "video": frame_urls,
        }
        if video_fps is not None:
            video_content["sample_fps"] = str(video_fps)
    else:
        video_content = {
            "type": "video",
            "video": f"file://{clip_path.resolve()}",
        }
        if video_fps is not None:
            video_content["fps"] = video_fps

    return [
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": prompt},
            ],
        }
    ]


def extract_frame_file_urls(
    clip_path: Path,
    *,
    video_fps: float,
    max_frames: int = MAX_FALLBACK_FRAMES,
) -> list[str]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="qwen_frames_"))
    out_pattern = tmp_dir / "frame_%04d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(clip_path),
        "-vf",
        f"fps={video_fps}",
        "-frames:v",
        str(max_frames),
        str(out_pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg frame extraction failed")

    frames = sorted(tmp_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("ffmpeg produced no frames")
    return [f"file://{frame.resolve()}" for frame in frames]


def run_vision_inference(
    model,
    processor,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    video_metadatas = None
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
    inputs = inputs.to(get_model_device(model))

    generate_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature

    generated_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0] if output_text else ""


def caption_scene(
    model,
    processor,
    *,
    clip_path: Path,
    prompt: str,
    temperature: float,
    max_tokens: int,
    video_fps: float | None,
) -> str:
    sampling_fps = video_fps if video_fps is not None else 2.0
    messages = build_video_messages(clip_path, prompt, video_fps)

    try:
        return run_vision_inference(
            model,
            processor,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as primary_error:
        try:
            frame_urls = extract_frame_file_urls(
                clip_path,
                video_fps=sampling_fps,
            )
            fallback_messages = build_video_messages(
                clip_path,
                prompt,
                video_fps,
                frame_urls=frame_urls,
            )
            return run_vision_inference(
                model,
                processor,
                fallback_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as fallback_error:
            raise RuntimeError(
                f"video decode failed ({primary_error}); "
                f"ffmpeg fallback failed ({fallback_error})"
            ) from fallback_error


def try_parse_json_caption(caption: str) -> dict | None:
    text = caption.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def discover_scenes_jsons(scenes_root: Path) -> list[Path]:
    return sorted(path for path in scenes_root.glob("*/scenes.json") if path.is_file())


def default_output_dir(manifest: dict, manifest_path: Path) -> Path:
    video_name = manifest.get("video_name") or Path(manifest["video_path"]).stem
    return DEFAULT_OUTPUT_DIR / video_name


def build_failed_caption(exc: Exception, scene: dict) -> str:
    return json.dumps(
        {
            "scene_summary": "Caption failed for this scene clip.",
            "error": str(exc),
            "clip_path": scene["clip_path"],
        },
        ensure_ascii=False,
    )


def build_scene_row(
    scene: dict,
    caption: str,
    *,
    prompt_style: str,
    caption_status: str = "success",
    caption_error: str | None = None,
) -> dict:
    parsed = (
        try_parse_json_caption(caption)
        if prompt_style == "storyboard" and caption_status == "success"
        else None
    )
    row = {
        "scene_id": scene["scene_id"],
        "start": scene["start"],
        "end": scene["end"],
        "start_seconds": scene["start_seconds"],
        "end_seconds": scene["end_seconds"],
        "duration_seconds": scene["duration_seconds"],
        "clip_path": scene["clip_path"],
        "caption_status": caption_status,
        "caption_error": caption_error or "",
        "caption": caption,
        "storyboard": parsed,
    }
    if parsed:
        row.update(
            {
                "shot_type": parsed.get("shot_type", ""),
                "camera": parsed.get("camera", ""),
                "characters": parsed.get("characters", ""),
                "action": parsed.get("action", ""),
                "environment": parsed.get("environment", ""),
                "emotion": parsed.get("emotion", ""),
            }
        )
    return row


def write_manifest_outputs(
    *,
    manifest_path: Path,
    manifest: dict,
    results: list[dict],
    output_dir: Path,
    model_id: str,
    model_path: Path,
    prompt_style: str,
    start_seconds: float | None,
    end_seconds: float | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_video": manifest.get("video_path"),
        "scenes_json": str(manifest_path),
        "model_id": model_id,
        "model_path": str(model_path),
        "prompt_style": prompt_style,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "scene_count": len(results),
        "failed_scene_count": sum(
            1 for row in results if row.get("caption_status") == "failed"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenes": results,
    }

    json_path = output_dir / "captions.json"
    csv_path = output_dir / "captions.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    write_csv(csv_path, results)

    print(f"Wrote {len(results)} caption(s) for {manifest_path.parent.name}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return payload


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "scene_id",
        "start",
        "end",
        "duration_seconds",
        "clip_path",
        "caption_status",
        "caption_error",
        "caption",
        "shot_type",
        "camera",
        "characters",
        "action",
        "environment",
        "emotion",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_caption_jobs(
    manifest_paths: list[Path],
    *,
    start_seconds: float | None,
    end_seconds: float | None,
    scene_ids: set[int] | None,
    output_dir_override: Path | None,
) -> tuple[list[dict], list[str]]:
    jobs = []
    skipped_manifests: list[str] = []
    for manifest_path in manifest_paths:
        video_name = manifest_path.parent.name
        try:
            manifest = load_manifest(manifest_path)
            scenes = select_scenes(
                manifest,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                scene_ids=scene_ids,
            )
            if not scenes:
                message = f"{video_name}: no scenes matched filters"
                print(f"Skipping {message}")
                skipped_manifests.append(message)
                continue

            output_dir = (
                output_dir_override
                if output_dir_override is not None
                else default_output_dir(manifest, manifest_path)
            )
            for scene in scenes:
                jobs.append(
                    {
                        "manifest_path": manifest_path,
                        "manifest": manifest,
                        "video_name": video_name,
                        "output_dir": output_dir,
                        "scene": scene,
                    }
                )
        except Exception as exc:
            message = f"{video_name}: {exc}"
            print(f"Skipping manifest {message}")
            skipped_manifests.append(message)
    return jobs, skipped_manifests


def run_captioning(args: argparse.Namespace) -> dict | list[dict]:
    start_seconds = parse_time(args.start)
    end_seconds = parse_time(args.end)
    if (
        start_seconds is not None
        and end_seconds is not None
        and start_seconds >= end_seconds
    ):
        raise ValueError("--start must be earlier than --end")

    scene_ids = None
    if args.scene_ids:
        scene_ids = {int(value) for value in args.scene_ids.split(",")}

    if args.scenes_root:
        manifest_paths = discover_scenes_jsons(
            Path(args.scenes_root).expanduser().resolve()
        )
        if not manifest_paths:
            raise ValueError(f"No scenes.json files found under {args.scenes_root}")
        print(f"Found {len(manifest_paths)} video manifest(s)")
    else:
        manifest_paths = [Path(args.scenes_json).expanduser().resolve()]

    models_dir = Path(args.models_dir).expanduser().resolve()
    model_path = ensure_model(args.model_id, models_dir)

    if args.download_only:
        print(f"Model ready at {model_path}")
        return {"model_path": str(model_path), "model_id": args.model_id}

    output_dir_override = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir and not args.scenes_root
        else None
    )
    jobs, skipped_manifests = collect_caption_jobs(
        manifest_paths,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        scene_ids=scene_ids,
        output_dir_override=output_dir_override,
    )
    if not jobs:
        raise ValueError("No scenes matched the requested filters.")

    prompt = STORYBOARD_PROMPT if args.prompt_style == "storyboard" else DETAILED_PROMPT
    try:
        model, processor = load_qwen3_vl(
            model_path,
            device=args.device,
            attn_implementation=args.attn_implementation,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load Qwen3-VL model: {exc}") from exc

    results_by_manifest: dict[Path, list[dict]] = {}
    manifest_meta: dict[Path, dict] = {}
    failed_scenes: list[str] = []
    succeeded_scenes = 0
    progress = tqdm(
        jobs,
        desc="Captioning all scenes",
        unit="scene",
        dynamic_ncols=True,
    )
    for job in progress:
        scene = job["scene"]
        manifest_path = job["manifest_path"]
        progress.set_postfix(
            video=job["video_name"],
            scene=scene["scene_id"],
            time=f"{scene['start']}->{scene['end']}",
            refresh=False,
        )

        caption_status = "success"
        caption_error = None
        try:
            caption = caption_scene(
                model,
                processor,
                clip_path=Path(scene["clip_path"]),
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                video_fps=args.video_fps,
            )
            succeeded_scenes += 1
        except Exception as exc:
            caption_status = "failed"
            caption_error = str(exc)
            failed_label = (
                f"{job['video_name']} scene {scene['scene_id']}: {exc}"
            )
            failed_scenes.append(failed_label)
            progress.write(f"Failed {failed_label}")
            caption = build_failed_caption(exc, scene)

        row = build_scene_row(
            scene,
            caption,
            prompt_style=args.prompt_style,
            caption_status=caption_status,
            caption_error=caption_error,
        )
        results_by_manifest.setdefault(manifest_path, []).append(row)
        manifest_meta[manifest_path] = {
            "manifest": job["manifest"],
            "output_dir": job["output_dir"],
        }

    payloads = []
    write_failures: list[str] = []
    for manifest_path, results in results_by_manifest.items():
        meta = manifest_meta[manifest_path]
        try:
            payload = write_manifest_outputs(
                manifest_path=manifest_path,
                manifest=meta["manifest"],
                results=results,
                output_dir=meta["output_dir"],
                model_id=args.model_id,
                model_path=model_path,
                prompt_style=args.prompt_style,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
            payloads.append(payload)
        except Exception as exc:
            message = f"{manifest_path.parent.name}: {exc}"
            print(f"Failed to write outputs for {message}")
            write_failures.append(message)

    summary = {
        "videos_written": len(payloads),
        "videos_skipped": len(skipped_manifests),
        "scenes_total": len(jobs),
        "scenes_succeeded": succeeded_scenes,
        "scenes_failed": len(failed_scenes),
        "write_failures": len(write_failures),
    }
    print("Run summary:")
    print(json.dumps(summary, indent=2))
    if skipped_manifests:
        print("Skipped manifests:")
        for message in skipped_manifests:
            print(f"  - {message}")
    if failed_scenes:
        print("Failed scenes:")
        for message in failed_scenes:
            print(f"  - {message}")
    if write_failures:
        print("Write failures:")
        for message in write_failures:
            print(f"  - {message}")

    if succeeded_scenes == 0:
        raise RuntimeError("No scenes were captioned successfully.")

    print(
        f"Finished {len(payloads)} video(s), "
        f"{succeeded_scenes}/{len(jobs)} scene(s) succeeded"
    )
    return payloads[0] if len(payloads) == 1 else payloads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Caption scene clips with local Qwen3-VL (HuggingFace Transformers)."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--scenes-json",
        help="Path to a single scenes.json produced by scene_cut.py",
    )
    input_group.add_argument(
        "--scenes-root",
        help="Directory containing per-video scene folders with scenes.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for captions.json and captions.csv",
    )
    parser.add_argument(
        "--models-dir",
        default=str(MODELS_DIR),
        help=f"Directory to store downloaded models (default: {MODELS_DIR})",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the model into models/, do not run captioning",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Only caption scenes overlapping this start time (seconds or HH:MM:SS)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Only caption scenes overlapping this end time (seconds or HH:MM:SS)",
    )
    parser.add_argument(
        "--scene-ids",
        default=None,
        help="Comma-separated scene IDs to caption (e.g. 1,3,5)",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["storyboard", "detailed"],
        default="storyboard",
        help="Caption prompt style (default: storyboard JSON)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per scene (default: 512)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=2.0,
        help="Video frame sampling FPS for Qwen3-VL (default: 2.0)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Torch device for inference (default: cuda). Use "cpu" to force CPU.',
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help='Optional attention backend, e.g. "flash_attention_2" for faster inference',
    )
    parser.add_argument(
        "--video-reader",
        choices=["decord", "torchvision", "torchcodec"],
        default=os.environ.get("FORCE_QWENVL_VIDEO_READER", "decord"),
        help="Video decode backend for qwen-vl-utils (default: decord)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    os.environ["FORCE_QWENVL_VIDEO_READER"] = args.video_reader

    try:
        run_captioning(args)
    except KeyboardInterrupt:
        print("\nCaptioning interrupted by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"Captioning run failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
