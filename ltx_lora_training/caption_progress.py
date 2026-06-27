#!/usr/bin/env python3
"""Progress helpers for multi-GPU LTX captioning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn


def count_json_entries(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    return len(entries) if isinstance(entries, list) else 0


def folder_caption_path(output_root: Path, video_dir: Path) -> Path:
    return output_root / video_dir.name / "captions.json"


def folder_tmp_path(video_dir: Path) -> Path:
    return video_dir / ".ltx_caption.tmp.json"


def load_status(status_path: Path) -> dict | None:
    if not status_path.is_file():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def count_clips_in_dirs(list_file: Path, extensions: tuple[str, ...] = ("mp4",)) -> int:
    total = 0
    if not list_file.is_file():
        return 0
    for line in list_file.read_text(encoding="utf-8").splitlines():
        video_dir = line.strip()
        if not video_dir:
            continue
        path = Path(video_dir)
        if not path.is_dir():
            continue
        for ext in extensions:
            total += len(list(path.glob(f"*.{ext}")))
    return total


def count_worker_clips(list_file: Path, output_root: Path) -> int:
    done = 0
    if not list_file.is_file():
        return 0
    for line in list_file.read_text(encoding="utf-8").splitlines():
        video_dir = line.strip()
        if not video_dir:
            continue
        path = Path(video_dir)
        if not path.is_dir():
            continue
        final_path = folder_caption_path(output_root, path)
        tmp_path = folder_tmp_path(path)
        done += count_json_entries(final_path)
        if not final_path.is_file():
            done += count_json_entries(tmp_path)
    return done


def finalize_caption_json(
    tmp_path: Path,
    scenes_root: Path,
    video_dir: Path,
    output_path: Path,
) -> int:
    entries = json.loads(tmp_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"Expected JSON list in {tmp_path}")

    folder_rel = video_dir.resolve().relative_to(scenes_root.resolve())
    fixed: list[dict[str, str]] = []
    for entry in entries:
        media_path = entry["media_path"]
        media = Path(media_path)
        if media.is_absolute() or ".." in media.parts:
            media_path = str(folder_rel / media.name)
        elif len(media.parts) == 1:
            media_path = str(folder_rel / media_path)
        fixed.append({"caption": entry["caption"], "media_path": media_path})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.unlink(missing_ok=True)
    return len(fixed)


def write_status(
    status_path: Path,
    *,
    worker_id: int,
    gpu_id: int,
    folders_done: int,
    folders_total: int,
    clips_total: int,
    list_file: Path,
    output_root: Path,
    current_folder: str = "",
    status: str = "running",
) -> None:
    clips_done = count_worker_clips(list_file, output_root)
    payload = {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "folders_done": folders_done,
        "folders_total": folders_total,
        "clips_done": clips_done,
        "clips_total": clips_total,
        "current_folder": current_folder,
        "status": status,
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_update(args: argparse.Namespace) -> None:
    write_status(
        args.status_file,
        worker_id=args.worker_id,
        gpu_id=args.gpu_id,
        folders_done=args.folders_done,
        folders_total=args.folders_total,
        clips_total=args.clips_total,
        list_file=args.list_file,
        output_root=args.output_root,
        current_folder=args.current_folder,
        status=args.status,
    )


def cmd_finalize(args: argparse.Namespace) -> None:
    count = finalize_caption_json(
        args.tmp_path,
        args.scenes_root,
        args.video_dir,
        args.output_path,
    )
    print(count)


def cmd_monitor(args: argparse.Namespace) -> None:
    console = Console(stderr=True)
    shards_dir = args.shards_dir
    output_root = args.output_root
    worker_ids = args.worker_ids
    refresh = args.refresh

    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("clips"),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    )

    task_ids: dict[int, int] = {}
    with Live(progress, console=console, refresh_per_second=4, transient=False):
        overall_task = progress.add_task(
            "[bold]Overall[/] clips 0/0 • folders 0/0",
            total=1,
        )
        for worker_id in worker_ids:
            status_path = shards_dir / f"worker{worker_id}.status.json"
            status = load_status(status_path) or {}
            clips_total = int(status.get("clips_total") or 0)
            task_ids[worker_id] = progress.add_task(
                f"GPU {status.get('gpu_id', worker_id)}",
                total=max(clips_total, 1),
            )

        while True:
            all_done = True
            total_clips_done = 0
            total_clips = 0
            total_folders_done = 0
            total_folders = 0

            for worker_id in worker_ids:
                status_path = shards_dir / f"worker{worker_id}.status.json"
                list_file = shards_dir / f"worker{worker_id}_dirs.txt"
                status = load_status(status_path) or {}

                folders_total = int(status.get("folders_total") or 0)
                folders_done = int(status.get("folders_done") or 0)
                clips_total = int(status.get("clips_total") or 0)
                clips_done = max(int(status.get("clips_done") or 0), count_worker_clips(list_file, output_root))
                worker_status = status.get("status", "pending")
                gpu_id = status.get("gpu_id", worker_id)
                current_folder = status.get("current_folder") or ""

                total_clips_done += clips_done
                total_clips += clips_total
                total_folders_done += folders_done
                total_folders += folders_total

                if worker_status not in {"done", "failed"}:
                    all_done = False

                folder_name = Path(current_folder).name if current_folder else "waiting"
                if len(folder_name) > 32:
                    folder_name = folder_name[:29] + "..."

                if worker_status == "done":
                    label = f"GPU {gpu_id} folders {folders_done}/{folders_total} [green]done[/]"
                elif worker_status == "failed":
                    label = f"GPU {gpu_id} folders {folders_done}/{folders_total} [red]failed[/]"
                elif current_folder:
                    label = f"GPU {gpu_id} folders {folders_done}/{folders_total} [dim]{folder_name}[/]"
                else:
                    label = f"GPU {gpu_id} folders {folders_done}/{folders_total}"

                progress.update(
                    task_ids[worker_id],
                    description=label,
                    completed=min(clips_done, clips_total or clips_done or 1),
                    total=max(clips_total, 1),
                )

            progress.update(
                overall_task,
                description=(
                    f"[bold]Overall[/] clips {total_clips_done}/{total_clips} • "
                    f"folders {total_folders_done}/{total_folders}"
                ),
                completed=min(total_clips_done, total_clips or total_clips_done or 1),
                total=max(total_clips, 1),
            )

            if all_done and total_folders > 0:
                break
            time.sleep(refresh)


def cmd_count_clips(args: argparse.Namespace) -> None:
    print(count_clips_in_dirs(args.list_file))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    update = sub.add_parser("update", help="Write worker status JSON")
    update.add_argument("--status-file", type=Path, required=True)
    update.add_argument("--worker-id", type=int, required=True)
    update.add_argument("--gpu-id", type=int, required=True)
    update.add_argument("--folders-done", type=int, required=True)
    update.add_argument("--folders-total", type=int, required=True)
    update.add_argument("--clips-total", type=int, required=True)
    update.add_argument("--list-file", type=Path, required=True)
    update.add_argument("--output-root", type=Path, required=True)
    update.add_argument("--current-folder", default="")
    update.add_argument(
        "--status",
        default="running",
        choices=("pending", "running", "done", "failed"),
    )
    update.set_defaults(func=cmd_update)

    finalize = sub.add_parser("finalize", help="Move temp captions JSON to scene_caption_op with fixed paths")
    finalize.add_argument("--tmp-path", type=Path, required=True)
    finalize.add_argument("--scenes-root", type=Path, required=True)
    finalize.add_argument("--video-dir", type=Path, required=True)
    finalize.add_argument("--output-path", type=Path, required=True)
    finalize.set_defaults(func=cmd_finalize)

    monitor = sub.add_parser("monitor", help="Live per-GPU progress display")
    monitor.add_argument("--shards-dir", type=Path, required=True)
    monitor.add_argument("--output-root", type=Path, required=True)
    monitor.add_argument("--worker-ids", type=int, nargs="+", required=True)
    monitor.add_argument("--refresh", type=float, default=2.0)
    monitor.set_defaults(func=cmd_monitor)

    count = sub.add_parser("count-clips", help="Count mp4 clips listed in a worker dir list")
    count.add_argument("--list-file", type=Path, required=True)
    count.set_defaults(func=cmd_count_clips)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
