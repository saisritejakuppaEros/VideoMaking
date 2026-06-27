#!/usr/bin/env bash
# Parallel LTX captioning by sharding scene folders across multiple GPUs.
# Writes one captions.json per video folder under scene_caption_op/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

if [[ ! -x "${LTX_PYTHON}" ]]; then
  echo "LTX trainer env missing. Run: ${SCRIPT_DIR}/setup_trainer.sh" >&2
  exit 1
fi

SCENES_INPUT="${SCENES_INPUT:-${SCENES_ROOT}}"
CAPTION_OUTPUT="${CAPTION_OUTPUT:-${SCENE_CAPTION_OUTPUT}}"
SHARDS_DIR="${CAPTION_SHARDS_DIR}"
USE_8BIT="${USE_8BIT:-1}"
CAPTION_FPS="${CAPTION_FPS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
PROGRESS_REFRESH="${PROGRESS_REFRESH:-2}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
NUM_GPUS="${#GPU_IDS[@]}"

if [[ ! -d "${SCENES_INPUT}" ]]; then
  echo "Scenes directory not found: ${SCENES_INPUT}" >&2
  exit 1
fi

mapfile -t VIDEO_DIRS < <(find "${SCENES_INPUT}" -mindepth 1 -maxdepth 1 -type d | sort)
TOTAL="${#VIDEO_DIRS[@]}"
if (( TOTAL == 0 )); then
  echo "No scene folders found under ${SCENES_INPUT}" >&2
  exit 1
fi

mkdir -p "${CAPTION_OUTPUT}" "${SHARDS_DIR}"
cd "${LTX_TRAINER}"

caption_args=(--fps "${CAPTION_FPS}")
if [[ "${USE_8BIT}" == "1" ]]; then
  caption_args+=(--use-8bit)
fi

declare -a PIDS=()
declare -a ACTIVE_WORKERS=()
MONITOR_PID=""

update_worker_status() {
  local worker_id="$1"
  local gpu_id="$2"
  local folders_done="$3"
  local folders_total="$4"
  local clips_total="$5"
  local list_file="$6"
  local current_folder="${7:-}"
  local status="${8:-running}"

  "${VENV_PYTHON}" "${SCRIPT_DIR}/caption_progress.py" update \
    --status-file "${SHARDS_DIR}/worker${worker_id}.status.json" \
    --worker-id "${worker_id}" \
    --gpu-id "${gpu_id}" \
    --folders-done "${folders_done}" \
    --folders-total "${folders_total}" \
    --clips-total "${clips_total}" \
    --list-file "${list_file}" \
    --output-root "${CAPTION_OUTPUT}" \
    --current-folder "${current_folder}" \
    --status "${status}"
}

cleanup_workers() {
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi

  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}

trap cleanup_workers INT TERM

run_worker() {
  local worker_id="$1"
  local gpu_id="$2"
  local list_file="$3"
  local folders_total="$4"
  local clips_total="$5"
  local log_file="${SHARDS_DIR}/worker${worker_id}.log"
  local folders_done=0
  local worker_status="running"
  local folder_failures=0

  {
    echo "Worker ${worker_id} on GPU ${gpu_id}"
    update_worker_status "${worker_id}" "${gpu_id}" 0 "${folders_total}" "${clips_total}" \
      "${list_file}" "" "running"

    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    while IFS= read -r video_dir; do
      [[ -z "${video_dir}" ]] && continue

      local folder_name final_output tmp_output
      folder_name="$(basename "${video_dir}")"
      final_output="${CAPTION_OUTPUT}/${folder_name}/captions.json"
      tmp_output="${video_dir}/.ltx_caption.tmp.json"

      if [[ "${SKIP_EXISTING}" == "1" && -f "${final_output}" ]]; then
        echo "  skip (exists) ${final_output}"
        folders_done=$((folders_done + 1))
        update_worker_status "${worker_id}" "${gpu_id}" "${folders_done}" "${folders_total}" \
          "${clips_total}" "${list_file}" "${video_dir}" "${worker_status}"
        continue
      fi

      echo "  -> ${video_dir} (${folders_done}/${folders_total})"
      update_worker_status "${worker_id}" "${gpu_id}" "${folders_done}" "${folders_total}" \
        "${clips_total}" "${list_file}" "${video_dir}" "running"

      rm -f "${tmp_output}"
      if ! uv run python scripts/caption_videos.py "${video_dir}" \
        --output "${tmp_output}" \
        "${caption_args[@]}"; then
        folder_failures=$((folder_failures + 1))
        rm -f "${tmp_output}"
        echo "Worker ${worker_id} failed on ${video_dir}" >&2
        if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
          worker_status="failed"
          break
        fi
        continue
      fi

      if ! "${VENV_PYTHON}" "${SCRIPT_DIR}/caption_progress.py" finalize \
        --tmp-path "${tmp_output}" \
        --scenes-root "${SCENES_INPUT}" \
        --video-dir "${video_dir}" \
        --output-path "${final_output}"; then
        folder_failures=$((folder_failures + 1))
        rm -f "${tmp_output}"
        echo "Worker ${worker_id} could not finalize ${video_dir}" >&2
        if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
          worker_status="failed"
          break
        fi
        continue
      fi

      echo "  saved ${final_output}"
      folders_done=$((folders_done + 1))
      update_worker_status "${worker_id}" "${gpu_id}" "${folders_done}" "${folders_total}" \
        "${clips_total}" "${list_file}" "${video_dir}" "${worker_status}"
    done < "${list_file}"

    if [[ "${worker_status}" == "running" && "${folder_failures}" -gt 0 ]]; then
      worker_status="failed"
    elif [[ "${worker_status}" == "running" ]]; then
      worker_status="done"
    fi
    update_worker_status "${worker_id}" "${gpu_id}" "${folders_done}" "${folders_total}" \
      "${clips_total}" "${list_file}" "" "${worker_status}"
    echo "Worker ${worker_id} finished (${worker_status}, ${folder_failures} folder errors)"
  } >"${log_file}" 2>&1
}

echo "Multi-GPU LTX captioning"
echo "  scenes root:   ${SCENES_INPUT}"
echo "  output root:   ${CAPTION_OUTPUT}/<video>/captions.json"
echo "  video dirs:    ${TOTAL}"
echo "  GPUs:          ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} workers)"
echo "  worker logs:   ${SHARDS_DIR}/worker*.log"
echo "  caption fps:   ${CAPTION_FPS}"
echo

for ((worker_id = 0; worker_id < NUM_GPUS; worker_id++)); do
  list_file="${SHARDS_DIR}/worker${worker_id}_dirs.txt"
  : > "${list_file}"
done

for i in "${!VIDEO_DIRS[@]}"; do
  worker_id=$((i % NUM_GPUS))
  list_file="${SHARDS_DIR}/worker${worker_id}_dirs.txt"
  printf '%s\n' "${VIDEO_DIRS[$i]}" >> "${list_file}"
done

for ((worker_id = 0; worker_id < NUM_GPUS; worker_id++)); do
  gpu_id="${GPU_IDS[$worker_id]}"
  list_file="${SHARDS_DIR}/worker${worker_id}_dirs.txt"
  folder_count="$(grep -c . "${list_file}" || true)"
  if (( folder_count == 0 )); then
    continue
  fi

  clip_count="$("${VENV_PYTHON}" "${SCRIPT_DIR}/caption_progress.py" count-clips --list-file "${list_file}")"
  update_worker_status "${worker_id}" "${gpu_id}" 0 "${folder_count}" "${clip_count}" \
    "${list_file}" "" "pending"

  run_worker "${worker_id}" "${gpu_id}" "${list_file}" "${folder_count}" "${clip_count}" &
  PIDS+=("$!")
  ACTIVE_WORKERS+=("${worker_id}")
  echo "Started worker ${worker_id} (GPU ${gpu_id}, ${folder_count} folders, ${clip_count} clips) -> ${SHARDS_DIR}/worker${worker_id}.log"
done

echo
echo "Live per-GPU progress (clips + folders):"
"${VENV_PYTHON}" "${SCRIPT_DIR}/caption_progress.py" monitor \
  --shards-dir "${SHARDS_DIR}" \
  --output-root "${CAPTION_OUTPUT}" \
  --worker-ids "${ACTIVE_WORKERS[@]}" \
  --refresh "${PROGRESS_REFRESH}" &
MONITOR_PID=$!

failures=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done

if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
  MONITOR_PID=""
fi

trap - INT TERM

captioned_folders="$(find "${CAPTION_OUTPUT}" -mindepth 2 -maxdepth 2 -name 'captions.json' | wc -l | tr -d ' ')"

if (( failures > 0 )); then
  echo "${failures} caption worker(s) failed. Check logs in ${SHARDS_DIR}" >&2
  echo "Partial output: ${captioned_folders} folders captioned under ${CAPTION_OUTPUT}" >&2
  exit 1
fi

echo
echo "Captioning complete: ${captioned_folders} folders under ${CAPTION_OUTPUT}"
echo "Each folder has its own captions.json (no merged dataset.json)."
