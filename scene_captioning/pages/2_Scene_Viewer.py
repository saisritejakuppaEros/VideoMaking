#!/usr/bin/env python3
"""Scene-by-scene viewer with slider navigation."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from viewer.components import render_sidebar
from viewer.data_loader import (
    STORYBOARD_FIELDS,
    get_storyboard,
    resolve_video_path,
    scene_label,
)

st.set_page_config(page_title="Scene Viewer | Scene Caption Viewer", layout="wide")

st.title("Scene Viewer")

selected_title, data, _json_path = render_sidebar()
if data is None:
    st.stop()

scenes = data.get("scenes", [])
if not scenes:
    st.warning("No scenes found for this video.")
    st.stop()

scene_ids = [int(scene.get("scene_id", index + 1)) for index, scene in enumerate(scenes)]
min_id = min(scene_ids)
max_id = max(scene_ids)

if st.session_state.get("scene_viewer_video") != selected_title:
    st.session_state.scene_viewer_video = selected_title
    st.session_state.scene_viewer_id = min_id

if "scene_viewer_id" not in st.session_state:
    st.session_state.scene_viewer_id = min_id

st.session_state.scene_viewer_id = int(
    st.slider(
        "Scene",
        min_value=min_id,
        max_value=max_id,
        value=int(st.session_state.scene_viewer_id),
        step=1,
        format="Scene %d",
        key="scene_slider",
    )
)
selected_scene_id = st.session_state.scene_viewer_id

scene = next(item for item in scenes if int(item.get("scene_id", -1)) == selected_scene_id)
storyboard = get_storyboard(scene)

st.caption(scene_label(scene))

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Video")
    video_path = resolve_video_path(scene, data)
    clip_path = scene.get("clip_path", "")

    if video_path and video_path.is_file():
        if clip_path and Path(clip_path).is_file():
            st.video(str(video_path))
            st.caption(f"Scene clip: `{clip_path}`")
        else:
            st.video(str(video_path))
            st.caption(f"Playing source video: `{video_path}`")
            st.info(
                f"This scene spans **{scene.get('start')} – {scene.get('end')}** "
                f"({scene.get('duration_seconds', 0):.2f}s). Scene clip not found."
            )
    else:
        st.error("Video file not found on disk.")
        if clip_path:
            st.code(clip_path, language=None)
        source = data.get("source_video", "")
        if source:
            st.code(source, language=None)

    timing_cols = st.columns(4)
    timing_cols[0].metric("Start", scene.get("start", ""))
    timing_cols[1].metric("End", scene.get("end", ""))
    timing_cols[2].metric("Duration (s)", f"{float(scene.get('duration_seconds', 0)):.2f}")
    timing_cols[3].metric("Status", scene.get("caption_status", "unknown"))

with right:
    st.subheader("Storyboard metadata")

    status = scene.get("caption_status", "")
    if status != "success":
        st.warning(f"Caption status: `{status}`")
        error = scene.get("caption_error", "")
        if error:
            st.error(error)

    summary = storyboard.get("scene_summary", "")
    if summary:
        st.markdown("**Scene summary**")
        st.write(summary)

    display_fields = [
        ("shot_type", "Shot type"),
        ("camera", "Camera"),
        ("characters", "Characters"),
        ("action", "Action"),
        ("objects", "Objects"),
        ("environment", "Environment"),
        ("visual_style", "Visual style"),
        ("emotion", "Emotion"),
    ]

    for key, label in display_fields:
        value = storyboard.get(key, "").strip()
        if value:
            st.markdown(f"**{label}**")
            st.write(value)

    with st.expander("Raw caption JSON"):
        raw = scene.get("caption", "")
        if raw:
            st.code(raw, language="json")
        else:
            st.json({key: storyboard.get(key, "") for key in STORYBOARD_FIELDS})

st.divider()

nav_cols = st.columns([1, 2, 1])
with nav_cols[0]:
    if selected_scene_id > min_id and st.button("Previous scene", use_container_width=True):
        st.session_state.scene_viewer_id = selected_scene_id - 1
        st.rerun()
with nav_cols[2]:
    if selected_scene_id < max_id and st.button("Next scene", use_container_width=True):
        st.session_state.scene_viewer_id = selected_scene_id + 1
        st.rerun()
