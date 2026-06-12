#!/usr/bin/env python3
"""Analysis dashboard for scene caption outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from viewer.charts import render_field_bar_chart
from viewer.components import render_sidebar
from viewer.data_loader import (
    FIELD_LABELS,
    META_CATEGORY_FIELDS,
    aggregate_field_counts,
    get_storyboard,
    video_duration_seconds,
)

st.set_page_config(page_title="Analysis | Scene Caption Viewer", layout="wide")

st.title("Caption Analysis")

selected_title, data, json_path = render_sidebar()
if data is None:
    st.stop()

scenes = data.get("scenes", [])
source_video = data.get("source_video", "")
duration = video_duration_seconds(scenes)
success_count = sum(1 for s in scenes if s.get("caption_status") == "success")
failed_count = int(data.get("failed_scene_count", 0))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Scenes", len(scenes))
col2.metric("Successful captions", success_count)
col3.metric("Failed captions", failed_count)
col4.metric("Video duration", f"{duration:.1f}s")

st.subheader("Run metadata")
meta_cols = st.columns(2)
with meta_cols[0]:
    st.markdown(f"**Source video:** `{source_video}`")
    st.markdown(f"**Scenes JSON:** `{data.get('scenes_json', '')}`")
    st.markdown(f"**Model:** `{data.get('model_id', '')}`")
    st.markdown(f"**Prompt style:** `{data.get('prompt_style', '')}`")
with meta_cols[1]:
    st.markdown(f"**Created:** `{data.get('created_at', '')}`")
    st.markdown(f"**Model path:** `{data.get('model_path', '')}`")
    start = data.get("start_seconds")
    end = data.get("end_seconds")
    if start is not None or end is not None:
        st.markdown(f"**Trim window:** `{start}` – `{end}` seconds")

if source_video and Path(source_video).is_file():
    with st.expander("Full source video", expanded=False):
        st.video(source_video)

st.subheader("Scene table")
rows = []
for scene in scenes:
    storyboard = get_storyboard(scene)
    rows.append(
        {
            "scene_id": scene.get("scene_id"),
            "start": scene.get("start"),
            "end": scene.get("end"),
            "duration_s": scene.get("duration_seconds"),
            "status": scene.get("caption_status"),
            "shot_type": storyboard.get("shot_type") or scene.get("shot_type", ""),
            "camera": storyboard.get("camera") or scene.get("camera", ""),
            "emotion": storyboard.get("emotion") or scene.get("emotion", ""),
            "summary": storyboard.get("scene_summary", "")[:120],
            "clip_exists": Path(scene.get("clip_path", "")).is_file(),
        }
    )

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Field breakdown")
st.caption(f"Metadata distribution for **{selected_title}**.")

priority_fields = ("shot_type", "camera")
ordered_fields = list(priority_fields) + [
    field for field in META_CATEGORY_FIELDS if field not in priority_fields
]

for row_start in range(0, len(ordered_fields), 2):
    breakdown_cols = st.columns(2)
    for col, field in zip(breakdown_cols, ordered_fields[row_start : row_start + 2]):
        with col:
            counts = aggregate_field_counts(scenes, field)
            render_field_bar_chart(
                counts,
                title=FIELD_LABELS.get(field, field),
                top_n=len(scenes) or 1,
            )

with st.expander("Export paths"):
    st.code(str(json_path), language=None)
    csv_path = json_path.parent / "captions.csv"
    if csv_path.is_file():
        st.code(str(csv_path), language=None)

with st.expander("Raw JSON"):
    st.json(data)
