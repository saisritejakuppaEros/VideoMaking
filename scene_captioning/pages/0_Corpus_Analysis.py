#!/usr/bin/env python3
"""Corpus-wide analysis across all captioned videos."""

from __future__ import annotations

import streamlit as st

from viewer.charts import render_field_bar_chart
from viewer.data_loader import (
    CAPTIONS_DIR,
    FIELD_LABELS,
    META_CATEGORY_FIELDS,
    aggregate_corpus_field_counts,
    load_all_caption_entries,
)

st.set_page_config(page_title="Corpus Analysis | Scene Caption Viewer", layout="wide")

st.title("Corpus Analysis")
st.markdown(
    "Distribution of storyboard metadata across **all** captioned videos in the corpus."
)


@st.cache_data(show_spinner="Loading caption outputs…")
def load_corpus():
    return load_all_caption_entries()


entries = load_corpus()
if not entries:
    st.error(f"No caption files found under `{CAPTIONS_DIR}`.")
    st.stop()

total_videos = len(entries)
total_scenes = sum(len(data.get("scenes", [])) for _, data, _ in entries)
successful_scenes = sum(
    1
    for _, data, _ in entries
    for scene in data.get("scenes", [])
    if scene.get("caption_status") == "success"
)

col1, col2, col3 = st.columns(3)
col1.metric("Videos", total_videos)
col2.metric("Total scenes", total_scenes)
col3.metric("Successful captions", successful_scenes)

top_n = st.slider(
    "Bars per chart",
    min_value=5,
    max_value=30,
    value=15,
    help="Show the most frequent values for each metadata field.",
)

st.subheader("Metadata distributions")
st.caption(
    "Each bar counts how many scenes use a given label. "
    "Values are grouped by exact caption text."
)

priority_fields = ("shot_type", "camera")
ordered_fields = list(priority_fields) + [
    field for field in META_CATEGORY_FIELDS if field not in priority_fields
]

for row_start in range(0, len(ordered_fields), 2):
    cols = st.columns(2)
    for col, field in zip(cols, ordered_fields[row_start : row_start + 2]):
        with col:
            counts = aggregate_corpus_field_counts(entries, field)
            render_field_bar_chart(
                counts,
                title=FIELD_LABELS.get(field, field),
                top_n=top_n,
            )

with st.expander("Videos in corpus"):
    for title, _, json_path in entries:
        st.markdown(f"- **{title}** — `{json_path}`")
