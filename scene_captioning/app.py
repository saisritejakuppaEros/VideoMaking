#!/usr/bin/env python3
"""Streamlit app entry point for scene caption outputs."""

from __future__ import annotations

import streamlit as st

from viewer.data_loader import CAPTIONS_DIR, list_caption_files

st.set_page_config(
    page_title="Scene Caption Viewer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Scene Caption Viewer")
st.markdown(
    """
Browse scene-level captions generated from video analysis.

Use the sidebar to pick a video, then open:

- **Corpus Analysis** — bar plots of shot types, camera, and other metadata across all videos
- **Analysis** — run metadata, scene table, and per-video field breakdowns
- **Scene Viewer** — slide through scenes with clip playback and storyboard metadata
"""
)

entries = list_caption_files()
if not entries:
    st.error(f"No caption files found under `{CAPTIONS_DIR}`.")
    st.stop()

st.subheader(f"{len(entries)} videos available")
for title, json_path in entries:
    st.markdown(f"- **{title}** — `{json_path.name}`")

st.info("Select a page from the sidebar to begin.")
