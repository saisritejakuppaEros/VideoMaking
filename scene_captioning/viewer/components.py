"""Shared Streamlit UI components."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from viewer.data_loader import CAPTIONS_DIR, list_caption_files, load_caption_data


def render_sidebar() -> tuple[str | None, dict | None, Path | None]:
    st.sidebar.title("Scene Caption Viewer")
    st.sidebar.caption(f"Data: `{CAPTIONS_DIR}`")

    entries = list_caption_files()
    if not entries:
        st.sidebar.warning("No caption outputs found.")
        return None, None, None

    titles = [title for title, _ in entries]
    selected_title = st.sidebar.selectbox("Video", titles, key="selected_video_title")

    json_path = dict(entries)[selected_title]
    data = load_caption_data(json_path)
    return selected_title, data, json_path
