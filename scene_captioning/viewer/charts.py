"""Chart helpers for caption analysis dashboards."""

from __future__ import annotations

from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st


def render_field_bar_chart(
    counts: Counter[str],
    *,
    title: str,
    top_n: int = 15,
) -> None:
    """Render a horizontal bar chart for categorical field counts."""
    if not counts:
        st.caption("No values recorded.")
        return

    items = counts.most_common(top_n)
    chart_df = pd.DataFrame(items, columns=["value", "count"])
    total_scenes = sum(counts.values())
    unique_values = len(counts)

    chart = (
        alt.Chart(chart_df)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("count:Q", title="Scene count", axis=alt.Axis(format="d")),
            y=alt.Y(
                "value:N",
                sort=alt.EncodingSortField(field="count", order="descending"),
                title="",
            ),
            tooltip=[
                alt.Tooltip("value:N", title="Value"),
                alt.Tooltip("count:Q", title="Scenes"),
            ],
        )
        .properties(
            height=max(180, len(items) * 28),
            title=title,
        )
    )
    st.altair_chart(chart, use_container_width=True)

    pct = 100 * sum(count for _, count in items) / total_scenes
    if unique_values > top_n:
        st.caption(
            f"Top {top_n} of {unique_values} unique values "
            f"({pct:.0f}% of {total_scenes} tagged scenes)."
        )
    else:
        st.caption(f"{unique_values} unique values across {total_scenes} tagged scenes.")
