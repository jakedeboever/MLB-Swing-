import streamlit as st
import pandas as pd
import numpy as np
import re

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("swing_plus_results.csv")

    # Make a clean Name column
    if "last_name, first_name" in df.columns:
        df.rename(columns={"last_name, first_name": "Name"}, inplace=True)

    # Round swing_plus to whole number
    if "swing_plus" in df.columns:
        df["swing_plus"] = df["swing_plus"].round(0).astype("Int64")

    # Reorder columns: Name, swing_plus, xWOBA-related, then rest
    col_order = ["Name"]
    if "swing_plus" in df.columns:
        col_order.append("swing_plus")

    # Find xwoba-related stats (case-insensitive search for "xwoba")
    xwoba_cols = [c for c in df.columns if "xwoba" in c.lower()]
    col_order.extend(xwoba_cols)

    remaining_cols = [c for c in df.columns if c not in col_order]
    col_order.extend(remaining_cols)
    df = df[col_order]

    return df

df = load_data()

st.title("Swing+ Results Viewer")

# Search bar - supports multiple names
search_query = st.text_input("Search by player name (comma-separated for multiple):")

# Team filter if available
team_filter = None
if "Team" in df.columns:
    teams = ["All"] + sorted(df["Team"].dropna().unique())
    team_filter = st.selectbox("Filter by Team:", teams)

# Year filter if available
year_filter = None
if "year" in df.columns:
    years = ["All"] + sorted(df["year"].dropna().unique())
    year_filter = st.selectbox("Filter by Year:", years)

# Numeric stat filter
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
stat_to_filter = st.selectbox("Select a stat to filter:", numeric_columns)
min_val = float(df[stat_to_filter].min())
max_val = float(df[stat_to_filter].max())
stat_range = st.slider(f"Filter {stat_to_filter} range:", min_val, max_val, (min_val, max_val))

# Apply filters
filtered_df = df.copy()

# Name filter with multiple names allowed
if search_query:
    names_list = [name.strip() for name in search_query.split(",") if name.strip()]
    regex_pattern = "|".join([re.escape(name) for name in names_list])
    filtered_df = filtered_df[filtered_df["Name"].str.contains(regex_pattern, case=False, na=False)]

# Team filter
if team_filter and team_filter != "All":
    filtered_df = filtered_df[filtered_df["Team"] == team_filter]

# Year filter
if year_filter and year_filter != "All":
    filtered_df = filtered_df[filtered_df["year"] == year_filter]

# Stat range filter
filtered_df = filtered_df[
    (filtered_df[stat_to_filter] >= stat_range[0]) &
    (filtered_df[stat_to_filter] <= stat_range[1])
]

# Format numeric columns for display
display_df = filtered_df.copy()
for col in display_df.select_dtypes(include=[np.number]).columns:
    if col == "swing_plus":
        display_df[col] = display_df[col].astype("Int64")
    elif display_df[col].between(0, 1).all():
        display_df[col] = display_df[col].round(3)
    else:
        display_df[col] = display_df[col].round(2)

# Show table
st.write(f"Showing {len(display_df)} results")
st.dataframe(display_df, use_container_width=True)

# CSV download
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered results as CSV",
    data=csv,
    file_name="filtered_swing_plus_results.csv",
    mime="text/csv"
)
