import pandas as pd
import streamlit as st
import plotly.express as px

@st.cache_data
def load_data():
    # Load CSV
    df = pd.read_csv("merged_fangraphs_swing_clean.csv")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Map possible names → pick whichever exists in the file
    col_map = {
        "player_name": ["player_name", "last_name, first_name"],
        "season": ["season", "year"],
        "team": ["team"],
        "swing_plus": ["swing_plus"],
        "xwobacon": ["xwobacon"],
        "predicted_xwobacon": ["predicted_xwobacon"],
        "xwoba_diff": ["xwoba_diff"],
    }

    # Create consistent column set
    for target, options in col_map.items():
        for opt in options:
            if opt in df.columns:
                df.rename(columns={opt: target}, inplace=True)
                break

    # Reorder for readability
    preferred = list(col_map.keys())
    cols = [c for c in preferred if c in df.columns]
    remaining = [c for c in df.columns if c not in cols]
    df = df[cols + remaining]

    return df

# Load data
df = load_data()

st.title("MLB Swing+ Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

# --- Search bar ---
search_query = st.sidebar.text_input("Search Player")
if search_query:
    df = df[df["player_name"].str.contains(search_query, case=False, na=False)]

# --- Team filter (multiselect but defaults to ALL) ---
if "team" in df.columns:
    team_options = df["team"].dropna().unique().tolist()
    team_options.sort()
    selected_teams = st.sidebar.multiselect("Select Teams", team_options, default=team_options)
    df = df[df["team"].isin(selected_teams)]

# --- Season filter (multiselect but defaults to ALL) ---
if "season" in df.columns:
    season_options = sorted(df["season"].dropna().unique().tolist())
    selected_seasons = st.sidebar.multiselect("Select Seasons", season_options, default=season_options)
    df = df[df["season"].isin(selected_seasons)]

# --- Numeric filters (sliders, default = full range) ---
st.sidebar.subheader("Numeric Filters")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
for col in numeric_cols:
    col_data = df[col].dropna()
    if col_data.empty:
        continue
    min_val, max_val = float(col_data.min()), float(col_data.max())
    if min_val == max_val:
        continue
    sel_min, sel_max = st.sidebar.slider(
        f"{col}",
        min_val,
        max_val,
        (min_val, max_val)
    )
    df = df[(df[col] >= sel_min) & (df[col] <= sel_max)]

# --- Data table ---
st.subheader("Filtered Data")
st.dataframe(df)

# --- Scatter plot ---
st.subheader("Scatter Plot")
numeric_choices = df.select_dtypes(include="number").columns.tolist()
if len(numeric_choices) >= 2:
    x_axis = st.selectbox("X-axis", numeric_choices, index=0)
    y_axis = st.selectbox("Y-axis", numeric_choices, index=1)
    color_var = st.selectbox("Color By", ["team", "season", "player_name"], index=0)

    fig = px.scatter(df, x=x_axis, y=y_axis, color=df[color_var] if color_var in df else None,
                     hover_data=["player_name", "team", "season"])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Not enough numeric columns for scatter plot.")
