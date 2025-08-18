import pandas as pd
import streamlit as st

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

# --- Categorical filters ---
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    options = df[col].dropna().unique().tolist()
    options.sort()
    selected = st.sidebar.multiselect(f"Select {col}", options, default=options)
    df = df[df[col].isin(selected)]

# --- Numeric filters ---
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

# Display table
st.subheader("Filtered Data")
st.dataframe(df)

