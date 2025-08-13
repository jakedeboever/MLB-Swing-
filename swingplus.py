import streamlit as st
import pandas as pd

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("swing_plus_results.csv")
    return df

df = load_data()

st.title("Swing+ Results Viewer")

# Multi-name search
search_query = st.text_input("Search by player name (comma-separated for multiple):")

# Dropdown filters
team_filter = None
year_filter = None

if "Team" in df.columns:
    teams = ["All"] + sorted(df["Team"].dropna().unique())
    team_filter = st.selectbox("Filter by Team:", teams)

if "Year" in df.columns:
    years = ["All"] + sorted(df["Year"].dropna().unique())
    year_filter = st.selectbox("Filter by Year:", years)

# Numeric stat filter
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
stat_to_filter = st.selectbox("Select a stat to filter:", numeric_columns)
min_val = float(df[stat_to_filter].min())
max_val = float(df[stat_to_filter].max())
stat_range = st.slider(f"Filter {stat_to_filter} range:", min_val, max_val, (min_val, max_val))

# Apply filters
filtered_df = df.copy()

# Name filter (supports multiple names)
if search_query:
    names_list = [name.strip() for name in search_query.split(",") if name.strip()]
    regex_pattern = "|".join([f"({name})" for name in names_list])
    filtered_df = filtered_df[filtered_df['Name'].str.contains(regex_pattern, case=False, na=False)]

# Team filter
if team_filter and team_filter != "All":
    filtered_df = filtered_df[filtered_df['Team'] == team_filter]

# Year filter
if year_filter and year_filter != "All":
    filtered_df = filtered_df[filtered_df['Year'] == year_filter]

# Stat range filter
filtered_df = filtered_df[
    (filtered_df[stat_to_filter] >= stat_range[0]) &
    (filtered_df[stat_to_filter] <= stat_range[1])
]

# Show results
st.write(f"Showing {len(filtered_df)} results")
st.dataframe(filtered_df, use_container_width=True)

# CSV download button
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download filtered results as CSV",
    data=csv,
    file_name="filtered_swing_plus_results.csv",
    mime="text/csv"
)
