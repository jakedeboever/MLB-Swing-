import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("swing_predictions.csv")
    # Create difference column
    df["xwoba_diff"] = df["xwobacon"] - df["predicted_xwobacon"]
    # Round swing_plus to whole number
    if "swing_plus" in df.columns:
        df["swing_plus"] = df["swing_plus"].round(0).astype(int)
    # Round xwobacons to 3 decimals
    if "xwobacon" in df.columns:
        df["xwobacon"] = df["xwobacon"].round(3)
    if "predicted_xwobacon" in df.columns:
        df["predicted_xwobacon"] = df["predicted_xwobacon"].round(3)
    if "xwoba_diff" in df.columns:
        df["xwoba_diff"] = df["xwoba_diff"].round(3)
    # Reorder columns
    cols = [
        "last_name, first_name",
        "year",
        "swing_plus",
        "xwobacon",
        "predicted_xwobacon",
        "xwoba_diff"
    ]
    remaining = [c for c in df.columns if c not in cols]
    df = df[cols + remaining]
    return df

# Load the dataframe
df = load_data()

st.title("Swing Predictions Explorer")

# Sidebar filters
st.sidebar.header("Filters")

# Year filter with aggregation option
years = sorted(df["year"].dropna().unique())
selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)
avg_by_player = st.sidebar.checkbox("Aggregate across all years per player")

# Player name filter
players = sorted(df["last_name, first_name"].dropna().unique())
selected_players = st.sidebar.multiselect("Select Player(s)", players)

# Apply filters
filtered_df = df[df["year"].isin(selected_years)]
if selected_players:
    filtered_df = filtered_df[filtered_df["last_name, first_name"].isin(selected_players)]

# Aggregate by player if selected
if avg_by_player:
    numeric_cols = filtered_df.select_dtypes(include="number").columns
    filtered_df = filtered_df.groupby("last_name, first_name", as_index=False)[numeric_cols].mean()
    filtered_df["year"] = "All"

# Numeric range filters
st.sidebar.subheader("Numeric Filters")
for col in filtered_df.select_dtypes(include="number").columns:
    min_val, max_val = float(filtered_df[col].min()), float(filtered_df[col].max())
    sel_min, sel_max = st.sidebar.slider(f"{col}", min_val, max_val, (min_val, max_val))
    filtered_df = filtered_df[(filtered_df[col] >= sel_min) & (filtered_df[col] <= sel_max)]

# Sorting
sort_col = st.sidebar.selectbox("Sort by", filtered_df.columns, index=2)
sort_asc = st.sidebar.radio("Order", ["Ascending", "Descending"]) == "Ascending"

filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_asc)

# Player search above table
search_term = st.text_input("Search player by name")
if search_term:
    filtered_df = filtered_df[filtered_df["last_name, first_name"].str.contains(search_term, case=False, na=False)]

# Show data
st.dataframe(filtered_df, use_container_width=True)

# Download button
st.download_button(
    label="Download filtered data as CSV",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_swing_predictions.csv",
    mime="text/csv"
)

# Scatterplot option
st.subheader("Scatterplot Explorer")
num_cols = filtered_df.select_dtypes(include="number").columns
if len(num_cols) >= 2:
    x_axis = st.selectbox("X-axis", num_cols, index=0)
    y_axis = st.selectbox("Y-axis", num_cols, index=1)
    fig = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        hover_data=["last_name, first_name", "year"]
    )

    # Fit regression line using statsmodels for stability
    try:
        X = sm.add_constant(filtered_df[x_axis])
        model = sm.OLS(filtered_df[y_axis], X).fit()
        slope = model.params[x_axis]
        intercept = model.params["const"]
        r2 = model.rsquared
        st.markdown(f"**Trendline equation:** y = {slope:.3f}x + {intercept:.3f}")
        st.markdown(f"**R² between {x_axis} and {y_axis}:** {r2:.3f}")
    except Exception as e:
        st.warning(f"Could not calculate trendline: {e}")

    st.plotly_chart(fig, use_container_width=True)
