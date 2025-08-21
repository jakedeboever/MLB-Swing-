import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import pearsonr  # <-- Add this import

# ... [rest of your code is unchanged] ...

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

        # Calculate Pearson correlation coefficient r
        r, _ = pearsonr(filtered_df[x_axis], filtered_df[y_axis])

        st.markdown(f"**Trendline equation:** y = {slope:.3f}x + {intercept:.3f}")
        st.markdown(f"**r (Pearson correlation):** {r:.3f}")
        st.markdown(f"**R² between {x_axis} and {y_axis}:** {r2:.3f}")
    except Exception as e:
        st.warning(f"Could not calculate trendline: {e}")

    st.plotly_chart(fig, use_container_width=True)
