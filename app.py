# app.py
# Run with:
#   pip install streamlit pandas numpy scikit-learn plotly statsmodels
#   streamlit run app.py

import io
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------
# App Config
# ---------------------------------------------------
st.set_page_config(
    page_title="From Notebook to App",
    page_icon="📊",
    layout="wide",
)

st.title("📊 From Notebook to App")
st.caption(
    "Build an interactive, shareable analytics app — no notebooks, no screenshots, no code edits."
)

# ---------------------------------------------------
# Sample Dataset (Advertising)
# ---------------------------------------------------
SAMPLE_DATA = """TV,Radio,Newspaper,Sales
230.1,37.8,69.2,22.1
44.5,39.3,45.1,10.4
17.2,45.9,69.3,9.3
151.5,41.3,58.5,18.5
180.8,10.8,58.4,12.9
8.7,48.9,75.0,7.2
57.5,32.8,23.5,11.8
120.2,19.6,11.6,13.2
8.6,2.1,1.0,4.8
199.8,2.6,21.2,10.6
"""

@st.cache_data
def load_sample():
    return pd.read_csv(io.StringIO(SAMPLE_DATA))

@st.cache_data
def load_uploaded(file):
    return pd.read_csv(file)

# ---------------------------------------------------
# Sidebar — Data Source
# ---------------------------------------------------
st.sidebar.header("1️⃣ Data Source")

data_choice = st.sidebar.radio(
    "Choose dataset",
    ["Use sample dataset", "Upload CSV"],
)

if data_choice == "Use sample dataset":
    df = load_sample()
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = load_uploaded(uploaded)
    else:
        st.stop()

# ---------------------------------------------------
# Sidebar — Columns
# ---------------------------------------------------
st.sidebar.header("2️⃣ Column Selection")

all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

target = st.sidebar.selectbox(
    "Target (what to predict)",
    options=numeric_cols,
    index=numeric_cols.index("Sales") if "Sales" in numeric_cols else 0,
)

features = st.sidebar.multiselect(
    "Features (inputs)",
    options=[c for c in numeric_cols if c != target],
    default=[c for c in numeric_cols if c != target],
)

if not features:
    st.warning("Select at least one feature.")
    st.stop()

df = df[[target] + features].dropna()

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Explore",
    "📈 Visualize",
    "🤖 Train Model",
    "🎯 Predict & Export"
])

# ---------------------------------------------------
# TAB 1 — Explore
# ---------------------------------------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download current data",
        data=csv,
        file_name="dataset.csv",
        mime="text/csv",
    )

# ---------------------------------------------------
# TAB 2 — Visualize (Interactive)
# ---------------------------------------------------
with tab2:
    st.subheader("Interactive Filters")

    filtered = df.copy()
    for col in numeric_cols:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        selected = st.slider(
            f"{col} range",
            min_val,
            max_val,
            (min_val, max_val),
        )
        filtered = filtered[
            (filtered[col] >= selected[0]) &
            (filtered[col] <= selected[1])
        ]

    st.write(f"Filtered rows: **{len(filtered)}**")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered,
            x=target,
            title=f"Distribution of {target}",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        x_axis = st.selectbox("X-axis", features)
        fig = px.scatter(
            filtered,
            x=x_axis,
            y=target,
            trendline="ols",
            title=f"{x_axis} vs {target}",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = filtered.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Trends")
    fig = px.line(
        filtered.reset_index(),
        x="index",
        y=features,
        title="Feature Trends",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# TAB 3 — Train Model
# ---------------------------------------------------
with tab3:
    st.subheader("Train a Model")

    model_type = st.selectbox(
        "Model",
        ["Linear Regression", "Ridge Regression", "Random Forest"],
    )

    test_size = st.slider("Test size", 0.1, 0.4, 0.2)
    random_state = st.number_input("Random seed", 0, 9999, 42)

    if st.button("🚀 Train Model", type="primary"):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Ridge Regression":
            model = Ridge(alpha=1.0)
        else:
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
            )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "R2": r2_score(y_test, preds),
            "RMSE": mean_squared_error(y_test, preds) ** 0.5,
            "MAE": mean_absolute_error(y_test, preds),
        }

        # ✅ STORE EVERYTHING NEEDED LATER
        st.session_state["trained_model"] = {
            "model": model,
            "features": features,
            "target": target,
            "metrics": metrics,
            "X_test": X_test,
            "y_test": y_test,
        }

    trained = st.session_state.get("trained_model")

    if trained:
        st.success("Model trained successfully")

        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{trained['metrics']['R2']:.3f}")
        c2.metric("RMSE", f"{trained['metrics']['RMSE']:.3f}")
        c3.metric("MAE", f"{trained['metrics']['MAE']:.3f}")

        X_test = trained["X_test"]
        y_test = trained["y_test"]

        y_test_pred = trained["model"].predict(X_test)

        fig = px.scatter(
            x=y_test,
            y=y_test_pred,
            labels={"x": "Actual", "y": "Predicted"},
            title="Actual vs Predicted",
        )
        fig.add_shape(
            type="line",
            x0=y_test.min(),
            y0=y_test.min(),
            x1=y_test.max(),
            y1=y_test.max(),
        )
        st.plotly_chart(fig, use_container_width=True)

        if hasattr(trained["model"], "feature_importances_"):
            imp = pd.DataFrame({
                "Feature": features,
                "Importance": trained["model"].feature_importances_,
            }).sort_values("Importance", ascending=False)

            fig = px.bar(
                imp,
                x="Feature",
                y="Importance",
                title="Feature Importance",
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# TAB 4 — Predict & Export
# ---------------------------------------------------
with tab4:
    trained = st.session_state.get("trained_model")

    if not trained:
        st.warning("Train a model first.")
        st.stop()

    st.subheader("Single Prediction")

    inputs = {}
    for f in trained["features"]:
        inputs[f] = st.number_input(f, value=float(df[f].median()))

    if st.button("🎯 Predict"):
        X_input = pd.DataFrame([inputs])
        prediction = trained["model"].predict(X_input)[0]
        st.success(f"Predicted {trained['target']}: **{prediction:.2f}**")

    st.subheader("Export Trained Model")

    model_bytes = pickle.dumps(trained)
    st.download_button(
        "⬇️ Download model (.pkl)",
        data=model_bytes,
        file_name="trained_model.pkl",
        mime="application/octet-stream",
    )

    st.code(
        """# Load later
import pickle

with open("trained_model.pkl", "rb") as f:
    payload = pickle.load(f)

model = payload["model"]
features = payload["features"]
""",
        language="python",
    )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.caption(
    "This app replaces notebooks, screenshots, and repeated code edits with one interactive, shareable interface."
)
