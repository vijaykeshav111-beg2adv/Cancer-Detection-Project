import streamlit as st
import requests

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
# API_URL = "http://localhost:8000/predict"  # change if backend is hosted elsewhere
API_URL = "http://cancer-backend:8000/predict" 
st.set_page_config(page_title="Cancer Detection App", layout="wide")

st.title("🩺 Breast Cancer Detection")
st.write(
    "Enter the tumor measurements below and click **Predict** to get the "
    "model's prediction from the FastAPI backend."
)

# ---------------------------------------------------------
# Feature groups (matches the Pydantic model in the backend)
# ---------------------------------------------------------
mean_features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
]

se_features = [
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
]

worst_features = [
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

all_features = mean_features + se_features + worst_features

# ---------------------------------------------------------
# Sidebar options
# ---------------------------------------------------------
st.sidebar.header("Settings")

api_url_input = st.sidebar.text_input(
    "Backend API URL",
    value=API_URL,
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Tip: run the backend with:\n\n"
    "```bash\n"
    "uvicorn main:app --reload\n"
    "```"
)

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
input_values = {}

with st.form("prediction_form"):

    tab_mean, tab_se, tab_worst = st.tabs(
        ["Mean", "Standard Error", "Worst"]
    )

    def render_inputs(tab, features):
        with tab:
            cols = st.columns(2)

            for i, feature in enumerate(features):
                label = feature.replace("_", " ").title()

                with cols[i % 2]:
                    input_values[feature] = st.number_input(
                        label,
                        value=0.0,
                        format="%.5f",
                        key=feature,
                    )

    render_inputs(tab_mean, mean_features)
    render_inputs(tab_se, se_features)
    render_inputs(tab_worst, worst_features)

    submitted = st.form_submit_button("🔍 Predict")

# ---------------------------------------------------------
# Call backend on submit
# ---------------------------------------------------------
if submitted:

    payload = {
        feature: input_values[feature]
        for feature in all_features
    }

    with st.spinner("Contacting model API..."):

        try:
            response = requests.post(
                api_url_input,
                json=payload,
                timeout=15,
            )

            response.raise_for_status()

            result = response.json()

            prediction = result.get("prediction")

            st.subheader("Result")
            st.success(f"Raw model output: {prediction}")

            # Optional: friendlier interpretation if prediction is 0/1
            try:
                pred_value = (
                    prediction[0]
                    if isinstance(prediction, list)
                    else prediction
                )

                if pred_value in (0, 1, 0.0, 1.0):
                    label = (
                        "Malignant"
                        if int(pred_value) == 1
                        else "Benign"
                    )
                    st.info(f"Interpreted prediction: **{label}**")

            except Exception:
                pass

        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not connect to the API at `{api_url_input}`. "
                "Make sure the FastAPI backend is running."
            )

        except requests.exceptions.HTTPError as e:
            st.error(f"API returned an error: {e}\n\n{response.text}")

        except Exception as e:
            st.error(f"Unexpected error: {e}")

st.markdown("---")

st.caption(
    "This tool is for demonstration purposes only and is not a substitute "
    "for professional medical diagnosis."
)