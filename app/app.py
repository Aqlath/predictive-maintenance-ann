from pathlib import Path
import base64
import json
import urllib.request
import urllib.error

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ann_baseline.keras"
SCALER_PATH = ROOT / "models" / "scaler_baseline.pkl"
FEATURE_PATH = ROOT / "models" / "feature_columns.pkl"
X_TRAIN_PATH = ROOT / "processed" / "X_train.csv"
Y_TRAIN_PATH = ROOT / "processed" / "y_train.csv"

FEATURES = [
    "setting_1", "setting_2",
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]
RUL_CAP = 125

st.set_page_config(
    page_title="Aircraft Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# BACKGROUND
# ============================================================

def find_background_image():
    candidates = [
        ROOT / "assets" / f"aircraft_background{ext}"
        for ext in (".jpg", ".jpeg", ".png", ".webp")
    ] + [
        ROOT / "app" / "assets" / f"aircraft_background{ext}"
        for ext in (".jpg", ".jpeg", ".png", ".webp")
    ]
    return next((path for path in candidates if path.exists()), None)


def apply_background():
    image_path = find_background_image()
    if image_path is None:
        return

    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(image_path.suffix.lower(), "image/jpeg")

    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(5,15,30,.88), rgba(5,15,30,.94)),
                url("data:{mime};base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        [data-testid="stSidebar"] {{
            background-color: rgba(5,18,35,.96);
        }}
        h1,h2,h3 {{ color: #EAF6FF; }}
        p,label,span {{ color: #D6E6F5; }}
        [data-testid="stMetric"] {{
            background: rgba(20,55,85,.55);
            padding: 14px;
            border-radius: 12px;
            border: 1px solid rgba(100,180,230,.25);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_background()


# ============================================================
# LOAD DATA AND SAVED MODEL ARTIFACTS
# ============================================================

@st.cache_data
def load_training_data():
    if not X_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training features not found: {X_TRAIN_PATH}")
    if not Y_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training target not found: {Y_TRAIN_PATH}")

    X = pd.read_csv(X_TRAIN_PATH)
    y_df = pd.read_csv(Y_TRAIN_PATH)

    missing = [f for f in FEATURES if f not in X.columns]
    if missing:
        raise ValueError(f"Missing columns in X_train.csv: {missing}")

    X = X[FEATURES].copy()
    y = y_df.iloc[:, 0].astype(float)

    if len(X) != len(y):
        raise ValueError("X_train.csv and y_train.csv have different row counts.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Training features contain NaN or infinite values.")

    return X, y


@st.cache_resource
def load_model_artifacts():
    for path in (MODEL_PATH, SCALER_PATH, FEATURE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    # Load the existing model only. This app never trains or saves a model.
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    scaler = joblib.load(SCALER_PATH)
    saved_features = list(joblib.load(FEATURE_PATH))

    if saved_features != FEATURES:
        raise ValueError("Saved feature order differs from the application's feature order.")
    if getattr(scaler, "n_features_in_", None) != len(FEATURES):
        raise ValueError("Saved scaler does not accept 17 features.")
    if model.input_shape[-1] != len(FEATURES):
        raise ValueError("Saved ANN model does not accept 17 features.")

    return model, scaler


# ============================================================
# TRAINING STATISTICS AND SAMPLE PROFILES
# ============================================================

@st.cache_data
def calculate_training_statistics():
    X, _ = load_training_data()
    result = {}

    for feature in FEATURES:
        s = X[feature]
        minimum = float(s.min())
        maximum = float(s.max())
        q1 = float(s.quantile(0.25))
        median = float(s.median())
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        span = maximum - minimum

        # Statistical screening boundary only; not a physical safety limit.
        if iqr > 0:
            margin = 6.0 * iqr
        else:
            margin = max(span * 1.5, abs(median) * 0.01, 1e-5)

        result[feature] = {
            "min": minimum,
            "max": maximum,
            "q1": q1,
            "median": median,
            "q3": q3,
            "iqr": iqr,
            "critical_min": q1 - margin,
            "critical_max": q3 + margin,
        }

    return result


@st.cache_data
def create_sample_profiles():
    X, y = load_training_data()
    bands = [(0, 25), (26, 50), (51, 75), (76, 100), (101, 125)]
    chosen = []

    # Pick up to two examples from each RUL band.
    for low, high in bands:
        indices = np.where((y.to_numpy() >= low) & (y.to_numpy() <= high))[0]
        if len(indices):
            positions = np.linspace(0, len(indices) - 1, min(2, len(indices)), dtype=int)
            for pos in positions:
                idx = int(indices[pos])
                if idx not in chosen:
                    chosen.append(idx)

    # Fill to ten profiles if a band had too few examples.
    if len(chosen) < min(10, len(X)):
        remaining = [i for i in range(len(X)) if i not in chosen]
        needed = min(10, len(X)) - len(chosen)
        if remaining and needed:
            positions = np.linspace(0, len(remaining) - 1, min(needed, len(remaining)), dtype=int)
            chosen.extend(remaining[pos] for pos in positions)

    profiles = {}
    for number, idx in enumerate(chosen[:10], start=1):
        profiles[f"Sample Engine {number:02d}"] = {
            "features": X.iloc[idx].to_dict(),
            "actual_rul": float(y.iloc[idx]),
        }
    return profiles


# ============================================================
# MULTIVARIATE STATISTICAL ANOMALY SCREEN
# ============================================================

@st.cache_resource
def build_multivariate_detector():
    X, _ = load_training_data()
    values = X[FEATURES].to_numpy(dtype=float)
    center = np.mean(values, axis=0)
    covariance = np.cov(values, rowvar=False)

    # Small regularization helps when sensor columns are highly correlated.
    diagonal_scale = float(np.mean(np.diag(covariance)))
    regularization = max(diagonal_scale * 1e-6, 1e-12)
    covariance = covariance + np.eye(len(FEATURES)) * regularization
    inverse_covariance = np.linalg.pinv(covariance)

    centered = values - center
    distances = np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered)
    threshold = float(np.quantile(distances, 0.995))
    return center, inverse_covariance, threshold


# ============================================================
# COMMON VALIDATION AND PREDICTION PIPELINE
# ============================================================

def validate_sensor_input(input_data):
    stats = calculate_training_statistics()
    center, inverse_covariance, threshold = build_multivariate_detector()

    values = np.array([float(input_data[f]) for f in FEATURES], dtype=float)
    if not np.isfinite(values).all():
        return {
            "quality": "CRITICAL",
            "warning_features": [],
            "critical_features": ["One or more values are NaN or infinite"],
            "multivariate_warning": False,
            "mahalanobis_distance": None,
            "distance_threshold": threshold,
            "stats": stats,
        }

    warning_features = []
    critical_features = []

    for feature, value in zip(FEATURES, values):
        limits = stats[feature]

        if value < limits["min"] or value > limits["max"]:
            warning_features.append(feature)

        if value < limits["critical_min"] or value > limits["critical_max"]:
            critical_features.append(feature)

    centered = values - center
    distance = float(centered @ inverse_covariance @ centered)
    multivariate_warning = distance > threshold

    if critical_features:
        quality = "CRITICAL"
    elif warning_features or multivariate_warning:
        quality = "WARNING"
    else:
        quality = "NORMAL"

    return {
        "quality": quality,
        "warning_features": warning_features,
        "critical_features": critical_features,
        "multivariate_warning": multivariate_warning,
        "mahalanobis_distance": distance,
        "distance_threshold": threshold,
        "stats": stats,
    }


def predict_rul(input_data, model, scaler):
    input_df = pd.DataFrame([[input_data[f] for f in FEATURES]], columns=FEATURES)
    scaled = scaler.transform(input_df)
    raw = float(np.asarray(model.predict(scaled, verbose=0)).reshape(-1)[0])

    if not np.isfinite(raw):
        raise ValueError("The ANN returned a non-finite prediction.")

    clipped = float(np.clip(raw, 0, RUL_CAP))
    return raw, clipped


def process_sensor_readings(input_data, source, model, scaler):
    st.session_state["last_prediction"] = None
    st.session_state["input_error"] = None

    try:
        # Enforce the exact expected input schema and numeric values.
        clean = {}
        for feature in FEATURES:
            if feature not in input_data:
                raise ValueError(f"Missing required feature: {feature}")
            if isinstance(input_data[feature], bool):
                raise ValueError(f"{feature} must be numeric.")
            clean[feature] = float(input_data[feature])
            if not np.isfinite(clean[feature]):
                raise ValueError(f"{feature} must be finite.")

        validation = validate_sensor_input(clean)

        if validation["quality"] == "CRITICAL":
            st.session_state["input_error"] = {
                "message": "Critical statistical outlier detected. Prediction was blocked.",
                "features": validation["critical_features"],
            }
            return

        raw, rul = predict_rul(clean, model, scaler)
        st.session_state["last_prediction"] = {
            "input": clean,
            "source": source,
            "raw_prediction": raw,
            "predicted_rul": rul,
            "validation": validation,
        }

    except Exception as exc:
        st.session_state["input_error"] = {
            "message": str(exc),
            "features": [],
        }


# ============================================================
# API RESPONSE PARSING AND HTTP GET
# ============================================================

def find_sensor_payload(data):
    if not isinstance(data, dict):
        raise ValueError("API response must be a JSON object.")

    if all(feature in data for feature in FEATURES):
        return data

    for key in ("data", "sensors", "readings", "latest", "payload"):
        nested = data.get(key)
        if isinstance(nested, dict):
            try:
                return find_sensor_payload(nested)
            except ValueError:
                pass

    raise ValueError(
        "Could not find all 17 required sensor fields in the API response."
    )


def convert_api_sensor_values(payload):
    values = {}
    missing = []

    for feature in FEATURES:
        if feature not in payload:
            missing.append(feature)
            continue

        value = payload[feature]
        if value is None or isinstance(value, bool):
            raise ValueError(f"API value for {feature} is invalid.")

        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"API value for {feature} is not numeric.")

        if not np.isfinite(number):
            raise ValueError(f"API value for {feature} is not finite.")

        values[feature] = number

    if missing:
        raise ValueError(f"API response is missing required fields: {missing}")

    return values


def fetch_sensor_data(api_url, auth_type, auth_token):
    api_url = api_url.strip()

    if not api_url:
        raise ValueError("Enter an API endpoint URL.")
    if not api_url.startswith(("https://", "http://")):
        raise ValueError("API endpoint must start with https:// or http://.")
    if auth_type != "No authentication" and not auth_token:
        raise ValueError("Enter the required API credential.")

    headers = {"Accept": "application/json"}
    if auth_type == "Bearer Token":
        headers["Authorization"] = f"Bearer {auth_token}"
    elif auth_type == "API Key Header":
        headers["X-API-Key"] = auth_token

    request = urllib.request.Request(api_url, headers=headers, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.status
            body = response.read().decode("utf-8")

        if status != 200:
            raise ValueError(f"API returned HTTP status {status}.")

        try:
            response_json = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("API did not return valid JSON.") from exc

        return convert_api_sensor_values(find_sensor_payload(response_json))

    except urllib.error.HTTPError as exc:
        raise ValueError(f"API HTTP error {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Could not connect to API: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ValueError("API request timed out after 15 seconds.") from exc


# ============================================================
# STATUS / RECOMMENDATIONS
# ============================================================

def get_engine_status(rul):
    if rul <= 15:
        return "CRITICAL", "error", "Immediate maintenance inspection is recommended."
    if rul <= 40:
        return "WARNING", "warning", "Schedule maintenance and monitor sensor readings closely."
    return "NORMAL", "success", "Continue routine monitoring and scheduled maintenance."


# ============================================================
# INITIALIZE APPLICATION
# ============================================================

st.title("✈️ Aircraft Predictive Maintenance")
st.markdown(
    """
    ### ENGINE HEALTH MONITORING & REMAINING USEFUL LIFE PREDICTION
    ANN-based Remaining Useful Life prediction using NASA C-MAPSS FD001.
    """
)
st.divider()

try:
    model, scaler = load_model_artifacts()
    X_train, y_train = load_training_data()
    sample_profiles = create_sample_profiles()
    training_stats = calculate_training_statistics()
except Exception as exc:
    st.error(f"Application initialization failed: {exc}")
    st.stop()


# ============================================================
# SIDEBAR: MANUAL INPUT OR API CONNECTION
# ============================================================

st.sidebar.title("✈️ ENGINE CONTROL PANEL")
input_mode = st.sidebar.radio(
    "Select input method",
    ["Manual Input", "API Connection"],
    key="input_mode",
)

st.sidebar.divider()

if input_mode == "Manual Input":
    st.sidebar.subheader("1. Manual Input")
    st.sidebar.caption("Enter readings manually or load a demonstration engine.")

    sample_name = st.sidebar.selectbox(
        "Choose a demonstration sample",
        list(sample_profiles.keys()),
        key="manual_sample_selector",
    )

    if st.sidebar.button("Load Sample Engine", use_container_width=True):
        selected = sample_profiles[sample_name]["features"]
        for feature in FEATURES:
            st.session_state[f"manual_{feature}"] = float(selected[feature])
        st.session_state["manual_loaded_sample"] = sample_name
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("input_error", None)
        st.rerun()

    if st.session_state.get("manual_loaded_sample"):
        st.sidebar.success(f"Loaded: {st.session_state['manual_loaded_sample']}")

    st.sidebar.info("After loading a sample, submit the manual form to predict RUL.")

else:
    st.sidebar.subheader("2. API Connection")
    st.sidebar.caption("Configure a future industrial sensor API.")

    st.sidebar.info(
        "No real API is required for the demo test. Use the simulated API "
        "sample test below to exercise the API parsing and prediction pipeline."
    )

    api_url = st.sidebar.text_input(
        "API Endpoint URL",
        placeholder="https://your-server.com/api/sensors/latest",
        key="api_endpoint",
    )

    auth_type = st.sidebar.selectbox(
        "Authentication",
        ["No authentication", "Bearer Token", "API Key Header"],
        key="api_auth_type",
    )

    auth_token = ""
    if auth_type != "No authentication":
        auth_token = st.sidebar.text_input(
            "API Credential",
            type="password",
            key="api_auth_token",
        )

    if st.sidebar.button("Fetch Latest Sensor Data", use_container_width=True):
        st.session_state["last_prediction"] = None
        st.session_state["input_error"] = None
        st.session_state["api_test_mode"] = False
        st.session_state["api_fetch_success"] = False

        try:
            with st.spinner("Connecting to the configured API..."):
                api_values = fetch_sensor_data(api_url, auth_type, auth_token)

            st.session_state["api_sensor_values"] = api_values
            process_sensor_readings(api_values, "Real API Connection", model, scaler)
            st.session_state["api_fetch_success"] = (
                st.session_state.get("last_prediction") is not None
            )
        except Exception as exc:
            st.session_state["input_error"] = {"message": str(exc), "features": []}

    st.sidebar.divider()
    st.sidebar.markdown("### Test API with Demo Samples")
    api_sample_name = st.sidebar.selectbox(
        "Select demo engine for API test",
        list(sample_profiles.keys()),
        key="api_demo_sample_selector",
    )

    if st.sidebar.button("Test API Using Sample", use_container_width=True):
        st.session_state["last_prediction"] = None
        st.session_state["input_error"] = None

        try:
            demo_values = sample_profiles[api_sample_name]["features"].copy()

            # Mimic the shape of a JSON response returned by a REST API.
            simulated_response = {"data": demo_values}
            extracted_payload = find_sensor_payload(simulated_response)
            api_values = convert_api_sensor_values(extracted_payload)

            st.session_state["api_sensor_values"] = api_values
            st.session_state["api_test_mode"] = True
            st.session_state["api_test_sample_name"] = api_sample_name

            process_sensor_readings(
                api_values,
                "Simulated API Sample",
                model,
                scaler,
            )
            st.session_state["api_fetch_success"] = (
                st.session_state.get("last_prediction") is not None
            )
        except Exception as exc:
            st.session_state["input_error"] = {"message": str(exc), "features": []}

    with st.sidebar.expander("Expected JSON response format"):
        example = {feature: training_stats[feature]["median"] for feature in FEATURES}
        st.code(json.dumps({"data": example}, indent=2), language="json")

    st.sidebar.caption(
        "Real API mode uses HTTP GET. Supported authentication: none, "
        "Bearer token, or X-API-Key header. Automatic polling is not enabled."
    )



m1, m2, m3 = st.columns(3)
m1.metric("Prediction Model", "ANN")
m2.metric("Input Features", "17")
m3.metric("Demo Profiles", str(len(sample_profiles)))




if input_mode == "Manual Input":
    st.subheader("🛠️ Manual Engine Sensor Input")
    st.write("Enter engine operating settings and sensor readings.")
    st.info(
        "Training min/max values are reference ranges, not certified "
        "physical sensor operating limits. Out-of-range readings may still "
        "be predicted if they do not cross the broad statistical block limits."
    )

    # Set defaults once, using the training median.
    for feature in FEATURES:
        key = f"manual_{feature}"
        if key not in st.session_state:
            st.session_state[key] = float(training_stats[feature]["median"])

    with st.form("manual_engine_form"):
        left, right = st.columns(2)
        manual_values = {}

        for idx, feature in enumerate(FEATURES):
            col = left if idx % 2 == 0 else right
            limits = training_stats[feature]

            with col:
                manual_values[feature] = st.number_input(
                    feature,
                    key=f"manual_{feature}",
                    format="%.6f",
                    step=max((limits["max"] - limits["min"]) / 100.0, 0.000001),
                    help=(
                        f"Observed training range: {limits['min']:.6f} "
                        f"to {limits['max']:.6f}"
                    ),
                )

        submitted = st.form_submit_button(
            "🚀 PREDICT REMAINING USEFUL LIFE",
            use_container_width=True,
        )

    if submitted:
        process_sensor_readings(
            {f: float(manual_values[f]) for f in FEATURES},
            "Manual Input",
            model,
            scaler,
        )



input_error = st.session_state.get("input_error")
if input_error:
    st.error(input_error["message"])
    for feature in input_error.get("features", []):
        st.write(f"- {feature}")



if input_mode == "API Connection" and st.session_state.get("api_fetch_success"):
    if st.session_state.get("api_test_mode"):
        st.info(
            "SIMULATED API TEST: the selected sample was wrapped in a JSON "
            "response and passed through API extraction, validation, scaling, "
            "and ANN prediction. No external API was contacted."
        )
    else:
        st.success("Sensor readings were successfully retrieved from the real API.")

    api_values = st.session_state.get("api_sensor_values", {})
    if api_values:
        with st.expander("View received API sensor readings", expanded=True):
            api_df = pd.DataFrame({
                "Feature": FEATURES,
                "Received Value": [api_values[f] for f in FEATURES],
            })
            st.dataframe(api_df, use_container_width=True, hide_index=True)



result = st.session_state.get("last_prediction")

if result is not None:
    st.divider()
    st.subheader("📊 Prediction Results")
    st.caption(f"Input source: {result['source']}")

    rul = result["predicted_rul"]
    raw = result["raw_prediction"]
    validation = result["validation"]

    if validation["quality"] == "NORMAL":
        st.success("Sensor Input Quality: NORMAL")
    elif validation["quality"] == "WARNING":
        st.warning(
            "Sensor Input Quality: WARNING — one or more values are outside "
            "the observed training range or the combined sensor pattern is unusual."
        )
        if validation["warning_features"]:
            st.write("Features outside observed training ranges:")
            for feature in validation["warning_features"]:
                lim = validation["stats"][feature]
                st.write(
                    f"- {feature}: {result['input'][feature]:.6f} "
                    f"(training range {lim['min']:.6f}–{lim['max']:.6f})"
                )
        if validation["multivariate_warning"]:
            st.warning(
                "The combination of readings is statistically unusual "
                "compared with the training data."
            )

    status, status_type, recommendation = get_engine_status(rul)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Predicted Remaining Useful Life",
        "125+ cycles" if rul >= 124.5 else f"{rul:.2f} cycles",
    )
    c2.metric("Engine Condition", status)
    c3.metric("Sensor Quality", validation["quality"])

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rul,
            title={"text": "Remaining Useful Life (Cycles)"},
            number={"suffix": " cycles", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, RUL_CAP]},
                "bar": {"color": "#19B5FE"},
                "steps": [
                    {"range": [0, 15], "color": "#F87171"},
                    {"range": [15, 40], "color": "#FBBF24"},
                    {"range": [40, 125], "color": "#34D399"},
                ],
            },
        )
    )
    gauge.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    st.plotly_chart(gauge, use_container_width=True)

    st.subheader("🔧 Maintenance Insights & Recommendations")
    if status_type == "error":
        st.error(recommendation)
    elif status_type == "warning":
        st.warning(recommendation)
    else:
        st.success(recommendation)

    if rul >= 124.5:
        st.info(
            "The ANN output reached its upper target cap. This represents "
            "125+ cycles in the capped target, not an exact estimate of 125 cycles."
        )

    with st.expander("View prediction and validation details"):
        st.write(f"Raw ANN output: {raw:.5f}")
        st.write(f"Displayed RUL after target-range clipping: {rul:.5f}")
        if validation["mahalanobis_distance"] is not None:
            st.write(
                "Mahalanobis squared distance: "
                f"{validation['mahalanobis_distance']:.5f}"
            )
        st.write(
            "Training-derived anomaly threshold: "
            f"{validation['distance_threshold']:.5f}"
        )
        st.caption(
            "These are statistical screening measures, not physical engine "
            "safety limits or calibrated prediction confidence."
        )

    with st.expander("View submitted sensor readings"):
        input_table = pd.DataFrame({
            "Feature": FEATURES,
            "Submitted Value": [result["input"][f] for f in FEATURES],
            "Training Minimum": [validation["stats"][f]["min"] for f in FEATURES],
            "Training Maximum": [validation["stats"][f]["max"] for f in FEATURES],
        })
        st.dataframe(input_table, use_container_width=True, hide_index=True)




with st.expander("📋 View sensor training reference ranges"):
    stats = calculate_training_statistics()
    reference_df = pd.DataFrame([
        {
            "Feature": f,
            "Training Min": stats[f]["min"],
            "Training Median": stats[f]["median"],
            "Training Max": stats[f]["max"],
            "Statistical Critical Min": stats[f]["critical_min"],
            "Statistical Critical Max": stats[f]["critical_max"],
        }
        for f in FEATURES
    ])
    st.dataframe(reference_df, use_container_width=True, hide_index=True)
    st.caption(
        "Broad statistical critical boundaries use six IQRs, with a fallback "
        "for nearly constant features. These are not manufacturer-defined limits."
    )


with st.expander("🔌 Future API integration information"):
    st.markdown(
        """
        **Current API client**
        - HTTP GET request, triggered manually by the Fetch Latest Sensor Data button.
        - JSON response parsing supports the feature fields at the root or under
          `data`, `sensors`, `readings`, `latest`, or `payload`.
        - Authentication options: none, Bearer token, or `X-API-Key` header.
        - All 17 required ANN features must be supplied as numeric values.
        - API and manual inputs share the same validation, scaler, and saved ANN.

        **Not implemented yet**
        - Continuous polling / streaming.
        - MQTT, OPC-UA, Modbus, or vendor-specific IoT protocols.
        - Automatic mapping of differently named provider fields.
        - Token refresh or OAuth workflows.

        When an actual provider is selected, its API documentation can be used
        to configure the endpoint, authentication, payload mapping, and polling.
        """
    )
    st.warning(
        "This is a research/demo application, not certified aircraft diagnostic "
        "software. Statistical screening is not a flight-safety mechanism."
    )

st.divider()
st.caption(
    "Aircraft Predictive Maintenance | NASA C-MAPSS FD001 | "
    "ANN-based Remaining Useful Life Prediction"
)
st.caption(
    "Research and demonstration only. Not certified for real aircraft "
    "maintenance or flight-safety decisions."
)
