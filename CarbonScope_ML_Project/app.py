import os
import json
import joblib
import pandas as pd

from flask import Flask, render_template, request, jsonify

BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

DATA = os.path.join(
    BASE,
    "dataset",
    "carbon_emissions.csv"
)

MODELS = os.path.join(BASE, "models")

PROC = os.path.join(
    BASE,
    "processed_data"
)

NUM = [
    "electricity_kwh",
    "natural_gas_m3",
    "vehicle_km",
    "flights_km",
    "waste_kg",
    "renewable_pct",
    "avg_temp_c",
    "occupancy",
    "industrial_output",
    "transport_load"
]

CAT = [
    "sector",
    "region"
]

FEATURES = NUM + CAT


# ---------------------------------------------------------
# LOAD PROJECT DATA
# ---------------------------------------------------------

def load_project_data():

    data = pd.read_csv(DATA)

    metrics = pd.read_csv(
        os.path.join(
            PROC,
            "model_metrics.csv"
        )
    )

    with open(
        os.path.join(
            MODELS,
            "metadata.json"
        ),
        "r"
    ) as f:
        meta = json.load(f)

    return data, metrics, meta


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.route("/")
def index():

    data, metrics, meta = load_project_data()

    summary = {
        "rows": len(data),

        "missing": int(
            data.isna().sum().sum()
        ),

        "target_mean": round(
            data["carbon_kg_co2e"].mean(),
            2
        ),

        "target_max": round(
            data["carbon_kg_co2e"].max(),
            2
        )
    }

    sectors = sorted(
        data["sector"]
        .dropna()
        .unique()
    )

    regions = sorted(
        data["region"]
        .dropna()
        .unique()
    )

    return render_template(
        "index.html",
        metrics=metrics.to_dict("records"),
        meta=meta,
        summary=summary,
        sectors=sectors,
        regions=regions
    )


# ---------------------------------------------------------
# PREDICTION API
# ---------------------------------------------------------

@app.post("/predict")
def predict():

    data = request.get_json(force=True)

    row = {
        key: data.get(key)
        for key in FEATURES
    }

    # Convert numerical values
    for key in NUM:

        if row[key] in (None, ""):
            row[key] = None
        else:
            row[key] = float(row[key])

    input_df = pd.DataFrame([row])

    results = {}

    prediction_models = [
        "linear_regression",
        "gradient_boosting",
        "random_forest"
    ]

    for model_name in prediction_models:

        model_path = os.path.join(
            MODELS,
            model_name + ".joblib"
        )

        if os.path.exists(model_path):

            model = joblib.load(
                model_path
            )

            prediction = model.predict(
                input_df
            )[0]

            results[model_name] = round(
                float(prediction),
                2
            )

    return jsonify({
        "predictions": results,
        "primary_model": "gradient_boosting"
    })


# ---------------------------------------------------------
# ANALYTICS API
# ---------------------------------------------------------

@app.get("/api/analytics")
def analytics():

    data, metrics, meta = load_project_data()

    # -----------------------------------------------------
    # CORRELATION
    # -----------------------------------------------------

    correlation = (
        data[
            NUM + ["carbon_kg_co2e"]
        ]
        .corr()["carbon_kg_co2e"]
        .drop("carbon_kg_co2e")
        .sort_values(
            ascending=False
        )
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    importance_file = os.path.join(
        PROC,
        "feature_importance.csv"
    )

    if os.path.exists(importance_file):

        importance_df = pd.read_csv(
            importance_file
        )

        importance = (
            importance_df
            .sort_values(
                "importance",
                ascending=False
            )
            .to_dict("records")
        )

    else:

        importance = []

    # -----------------------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------------------

    actual_predicted = []

    test_file = os.path.join(
        PROC,
        "test_predictions.csv"
    )

    if os.path.exists(test_file):

        prediction_df = pd.read_csv(
            test_file
        )

        for _, row in prediction_df.iterrows():

            actual_predicted.append({
                "actual": float(
                    row["actual"]
                ),
                "predicted": float(
                    row["predicted"]
                )
            })

    else:

        # Generate predictions directly if the
        # processed prediction file doesn't exist.

        try:

            model_path = os.path.join(
                MODELS,
                "gradient_boosting.joblib"
            )

            model = joblib.load(
                model_path
            )

            target = "carbon_kg_co2e"

            X = data[
                FEATURES
            ]

            y = data[
                target
            ]

            predictions = model.predict(X)

            for actual, predicted in zip(
                y,
                predictions
            ):

                actual_predicted.append({
                    "actual": float(actual),
                    "predicted": float(predicted)
                })

        except Exception:

            actual_predicted = []

    # Limit graph points for a clean graph
    actual_predicted = actual_predicted[:300]

    return jsonify({

        "correlation":
            correlation.to_dict(),

        "importance":
            importance,

        "metrics":
            metrics.to_dict("records"),

        "actual_predicted":
            actual_predicted
    })


# ---------------------------------------------------------
# UNSUPERVISED ML
# ---------------------------------------------------------

@app.get("/api/unsupervised")
def unsupervised():

    result_file = os.path.join(
        PROC,
        "unsupervised_results.json"
    )

    if os.path.exists(result_file):

        with open(
            result_file,
            "r"
        ) as f:

            return jsonify(
                json.load(f)
            )

    return jsonify({})


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

@app.get("/api/evaluation")
def evaluation():

    result_file = os.path.join(
        PROC,
        "evaluation_results.json"
    )

    if os.path.exists(result_file):

        with open(
            result_file,
            "r"
        ) as f:

            return jsonify(
                json.load(f)
            )

    return jsonify({})


# ---------------------------------------------------------
# RETRAIN
# ---------------------------------------------------------

@app.post("/api/retrain")
def retrain():

    from train import main

    main()

    from analyze import main as analyze_main

    analyze_main()

    from evaluate import main as evaluate_main

    evaluate_main()

    from unsupervised import main as unsupervised_main

    unsupervised_main()

    return jsonify({
        "status": "success",
        "message":
            "Models retrained and artifacts refreshed."
    })


# ---------------------------------------------------------
# RUN FLASK
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )