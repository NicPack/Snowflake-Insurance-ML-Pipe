# backend.py (Secure Server)
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from snowflake.ml.registry import registry
from snowflake.snowpark import Session

load_dotenv()

app = Flask(__name__)


def get_snowflake_session():
    return Session.builder.configs(
        {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "warehouse": "COMPUTE_WH",
            "database": "INSURANCE",
            "schema": "ML_PIPE",
        }
    ).create()


@app.route("/api/gold_data", methods=["GET"])
def get_gold_data():
    session = get_snowflake_session()
    try:
        df = session.table("INSURANCE_GOLD").limit(600).to_pandas()
        return jsonify(df.to_dict(orient="records"))
    finally:
        session.close()


@app.route("/api/predict", methods=["POST"])
def predict():
    session = get_snowflake_session()
    try:
        data = request.json
        model_registry = registry.Registry(
            session=session, database_name="INSURANCE", schema_name="ML_PIPE"
        )
        model_version = model_registry.get_model("INSURANCE_CHARGES_PREDICTION").default
        prediction = model_version.run(data, function_name="predict")
        return jsonify({"prediction": prediction["PREDICTED_CHARGES"]})
    finally:
        session.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
