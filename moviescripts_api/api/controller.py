import json

from flask import request, jsonify, Response
from moviescripts.prediction_model import predict_class


def health():
    if request.method == "GET":
        return jsonify({"status": "ok"})


def predict():
    if request.method == "POST":
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()

        # Step 2: Access the model prediction function (also validates data)
        result = predict_class(input_data=json_data)

        # Step 3: Handle errors
        errors = result.get("errors")
        if errors == "invalid":
            return Response(json.dumps(errors), status=400)

        # Step 4: Split out results
        predictions = result.get("predictions")
        version = result.get("version")

        # Step 5: Prepare prediction response
        return jsonify(
            {"predictions": predictions, "version": version, "errors": errors}
        )

