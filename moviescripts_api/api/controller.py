import json
import os
print("controler listdir ",os.listdir())

from flask import request, jsonify, Response
from moviescripts.prediction_model import predict_class
from prometheus_client import Counter, Gauge, Histogram, Summary


def health():
    if request.method == "GET":
        return jsonify({"status": "ok"})

REQUEST_COUNTER = Counter("request_count", "The number of requests")
TARGET_GAUGE = Gauge("target_sum_sign", "Predicted class: +1 if class=1 else -1")
WAREA_HIST = Histogram("worst_area_hist", "worst_area histogram")
REQUEST_TIME = Summary("request_processing_seconds", "Time spent processing request")

@REQUEST_TIME.time()
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


        REQUEST_COUNTER.inc()  # Increments by 1.
        
        # if predictions == 1:
        #     TARGET_GAUGE.inc()
        # else:
        #     TARGET_GAUGE.dec()
        
        # WAREA_HIST.observe(json_data)
        # Step 5: Prepare prediction response
        return jsonify(
            {"predictions": predictions, "version": version, "errors": errors}
        )

