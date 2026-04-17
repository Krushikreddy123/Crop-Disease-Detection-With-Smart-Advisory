"""Minimal Flask app with routes only."""

import os
import traceback

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from advisory import enrich_farmer_advisory, get_leaf_advisory
from model import create_heatmap_image, predict_disease
from utils import UPLOAD_DIR, clean_name, ensure_static_dirs, is_healthy

app = Flask(__name__)
ensure_static_dirs()


@app.route("/")
def index():
    """Render the main dashboard page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload, prediction, and advisory generation."""
    try:
        uploaded_file = request.files["image"]
        file_name = secure_filename(uploaded_file.filename)
        image_path = os.path.join(UPLOAD_DIR, file_name)
        uploaded_file.save(image_path)

        prediction = predict_disease(image_path)
        if not prediction["is_valid_leaf"]:
            return jsonify({
                "error": "Please upload a clear crop leaf image. The current image does not look like a valid leaf sample.",
                "validation": prediction["validation_details"],
            }), 400

        class_name = prediction["class_name"]
        confidence = prediction["confidence"]
        healthy = is_healthy(class_name)

        advisory = get_leaf_advisory(class_name, confidence)
        advisory = enrich_farmer_advisory(class_name, confidence, advisory, healthy)

        response = {
            "class": clean_name(class_name),
            "confidence": confidence,
            "healthy": healthy,
            "image": "/" + image_path,
            "advisory": advisory,
            "ai_note": advisory["advisory_note"],
            "advisory_source": advisory["source"],
            "advisory_links": advisory["sources"],
        }

        if not healthy:
            try:
                response["heatmap"] = create_heatmap_image(
                    image_path,
                    file_name,
                    prediction["class_index"],
                    prediction["image_array"],
                )
            except Exception:
                response["heatmap"] = None

        return jsonify(response)
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(debug=True)
