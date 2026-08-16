"""Minimal Flask app with routes only."""

import os
import traceback

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from advisory import enrich_farmer_advisory, get_leaf_advisory
from model import InvalidImageError, create_heatmap_image, predict_disease
from utils import UPLOAD_DIR, clean_name, ensure_static_dirs, is_healthy

app = Flask(__name__)
ensure_static_dirs()


def _parse_optional_float(value):
    """Parse optional numeric form values safely."""
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@app.route("/")
def index():
    """Render the main dashboard page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload, prediction, and advisory generation."""
    try:
        uploaded_file = request.files.get("image")
        if uploaded_file is None or not uploaded_file.filename:
            return jsonify({
                "status": "invalid",
                "error": "Invalid image file",
            }), 400

        file_name = secure_filename(uploaded_file.filename)
        if not file_name:
            return jsonify({
                "status": "invalid",
                "error": "Invalid image file",
            }), 400

        image_path = os.path.join(UPLOAD_DIR, file_name)
        uploaded_file.save(image_path)

        prediction = predict_disease(image_path)
        if prediction.get("status") == "invalid":
            return jsonify({
                "status": "invalid",
                "error": prediction["error"],
            }), 400

        if not prediction["is_valid_leaf"]:
            return jsonify({
                "status": "invalid",
                "error": "Invalid image file",
            }), 400

        class_name = prediction["class_name"]
        confidence = prediction["confidence"]
        healthy = is_healthy(class_name)
        latitude = _parse_optional_float(request.form.get("latitude"))
        longitude = _parse_optional_float(request.form.get("longitude"))

        advisory = get_leaf_advisory(class_name, confidence, latitude=latitude, longitude=longitude)
        advisory = enrich_farmer_advisory(class_name, confidence, advisory, healthy)

        response = {
            "status": "success",
            "prediction": clean_name(class_name),
            "class": clean_name(class_name),
            "confidence": confidence,
            "warnings": prediction["warnings"],
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
    except InvalidImageError:
        return jsonify({
            "status": "invalid",
            "error": "Invalid image file",
        }), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(debug=True)
