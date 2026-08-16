"""Prediction and Grad-CAM helpers for the crop disease model."""

import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from utils import CLASS_NAMES, HEATMAP_DIR

MODEL = tf.keras.models.load_model("crop_disease_model.keras")
LEAF_MODEL = tf.keras.models.load_model("leaf_classifier.keras")
BLUR_WARNING_THRESHOLD = 80.0
LOW_CONFIDENCE_THRESHOLD = 60.0
UNCERTAINTY_MARGIN_THRESHOLD = 0.2
LOW_BRIGHTNESS_THRESHOLD = 35.0
HIGH_BRIGHTNESS_THRESHOLD = 220.0


class InvalidImageError(ValueError):
    """Raised when an uploaded file is not a usable image."""


def preprocess_image(image_path):
    """Load and preprocess an uploaded image for model inference."""
    try:
        image = Image.open(image_path)
        image.load()
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError("Invalid image file") from error

    image = image.convert("RGB").resize((224, 224))
    image_array = np.array(image, dtype=np.float32)

    if image_array.size == 0 or image_array.ndim != 3 or image_array.shape[2] != 3:
        raise InvalidImageError("Invalid image file")
    if not np.isfinite(image_array).all():
        raise InvalidImageError("Invalid image file")

    return np.expand_dims(image_array / 255.0, axis=0)


def is_leaf_model(image_array):
    """Use the binary leaf classifier to reject non-leaf images before disease inference."""
    if image_array is None or not isinstance(image_array, np.ndarray) or image_array.size == 0:
        return False

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        return False

    img = cv2.resize(image_array, (224, 224))
    img = img.astype(np.float32)
    img = img.reshape(1, 224, 224, 3)

    prob = float(LEAF_MODEL.predict(img, verbose=0)[0][0])
    print("Leaf probability:", prob)

    return prob < 0.6


def validate_leaf_image(image_array):
    """Soft-validate a leaf image and return a quality label."""
    if image_array is None or not isinstance(image_array, np.ndarray) or image_array.size == 0:
        return False, "invalid"
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        return False, "invalid"
    if not np.isfinite(image_array).all():
        return False, "invalid"

    pixel_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    gray_image = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2GRAY)

    if gray_image.size == 0:
        return False, "invalid"

    blur_variance = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())
    brightness = float(np.mean(gray_image))

    if blur_variance < BLUR_WARNING_THRESHOLD:
        return True, "blurry"
    if brightness < LOW_BRIGHTNESS_THRESHOLD or brightness > HIGH_BRIGHTNESS_THRESHOLD:
        return True, "low_quality"
    return True, "valid"


def build_prediction_warnings(validation_result, confidence, predictions):
    """Build user-facing warning messages without blocking prediction."""
    warnings = []

    if validation_result == "blurry":
        warnings.append("Image is slightly blurry. Results may be less accurate.")
    elif validation_result == "low_quality":
        warnings.append("Image quality is low due to lighting. Results may be less accurate.")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        warnings.append("Low confidence prediction. Please verify results.")

    if len(predictions) > 1:
        top_two = np.sort(np.asarray(predictions))[-2:]
        if float(top_two[-1] - top_two[-2]) < UNCERTAINTY_MARGIN_THRESHOLD:
            warnings.append("Model is uncertain between multiple diseases.")

    return warnings


def predict_disease(image_path):
    """Run model inference and return prediction metadata."""
    image_array = preprocess_image(image_path)

    if not is_leaf_model(image_array[0]):
        return {
            "status": "invalid",
            "error": "This does not appear to be a crop leaf image. Please upload a valid leaf image.",
            "is_valid_leaf": False,
            "validation_details": "not_leaf",
            "warnings": [],
        }

    predictions = MODEL.predict(image_array)[0]

    class_index = int(np.argmax(predictions))
    confidence = round(float(np.max(predictions) * 100), 2)
    is_valid_leaf, validation_details = validate_leaf_image(image_array[0])
    warnings = build_prediction_warnings(validation_details, confidence, predictions)

    return {
        "class_index": class_index,
        "class_name": CLASS_NAMES[class_index],
        "confidence": confidence,
        "image_array": image_array,
        "predictions": predictions,
        "is_valid_leaf": is_valid_leaf,
        "validation_details": validation_details,
        "warnings": warnings,
    }


def generate_gradcam(image_array, class_index):
    """Generate a Grad-CAM heatmap for the predicted disease class."""
    last_conv_layer = MODEL.get_layer("Conv_1")
    grad_model = tf.keras.models.Model(inputs=MODEL.input, outputs=[last_conv_layer.output, MODEL.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_array)
        predictions = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
        loss = predictions[:, class_index]

    gradients = tape.gradient(loss, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return cv2.resize(heatmap.numpy(), (224, 224))


def overlay_heatmap(image_path, heatmap):
    """Blend a heatmap over the uploaded leaf image."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    colored_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.6, colored_heatmap, 0.4, 0)


def create_heatmap_image(image_path, file_name, class_index, image_array):
    """Create and save a Grad-CAM overlay image for the current prediction."""
    heatmap = generate_gradcam(image_array, class_index)
    overlay = overlay_heatmap(image_path, heatmap)
    heatmap_path = os.path.join(HEATMAP_DIR, file_name)
    cv2.imwrite(heatmap_path, overlay)
    return "/" + heatmap_path
