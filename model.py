"""Prediction and Grad-CAM helpers for the crop disease model."""

import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from utils import CLASS_NAMES, HEATMAP_DIR

MODEL = tf.keras.models.load_model("crop_disease_model.keras")
MIN_CONFIDENCE_FOR_VALID_LEAF = 45.0
MIN_MARGIN_FOR_VALID_LEAF = 0.12
MIN_NATURAL_COLOR_RATIO = 0.06
MIN_TEXTURE_STD = 0.05


def preprocess_image(image_path):
    """Load and preprocess an uploaded image for model inference."""
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


def validate_leaf_image(image_array, predictions):
    """Reject obvious non-leaf uploads using simple image and prediction heuristics."""
    rgb_image = np.uint8(np.clip(image_array * 255, 0, 255))
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    green_mask = cv2.inRange(hsv_image, np.array([25, 35, 25]), np.array([95, 255, 255]))
    yellow_brown_mask = cv2.inRange(hsv_image, np.array([5, 25, 20]), np.array([35, 255, 255]))
    natural_color_ratio = float(np.mean((green_mask > 0) | (yellow_brown_mask > 0)))

    edge_density = float(np.mean(cv2.Canny(grayscale_image, 80, 160) > 0))
    texture_std = float(np.std(image_array))

    sorted_predictions = np.sort(predictions)[::-1]
    top_confidence = float(sorted_predictions[0] * 100)
    runner_up = float(sorted_predictions[1]) if len(sorted_predictions) > 1 else 0.0
    confidence_margin = float(sorted_predictions[0] - runner_up)

    looks_like_leaf = (
        top_confidence >= MIN_CONFIDENCE_FOR_VALID_LEAF
        and confidence_margin >= MIN_MARGIN_FOR_VALID_LEAF
        and natural_color_ratio >= MIN_NATURAL_COLOR_RATIO
        and texture_std >= MIN_TEXTURE_STD
        and 0.01 <= edge_density <= 0.35
    )

    details = {
        "top_confidence": round(top_confidence, 2),
        "confidence_margin": round(confidence_margin, 3),
        "natural_color_ratio": round(natural_color_ratio, 3),
        "texture_std": round(texture_std, 3),
        "edge_density": round(edge_density, 3),
    }
    return looks_like_leaf, details


def predict_disease(image_path):
    """Run model inference and return prediction metadata."""
    image_array = preprocess_image(image_path)
    predictions = MODEL.predict(image_array)[0]
    class_index = int(np.argmax(predictions))
    confidence = round(float(np.max(predictions) * 100), 2)
    is_valid_leaf, validation_details = validate_leaf_image(image_array[0], predictions)
    return {
        "class_index": class_index,
        "class_name": CLASS_NAMES[class_index],
        "confidence": confidence,
        "image_array": image_array,
        "predictions": predictions,
        "is_valid_leaf": is_valid_leaf,
        "validation_details": validation_details,
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
