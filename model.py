"""Prediction and Grad-CAM helpers for the crop disease model."""

import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from utils import CLASS_NAMES, HEATMAP_DIR

MODEL = tf.keras.models.load_model("crop_disease_model.keras")


def preprocess_image(image_path):
    """Load and preprocess an uploaded image for model inference."""
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


def validate_leaf_image(image_array, predictions):
    """Return a relaxed leaf validation result."""
    return True, {"validation": "strict checks removed"}


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
