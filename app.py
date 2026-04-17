from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import traceback
from PIL import Image
from werkzeug.utils import secure_filename

# ---------------- APP SETUP ----------------

app = Flask(__name__)

UPLOAD = "static/uploads"
HEATMAP = "static/heatmaps"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(HEATMAP, exist_ok=True)

# ---------------- LOAD MODEL ----------------

model = tf.keras.models.load_model("crop_disease_model.keras")

# ---------------- CLASS NAMES ----------------

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# ---------------- IMAGE PREPROCESS ----------------

def preprocess(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------- CLEAN NAME ----------------

def clean_name(name):
    return name.replace("_", " ")

# ---------------- HEALTH CHECK ----------------

def is_healthy(name):
    return "healthy" in name.lower()

# ---------------- SEVERITY ----------------

def get_severity_band(conf):

    if conf <= 50:
        return "low"
    if conf <= 75:
        return "medium"
    return "high"

# ---------------- ADVISORY RULES ----------------

CLASS_TO_ADVISORY_KEY = {
    "Pepper__bell___Bacterial_spot": "pepper_bacterial_spot",
    "Pepper__bell___healthy": "pepper_healthy",
    "Potato___Early_blight": "potato_early_blight",
    "Potato___Late_blight": "potato_late_blight",
    "Potato___healthy": "potato_healthy",
    "Tomato_Bacterial_spot": "tomato_bacterial_spot",
    "Tomato_Early_blight": "tomato_early_blight",
    "Tomato_Late_blight": "tomato_late_blight",
    "Tomato_Leaf_Mold": "tomato_leaf_mold",
    "Tomato_Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "tomato_spider_mites",
    "Tomato__Target_Spot": "tomato_target_spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "tomato_yellow_leaf_curl_virus",
    "Tomato__Tomato_mosaic_virus": "tomato_mosaic_virus",
    "Tomato_healthy": "tomato_healthy"
}

ADVISORY_RULES_EN = {
    "pepper_bacterial_spot": {
        "low": ["Prune spotted leaves and avoid overhead irrigation.", "Spray a copper-based bactericide in the evening."],
        "medium": ["Remove heavily infected leaves and sanitize tools after each plant.", "Use copper + mancozeb rotation every 7 days."],
        "high": ["Isolate infected plants immediately to reduce spread.", "Destroy severely infected plants and disinfect the area."]
    },
    "pepper_healthy": {
        "all": ["Leaf looks healthy. Keep scouting twice per week.", "Maintain balanced NPK and avoid excess leaf wetness."]
    },
    "potato_early_blight": {
        "low": ["Start preventive fungicide spray and remove lower infected leaves.", "Improve airflow by spacing and weed control."],
        "medium": ["Rotate fungicides (chlorothalonil / mancozeb classes) every 7 days.", "Irrigate at soil level and avoid evening wet foliage."],
        "high": ["Remove highly infected plants to stop field spread.", "Plan crop rotation for next season and avoid potato residue."]
    },
    "potato_late_blight": {
        "low": ["Begin anti-oomycete spray immediately.", "Avoid water splash and keep field drainage clear."],
        "medium": ["Spray systemic + contact fungicide mix as per label.", "Remove nearby infected plants and monitor daily."],
        "high": ["Emergency control needed: rogue badly infected patches.", "Do not move infected foliage through healthy rows."]
    },
    "potato_healthy": {
        "all": ["Leaf looks healthy. Continue preventive scouting.", "Use drip irrigation and maintain clean field borders."]
    },
    "tomato_bacterial_spot": {
        "low": ["Remove affected leaves and avoid touching plants when wet.", "Use copper spray with spreader sticker."],
        "medium": ["Disinfect pruning tools and stakes daily.", "Apply bactericide rotation every 5-7 days."],
        "high": ["Remove heavily infected plants and bag plant waste.", "Do not compost infected tomato debris."]
    },
    "tomato_early_blight": {
        "low": ["Remove bottom leaves touching soil.", "Start protective fungicide program."],
        "medium": ["Spray in rotation and improve canopy ventilation.", "Mulch soil to reduce spore splash."],
        "high": ["Remove severe plants and protect nearby healthy plants quickly.", "Follow strict 2-3 year crop rotation."]
    },
    "tomato_late_blight": {
        "low": ["Start immediate late blight spray schedule.", "Avoid overhead watering and long leaf wetness."],
        "medium": ["Use recommended systemic fungicides and monitor every day.", "Remove infected tissue as soon as seen."],
        "high": ["Urgent containment: remove entire infected plants.", "Restrict field movement to prevent disease transfer."]
    },
    "tomato_leaf_mold": {
        "low": ["Reduce humidity in canopy and improve ventilation.", "Remove first infected leaves."],
        "medium": ["Spray labeled fungicide and avoid dense pruning wounds.", "Water early morning to dry leaves faster."],
        "high": ["Thin canopy aggressively and remove severe plants.", "Sanitize greenhouse or stakes after harvest."]
    },
    "tomato_septoria_leaf_spot": {
        "low": ["Pick infected lower leaves and keep soil mulched.", "Begin preventive fungicide applications."],
        "medium": ["Continue spray program at 7-day interval.", "Avoid handling plants when wet."],
        "high": ["Remove heavily infected plants.", "Destroy residues after harvest to break lifecycle."]
    },
    "tomato_spider_mites": {
        "low": ["Spray water mist under leaves in morning to reduce mites.", "Release or conserve beneficial predators if available."],
        "medium": ["Apply approved miticide and rotate active ingredients.", "Remove worst infested leaves."],
        "high": ["Isolate hotspot plants and apply full-coverage miticide.", "Repeat follow-up spray as label recommends."]
    },
    "tomato_target_spot": {
        "low": ["Remove spotted leaves and improve airflow.", "Start protectant fungicide schedule."],
        "medium": ["Increase spray coverage on lower canopy.", "Avoid overhead irrigation in evening."],
        "high": ["Remove severely affected plants and sanitize tools.", "Use strict residue cleanup after harvest."]
    },
    "tomato_yellow_leaf_curl_virus": {
        "low": ["Control whiteflies using yellow sticky traps.", "Remove plants with strong curling symptoms."],
        "medium": ["Use insect-proof netting and vector control spray program.", "Avoid mixing infected and healthy nursery seedlings."],
        "high": ["Rogue infected plants immediately.", "Focus on aggressive whitefly suppression in whole plot."]
    },
    "tomato_mosaic_virus": {
        "low": ["Avoid tobacco contact before handling plants.", "Disinfect hands and tools regularly."],
        "medium": ["Remove infected plants and sanitize supports.", "Use only clean seedling sources."],
        "high": ["Uproot severely infected plants and dispose safely.", "Disinfect entire working area and tools."]
    },
    "tomato_healthy": {
        "all": ["Leaf looks healthy. Keep monitoring every 3-4 days.", "Maintain balanced nutrition and preventive hygiene."]
    }
}

def get_leaf_advisory(class_name, confidence, lang):
    advisory_key = CLASS_TO_ADVISORY_KEY.get(class_name)
    if not advisory_key:
        return ["No advisory available for this class."]

    lang_key = "en" if lang not in {"en", "te"} else lang
    # Telugu fallback currently mirrors English to keep behavior stable.
    advisory_store = ADVISORY_RULES_EN

    advisory = advisory_store.get(advisory_key, {})
    if not advisory:
        return ["No advisory available for this class."]

    if "all" in advisory:
        return advisory["all"]

    severity_band = get_severity_band(confidence)
    return advisory.get(severity_band, ["No advisory available for this severity."])

# ---------------- GRADCAM ----------------

def generate_gradcam(img_array, class_idx):

    last_conv = model.get_layer("Conv_1")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_out, preds = grad_model(img_array)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]

    heatmap = conv_out @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap.numpy(), (224, 224))

    return heatmap

# ---------------- HEATMAP OVERLAY ----------------

def overlay_heatmap(img_path, heatmap):

    img = cv2.imread(img_path)

    img = cv2.resize(img, (224, 224))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# ---------------- ROUTES ----------------

@app.route("/")
def index():
    return render_template("index.html")

# ---------------- PREDICT ----------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        file = request.files["image"]
        lang = request.form.get("lang", "en")

        filename = secure_filename(file.filename)

        img_path = os.path.join(UPLOAD, filename)

        file.save(img_path)

        img_array = preprocess(img_path)

        preds = model.predict(img_array)

        idx = int(np.argmax(preds))

        class_name = CLASS_NAMES[idx]

        confidence = round(float(np.max(preds) * 100), 2)

        healthy = is_healthy(class_name)

        response = {
            "class": clean_name(class_name),
            "confidence": confidence,
            "healthy": healthy,
            "image": "/" + img_path,
            "advisory": get_leaf_advisory(class_name, confidence, lang)
        }

        # GradCAM only for diseased leaves
        if not healthy:

            try:

                heatmap = generate_gradcam(img_array, idx)

                overlay = overlay_heatmap(img_path, heatmap)

                heatmap_path = os.path.join(HEATMAP, filename)

                cv2.imwrite(heatmap_path, overlay)

                response["heatmap"] = "/" + heatmap_path

            except:
                response["heatmap"] = None

        return jsonify(response)

    except Exception:

        traceback.print_exc()

        return jsonify({"error": "Prediction failed"}), 500

# ---------------- RUN SERVER ----------------

if __name__ == "__main__":
    app.run(debug=True)
