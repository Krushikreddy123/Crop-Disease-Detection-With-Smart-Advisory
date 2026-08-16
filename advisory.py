"""Advisory generation helpers for disease explanations and treatment guidance."""

import csv
from datetime import datetime
import json
import os
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from utils import (
    ADVISORY_CSV_PATH,
    ADVISORY_RULES_EN,
    CLASS_TO_ADVISORY_KEY,
    best_spray_time_from_severity,
    clean_name,
    confidence_to_csv_band,
    extract_crop_name,
    fallback_buy_links,
    fallback_source_links,
    get_severity_band,
    normalize_disease_name,
    normalize_text,
    recommendation_library,
    symptom_library,
)

ADVISORY_CACHE = {}
AI_API_TIMEOUT_SECONDS = 10
AI_RESPONSE_MAX_TOKENS = 700
WEATHER_API_TIMEOUT_SECONDS = 8
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_DOCS_URL = "https://open-meteo.com/en/docs"

AI_ADVISORY_PROMPT = """Act as an expert agricultural scientist.

Disease: {disease_name}
Severity: {severity}
Crop: {crop_name}

Provide accurate, practical, non-repetitive advisory for farmers.
Provide unique and non-overlapping information in each field.
Ensure all fields are detailed, non-overlapping, and based on real agricultural practices.

Example:
Cause: "Fungal infection caused by Alternaria solani"
Symptoms: ["Brown circular spots", "Yellowing leaves"]
Chemical treatment: ["Spray Mancozeb 2-2.5 g/L every 5-7 days", "Rotate with Chlorothalonil where labels allow"]
Organic treatment: ["Spray neem oil 3-5 ml/L with proper emulsifier", "Use Bacillus-based bio-control or compost extract after removing infected leaves"]

Return ONLY JSON in this exact format:

{{
"cause": "Explain actual biological cause (fungal/bacterial/viral).",
"symptoms": ["clear symptom 1", "clear symptom 2", "clear symptom 3"],
"treatment": {{
"chemical": ["specific fungicide/pesticide actions", "spray interval"],
"organic": ["real organic methods like neem oil, compost extract"]
}},
"prevention": ["specific prevention step 1", "step 2", "step 3"],
"best_spray_time": "clear spray timing",
"recommended_products": ["Mancozeb", "Chlorothalonil"],
"dosage": "exact dosage",
"irrigation_advice": "clear irrigation advice",
"fertilizer_advice": "nutrient advice",
"weather_precautions": "weather risks",
"spread_risk": "how disease spreads",
"next_7_days_prediction": "short-term forecast",
"safety_precautions": ["safety step 1", "step 2"]
}}

Rules:

* No repetition
* No generic lines like "continue spray program"
* Do not mix fields
* Return ONLY JSON"""


def load_csv_advisory_store():
    """Load advisory rows from the local CSV into a lookup dictionary."""
    advisory_store = {}
    if not os.path.exists(ADVISORY_CSV_PATH):
        return advisory_store

    with open(ADVISORY_CSV_PATH, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            disease = row.get("disease", "").strip()
            severity = row.get("severity", "").strip()
            language = row.get("lang", "en").strip()
            points = [(row.get(f"point{index}") or "").strip() for index in range(1, 16)]
            points = [point for point in points if point]
            if disease and severity and points:
                advisory_store[(disease, severity, language)] = points
    return advisory_store


CSV_ADVISORY_STORE = load_csv_advisory_store()


def _fallback_text(value):
    """Return the first useful fallback string from text or list data."""
    if isinstance(value, str) and value.strip():
        return normalize_text(value)
    if isinstance(value, list):
        for item in value:
            if str(item).strip():
                return normalize_text(str(item))
    return ""


def _as_text(value, fallback):
    """Normalize a value into a compact text string."""
    if isinstance(value, str) and value.strip():
        return normalize_text(value)
    if isinstance(value, list):
        joined = "; ".join(normalize_text(str(item)) for item in value if str(item).strip())
        if joined:
            return joined
    return fallback


def _as_list(value, fallback, limit=None):
    """Normalize a value into a list of strings."""
    if isinstance(value, list):
        cleaned = [normalize_text(str(item)) for item in value if str(item).strip()]
    elif isinstance(value, str) and value.strip():
        cleaned = [normalize_text(item) for item in re.split(r"[;\n]+", value) if item.strip()]
    else:
        cleaned = []

    cleaned = cleaned or list(fallback)
    return cleaned[:limit] if limit else cleaned


def _extract_json_object(raw_text):
    """Parse AI output and recover the first JSON object if wrapped in extra text."""
    content = (raw_text or "").strip()
    if not content:
        raise ValueError("Empty AI response")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("AI response is not a JSON object")
    return parsed


def _extract_chat_content(response_json):
    """Extract the assistant text content from a chat completions response."""
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError("AI response has no choices")

    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts)
    return ""


def _safe_float(value):
    """Convert API values into floats when possible."""
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_day_name(raw_date):
    """Convert an ISO date into a short weekday label."""
    try:
        return datetime.fromisoformat(str(raw_date)).strftime("%a")
    except ValueError:
        return str(raw_date)


def _disease_family(class_name):
    """Map a model class name into a disease family for weather logic."""
    lowered = class_name.lower()
    if "healthy" in lowered:
        return "healthy"
    if "spider_mites" in lowered or "mite" in lowered:
        return "mites"
    if "virus" in lowered or "yellowleaf" in lowered or "mosaic" in lowered:
        return "virus"
    if "bacterial" in lowered:
        return "bacterial"
    return "fungal"


def _build_weather_source_url(latitude, longitude):
    """Build a reusable Open-Meteo URL for the chosen location."""
    params = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
        ]),
        "forecast_days": 7,
        "timezone": "auto",
    }
    return f"{OPEN_METEO_FORECAST_URL}?{urlencode(params)}"


def fetch_weather_context(latitude, longitude):
    """Fetch a free 7-day weather forecast from Open-Meteo."""
    if latitude is None or longitude is None:
        raise ValueError("Latitude and longitude are required for live weather advisory")

    source_url = _build_weather_source_url(latitude, longitude)
    request = Request(
        source_url,
        headers={"User-Agent": "crop-disease-advisory/1.0"},
        method="GET",
    )

    with urlopen(request, timeout=WEATHER_API_TIMEOUT_SECONDS) as response:
        response_body = response.read().decode("utf-8", errors="ignore")

    weather_json = json.loads(response_body)
    current = weather_json.get("current") or {}
    daily = weather_json.get("daily") or {}
    dates = daily.get("time") or []

    forecast_days = []
    for index, raw_date in enumerate(dates[:7]):
        forecast_days.append({
            "date": str(raw_date),
            "day_name": _safe_day_name(raw_date),
            "temp_max": _safe_float((daily.get("temperature_2m_max") or [None])[index] if index < len(daily.get("temperature_2m_max") or []) else None),
            "temp_min": _safe_float((daily.get("temperature_2m_min") or [None])[index] if index < len(daily.get("temperature_2m_min") or []) else None),
            "humidity_mean": _safe_float((daily.get("relative_humidity_2m_mean") or [None])[index] if index < len(daily.get("relative_humidity_2m_mean") or []) else None),
            "precipitation_sum": _safe_float((daily.get("precipitation_sum") or [None])[index] if index < len(daily.get("precipitation_sum") or []) else None),
            "precipitation_probability_max": _safe_float((daily.get("precipitation_probability_max") or [None])[index] if index < len(daily.get("precipitation_probability_max") or []) else None),
            "wind_speed_max": _safe_float((daily.get("wind_speed_10m_max") or [None])[index] if index < len(daily.get("wind_speed_10m_max") or []) else None),
        })

    return {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": weather_json.get("timezone", ""),
        "current": {
            "temperature": _safe_float(current.get("temperature_2m")),
            "humidity": _safe_float(current.get("relative_humidity_2m")),
            "precipitation": _safe_float(current.get("precipitation")),
            "wind_speed": _safe_float(current.get("wind_speed_10m")),
        },
        "days": forecast_days,
        "source_url": source_url,
    }


def _summarize_weather_risk(class_name, weather_context):
    """Turn weather data into disease-specific risk language."""
    family = _disease_family(class_name)
    days = weather_context.get("days") or []
    current = weather_context.get("current") or {}

    wet_days = sum(
        1
        for day in days
        if (day.get("precipitation_sum") or 0) >= 3
        or (day.get("precipitation_probability_max") or 0) >= 60
        or (day.get("humidity_mean") or 0) >= 82
    )
    hot_dry_days = sum(
        1
        for day in days
        if (day.get("temp_max") or 0) >= 30
        and (day.get("humidity_mean") or 100) <= 60
        and (day.get("precipitation_probability_max") or 100) <= 30
    )
    windy_days = sum(1 for day in days if (day.get("wind_speed_max") or 0) >= 22)

    current_temp = current.get("temperature")
    current_humidity = current.get("humidity")

    if family == "mites":
        if hot_dry_days >= 2:
            level = "high"
            reason = f"hot, dry weather is forecast on {hot_dry_days} of the next 7 days"
        elif hot_dry_days == 1:
            level = "medium"
            reason = "one hot and dry day may help mites build up"
        else:
            level = "low"
            reason = "the coming week is not strongly favorable for rapid mite flare-up"
    elif family == "virus":
        if hot_dry_days >= 2 or windy_days >= 2:
            level = "high"
            reason = "warm, dry or windy conditions can increase vector movement and field spread"
        elif wet_days >= 3:
            level = "medium"
            reason = "wet weather may slow vectors a bit, but infected plants can still act as a source"
        else:
            level = "medium"
            reason = "vector pressure should still be monitored closely this week"
    elif family == "bacterial":
        if wet_days >= 2:
            level = "high"
            reason = f"leaf wetness, splash, and humidity are favorable on {wet_days} of the next 7 days"
        elif wet_days == 1:
            level = "medium"
            reason = "a wet day can trigger fresh bacterial spotting and splash spread"
        else:
            level = "low"
            reason = "the coming week looks relatively dry, which helps slow splash-borne spread"
    elif family == "healthy":
        if wet_days >= 3:
            level = "medium"
            reason = "the crop is currently healthy, but several wet days can raise disease pressure"
        else:
            level = "low"
            reason = "weather pressure remains fairly low if scouting continues"
    else:
        if wet_days >= 2:
            level = "high"
            reason = f"humid or rainy conditions are forecast on {wet_days} of the next 7 days"
        elif wet_days == 1:
            level = "medium"
            reason = "one wet or highly humid day may allow new lesions to expand"
        else:
            level = "low"
            reason = "the next week looks drier, which should slow most fungal spread"

    summary = []
    if current_temp is not None:
        summary.append(f"current temperature is around {current_temp:.0f} C")
    if current_humidity is not None:
        summary.append(f"humidity is near {current_humidity:.0f}%")

    return {
        "level": level,
        "reason": reason,
        "current_summary": ", ".join(summary),
        "wet_days": wet_days,
        "hot_dry_days": hot_dry_days,
        "windy_days": windy_days,
    }


def _best_spray_window(weather_context):
    """Pick the earliest safer spray window from the forecast."""
    days = weather_context.get("days") or []
    if not days:
        return ""

    best_day = None
    best_score = None
    for day in days[:5]:
        rain_risk = (day.get("precipitation_probability_max") or 100) / 5
        rain_total = (day.get("precipitation_sum") or 20) * 4
        wind = (day.get("wind_speed_max") or 40) * 2
        score = rain_risk + rain_total + wind
        if best_score is None or score < best_score:
            best_day = day
            best_score = score

    if not best_day:
        return ""

    day_name = best_day.get("day_name", "the next suitable day")
    rain_probability = int(best_day.get("precipitation_probability_max") or 0)
    wind_speed = int(best_day.get("wind_speed_max") or 0)

    if rain_probability <= 35 and wind_speed <= 20:
        return f"Plan spraying on {day_name} in the early morning or late afternoon; rain chance is about {rain_probability}% and wind may stay near {wind_speed} km/h."
    return f"Use the driest available window on {day_name}, preferably in the early morning; forecast rain chance is about {rain_probability}% and wind may reach {wind_speed} km/h."


def build_free_dynamic_advisory(class_name, confidence, latitude, longitude):
    """Build advisory text from local disease knowledge plus free live weather."""
    payload = get_local_advisory(class_name, confidence)
    weather_context = fetch_weather_context(latitude, longitude)
    risk = _summarize_weather_risk(class_name, weather_context)
    spray_window = _best_spray_window(weather_context)
    current_summary = risk.get("current_summary")
    family = _disease_family(class_name)
    days = weather_context.get("days") or []
    first_day = days[0] if days else {}
    rain_today = first_day.get("precipitation_probability_max")
    week_rain = sum((day.get("precipitation_sum") or 0) for day in days)

    if family == "mites":
        irrigation_advice = (
            f"Keep soil moisture even and reduce crop stress because {risk['reason']}. "
            "If irrigation is needed, use morning irrigation and wash dusty hotspots lightly where practical."
        )
        forecast_note = (
            f"Mite pressure is likely to stay {risk['level']} over the next 7 days because {risk['reason']}."
        )
    elif family == "virus":
        irrigation_advice = (
            "Keep irrigation uniform to reduce plant stress and remove severely affected plants early; "
            "stressed crops usually decline faster under virus pressure."
        )
        forecast_note = (
            f"Vector-driven spread risk is {risk['level']} this week because {risk['reason']}."
        )
    else:
        irrigation_advice = (
            f"Base irrigation on the local forecast. About {week_rain:.1f} mm rain is expected in the next 7 days, "
            "so avoid unnecessary overhead irrigation and keep leaf wetness periods short."
        )
        forecast_note = (
            f"Disease pressure is likely to stay {risk['level']} over the next 7 days because {risk['reason']}."
        )

    weather_precautions = (
        f"Live forecast check: {forecast_note} "
        f"{current_summary + '. ' if current_summary else ''}"
        f"{'Rain is possible today, so avoid spraying just before showers. ' if (rain_today or 0) >= 40 else ''}"
        f"{'Use calm early-morning spray windows when wind drops.' if risk['windy_days'] else 'Prefer calm early-morning or late-afternoon application windows.'}"
    ).strip()

    prevention = list(payload.get("prevention") or [])
    live_prevention = f"Use the live forecast to plan scouting and spraying because {risk['reason']}."
    if live_prevention not in prevention:
        prevention.insert(0, live_prevention)

    sources = [
        {
            "title": "Open-Meteo forecast used for this advisory",
            "url": weather_context["source_url"],
            "domain": "open-meteo.com",
        },
        {
            "title": "Open-Meteo API documentation",
            "url": OPEN_METEO_DOCS_URL,
            "domain": "open-meteo.com",
        },
    ]

    payload.update({
        "best_spray_time": spray_window or payload.get("best_spray_time"),
        "irrigation_advice": irrigation_advice,
        "weather_precautions": weather_precautions,
        "spread_risk": f"{risk['level'].capitalize()} weather-linked spread risk: {risk['reason']}.",
        "next_7_days_prediction": forecast_note,
        "prevention": prevention[:5],
        "source": "free-dynamic-weather",
        "sources": sources,
        "weather_summary": current_summary,
        "location_label": (weather_context.get("timezone") or "").replace("_", " "),
    })
    return payload


def _build_default_payload(class_name, confidence):
    """Build a safe advisory payload from local recommendation data."""
    disease_name = clean_name(class_name)
    severity = get_severity_band(confidence)
    recommendation = recommendation_library(class_name)

    chemical_steps = recommendation.get("chemical_treatment", [])
    organic_steps = recommendation.get("organic_treatment", [])

    return {
        "disease_name": disease_name,
        "cause": f"{disease_name} is affecting the crop and needs timely management to protect new growth and limit field spread.",
        "symptoms": list(symptom_library(class_name)),
        "treatment": {
            "chemical": _fallback_text(chemical_steps) or "Follow the product label and local agronomy guidance.",
            "organic": _fallback_text(organic_steps) or "Use sanitation and organic practices where possible.",
        },
        "chemical_treatment": list(chemical_steps),
        "organic_treatment": list(organic_steps),
        "prevention": [
            "Remove infected plant parts promptly and keep the area clean to reduce inoculum in the field.",
            "Avoid overhead irrigation when disease pressure is high so foliage dries faster.",
            "Scout nearby plants every 2-3 days and respond early if new lesions appear.",
        ],
        "best_spray_time": recommendation.get("best_spray_time") or best_spray_time_from_severity(severity),
        "recommended_fungicides_or_pesticides": list(recommendation.get("recommended", [])),
        "dosage": recommendation.get("dosage", "Use the selected product only at the labeled dose."),
        "irrigation_advice": recommendation.get("irrigation", "Maintain even soil moisture and avoid prolonged leaf wetness."),
        "fertilizer_advice": recommendation.get("fertilizer", "Use balanced nutrition and avoid excess nitrogen during disease pressure."),
        "weather_precautions": recommendation.get("weather", "Avoid spraying before rainfall or during strong wind."),
        "spread_risk": recommendation.get("spread", "Spread risk depends on crop moisture, weather, and sanitation."),
        "next_7_days_prediction": recommendation.get("prediction", "Disease pressure may increase if favorable conditions continue."),
        "safety_precautions": list(recommendation.get("safety_precautions", [])),
        "severity_level": severity,
        "pesticide_products": fallback_buy_links(class_name),
        "source": "fallback",
        "sources": [],
    }


def fetch_structured_web_advisory(class_name, confidence):
    """Build a structured advisory from an AI API with strict JSON parsing and local caching."""
    cache_key = f"ai:{class_name}:{get_severity_band(confidence)}"
    if cache_key in ADVISORY_CACHE:
        return ADVISORY_CACHE[cache_key]

    disease_name = clean_name(class_name)
    severity = get_severity_band(confidence)
    crop_name = extract_crop_name(class_name)
    recommendation = recommendation_library(class_name)

    api_key = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("AI API key is not configured")

    prompt = AI_ADVISORY_PROMPT.format(
        disease_name=disease_name,
        severity=severity,
        crop_name=crop_name,
    )

    request_body = {
        "model": os.getenv("AI_API_MODEL", "gpt-4o-mini"),
        "temperature": 0.2,
        "max_tokens": AI_RESPONSE_MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": """You are an expert agricultural scientist and advisor.

Rules:

* Each field must contain UNIQUE information (no repetition)
* Avoid generic phrases like 'continue spray program'
* Be specific with real agricultural practices
* Include actual chemical names (Mancozeb, Chlorothalonil, etc.)
* Keep advice practical and farmer-friendly
* Maintain consistency across fields
* Return STRICT JSON only""",
            },
            {"role": "user", "content": prompt},
        ],
    }

    request = Request(
        os.getenv("AI_API_URL", "https://api.openai.com/v1/chat/completions"),
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urlopen(request, timeout=AI_API_TIMEOUT_SECONDS) as response:
        response_body = response.read().decode("utf-8", errors="ignore")

    response_json = json.loads(response_body)
    parsed = _extract_json_object(_extract_chat_content(response_json))

    chemical_steps = _as_list(
        (parsed.get("treatment") or {}).get("chemical"),
        recommendation.get("chemical_treatment", []),
        limit=4,
    )
    organic_steps = _as_list(
        (parsed.get("treatment") or {}).get("organic"),
        recommendation.get("organic_treatment", []),
        limit=4,
    )

    payload = {
        "disease_name": disease_name,
        "cause": _as_text(
            parsed.get("cause"),
            f"{disease_name} is affecting the crop and needs timely management.",
        ),
        "symptoms": _as_list(parsed.get("symptoms"), symptom_library(class_name), limit=5),
        "treatment": {
            "chemical": _as_text(chemical_steps, "Follow the product label and local agronomy guidance."),
            "organic": _as_text(organic_steps, "Use sanitation and organic practices where possible."),
        },
        "chemical_treatment": chemical_steps,
        "organic_treatment": organic_steps,
        "prevention": _as_list(
            parsed.get("prevention"),
            [
                "Remove infected plant parts promptly and keep the area clean.",
                "Avoid overhead irrigation when disease pressure is high.",
                "Monitor nearby plants regularly and act early if symptoms increase.",
            ],
            limit=5,
        ),
        "best_spray_time": _as_text(
            parsed.get("best_spray_time"),
            recommendation.get("best_spray_time") or best_spray_time_from_severity(severity),
        ),
        "recommended_fungicides_or_pesticides": _as_list(
            parsed.get("recommended_products"),
            recommendation.get("recommended", []),
            limit=5,
        ),
        "dosage": _as_text(parsed.get("dosage"), recommendation.get("dosage", "Use the selected product only at the labeled dose.")),
        "irrigation_advice": _as_text(
            parsed.get("irrigation_advice"),
            recommendation.get("irrigation", "Maintain even soil moisture and avoid prolonged leaf wetness."),
        ),
        "fertilizer_advice": _as_text(
            parsed.get("fertilizer_advice"),
            recommendation.get("fertilizer", "Use balanced nutrition and avoid excess nitrogen during disease pressure."),
        ),
        "weather_precautions": _as_text(
            parsed.get("weather_precautions"),
            recommendation.get("weather", "Avoid spraying before rainfall or during strong wind."),
        ),
        "spread_risk": _as_text(
            parsed.get("spread_risk"),
            recommendation.get("spread", "Spread risk depends on crop moisture, weather, and sanitation."),
        ),
        "next_7_days_prediction": _as_text(
            parsed.get("next_7_days_prediction"),
            recommendation.get("prediction", "Disease pressure may increase if favorable conditions continue."),
        ),
        "safety_precautions": _as_list(
            parsed.get("safety_precautions"),
            recommendation.get("safety_precautions", [])
            or [
                "Wear gloves, mask and full sleeves during mixing and spraying.",
                "Do not spray during strong wind or peak afternoon heat.",
                "Keep children, animals and food items away from the spray area.",
                "Follow re-entry and pre-harvest interval mentioned on the product label.",
            ],
            limit=5,
        ),
        "severity_level": severity,
        "pesticide_products": fallback_buy_links(class_name),
        "source": "ai",
        "sources": [],
    }

    ADVISORY_CACHE[cache_key] = payload
    return payload


def get_local_advisory(class_name, confidence):
    """Build a structured advisory from the local CSV and fallback rules."""
    csv_points = CSV_ADVISORY_STORE.get((class_name, confidence_to_csv_band(confidence), "en"))
    severity = get_severity_band(confidence)

    if csv_points:
        payload = _build_default_payload(class_name, confidence)
        payload.update({
            "cause": csv_points[0],
            "prevention": csv_points[1:4] if len(csv_points) >= 4 else csv_points[:3],
            "treatment": {
                "chemical": csv_points[6] if len(csv_points) > 6 else payload["treatment"]["chemical"],
                "organic": csv_points[7] if len(csv_points) > 7 else payload["treatment"]["organic"],
            },
            "source": "local-csv",
        })
        payload["symptoms"] = list(payload.get("symptoms") or symptom_library(class_name))
        payload["cause"] = payload["cause"] or f"{clean_name(class_name)} is active and should be managed before symptoms spread further."
        return payload

    advisory_key = CLASS_TO_ADVISORY_KEY.get(class_name)
    advisory = ADVISORY_RULES_EN.get(advisory_key, {})
    payload = _build_default_payload(class_name, confidence)

    if not advisory_key or not advisory:
        return payload

    if "all" in advisory:
        all_points = advisory["all"]
        payload.update({
            "cause": all_points[0],
            "treatment": {
                "chemical": "No pesticide needed at this stage.",
                "organic": all_points[1] if len(all_points) > 1 else payload["treatment"]["organic"],
            },
            "prevention": all_points[:5],
            "best_spray_time": "No curative spray needed unless symptoms appear.",
        })
        return payload

    points = advisory.get(severity, ["No advisory available for this severity."])
    payload.update({
        "cause": points[0],
        "treatment": {
            "chemical": points[1] if len(points) > 1 else payload["treatment"]["chemical"],
            "organic": payload["treatment"]["organic"],
        },
        "prevention": points[:5],
    })
    return payload


def get_leaf_advisory(class_name, confidence, latitude=None, longitude=None):
    """Return a free dynamic advisory when possible, otherwise fall back safely."""
    try:
        if latitude is not None and longitude is not None:
            result = build_free_dynamic_advisory(class_name, confidence, latitude, longitude)
            return result
    except Exception as error:
        print("Dynamic weather advisory failed:", error)

    if os.getenv("ENABLE_PAID_AI_ADVISORY") == "1":
        try:
            result = fetch_structured_web_advisory(class_name, confidence)
            if result.get("source") == "ai":
                return result
        except Exception as error:
            print("AI failed:", error)

    return get_local_advisory(class_name, confidence)


def generate_ai_advisory_note(advisory_payload, confidence, healthy):
    """Convert advisory content into a short AI-style field note."""
    disease_name = advisory_payload.get("disease_name", "Unknown disease")
    cause = advisory_payload.get("cause", "Cause details are not available.")
    first_prevention = (advisory_payload.get("prevention") or ["Monitor nearby plants closely."])[0]
    spray_time = advisory_payload.get("best_spray_time", "Follow the product label timing.")
    severity = advisory_payload.get("severity_level", "low")
    weather_summary = advisory_payload.get("weather_summary")
    weather_suffix = f" Live weather signal: {weather_summary.lower()}." if weather_summary else ""

    healthy_variants = [
        f"Detected {disease_name} with {confidence}% confidence, but the crop appears stable. Keep monitoring closely and focus on {first_prevention.lower()}{weather_suffix}",
        f"{disease_name} is predicted at {confidence}% confidence and the leaf still looks stable. Continue routine scouting and make sure you {first_prevention.lower()}{weather_suffix}",
    ]
    active_variants = [
        f"Detected {disease_name} ({confidence}% confidence, {severity} severity). This is usually linked to {cause.lower()} Start with {first_prevention.lower()} Best time to spray is {spray_time.lower()}{weather_suffix}",
        f"{disease_name} is the likely issue at {confidence}% confidence with {severity} severity. The problem is commonly driven by {cause.lower()} Act first by {first_prevention.lower()} and plan sprays for {spray_time.lower()}{weather_suffix}",
    ]

    variant_index = sum(ord(char) for char in disease_name) % 2

    if healthy:
        return healthy_variants[variant_index]

    return active_variants[variant_index]


def enrich_farmer_advisory(class_name, confidence, advisory_payload, healthy):
    """Add disease-specific farmer guidance fields without overwriting AI-provided data."""
    recommendation = recommendation_library(class_name)
    prevention = list(advisory_payload.get("prevention", []))

    for item in [
        "Remove infected leaves or plants early to reduce further spread.",
        "Keep field sanitation clean and destroy infected crop debris away from the field.",
        "Continue scouting every 2-3 days and act quickly if symptoms increase.",
    ]:
        if len(prevention) >= 5:
            break
        if item not in prevention:
            prevention.append(item)

    default_safety = [
        "Wear gloves, mask and full sleeves during mixing and spraying.",
        "Do not spray during strong wind or peak afternoon heat.",
        "Keep children, animals and food items away from the spray area.",
        "Follow re-entry and pre-harvest interval mentioned on the product label.",
    ]

    advisory_payload.update({
        "crop_name": extract_crop_name(class_name),
        "disease_name": normalize_disease_name(class_name),
        "severity": advisory_payload.get("severity_level", get_severity_band(confidence)).capitalize(),
        "cause": advisory_payload.get("cause") or f"{normalize_disease_name(class_name)} is affecting the crop and needs timely management.",
        "symptoms": advisory_payload.get("symptoms") or symptom_library(class_name),
        "advisory_note": generate_ai_advisory_note(advisory_payload, confidence, healthy),
        "chemical_treatment": advisory_payload.get("chemical_treatment") or recommendation.get("chemical_treatment", []),
        "organic_treatment": advisory_payload.get("organic_treatment") or recommendation.get("organic_treatment", []),
        "recommended_fungicides_or_pesticides": advisory_payload.get("recommended_fungicides_or_pesticides") or recommendation.get("recommended", []),
        "best_spray_time": advisory_payload.get("best_spray_time") or recommendation.get("best_spray_time") or best_spray_time_from_severity(advisory_payload.get("severity_level", get_severity_band(confidence))),
        "dosage": advisory_payload.get("dosage") or recommendation.get("dosage"),
        "irrigation_advice": advisory_payload.get("irrigation_advice") or recommendation.get("irrigation"),
        "fertilizer_advice": advisory_payload.get("fertilizer_advice") or recommendation.get("fertilizer"),
        "weather_precautions": advisory_payload.get("weather_precautions") or recommendation.get("weather"),
        "spread_risk": advisory_payload.get("spread_risk") or recommendation.get("spread"),
        "next_7_days_prediction": advisory_payload.get("next_7_days_prediction") or recommendation.get("prediction"),
        "prevention": prevention[:5],
        "sources": advisory_payload.get("sources") or fallback_source_links(class_name),
        "safety_precautions": advisory_payload.get("safety_precautions") or recommendation.get("safety_precautions") or default_safety,
    })
    return advisory_payload
