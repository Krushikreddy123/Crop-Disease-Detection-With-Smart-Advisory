"""Advisory generation helpers for disease explanations and treatment guidance."""

import csv
import os
import re
import socket
from html import unescape
from urllib.parse import quote
from urllib.request import Request, urlopen

from utils import (
    ADVISORY_CSV_PATH,
    ADVISORY_RULES_EN,
    CLASS_TO_ADVISORY_KEY,
    WEB_TIMEOUT_SECONDS,
    best_spray_time_from_severity,
    clean_name,
    confidence_to_csv_band,
    extract_crop_name,
    extract_domain,
    fallback_buy_links,
    fallback_source_links,
    get_severity_band,
    is_healthy,
    normalize_disease_name,
    normalize_text,
    product_search_suffix,
    recommendation_library,
    symptom_library,
    tokenize_label,
)

ADVISORY_CACHE = {}


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


def fetch_search_results(query):
    """Fetch lightweight search results for a disease-specific query."""
    request = Request(
        f"https://html.duckduckgo.com/html/?q={quote(query)}",
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123 Safari/537.36"},
    )
    socket.setdefaulttimeout(WEB_TIMEOUT_SECONDS)
    with urlopen(request, timeout=WEB_TIMEOUT_SECONDS) as response:
        body = response.read().decode("utf-8", errors="ignore")

    results, seen_urls = [], set()
    blocks = re.findall(r'(<div[^>]+class="[^"]*result[^"]*"[^>]*>.*?</div>\s*</div>)', body, re.IGNORECASE | re.DOTALL)
    for block in blocks:
        link_match = re.search(r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', block, re.IGNORECASE | re.DOTALL)
        snippet_match = re.search(r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>|<div[^>]*class="result__snippet"[^>]*>(?P<divsnippet>.*?)</div>', block, re.IGNORECASE | re.DOTALL)
        if not link_match:
            continue
        href = unescape(link_match.group("href"))
        if href in seen_urls:
            continue
        results.append({
            "title": normalize_text(link_match.group("title")),
            "url": href,
            "domain": extract_domain(href),
            "snippet": normalize_text((snippet_match.group("snippet") if snippet_match else "") or (snippet_match.group("divsnippet") if snippet_match else "")),
        })
        seen_urls.add(href)
    return results


def score_result(result, disease_tokens, keywords):
    """Score a web result by disease-name and keyword relevance."""
    haystack = f"{result['title']} {result['snippet']} {result['domain']}".lower()
    score = sum(3 for token in disease_tokens if token in haystack) + sum(2 for keyword in keywords if keyword in haystack)
    trusted_domains = (".edu", ".gov", "extension", "agri", "agriculture", "cabi", "syngenta", "bayer", "upl", "ufl.edu", "cornell")
    if any(item in result["domain"] for item in trusted_domains):
        score += 2
    return score + (1 if result["snippet"] else 0)


def best_result_for_query(query, disease_label, keywords, min_score=3):
    """Return the single best web result for a disease query."""
    disease_tokens = tokenize_label(disease_label)
    candidates = [(score_result(result, disease_tokens, keywords), result) for result in fetch_search_results(query)]
    candidates = [item for item in candidates if item[0] >= min_score]
    if not candidates:
        raise ValueError(f"No strong results found for query: {query}")
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]


def top_results_for_query(query, disease_label, keywords, limit=3, min_score=2):
    """Return the top matching web results for a disease query."""
    disease_tokens = tokenize_label(disease_label)
    candidates = [(score_result(result, disease_tokens, keywords), result) for result in fetch_search_results(query)]
    candidates = [item for item in candidates if item[0] >= min_score]
    return [item[1] for item in sorted(candidates, key=lambda item: item[0], reverse=True)[:limit]]


def fetch_structured_web_advisory(class_name, confidence):
    """Build a structured advisory from live web search snippets."""
    cache_key = f"{class_name}:{get_severity_band(confidence)}"
    if cache_key in ADVISORY_CACHE:
        return ADVISORY_CACHE[cache_key]

    disease_label = clean_name(class_name)
    severity = get_severity_band(confidence)
    if is_healthy(class_name):
        maintenance = best_result_for_query(f"{disease_label} maintenance care best practices", disease_label, {"maintenance", "care", "healthy", "nutrition", "irrigation", "monitor"})
        payload = {"disease_name": disease_label, "cause": "No visible disease detected. The leaf appears healthy based on the current prediction.", "treatment": {"chemical": maintenance["snippet"] or "No pesticide recommended for a healthy leaf.", "organic": "Use compost, balanced irrigation, and regular scouting to keep plants healthy."}, "prevention": ["Keep field hygiene clean and remove plant debris.", "Avoid overwatering and long periods of leaf wetness.", "Continue monitoring plants every few days for new symptoms."], "best_spray_time": "Not needed unless symptoms appear; if using foliar nutrition, spray in early morning or late afternoon.", "severity_level": severity, "pesticide_products": top_results_for_query(f"{disease_label} {product_search_suffix(class_name)}", disease_label, {"buy", "shop", "product", "spray", "nutrient", "fungicide"}, limit=3, min_score=1) or fallback_buy_links(class_name), "source": "web", "sources": [maintenance]}
    else:
        cause_result = best_result_for_query(f"{disease_label} cause symptoms crop disease", disease_label, {"cause", "symptoms", "disease", "fungal", "bacterial", "viral", "mite"})
        chemical_result = best_result_for_query(f"{disease_label} treatment fungicide bactericide miticide", disease_label, {"treatment", "control", "fungicide", "bactericide", "miticide", "spray"})
        organic_result = best_result_for_query(f"{disease_label} organic treatment neem biocontrol", disease_label, {"organic", "neem", "biocontrol", "cultural", "natural"})
        prevention_results = top_results_for_query(f"{disease_label} prevention management", disease_label, {"prevention", "management", "avoid", "remove", "monitor", "sanitation"}, limit=4)
        prevention = [item["snippet"] for item in prevention_results if item["snippet"]][:3] or ["Remove infected plant parts promptly and keep the area clean.", "Avoid overhead irrigation when disease pressure is high.", "Monitor nearby plants regularly and follow label directions for sprays."]
        payload = {"disease_name": disease_label, "cause": cause_result["snippet"] or cause_result["title"], "treatment": {"chemical": chemical_result["snippet"] or chemical_result["title"], "organic": organic_result["snippet"] or organic_result["title"]}, "prevention": prevention, "best_spray_time": best_spray_time_from_severity(severity), "severity_level": severity, "pesticide_products": top_results_for_query(f"{disease_label} {product_search_suffix(class_name)}", disease_label, {"buy", "shop", "fungicide", "bactericide", "miticide", "pesticide", "product"}, limit=3, min_score=1) or fallback_buy_links(class_name), "source": "web", "sources": [cause_result, chemical_result, organic_result] + prevention_results[:2]}
    ADVISORY_CACHE[cache_key] = payload
    return payload


def get_local_advisory(class_name, confidence):
    """Build a structured advisory from the local CSV and fallback rules."""
    csv_points = CSV_ADVISORY_STORE.get((class_name, confidence_to_csv_band(confidence), "en"))
    if csv_points:
        return {"disease_name": clean_name(class_name), "cause": csv_points[0], "treatment": {"chemical": csv_points[6] if len(csv_points) > 6 else "Consult the product label before spraying.", "organic": csv_points[7] if len(csv_points) > 7 else "Use field sanitation and organic practices where possible."}, "prevention": csv_points[1:4] if len(csv_points) >= 4 else csv_points[:3], "best_spray_time": best_spray_time_from_severity(get_severity_band(confidence)), "severity_level": get_severity_band(confidence), "pesticide_products": fallback_buy_links(class_name), "source": "local-csv", "sources": []}

    advisory_key = CLASS_TO_ADVISORY_KEY.get(class_name)
    advisory = ADVISORY_RULES_EN.get(advisory_key, {})
    if not advisory_key or not advisory:
        return {"disease_name": clean_name(class_name), "cause": "No advisory available for this class.", "treatment": {"chemical": "No chemical recommendation available.", "organic": "No organic recommendation available."}, "prevention": ["Continue observation and consult an agriculture expert if symptoms spread."], "best_spray_time": best_spray_time_from_severity(get_severity_band(confidence)), "severity_level": get_severity_band(confidence), "pesticide_products": fallback_buy_links(class_name), "source": "fallback", "sources": []}

    if "all" in advisory:
        return {"disease_name": clean_name(class_name), "cause": advisory["all"][0], "treatment": {"chemical": "No pesticide needed at this stage.", "organic": advisory["all"][1] if len(advisory["all"]) > 1 else "Maintain balanced nutrition and hygiene."}, "prevention": advisory["all"], "best_spray_time": "No curative spray needed unless symptoms appear.", "severity_level": get_severity_band(confidence), "pesticide_products": fallback_buy_links(class_name), "source": "fallback", "sources": []}

    severity_band = get_severity_band(confidence)
    points = advisory.get(severity_band, ["No advisory available for this severity."])
    return {"disease_name": clean_name(class_name), "cause": points[0], "treatment": {"chemical": points[1] if len(points) > 1 else "Follow the pesticide label and local agronomy guidance.", "organic": "Use sanitation, removal of infected tissue, and airflow improvement as organic support."}, "prevention": points, "best_spray_time": best_spray_time_from_severity(severity_band), "severity_level": severity_band, "pesticide_products": fallback_buy_links(class_name), "source": "fallback", "sources": []}


def get_leaf_advisory(class_name, confidence):
    """Return a live advisory when possible, otherwise fall back to local content."""
    try:
        return fetch_structured_web_advisory(class_name, confidence)
    except Exception:
        return get_local_advisory(class_name, confidence)


def generate_ai_advisory_note(advisory_payload, confidence, healthy):
    """Convert advisory content into a short AI-style field note."""
    disease_name = advisory_payload.get("disease_name", "Unknown disease")
    cause = advisory_payload.get("cause", "Cause details are not available.")
    chemical = advisory_payload.get("treatment", {}).get("chemical", "No chemical treatment available.")
    organic = advisory_payload.get("treatment", {}).get("organic", "No organic alternative available.")
    first_prevention = (advisory_payload.get("prevention") or ["Monitor nearby plants closely."])[0]
    spray_time = advisory_payload.get("best_spray_time", "Follow the product label timing.")
    if healthy:
        return f"AI Advisory Note: The leaf is currently predicted as {disease_name} with {confidence}% confidence, which suggests the plant is in a stable condition. Keep regular monitoring in place, maintain clean field hygiene, and use foliar support only if needed. Priority action: {first_prevention}"
    return f"AI Advisory Note: The model predicts {disease_name} with {confidence}% confidence and a {advisory_payload.get('severity_level', 'low')} severity pattern. This usually indicates: {cause} Immediate field action should focus on {chemical.lower()} If you want a lower-input option, consider {organic.lower()} The next best preventive step is: {first_prevention} Recommended spray window: {spray_time}"


def enrich_farmer_advisory(class_name, confidence, advisory_payload, healthy):
    """Add disease-specific farmer guidance fields to the advisory payload."""
    prevention = list(advisory_payload.get("prevention", []))
    for item in ["Remove infected leaves or plants early to reduce further spread.", "Keep field sanitation clean and destroy infected crop debris away from the field.", "Continue scouting every 2-3 days and act quickly if symptoms increase."]:
        if len(prevention) >= 3:
            break
        if item not in prevention:
            prevention.append(item)

    recommendation = recommendation_library(class_name)
    advisory_payload.update({
        "crop_name": extract_crop_name(class_name),
        "disease_name": normalize_disease_name(class_name),
        "severity": advisory_payload.get("severity_level", get_severity_band(confidence)).capitalize(),
        "cause": advisory_payload.get("cause") or f"{normalize_disease_name(class_name)} is affecting the crop and needs timely management.",
        "symptoms": symptom_library(class_name),
        "advisory_note": generate_ai_advisory_note(advisory_payload, confidence, healthy),
        "chemical_treatment": recommendation["chemical_treatment"],
        "organic_treatment": recommendation["organic_treatment"],
        "recommended_fungicides_or_pesticides": recommendation["recommended"],
        "best_spray_time": advisory_payload.get("best_spray_time") or best_spray_time_from_severity(advisory_payload.get("severity_level", get_severity_band(confidence))),
        "dosage": advisory_payload.get("dosage") or recommendation["dosage"],
        "irrigation_advice": advisory_payload.get("irrigation_advice") or recommendation["irrigation"],
        "fertilizer_advice": advisory_payload.get("fertilizer_advice") or recommendation["fertilizer"],
        "weather_precautions": advisory_payload.get("weather_precautions") or recommendation["weather"],
        "spread_risk": advisory_payload.get("spread_risk") or recommendation["spread"],
        "next_7_days_prediction": advisory_payload.get("next_7_days_prediction") or recommendation["prediction"],
        "prevention": prevention[:5],
        "sources": advisory_payload.get("sources") or fallback_source_links(class_name),
        "safety_precautions": ["Wear gloves, mask and full sleeves during mixing and spraying.", "Do not spray during strong wind or peak afternoon heat.", "Keep children, animals and food items away from the spray area.", "Follow re-entry and pre-harvest interval mentioned on the product label."],
    })
    return advisory_payload
