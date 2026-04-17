"""Shared helpers and constants for the crop disease app."""

import html
import os
import re
from urllib.parse import quote, urlparse

UPLOAD_DIR = "static/uploads"
HEATMAP_DIR = "static/heatmaps"
WEB_TIMEOUT_SECONDS = 2.5
ADVISORY_CSV_PATH = "crop_disease_full_advisory.csv"

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
    "Tomato_healthy",
]

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
    "Tomato_healthy": "tomato_healthy",
}

ADVISORY_RULES_EN = {
    "pepper_bacterial_spot": {"low": ["Prune spotted leaves and avoid overhead irrigation.", "Spray a copper-based bactericide in the evening."], "medium": ["Remove heavily infected leaves and sanitize tools after each plant.", "Use copper + mancozeb rotation every 7 days."], "high": ["Isolate infected plants immediately to reduce spread.", "Destroy severely infected plants and disinfect the area."]},
    "pepper_healthy": {"all": ["Leaf looks healthy. Keep scouting twice per week.", "Maintain balanced NPK and avoid excess leaf wetness."]},
    "potato_early_blight": {"low": ["Start preventive fungicide spray and remove lower infected leaves.", "Improve airflow by spacing and weed control."], "medium": ["Rotate fungicides (chlorothalonil / mancozeb classes) every 7 days.", "Irrigate at soil level and avoid evening wet foliage."], "high": ["Remove highly infected plants to stop field spread.", "Plan crop rotation for next season and avoid potato residue."]},
    "potato_late_blight": {"low": ["Begin anti-oomycete spray immediately.", "Avoid water splash and keep field drainage clear."], "medium": ["Spray systemic + contact fungicide mix as per label.", "Remove nearby infected plants and monitor daily."], "high": ["Emergency control needed: rogue badly infected patches.", "Do not move infected foliage through healthy rows."]},
    "potato_healthy": {"all": ["Leaf looks healthy. Continue preventive scouting.", "Use drip irrigation and maintain clean field borders."]},
    "tomato_bacterial_spot": {"low": ["Remove affected leaves and avoid touching plants when wet.", "Use copper spray with spreader sticker."], "medium": ["Disinfect pruning tools and stakes daily.", "Apply bactericide rotation every 5-7 days."], "high": ["Remove heavily infected plants and bag plant waste.", "Do not compost infected tomato debris."]},
    "tomato_early_blight": {"low": ["Remove bottom leaves touching soil.", "Start protective fungicide program."], "medium": ["Spray in rotation and improve canopy ventilation.", "Mulch soil to reduce spore splash."], "high": ["Remove severe plants and protect nearby healthy plants quickly.", "Follow strict 2-3 year crop rotation."]},
    "tomato_late_blight": {"low": ["Start immediate late blight spray schedule.", "Avoid overhead watering and long leaf wetness."], "medium": ["Use recommended systemic fungicides and monitor every day.", "Remove infected tissue as soon as seen."], "high": ["Urgent containment: remove entire infected plants.", "Restrict field movement to prevent disease transfer."]},
    "tomato_leaf_mold": {"low": ["Reduce humidity in canopy and improve ventilation.", "Remove first infected leaves."], "medium": ["Spray labeled fungicide and avoid dense pruning wounds.", "Water early morning to dry leaves faster."], "high": ["Thin canopy aggressively and remove severe plants.", "Sanitize greenhouse or stakes after harvest."]},
    "tomato_septoria_leaf_spot": {"low": ["Pick infected lower leaves and keep soil mulched.", "Begin preventive fungicide applications."], "medium": ["Continue spray program at 7-day interval.", "Avoid handling plants when wet."], "high": ["Remove heavily infected plants.", "Destroy residues after harvest to break lifecycle."]},
    "tomato_spider_mites": {"low": ["Spray water mist under leaves in morning to reduce mites.", "Release or conserve beneficial predators if available."], "medium": ["Apply approved miticide and rotate active ingredients.", "Remove worst infested leaves."], "high": ["Isolate hotspot plants and apply full-coverage miticide.", "Repeat follow-up spray as label recommends."]},
    "tomato_target_spot": {"low": ["Remove spotted leaves and improve airflow.", "Start protectant fungicide schedule."], "medium": ["Increase spray coverage on lower canopy.", "Avoid overhead irrigation in evening."], "high": ["Remove severely affected plants and sanitize tools.", "Use strict residue cleanup after harvest."]},
    "tomato_yellow_leaf_curl_virus": {"low": ["Control whiteflies using yellow sticky traps.", "Remove plants with strong curling symptoms."], "medium": ["Use insect-proof netting and vector control spray program.", "Avoid mixing infected and healthy nursery seedlings."], "high": ["Rogue infected plants immediately.", "Focus on aggressive whitefly suppression in whole plot."]},
    "tomato_mosaic_virus": {"low": ["Avoid tobacco contact before handling plants.", "Disinfect hands and tools regularly."], "medium": ["Remove infected plants and sanitize supports.", "Use only clean seedling sources."], "high": ["Uproot severely infected plants and dispose safely.", "Disinfect entire working area and tools."]},
    "tomato_healthy": {"all": ["Leaf looks healthy. Keep monitoring every 3-4 days.", "Maintain balanced nutrition and preventive hygiene."]},
}

SYMPTOM_LIBRARY = {
    "bacterial_spot": ["Small dark brown to black water-soaked spots appear on leaves.", "Spots may develop yellow halos and later join together.", "Older leaves dry early and plant vigor starts dropping."],
    "early_blight": ["Brown circular spots with concentric ring pattern appear first on older leaves.", "Lower leaves turn yellow and dry from the edges.", "Defoliation increases as the disease moves upward."],
    "late_blight": ["Large water-soaked lesions spread quickly on leaves.", "White fungal growth may appear on the lower leaf surface in humid weather.", "Leaves collapse rapidly and infection can move to stems and fruits or tubers."],
    "leaf_mold": ["Pale green to yellow patches appear on the upper leaf surface.", "Olive green to grey mold develops on the underside of leaves.", "Leaves curl, dry and drop when humidity stays high."],
    "septoria": ["Many small round spots with grey centers appear on older leaves.", "Dark margins form around each spot and leaves yellow gradually.", "Heavy spotting leads to fast leaf drop from the lower canopy."],
    "spider_mites": ["Tiny yellow specks appear across the leaf surface.", "Fine webbing may be seen on the underside of leaves.", "Leaves bronze, dry and curl under heavy infestation."],
    "target_spot": ["Brown target-like lesions appear with clear circular zoning.", "Spots enlarge and merge under warm humid conditions.", "Leaf drop increases and fruits may also show lesions."],
    "yellowleaf": ["Leaves curl upward and become smaller than normal.", "Yellowing appears between veins and new growth becomes stunted.", "Plant growth slows sharply and fruit set is reduced."],
    "mosaic_virus": ["Mosaic light and dark green pattern appears on leaves.", "Leaves become narrow, puckered or distorted.", "Plants show stunting and uneven canopy growth."],
    "healthy": ["Leaves are uniformly green without active lesions.", "No strong spotting, curling or fungal growth is visible.", "Plant tissue looks stable at the current prediction stage."],
}

RECOMMENDATION_LIBRARY = {
    "bacterial_spot": {"chemical_treatment": ["Spray copper oxychloride 2.5-3 g per liter of water.", "Rotate with copper hydroxide plus mancozeb where labels allow.", "Repeat spray at 7 day interval during active spread."], "organic_treatment": ["Spray neem oil 3-5 ml per liter with proper emulsifier.", "Use Bacillus-based bio-products if locally available.", "Remove badly infected leaves before organic spray."], "recommended": ["Copper oxychloride", "Copper hydroxide", "Mancozeb"], "dosage": "Copper oxychloride 2.5-3 g/L or copper hydroxide as per label; repeat every 7 days if humidity stays high.", "irrigation": "Avoid overhead irrigation and keep foliage dry as much as possible.", "fertilizer": "Avoid excess nitrogen; use balanced NPK with enough calcium to strengthen foliage.", "weather": "Do not spray before rainfall; restart protection after heavy rain washes deposits away.", "spread": "Spread risk is high through splashing water, tools and leaf contact.", "prediction": "If wet weather continues for 7 days, infection can move quickly to fresh leaves."},
    "early_blight": {"chemical_treatment": ["Start protectant fungicide spray with mancozeb 2-2.5 g/L.", "Rotate with chlorothalonil or azoxystrobin as per local label.", "Maintain 7 day spray interval under strong disease pressure."], "organic_treatment": ["Spray neem extract or compost tea only after removing infected lower leaves.", "Mulch soil to reduce spore splash to new leaves.", "Improve airflow by pruning dense lower canopy."], "recommended": ["Mancozeb", "Chlorothalonil", "Azoxystrobin"], "dosage": "Mancozeb 2-2.5 g/L or chlorothalonil as per label; repeat in 5-7 days under humid conditions.", "irrigation": "Use furrow or drip irrigation and avoid evening leaf wetness.", "fertilizer": "Use potash-rich balanced nutrition and avoid pushing soft growth with excess urea.", "weather": "Warm humid weather increases infection, so maintain preventive cover before long cloudy spells.", "spread": "Spread risk is medium to high because spores move from lower infected leaves and crop debris.", "prediction": "Without removal of infected leaves and timely spray, early blight will intensify over the next 7 days."},
    "late_blight": {"chemical_treatment": ["Use systemic plus contact fungicide mix recommended for late blight.", "Common options include metalaxyl + mancozeb or cymoxanil + mancozeb as per label.", "Repeat at 5-7 day interval during cool wet periods."], "organic_treatment": ["Remove infected leaves immediately to lower inoculum.", "Improve drainage and reduce standing moisture around plants.", "Use copper-based organic-compatible sprays where permitted."], "recommended": ["Metalaxyl + Mancozeb", "Cymoxanil + Mancozeb", "Copper oxychloride"], "dosage": "Apply late blight fungicide strictly as per product label; keep interval short during continuous wet weather.", "irrigation": "Stop overhead irrigation and keep field drainage open.", "fertilizer": "Do not apply heavy nitrogen now; support crop recovery with balanced potassium and micronutrients.", "weather": "Cool cloudy and wet weather can trigger explosive spread; maintain protective spray before such windows.", "spread": "Spread risk is very high under wet and cool conditions.", "prediction": "Next 7 days can show rapid field spread if humidity and leaf wetness remain high."},
    "spider_mites": {"chemical_treatment": ["Use a registered miticide and rotate active ingredients to avoid resistance.", "Ensure good spray coverage on the underside of leaves.", "Repeat only according to label interval after scouting."], "organic_treatment": ["Spray neem oil 3-5 ml/L targeting lower leaf surface.", "Wash hotspots with plain water in the morning where feasible.", "Conserve predatory mites and beneficial insects."], "recommended": ["Abamectin", "Spiromesifen", "Neem oil"], "dosage": "Use miticide exactly as per label and achieve full underside coverage for best control.", "irrigation": "Maintain even moisture; drought stress usually worsens mite outbreaks.", "fertilizer": "Avoid excess nitrogen because lush soft growth can favor pest buildup.", "weather": "Hot dry weather favors mites, so scout daily in such periods.", "spread": "Spread risk is medium and usually increases from hotspot plants outward.", "prediction": "If hot dry conditions continue for 7 days, mite pressure may rise quickly to a more damaging level."},
    "virus": {"chemical_treatment": ["No direct curative chemical exists for virus-infected tissue.", "Control vectors such as whiteflies or aphids with recommended insecticides.", "Remove heavily infected plants to reduce secondary spread."], "organic_treatment": ["Use yellow sticky traps to suppress vector movement.", "Spray neem oil 3-5 ml/L to reduce vector feeding pressure.", "Rogue badly affected plants early and destroy them outside the field."], "recommended": ["Imidacloprid", "Thiamethoxam", "Neem oil"], "dosage": "Use vector control insecticide strictly as per label; repeat only after scouting and threshold check.", "irrigation": "Keep irrigation balanced and avoid crop stress because weak plants decline faster under virus attack.", "fertilizer": "Use balanced fertilizers with adequate potash; avoid excessive nitrogen that attracts vectors.", "weather": "Warm dry weather often favors vector movement, so increase monitoring during such weeks.", "spread": "Spread risk is high through insect vectors and infected planting material.", "prediction": "Over the next 7 days, new plants may show symptoms if vector control is delayed."},
    "healthy": {"chemical_treatment": ["No curative chemical spray is needed right now.", "Use preventive fungicide only if the local forecast strongly favors disease outbreak.", "Continue scouting before starting any pesticide."], "organic_treatment": ["Use compost and biostimulants to maintain plant vigor.", "Keep field sanitation clean and remove weeds regularly.", "Use neem-based preventive spray only if local pest pressure rises."], "recommended": ["No immediate pesticide required", "Micronutrient foliar spray if deficiency appears", "Neem oil if pest pressure starts"], "dosage": "No disease-control dosage needed now; spray only when symptoms or pest pressure appear.", "irrigation": "Maintain uniform irrigation and avoid waterlogging or long dry stress.", "fertilizer": "Continue balanced nutrition based on crop stage and soil test.", "weather": "After rain or humid spells, scout carefully for first symptoms.", "spread": "Current spread risk is low because no active disease is strongly indicated.", "prediction": "If weather remains normal and field hygiene is maintained, crop condition should stay stable over the next 7 days."},
    "default": {"chemical_treatment": ["Use a locally recommended fungicide or pesticide matched to the disease group.", "Rotate active ingredients to reduce resistance pressure.", "Repeat spray only at labeled interval after field scouting."], "organic_treatment": ["Remove infected leaves and keep the field clean.", "Use neem-based or bio-control products where available.", "Improve airflow and reduce moisture on leaf surface."], "recommended": ["Consult local label-recommended product", "Neem oil", "Copper-based protectant where suitable"], "dosage": "Use the label dosage of the selected product and maintain correct water volume for full coverage.", "irrigation": "Avoid prolonged leaf wetness and maintain even soil moisture.", "fertilizer": "Use balanced NPK and avoid excess nitrogen during disease pressure.", "weather": "Avoid spraying during strong wind or just before rain.", "spread": "Spread risk depends on humidity, crop density and delay in control measures.", "prediction": "If conditions remain favorable, symptoms may expand during the next 7 days."},
}


def ensure_static_dirs():
    """Create upload and heatmap folders if they do not exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(HEATMAP_DIR, exist_ok=True)


def clean_name(name):
    """Convert model class names into readable text."""
    return name.replace("_", " ")


def is_healthy(name):
    """Return True when the predicted class represents a healthy leaf."""
    return "healthy" in name.lower()


def get_severity_band(confidence):
    """Map confidence to a low, medium, or high severity band."""
    if confidence <= 50:
        return "low"
    if confidence <= 75:
        return "medium"
    return "high"


def confidence_to_csv_band(confidence):
    """Map confidence into the CSV severity bucket format."""
    if confidence <= 25:
        return "0-25"
    if confidence <= 50:
        return "25-50"
    if confidence <= 75:
        return "50-75"
    return "75-100"


def strip_html_tags(raw_html):
    """Remove HTML tags from a search snippet."""
    return re.sub(r"<[^>]+>", " ", raw_html)


def normalize_text(text):
    """Normalize web text so it is easier to display in the advisory."""
    return re.sub(r"\s+", " ", html.unescape(strip_html_tags(text))).strip()


def extract_domain(url):
    """Extract a clean domain from a URL."""
    hostname = urlparse(url).netloc.lower()
    return hostname[4:] if hostname.startswith("www.") else hostname


def tokenize_label(label):
    """Turn a disease name into searchable keyword tokens."""
    return {token for token in re.split(r"[^a-z0-9]+", label.lower()) if len(token) > 2 and token not in {"leaf", "plant", "crop"}}


def product_search_suffix(class_name):
    """Pick a product query suffix that matches the disease type."""
    lowered = class_name.lower()
    if "spider_mites" in lowered or "mite" in lowered:
        return "miticide buy online"
    if "virus" in lowered:
        return "vector control insecticide buy online"
    if "bacterial" in lowered:
        return "bactericide copper pesticide buy online"
    if "healthy" in lowered:
        return "plant care nutrient spray buy online"
    return "fungicide pesticide buy online"


def best_spray_time_from_severity(severity):
    """Return a spray-time recommendation for the severity level."""
    if severity == "high":
        return "Spray immediately at early morning or late afternoon when wind is low."
    if severity == "medium":
        return "Spray in the early morning or late afternoon and repeat on the label schedule."
    return "Start preventive spray in the early morning after dew dries or in the late afternoon."


def fallback_buy_links(class_name):
    """Provide a fallback product search link when no direct links are available."""
    query = f"{clean_name(class_name)} {product_search_suffix(class_name)}"
    return [{"title": f"Search products for {clean_name(class_name)}", "url": f"https://duckduckgo.com/?q={quote(query)}", "domain": "duckduckgo.com"}]


def fallback_source_links(class_name):
    """Provide fallback source links when live web citations are unavailable."""
    disease_label = clean_name(class_name)
    return [
        {"title": f"Disease management search for {disease_label}", "url": f"https://duckduckgo.com/?q={quote(disease_label + ' disease management crop')}", "domain": "duckduckgo.com"},
        {"title": f"Treatment search for {disease_label}", "url": f"https://duckduckgo.com/?q={quote(disease_label + ' treatment fungicide pesticide')}", "domain": "duckduckgo.com"},
    ]


def extract_crop_name(class_name):
    """Extract the crop name from a model class label."""
    return class_name.replace("___", "_").replace("__", "_").split("_")[0].capitalize()


def normalize_disease_name(class_name):
    """Remove the crop prefix from a readable disease label."""
    crop_name = extract_crop_name(class_name)
    readable = clean_name(class_name)
    if readable.lower().startswith(crop_name.lower()):
        suffix = readable[len(crop_name):].strip()
        return suffix if suffix else "Healthy leaf"
    return readable


def symptom_library(class_name):
    """Return disease-specific symptom bullets for the UI."""
    lowered = class_name.lower()
    for key, symptoms in SYMPTOM_LIBRARY.items():
        if key in lowered:
            return symptoms
    return ["Visible leaf symptoms are present and should be monitored closely.", "Disease pressure may increase if weather remains favorable for spread.", "Check nearby plants for similar early symptoms."]


def recommendation_library(class_name):
    """Return disease-specific recommendation content for advisory enrichment."""
    lowered = class_name.lower()
    for key, recommendation in RECOMMENDATION_LIBRARY.items():
        if key != "default" and key in lowered:
            return recommendation
    return RECOMMENDATION_LIBRARY["default"]
