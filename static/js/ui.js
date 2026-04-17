// UI layer for dashboard interactions and DOM updates.

const predictButton = document.getElementById("predictBtn");
const speakAdvisoryButton = document.getElementById("speakAdvisoryBtn");
const stopVoiceButton = document.getElementById("stopVoiceBtn");
const statusMessage = document.getElementById("statusMessage");
const resultsShell = document.getElementById("resultsShell");
const advisoryCard = document.getElementById("advisoryCard");
const imageInput = document.getElementById("image");
let lastSpeechText = "";

// Toggle the button loading state.
function setLoadingState(isLoading) {
  predictButton.disabled = isLoading;
  predictButton.querySelector(".button-text").innerText = isLoading ? "Predicting..." : "Predict";
  statusMessage.innerText = isLoading ? "Running detection and generating advisory..." : statusMessage.innerText;
}

// Render the hero summary section.
function renderHero(advisory, data) {
  const severity = (advisory.severity_level || "low").toLowerCase();
  const severityBadge = document.getElementById("severityBadge");

  document.getElementById("cropName").innerText = advisory.crop_name ? `Crop: ${advisory.crop_name}` : "Crop analysis";
  document.getElementById("diseaseName").innerText = advisory.disease_name || data.class;
  document.getElementById("confidenceBadge").innerText = `${data.confidence}%`;

  severityBadge.className = `badge severity-badge ${severityClassMap[severity] || severityClassMap.low}`;
  severityBadge.innerText = `${(advisory.severity || severity).toString()} Severity`;
}

// Render the AI note as short bullets.
function renderAiNote(advisory, data) {
  const aiBullets = splitNoteIntoBullets(
    advisory.advisory_note || data.ai_note,
    "No AI advisory note is available."
  ).map(emphasizeActions);

  lastSpeechText = splitNoteIntoBullets(
    advisory.advisory_note || data.ai_note,
    "No AI advisory note is available."
  ).join(" ");

  renderList("aiNoteList", aiBullets, "No AI advisory note is available.", true);
}

// Render all advisory cards from the response payload.
function renderAdvisoryCards(advisory, data) {
  renderList("causeList", toArray(advisory.cause, "No cause details available."), "No cause details available.");
  renderList("symptomsList", advisory.symptoms, "No symptoms listed.");
  renderList("treatmentSummaryList", [
    advisory.treatment?.chemical ? `Chemical: ${advisory.treatment.chemical}` : null,
    advisory.treatment?.organic ? `Organic: ${advisory.treatment.organic}` : null
  ].filter(Boolean), "No treatment overview available.");
  renderList("chemicalTreatmentList", advisory.chemical_treatment, "No chemical treatment steps available.");
  renderList("organicTreatmentList", advisory.organic_treatment, "No organic treatment steps available.");
  renderList("preventionList", advisory.prevention, "No prevention steps available.");
  renderList("sprayTimeList", toArray(advisory.best_spray_time, "No spray timing advice available."), "No spray timing advice available.");
  renderList("recommendedList", advisory.recommended_fungicides_or_pesticides, "No recommended pesticide list available.");
  renderList("dosageList", toArray(advisory.dosage, "No dosage details available."), "No dosage details available.");
  renderList("irrigationList", toArray(advisory.irrigation_advice, "No irrigation advice available."), "No irrigation advice available.");
  renderList("fertilizerList", toArray(advisory.fertilizer_advice, "No fertilizer advice available."), "No fertilizer advice available.");
  renderList("weatherList", toArray(advisory.weather_precautions, "No weather precautions available."), "No weather precautions available.");
  renderList("spreadRiskList", toArray(advisory.spread_risk, "No spread risk details available."), "No spread risk details available.");
  renderList("predictionList", toArray(advisory.next_7_days_prediction, "No 7-day prediction available."), "No 7-day prediction available.");
  renderList("safetyList", advisory.safety_precautions, "No safety precautions available.");
  renderLinks("productList", advisory.pesticide_products, "No product links found.");
  renderLinks("advisoryLinks", data.advisory_links, "No source links available.");
}

// Render the image and Grad-CAM media panels.
function renderMedia(data) {
  document.getElementById("leaf").src = data.image;
  if (data.healthy) {
    document.getElementById("heatmapBox").style.display = "none";
  } else {
    document.getElementById("heatmapBox").style.display = "block";
    document.getElementById("heatmap").src = data.heatmap;
  }
}

// Render the full prediction result into the dashboard.
function renderPrediction(data) {
  const advisory = data.advisory || {};
  resultsShell.hidden = false;
  advisoryCard.hidden = false;

  renderHero(advisory, data);
  renderAiNote(advisory, data);
  renderAdvisoryCards(advisory, data);
  renderMedia(data);
}

// Handle the Predict button click and connect the UI to the API.
async function handlePredictClick() {
  const file = imageInput.files[0];
  if (!file) {
    alert("Please upload an image");
    return;
  }

  try {
    setLoadingState(true);
    const data = await predictLeaf(file);
    renderPrediction(data);
    statusMessage.innerText = "Prediction completed.";
  } catch (error) {
    statusMessage.innerText = error.message || "Prediction failed.";
    alert(error.message || "Prediction failed");
  } finally {
    setLoadingState(false);
  }
}

predictButton.addEventListener("click", handlePredictClick);

// Read the latest advisory note aloud using the browser speech engine.
function speakAdvisory() {
  if (!lastSpeechText || !("speechSynthesis" in window)) {
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(lastSpeechText);
  utterance.lang = document.documentElement.lang || "en-US";
  utterance.rate = 0.95;
  utterance.pitch = 1;
  window.speechSynthesis.speak(utterance);
}

// Stop the current advisory voice output.
function stopAdvisoryVoice() {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

speakAdvisoryButton.addEventListener("click", speakAdvisory);
stopVoiceButton.addEventListener("click", stopAdvisoryVoice);
