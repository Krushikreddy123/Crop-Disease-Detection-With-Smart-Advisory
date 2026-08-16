// Shared frontend helpers for advisory rendering.

const severityClassMap = {
  low: "severity-low",
  medium: "severity-medium",
  high: "severity-high"
};

// Convert a value into a display-safe list.
function toArray(value, fallback) {
  if (Array.isArray(value) && value.length) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parts = value
      .split(/(?<=[.!?])\s+/)
      .map(item => item.trim())
      .filter(Boolean);
    return parts.length ? parts : [value.trim()];
  }
  return fallback ? [fallback] : [];
}

// Split a long advisory note into shorter bullet points.
function splitNoteIntoBullets(note, fallback) {
  if (!note) {
    return [fallback];
  }

  const parts = note
    .split(/(?<=[.!?])\s+/)
    .map(item => item.trim())
    .filter(Boolean);

  return parts.length ? parts : [note];
}

// Add emphasis to important AI note actions.
function emphasizeActions(text) {
  return text
    .replace(/Immediate field action should focus on/gi, "<strong>Immediate field action should focus on</strong>")
    .replace(/Priority action:/gi, "<strong>Priority action:</strong>")
    .replace(/Recommended spray window:/gi, "<strong>Recommended spray window:</strong>")
    .replace(/The next best preventive step is:/gi, "<strong>The next best preventive step is:</strong>");
}

// Render a plain list into a target element.
function renderList(targetId, items, emptyText, allowMarkup = false) {
  const list = document.getElementById(targetId);
  list.innerHTML = "";

  const finalItems = items && items.length ? items : [emptyText];
  finalItems.forEach(item => {
    const li = document.createElement("li");
    if (allowMarkup) {
      li.innerHTML = item;
    } else {
      li.textContent = item;
    }
    list.appendChild(li);
  });
}

// Render a list of links into a target element.
function renderLinks(targetId, items, emptyText) {
  const list = document.getElementById(targetId);
  list.innerHTML = "";

  if (!items || !items.length) {
    const li = document.createElement("li");
    li.textContent = emptyText;
    list.appendChild(li);
    return;
  }

  items.forEach(item => {
    const li = document.createElement("li");
    const anchor = document.createElement("a");
    anchor.href = item.url;
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    anchor.textContent = item.title || item.domain;
    li.appendChild(anchor);
    list.appendChild(li);
  });
}
