// API helpers for talking to Flask routes.

// Submit the selected leaf image to the Flask prediction route.
async function predictLeaf(file) {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch("/predict", {
    method: "POST",
    body: formData
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Prediction failed");
  }

  return data;
}
