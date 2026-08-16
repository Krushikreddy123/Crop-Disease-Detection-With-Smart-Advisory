// API helpers for talking to Flask routes.

// Submit the selected leaf image to the Flask prediction route.
async function predictLeaf(file, location) {
  const formData = new FormData();
  formData.append("image", file);
  if (location?.latitude != null && location?.longitude != null) {
    formData.append("latitude", String(location.latitude));
    formData.append("longitude", String(location.longitude));
  }

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

// Request the user's coordinates for live weather-aware advisory.
function getCurrentCoordinates() {
  if (!("geolocation" in navigator)) {
    return Promise.resolve(null);
  }

  return new Promise(resolve => {
    navigator.geolocation.getCurrentPosition(
      position => {
        resolve({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude
        });
      },
      () => resolve(null),
      {
        enableHighAccuracy: false,
        timeout: 6000,
        maximumAge: 300000
      }
    );
  });
}
