document.getElementById("predictBtn").onclick = () => {
  const file = document.getElementById("image").files[0];
  if (!file) {
    alert("Please upload an image");
    return;
  }

  const lang = document.getElementById("langSelect").value;
  const formData = new FormData();
  formData.append("image", file);
  formData.append("lang", lang);

  fetch("/predict", {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        alert(data.error);
        return;
      }

      document.getElementById("leaf").src = data.image;
      document.getElementById("result").innerText =
        `${data.class} (${data.confidence}%)`;

      const advice = document.getElementById("advice");
      advice.innerHTML = "";

      data.advisory.forEach(point => {
        advice.innerHTML += `<li>${point}</li>`;
      });

      if (data.healthy) {
        document.getElementById("heatmapBox").style.display = "none";
      } else {
        document.getElementById("heatmapBox").style.display = "block";
        document.getElementById("heatmap").src = data.heatmap;
      }
    })
    .catch(() => alert("Prediction failed"));
};
