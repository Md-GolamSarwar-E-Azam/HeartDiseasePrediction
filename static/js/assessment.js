(function () {
  const form = document.getElementById("assessment-form");
  if (!form) return;

  const steps = Array.from(form.querySelectorAll(".form-step"));
  const progressItems = Array.from(document.querySelectorAll(".progress-step"));
  const reviewList = document.getElementById("review-list");
  const demoButton = document.getElementById("demo-fill");
  let currentStep = 0;

  function showStep(index) {
    currentStep = index;
    steps.forEach((step, i) => {
      step.hidden = i !== index;
      step.classList.toggle("is-active", i === index);
    });
    progressItems.forEach((item, i) => {
      item.classList.toggle("is-active", i === index);
      item.classList.toggle("is-complete", i < index);
    });
    if (index === steps.length - 1) {
      updateReview();
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function stepFields(stepEl) {
    return Array.from(stepEl.querySelectorAll("input, select")).filter(
      (el) => el.name && !el.disabled
    );
  }

  function validateStep(index) {
    const fields = stepFields(steps[index]);
    for (const field of fields) {
      if (!field.checkValidity()) {
        field.reportValidity();
        return false;
      }
    }
    return true;
  }

  function displayValue(field) {
    if (!field) return "—";
    if (field.tagName === "SELECT") {
      const option = field.options[field.selectedIndex];
      return option && option.value !== "" ? option.textContent.trim() : "—";
    }
    return field.value ? field.value : "—";
  }

  function updateReview() {
    if (!reviewList) return;
    const rows = [
      ["Age", "age"],
      ["Sex", "sex"],
      ["Resting blood pressure", "trestbps"],
      ["Cholesterol", "chol"],
      ["Fasting blood sugar", "fbs"],
      ["Chest pain type", "cp"],
      ["Resting ECG", "restecg"],
      ["Max heart rate", "thalach"],
      ["Exercise angina", "exang"],
      ["ST depression", "oldpeak"],
      ["ST slope", "slope"],
      ["Major vessels", "ca"],
      ["Thallium result", "thal"],
    ];
    reviewList.innerHTML = rows
      .map(([label, name]) => {
        const field = form.elements.namedItem(name);
        return `<li><span>${label}</span><strong>${displayValue(field)}</strong></li>`;
      })
      .join("");
  }

  form.addEventListener("click", (event) => {
    const nextBtn = event.target.closest("[data-next]");
    const prevBtn = event.target.closest("[data-prev]");
    if (nextBtn) {
      event.preventDefault();
      if (!validateStep(currentStep)) return;
      if (currentStep < steps.length - 1) showStep(currentStep + 1);
    }
    if (prevBtn) {
      event.preventDefault();
      if (currentStep > 0) showStep(currentStep - 1);
    }
  });

  if (demoButton) {
    demoButton.addEventListener("click", () => {
      const demo = {
        age: "58",
        sex: "1",
        trestbps: "140",
        chol: "250",
        fbs: "1",
        cp: "0",
        restecg: "1",
        thalach: "130",
        exang: "1",
        oldpeak: "2.0",
        slope: "1",
        ca: "2",
        thal: "3",
      };
      Object.entries(demo).forEach(([name, value]) => {
        const field = form.elements.namedItem(name);
        if (field) field.value = value;
      });
      showStep(0);
    });
  }

  showStep(0);
})();
