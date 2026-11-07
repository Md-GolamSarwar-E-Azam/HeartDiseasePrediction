from datetime import datetime


SEX_LABELS = {"0": "Female", "1": "Male"}
CP_LABELS = {
    "0": "Typical angina",
    "1": "Atypical angina",
    "2": "Non-anginal pain",
    "3": "Asymptomatic",
}
FBS_LABELS = {"0": "No (not above 120 mg/dL)", "1": "Yes (above 120 mg/dL)"}
RESTECG_LABELS = {
    "0": "Normal / no notable finding",
    "1": "ST-T wave abnormality",
    "2": "Left ventricular hypertrophy",
}
EXANG_LABELS = {"0": "No", "1": "Yes"}
SLOPE_LABELS = {
    "0": "Upsloping",
    "1": "Flatsloping",
    "2": "Downsloping",
}
CA_LABELS = {
    "0": "0 vessels",
    "1": "1 vessel",
    "2": "2 vessels",
    "3": "3 vessels",
    "4": "4 vessels",
}
THAL_LABELS = {
    "0": "Unknown / not coded",
    "1": "Normal",
    "2": "Fixed defect",
    "3": "Reversible defect",
}

STATUS_WITHIN = "within"
STATUS_OUTSIDE = "outside"
STATUS_REVIEW = "review"
STATUS_CONTEXT = "context"


def _label(mapping, value, fallback=None):
    key = str(value)
    if key in mapping:
        return mapping[key]
    return fallback if fallback is not None else key


def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_screening_report(form_data: dict, prediction) -> dict:
    """
    Build an educational screening report from submitted form values.
    Range flags are heuristics for UX, not model feature importance.
    """
    age = _to_float(form_data.get("age"))
    trestbps = _to_float(form_data.get("trestbps"))
    chol = _to_float(form_data.get("chol"))
    thalach = _to_float(form_data.get("thalach"))
    oldpeak = _to_float(form_data.get("oldpeak"))
    ca = _to_float(form_data.get("ca"))
    fbs = str(form_data.get("fbs"))
    cp = str(form_data.get("cp"))
    exang = str(form_data.get("exang"))
    slope = str(form_data.get("slope"))
    thal = str(form_data.get("thal"))
    restecg = str(form_data.get("restecg"))
    sex = str(form_data.get("sex"))

    predicted = int(float(prediction))
    higher_concern = predicted == 1

    expected_max_hr = int(220 - age) if age is not None else None

    findings = []

    findings.append(
        {
            "key": "age",
            "name": "Age",
            "value": f"{int(age)} years" if age is not None else "—",
            "reference": "Context only for this screening tool",
            "status": STATUS_CONTEXT,
            "status_label": "Context",
            "explainer": None,
        }
    )
    findings.append(
        {
            "key": "sex",
            "name": "Sex",
            "value": _label(SEX_LABELS, sex),
            "reference": "Recorded demographic category",
            "status": STATUS_CONTEXT,
            "status_label": "Context",
            "explainer": None,
        }
    )

    if trestbps is None:
        bp_status, bp_label, bp_explain = STATUS_REVIEW, "Review", None
    elif trestbps >= 130:
        bp_status, bp_label = STATUS_OUTSIDE, "Outside range"
        bp_explain = (
            "Resting blood pressure is above the common reference band "
            "used in this tool (desirable under 120 mm Hg; review often from 130)."
        )
    elif trestbps >= 120:
        bp_status, bp_label = STATUS_REVIEW, "Review"
        bp_explain = (
            "Resting blood pressure is in an elevated educational band "
            "(120–129 mm Hg) relative to a desirable reading under 120."
        )
    else:
        bp_status, bp_label, bp_explain = STATUS_WITHIN, "Within range", None
    findings.append(
        {
            "key": "trestbps",
            "name": "Resting blood pressure",
            "value": f"{int(trestbps)} mm Hg" if trestbps is not None else "—",
            "reference": "Desirable < 120 mm Hg; review ≥ 130",
            "status": bp_status,
            "status_label": bp_label,
            "explainer": bp_explain,
        }
    )

    if chol is None:
        chol_status, chol_label, chol_explain = STATUS_REVIEW, "Review", None
    elif chol >= 240:
        chol_status, chol_label = STATUS_OUTSIDE, "Outside range"
        chol_explain = (
            "Serum cholesterol is in a high educational band "
            "(≥ 240 mg/dL) compared with a desirable reading under 200."
        )
    elif chol >= 200:
        chol_status, chol_label = STATUS_REVIEW, "Review"
        chol_explain = (
            "Serum cholesterol is above the desirable educational band "
            "(under 200 mg/dL) used in this tool."
        )
    else:
        chol_status, chol_label, chol_explain = STATUS_WITHIN, "Within range", None
    findings.append(
        {
            "key": "chol",
            "name": "Serum cholesterol",
            "value": f"{int(chol)} mg/dL" if chol is not None else "—",
            "reference": "Desirable < 200 mg/dL; high ≥ 240",
            "status": chol_status,
            "status_label": chol_label,
            "explainer": chol_explain,
        }
    )

    if fbs == "1":
        fbs_status, fbs_label = STATUS_OUTSIDE, "Outside range"
        fbs_explain = (
            "Fasting blood sugar was recorded as above 120 mg/dL, "
            "which this tool treats as outside its reference definition."
        )
    else:
        fbs_status, fbs_label, fbs_explain = STATUS_WITHIN, "Within range", None
    findings.append(
        {
            "key": "fbs",
            "name": "Fasting blood sugar > 120 mg/dL",
            "value": _label(FBS_LABELS, fbs),
            "reference": "No (not above 120 mg/dL)",
            "status": fbs_status,
            "status_label": fbs_label,
            "explainer": fbs_explain,
        }
    )

    if cp == "0":
        cp_status, cp_label = STATUS_REVIEW, "Review"
        cp_explain = (
            "Chest pain was recorded as typical angina. "
            "A clinician should interpret what that category means for you."
        )
    else:
        cp_status, cp_label, cp_explain = STATUS_CONTEXT, "Context", None
    findings.append(
        {
            "key": "cp",
            "name": "Chest pain type",
            "value": _label(CP_LABELS, cp),
            "reference": "Use the category from the medical record",
            "status": cp_status,
            "status_label": cp_label,
            "explainer": cp_explain,
        }
    )

    if restecg in {"1", "2"}:
        rest_status, rest_label = STATUS_REVIEW, "Review"
        rest_explain = (
            "The resting ECG category is not the ‘normal / no notable finding’ "
            "band used as a simple reference in this tool."
        )
    else:
        rest_status, rest_label, rest_explain = STATUS_WITHIN, "Within range", None
    findings.append(
        {
            "key": "restecg",
            "name": "Resting ECG result",
            "value": _label(RESTECG_LABELS, restecg),
            "reference": "Normal / no notable finding",
            "status": rest_status,
            "status_label": rest_label,
            "explainer": rest_explain,
        }
    )

    if thalach is None or expected_max_hr is None:
        thr_status, thr_label, thr_explain = STATUS_REVIEW, "Review", None
        thr_ref = "Rough expected max ≈ 220 − age"
    else:
        thr_ref = f"Rough expected max ≈ {expected_max_hr} bpm (220 − age)"
        if thalach < expected_max_hr * 0.7:
            thr_status, thr_label = STATUS_OUTSIDE, "Outside range"
            thr_explain = (
                "Maximum heart rate achieved is much lower than the rough "
                "age-based expectation used in this educational comparison."
            )
        elif thalach < expected_max_hr * 0.85:
            thr_status, thr_label = STATUS_REVIEW, "Review"
            thr_explain = (
                "Maximum heart rate achieved is somewhat lower than the rough "
                "age-based expectation used in this tool."
            )
        else:
            thr_status, thr_label, thr_explain = STATUS_WITHIN, "Within range", None
    findings.append(
        {
            "key": "thalach",
            "name": "Maximum heart rate achieved",
            "value": f"{int(thalach)} bpm" if thalach is not None else "—",
            "reference": thr_ref,
            "status": thr_status,
            "status_label": thr_label,
            "explainer": thr_explain,
        }
    )

    if exang == "1":
        ex_status, ex_label = STATUS_OUTSIDE, "Outside range"
        ex_explain = (
            "Exercise-induced angina was recorded as yes on the test report "
            "used for this screening."
        )
    else:
        ex_status, ex_label, ex_explain = STATUS_WITHIN, "Within range", None
    findings.append(
        {
            "key": "exang",
            "name": "Exercise-induced angina",
            "value": _label(EXANG_LABELS, exang),
            "reference": "No",
            "status": ex_status,
            "status_label": ex_label,
            "explainer": ex_explain,
        }
    )

    if oldpeak is None:
        op_status, op_label, op_explain = STATUS_REVIEW, "Review", None
    elif oldpeak >= 2.0:
        op_status, op_label = STATUS_OUTSIDE, "Outside range"
        op_explain = (
            "ST depression (oldpeak) is at or above 2.0, which this tool "
            "treats as outside its common educational reference band."
        )
    elif oldpeak >= 1.0:
        op_status, op_label = STATUS_REVIEW, "Review"
        op_explain = (
            "ST depression (oldpeak) is at or above 1.0, which may warrant "
            "clinician review alongside the rest of the report."
        )
    else:
        op_status, op_label, op_explain = STATUS_WITHIN, "Within range", None
    findings.append(
        {
            "key": "oldpeak",
            "name": "ST depression (oldpeak)",
            "value": f"{oldpeak}" if oldpeak is not None else "—",
            "reference": "Commonly lower concern under 1.0",
            "status": op_status,
            "status_label": op_label,
            "explainer": op_explain,
        }
    )

    if slope == "2":
        slope_status, slope_label = STATUS_REVIEW, "Review"
        slope_explain = (
            "Peak exercise ST segment slope was recorded as downsloping, "
            "a category often reviewed carefully with a clinician."
        )
    else:
        slope_status, slope_label, slope_explain = STATUS_CONTEXT, "Context", None
    findings.append(
        {
            "key": "slope",
            "name": "Peak exercise ST slope",
            "value": _label(SLOPE_LABELS, slope),
            "reference": "Use the category from the stress-test report",
            "status": slope_status,
            "status_label": slope_label,
            "explainer": slope_explain,
        }
    )

    if ca is None:
        ca_status, ca_label, ca_explain = STATUS_REVIEW, "Review", None
    elif ca >= 1:
        ca_status, ca_label = STATUS_OUTSIDE, "Outside range"
        ca_explain = (
            "More than zero major vessels were noted on fluoroscopy in the "
            "values provided to this tool."
        )
    else:
        ca_status, ca_label, ca_explain = STATUS_WITHIN, "Within range", None
    findings.append(
        {
            "key": "ca",
            "name": "Major vessels (fluoroscopy)",
            "value": _label(CA_LABELS, int(ca) if ca is not None else ca, fallback="—"),
            "reference": "0 vessels",
            "status": ca_status,
            "status_label": ca_label,
            "explainer": ca_explain,
        }
    )

    if thal in {"2", "3"}:
        thal_status, thal_label = STATUS_OUTSIDE, "Outside range"
        thal_explain = (
            "The thallium stress category was recorded as a defect pattern "
            "(fixed or reversible) rather than normal."
        )
    elif thal == "0":
        thal_status, thal_label = STATUS_REVIEW, "Review"
        thal_explain = (
            "The thallium category is uncommon/unclear in this dataset coding; "
            "confirm the report value with a clinician."
        )
    else:
        thal_status, thal_label, thal_explain = STATUS_WITHIN, "Within range", None
    findings.append(
        {
            "key": "thal",
            "name": "Thallium stress result",
            "value": _label(THAL_LABELS, thal),
            "reference": "Normal",
            "status": thal_status,
            "status_label": thal_label,
            "explainer": thal_explain,
        }
    )

    flagged = [
        item
        for item in findings
        if item["status"] in {STATUS_OUTSIDE, STATUS_REVIEW} and item["explainer"]
    ]

    if higher_concern:
        if flagged:
            explainer_intro = (
                "The model returned a higher screening concern. "
                "These educational comparisons were outside or near the "
                "reference bands used in this tool and may be useful to discuss "
                "with a clinician:"
            )
            explainer_bullets = [item["explainer"] for item in flagged]
        else:
            explainer_intro = (
                "The model returned a higher screening concern. "
                "Individual values were mostly within this tool’s simple "
                "reference bands; patterns across features can still matter."
            )
            explainer_bullets = []
    else:
        explainer_intro = (
            "The model returned a lower screening concern. "
            "That does not rule out heart disease. Seek care if you have "
            "symptoms or clinician advice to follow up."
        )
        explainer_bullets = [
            item["explainer"]
            for item in flagged
            if item["status"] == STATUS_OUTSIDE
        ]

    return {
        "generated_at": datetime.now().strftime("%d %b %Y · %H:%M"),
        "prediction": predicted,
        "higher_concern": higher_concern,
        "signal_title": "Higher screening concern" if higher_concern else "Lower screening concern",
        "signal_summary": (
            "Based on the submitted measurements, the model identified a higher "
            "heart-disease screening signal."
            if higher_concern
            else "Based on the submitted measurements, the model did not identify "
            "a higher heart-disease screening signal."
        ),
        "findings": findings,
        "explainer_intro": explainer_intro,
        "explainer_bullets": explainer_bullets,
        "honesty_note": (
            "Range flags are educational heuristics for readability. "
            "They are not the model’s internal feature weights and are not a diagnosis."
        ),
    }
