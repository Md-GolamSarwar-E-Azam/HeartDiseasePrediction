from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipelline
from src.report_builder import build_screening_report

application = Flask(__name__)

app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("index.html")

    form_data = {
        "age": request.form.get("age"),
        "sex": request.form.get("sex"),
        "cp": request.form.get("cp"),
        "trestbps": request.form.get("trestbps"),
        "chol": request.form.get("chol"),
        "fbs": request.form.get("fbs"),
        "restecg": request.form.get("restecg"),
        "thalach": request.form.get("thalach"),
        "exang": request.form.get("exang"),
        "oldpeak": request.form.get("oldpeak"),
        "slope": request.form.get("slope"),
        "ca": request.form.get("ca"),
        "thal": request.form.get("thal"),
    }

    data = CustomData(**form_data)
    pred_df = data.get_data_as_frame()
    predict_pipeline = PredictPipelline()
    results = predict_pipeline.predict(pred_df)
    report = build_screening_report(form_data, results[0])

    return render_template("home.html", results=results[0], report=report)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
