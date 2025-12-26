from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# load the tuned SVM model
model = joblib.load("best_fake_review_svm.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("review_text", "")
        if input_text.strip():
            prediction = model.predict([input_text])[0]

    return render_template("index.html",
                           prediction=prediction,
                           input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
