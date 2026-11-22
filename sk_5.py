import joblib
model = joblib.load("student_model.pkl")
joblib.dump(model, "student_model.pkl")
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("student_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    hours = float(data["study_hours"])

    result = model.predict([[hours]])

    return jsonify({
        "predicted_score": float(result[0])
    })

if __name__ == "__main__":
    app.run()
