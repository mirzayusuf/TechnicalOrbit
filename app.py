from re import X
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def iris_prediction():
    if request.method == 'GET':
        return render_template("Prediksi-Berat-Badan.html")
    elif request.method == 'POST':
        print(dict(request.form))
        weight_height_features = dict(request.form).values()
        weight_height_features = np.array([float(x) for x in weight_height_features])
        model, x = joblib.load("model-development/weight-height-linier-regression.pkl")
        weight_height_features = x.transform([weight_height_features])
        print(weight_height_features)
        result = model.predict(weight_height_features)
        
        return render_template('Prediksi-Berat-Badan.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)