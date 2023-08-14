import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
forest_model = None


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    json_data = request.json
    pixels = np.array(json_data['pixels']).T.reshape(28 * 28)
    pixels *= 255
    prediction = forest_model.predict([pixels, ])[0]
    probs = forest_model.predict_proba([pixels])[0]
    probs = list(zip(range(10), probs))
    probs = sorted(probs, key=lambda x: x[1], reverse=True)
    return jsonify({'predicted': str(prediction), 'probs': probs[:5]})



if __name__ == '__main__':
    with open('rf_model.pickle', 'rb') as f:
        forest_model = pickle.load(f)
    app.run()
