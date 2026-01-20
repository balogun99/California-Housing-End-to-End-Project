import pickle
from flask import Flask,requests,jsonify,app,url_for,render_template
import numpy as np
import pandas as pd

# load the pickle file
app=Flask(__name__)
# load the pickle model file
regmodel = pickle.load(open('california_model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# create the flask endpoint
@app.route('/')
def home():
    return render_template('home.htm')

# create and predict the standard template
@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = requests.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in requests.form_values()]
    final_input = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.htm", prediction_text="The House Prediction Price is: {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)