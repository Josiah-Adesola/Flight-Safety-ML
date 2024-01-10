import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle


app = Flask(__name__)
model = pickle.load(open('logreg_model.pkl','rb'))


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    map_val = {
            "0": "minor fault",
            "1": "healthy",
            "2": "major fault"
        }
    pred_val = map_val.get(str(prediction[0]), "Unknown")
    print(pred_val)

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="The flight is {}".format(pred_val))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)
