import numpy as np
from flask import Flask,render_template,request,jsonify
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index80.html')

@app.route("/predict",methods=['POST','GET'])
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            Phase_current_A=float(request.form['Phase_current_A'])
            Phase_voltage_A=float(request.form['Phase_voltage_A'])
            Phase_voltage_B=float(request.form['Phase_voltage_B'])
            Radiation=float(request.form['Radiation'])
            Temperature=float(request.form['Temperature'])
            model = pickle.load(open('model.pkl', 'rb'))
            prediction=model.predict([[Phase_current_A,Phase_voltage_A,Phase_voltage_B,Radiation,Temperature]])
            return render_template('index80.html',prediction_text='Predicted Active Power is {}'.format(prediction))
        except Exception as e:
            print('The Exception message is:',e)
            return 'something went wrong'
        else:
            return render_template('index80.html')


if __name__=="__main__":
    app.run(debug=True)