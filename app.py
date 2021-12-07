from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import pickle
import pandas as pd


app = Flask(__name__)
boston = datasets.load_boston()
features = pd.DataFrame(boston.data,columns=boston.feature_names)
sd_sc = StandardScaler()
sd_sc.fit_transform(features)
@app.route('/' , methods =['GET'])
@cross_origin()
def homepage():
    return render_template('index.html')

@app.route('/predict' , methods =['GET','POST'])
@cross_origin()
def Prediction():
    if request.method == 'POST':
        try:
            CRIM = float(request.form['CRIM'])
            ZN = float(request.form['ZN'])
            INDUS = float(request.form['INDUS'])
            CHAS = float(request.form['CHAS'])
            NOX = float(request.form['NOX'])
            RM = float(request.form['RM'])
            AGE = float(request.form['AGE'])
            DIS = float(request.form['DIS'])
            RAD = float(request.form['RAD'])
            TAX = float(request.form['TAX'])
            PTRATIO = float(request.form['PTRATIO'])
            B = float(request.form['B'])
            LSTAT = float(request.form['LSTAT'])

            filename = 'Rd-assign.pickle'
            load_model = pickle.load(open(filename, 'rb'))
            prediction = load_model.predict(sd_sc.transform([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]]))
            return render_template('results.html', prediction=round(prediction[0]))
        except Exception as e:
            print('the exception msg is : ' , e)
            return 'Something went wrong'
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug =True)