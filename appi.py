import sys
import logging  # Correct import statement for logging
from flask import Flask, render_template, redirect, request, session, url_for
import numpy as np
from src.Parkinson.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/', methods=['GET'])
def homepage():
    return render_template("index.html")


@app.route('/results', methods=['GET'])
def results():
    # Fetch prediction result from session
    prediction_result = session.get('prediction_result')
    
    # Render the results.html template with the prediction result
    return render_template("results.html", prediction=prediction_result)


    
@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
            # Reading the inputs given by the user
            MDVP_Fo = float(request.form['MDVP_Fo'])
            MDVP_Fhi = float(request.form['MDVP_Fhi'])
            MDVP_Flo = float(request.form['MDVP_Flo'])
            MDVP_Jitter = float(request.form['MDVP_Jitter'])
            MDVP_Jitter_Abs = float(request.form['MDVP_Jitter_Abs'])
            MDVP_RAP = float(request.form['MDVP_RAP'])
            MDVP_PPQ = float(request.form['MDVP_PPQ'])
            Jitter_DDP = float(request.form['Jitter_DDP'])
            MDVP_Shimmer = float(request.form['MDVP_Shimmer'])
            MDVP_Shimmer_dB = float(request.form['MDVP_Shimmer_dB'])
            Shimmer_APQ3 = float(request.form['Shimmer_APQ3'])
            Shimmer_APQ5 = float(request.form['Shimmer_APQ5'])
            MDVP_APQ = float(request.form['MDVP_APQ'])
            Shimmer_DDA = float(request.form['Shimmer_DDA'])
            NHR = float(request.form['NHR'])
            HNR = float(request.form['HNR'])
            RPDE = float(request.form['RPDE'])
            DFA = float(request.form['DFA'])
            spread1 = float(request.form['spread1'])
            spread2 = float(request.form['spread2'])
            D2 = float(request.form['D2'])
            PPE = float(request.form['PPE'])
            
            data = [MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Jitter_Abs,
                    MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB,
                    Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR,
                    HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            data = np.array(data).reshape(1, 22)  # Reshape to a 2D array
            
            obj = PredictionPipeline()
            predict = obj.predict(data)
            
            print(predict)

            # Log prediction result
            logging.info(f"Prediction successful. Result: {predict}")

            return render_template('results.html', prediction=predict)

    else:
        return render_template('prediction.html')
    
# Importing the PredictionPipeline class from prediction.py
from src.Parkinson.pipeline.prediction import PredictionPipeline
import numpy as np

# Creating an instance of the PredictionPipeline class
pipeline = PredictionPipeline()

# Assuming you have new data stored in a NumPy array called new_data
new_data = np.array([[119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,
                             0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,
                             0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654]])  # Add values for each feature

# M # Replace ... with the rest of your data

# Making predictions using the predict method
predictions = pipeline.predict(new_data)

# Print or use the predictions as needed
print("Predictions:", predictions)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)