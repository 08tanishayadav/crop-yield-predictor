from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and column names
with open("crop_yield_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        crop = request.form['Crop']
        season = request.form['Season']
        state = request.form['State']
        area = float(request.form['Area'])
        rainfall = float(request.form['Annual_Rainfall'])
        fertilizer = float(request.form['Fertilizer'])
        pesticide = float(request.form['Pesticide'])

        # Create input DataFrame
        input_dict = {
            'Area': area,
            'Annual_Rainfall': rainfall,
            'Fertilizer': fertilizer,
            'Pesticide': pesticide,
            f'Crop_{crop}': 1,
            f'Season_{season}': 1,
            f'State_{state}': 1
        }

        # Set all columns to 0 initially
        input_df = pd.DataFrame([np.zeros(len(model_columns))], columns=model_columns)
        for key in input_dict:
            if key in input_df.columns:
                input_df.at[0, key] = input_dict[key]

        prediction = model.predict(input_df)[0]
        return render_template('index.html', prediction_text=f"Predicted Yield: {prediction:.2f} tons/hectare")

    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)
