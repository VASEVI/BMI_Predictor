from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'bmi_model.pkl')
model = joblib.load(model_path)

# Health recommendations based on BMI category
RECOMMENDATIONS = {
    'Underweight': {
        'advice': 'Increase calorie intake with nutrient-dense foods and include resistance training to gain healthy weight.',
        'diet': 'Eat more whole-milk dairy, nuts, healthy oils, and frequent snacks. Include protein with each meal.',
        'exercise': 'Focus on resistance training and progressive overload to build muscle mass.'
    },
    'Normal': {
        'advice': 'Maintain your healthy weight through balanced diet and regular exercise. Keep monitoring your lifestyle habits.',
        'diet': 'Balanced macronutrients and plenty of vegetables and lean protein.',
        'exercise': 'Mix of cardio and strength training; aim for 150 min moderate activity weekly.'
    },
    'Overweight': {
        'advice': 'Reduce caloric intake slightly, increase physical activity, and focus on sustainable habit changes.',
        'diet': 'Reduce processed foods and sugary drinks, choose whole grains and lean protein.',
        'exercise': 'Combine cardio and resistance training; aim for 150–300 min active exercise weekly.'
    },
    'Obese': {
        'advice': 'Seek medical/dietary supervision. Begin with a gradual calorie deficit and steady increase in activity.',
        'diet': 'Structured calorie-controlled plan, focus on high-fiber vegetables and lean proteins.',
        'exercise': 'Start slow (low-impact cardio) and progress to combined training under guidance.'
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract and cast inputs
        age = float(data.get('age', 0))
        gender = data.get('gender', 'Male')
        height_cm = float(data.get('height_cm', 0))
        weight_kg = float(data.get('weight_kg', 0))
        diet = data.get('diet', 'Non-Veg')
        exercise = float(data.get('exercise_min_per_week', 0))
        smoking = data.get('smoking', 'No')
        drinking = data.get('drinking', 'No')

        # Compute BMI
        height_m = height_cm / 100.0 if height_cm > 0 else 1.0
        bmi = weight_kg / (height_m ** 2)

        # Encode categorical variables
        gender_map = {'Male': 1, 'Female': 0}
        diet_map = {'Non-Veg': 2, 'Veg': 1, 'Vegan': 0}
        yes_no_map = {'Yes': 1, 'No': 0}

        # Prepare DataFrame for model prediction
        model_input = pd.DataFrame([{
            'age': age,
            'gender': gender_map.get(gender, 0),
            'height_cm': height_cm,
            'weight_kg': weight_kg,
            'diet': diet_map.get(diet, 0),
            'exercise_min_per_week': exercise,  # <- NO leading space
            'smoking': yes_no_map.get(smoking, 0),
            'drinking': yes_no_map.get(drinking, 0),
            'BMI': bmi
        }])

        # Make prediction
        pred_encoded = model.predict(model_input)[0]

        # Decode numeric label to text
        categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
        pred = categories[int(pred_encoded)] if isinstance(pred_encoded, (np.integer, int, float)) else str(pred_encoded)

        # Get recommendations
        rec = RECOMMENDATIONS.get(pred, {})

        # Convert any numpy types to Python native types for JSON
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return obj

        response = {
            'prediction': str(pred),
            'bmi': float(round(bmi, 2)),
            'recommendations': {k: convert_types(v) for k, v in rec.items()}
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
