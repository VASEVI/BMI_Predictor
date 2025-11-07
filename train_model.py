import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# 1️⃣ Load and prepare dataset
def load_and_prepare():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'weight_dataset.csv')
    data = pd.read_csv(data_path)
    print("✅ Dataset loaded successfully!")
    print("Columns:", list(data.columns))

    # Ensure necessary columns exist
    if {'height_cm', 'weight_kg'}.issubset(data.columns):
        # Calculate BMI
        data['BMI'] = data['weight_kg'] / ((data['height_cm'] / 100) ** 2)

        # Create BMI categories
        def categorize_bmi(bmi):
            if bmi < 18.5:
                return 'Underweight'
            elif 18.5 <= bmi < 24.9:
                return 'Normal'
            elif 25 <= bmi < 29.9:
                return 'Overweight'
            else:
                return 'Obese'

        data['bmi_category'] = data['BMI'].apply(categorize_bmi)
        print("🧮 BMI and bmi_category columns created successfully!")
    else:
        raise KeyError("❌ Missing 'height_cm' or 'weight_kg' column in dataset!")

    # Separate features (X) and target (y)
    X = data.drop('bmi_category', axis=1)
    y = data['bmi_category']

    # Label encode categorical columns
    label_encoder = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = label_encoder.fit_transform(X[col])

    # Encode labels
    y = label_encoder.fit_transform(y)

    return X, y

# 2️⃣ Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model trained successfully with accuracy: {acc * 100:.2f}%")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model

# 3️⃣ Save model
def save_model(model):
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'bmi_model.pkl')
    joblib.dump(model, model_path)
    print(f"💾 Model saved at: {model_path}")

# 4️⃣ Main Execution
if __name__ == "__main__":
    try:
        X, y = load_and_prepare()
        model = train_model(X, y)
        save_model(model)
        print("\n🎉 Training complete and model saved successfully!")
    except Exception as e:
        print("\n❌ Error during training:", e)
