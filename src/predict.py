import joblib
import pandas as pd

CLASS_ORDER = ["Low Risk", "Prediabetes", "High Risk"]

def predict_risk(model_path: str, patient_data: dict) -> dict:
    model = joblib.load(model_path)
    df = pd.DataFrame([patient_data])

    # Add engineered features
    df["bmi_age_ratio"]        = df["bmi"] / df["age"]
    df["glucose_insulin_ratio"] = df["fasting_glucose_level"] / (df["insulin_level"] + 1e-5)
    df["waist_bmi_ratio"]      = df["waist_circumference_cm"] / df["bmi"]
    activity_map = {"Low": 1, "Moderate": 2, "High": 3}
    df["calorie_activity"] = df["daily_calorie_intake"] / df["physical_activity_level"].map(activity_map)

    pred  = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    return {
        "prediction":   CLASS_ORDER[pred],
        "probabilities": dict(zip(CLASS_ORDER, proba.round(3)))
    }

if __name__ == "__main__":
    result = predict_risk("outputs/diabetes_best_model.pkl", {
        "age": 55, "gender": "Female", "bmi": 32.0,
        "blood_pressure": 145, "fasting_glucose_level": 115,
        "insulin_level": 12.0, "HbA1c_level": 6.1,
        "cholesterol_level": 220, "triglycerides_level": 180,
        "physical_activity_level": "Low", "daily_calorie_intake": 2400,
        "sugar_intake_grams_per_day": 85.0, "sleep_hours": 6.5,
        "stress_level": 7, "family_history_diabetes": "Yes",
        "waist_circumference_cm": 98.0
    })
    print(result)