import joblib
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load test data
X_train, X_test, y_train, y_test = joblib.load(
    "models/train_test_data.pkl"
)

# Load all trained models
models = joblib.load(
    "models/all_models.pkl"
)

print("\nDETAILED MODEL EVALUATION")
print("=" * 60)

results = []

for name, model in models.items():

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

# Display comparison
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for result in results:
    print(
        f"{result['Model']:<20} "
        f"MAE: {result['MAE']:.2f} | "
        f"RMSE: {result['RMSE']:.2f} | "
        f"R2: {result['R2']:.4f}"
    )

print("\nFinal Selected Model: Random Forest")