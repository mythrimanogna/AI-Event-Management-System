import os
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load preprocessed data
X_train, X_test, y_train, y_test = joblib.load(
    "models/train_test_data.pkl"
)

# Define ML models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
}

results = {}
trained_models = {}

print("\nMODEL TRAINING AND COMPARISON")
print("=" * 50)

for name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results[name] = mae
    trained_models[name] = model

    print(f"\n{name}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

# Select model with lowest MAE
best_model_name = min(results, key=results.get)
best_model = trained_models[best_model_name]

# Save all models
os.makedirs("models", exist_ok=True)

joblib.dump(
    trained_models,
    "models/all_models.pkl"
)

# Save best model
joblib.dump(
    best_model,
    "models/event_attendance_model.pkl"
)

print("\n" + "=" * 50)
print("Best Model:", best_model_name)
print("Best MAE:", round(results[best_model_name], 2))
print("Best model saved successfully!")