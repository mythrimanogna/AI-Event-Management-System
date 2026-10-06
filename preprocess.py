import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# Load dataset
df = pd.read_csv("events.csv")

# Separate input and target
X = df.drop("Actual_Attendance", axis=1)
y = df["Actual_Attendance"]

# Categorical column
categorical_columns = ["Event_Type"]

# Encode Event_Type
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded_data = encoder.fit_transform(
    X[categorical_columns]
)

encoded_columns = encoder.get_feature_names_out(
    categorical_columns
)

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoded_columns
)

# Remove original text column
X_numeric = X.drop(
    categorical_columns,
    axis=1
).reset_index(drop=True)

# Combine numerical + encoded columns
X_final = pd.concat(
    [X_numeric, encoded_df],
    axis=1
)

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.2,
    random_state=42
)

# Save encoder
joblib.dump(
    encoder,
    "models/event_encoder.pkl"
)

# Save feature names
joblib.dump(
    list(X_final.columns),
    "models/feature_columns.pkl"
)

# Save train/test data
joblib.dump(
    (X_train, X_test, y_train, y_test),
    "models/train_test_data.pkl"
)

print("Preprocessing completed successfully!")

print("\nOriginal dataset shape:")
print(df.shape)

print("\nProcessed feature shape:")
print(X_final.shape)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)

print("\nFeature columns:")
print(X_final.columns.tolist())