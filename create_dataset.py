import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

event_types = [
    "Conference",
    "Workshop",
    "Concert",
    "Seminar",
    "Exhibition",
    "Sports",
    "Wedding",
    "Corporate"
]

event_type = np.random.choice(event_types, n)

registrations = np.random.randint(50, 1501, n)

ticket_price = np.random.randint(100, 2001, n)

marketing_budget = np.random.randint(2000, 100001, n)

venue_capacity = registrations + np.random.randint(50, 501, n)

duration_hours = np.random.randint(2, 13, n)

previous_attendance = np.maximum(
    registrations + np.random.randint(-100, 101, n),
    20
)

social_media_reach = np.random.randint(500, 50001, n)

event_rating = np.round(
    np.random.uniform(3.0, 5.0, n),
    1
)

# Generate realistic attendance
attendance_rate = (
    0.55
    + 0.000003 * social_media_reach
    + 0.000001 * marketing_budget
    + 0.02 * (event_rating - 3)
    + np.random.normal(0, 0.05, n)
)

attendance_rate = np.clip(attendance_rate, 0.45, 0.98)

actual_attendance = (
    registrations * attendance_rate
    + 0.15 * previous_attendance
    + np.random.normal(0, 20, n)
)

actual_attendance = np.clip(
    actual_attendance,
    10,
    venue_capacity
)

actual_attendance = actual_attendance.astype(int)

df = pd.DataFrame({
    "Event_Type": event_type,
    "Registrations": registrations,
    "Ticket_Price": ticket_price,
    "Marketing_Budget": marketing_budget,
    "Venue_Capacity": venue_capacity,
    "Duration_Hours": duration_hours,
    "Previous_Attendance": previous_attendance,
    "Social_Media_Reach": social_media_reach,
    "Event_Rating": event_rating,
    "Actual_Attendance": actual_attendance
})

df.to_csv("events.csv", index=False)

print("Dataset created successfully!")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())