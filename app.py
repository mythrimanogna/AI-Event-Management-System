import streamlit as st
import pandas as pd
import joblib
import os

# Import chatbot
from chatbot import event_chatbot


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Event Management System",
    page_icon="🎪",
    layout="wide"
)


# =========================================================
# LOAD MACHINE LEARNING FILES
# =========================================================

MODEL_PATH = "models/event_attendance_model.pkl"
ENCODER_PATH = "models/event_encoder.pkl"
FEATURE_PATH = "models/feature_columns.pkl"

model = joblib.load(MODEL_PATH)

encoder = joblib.load(
    ENCODER_PATH
)

feature_columns = joblib.load(
    FEATURE_PATH
)


# =========================================================
# EVENT RECORD FILE
# =========================================================

EVENT_FILE = "event_records.csv"


if os.path.exists(EVENT_FILE):

    events_df = pd.read_csv(
        EVENT_FILE
    )

else:

    events_df = pd.DataFrame(
        columns=[
            "Event_Name",
            "Event_Type",
            "Date",
            "Venue",
            "Venue_Capacity",
            "Registrations",
            "Ticket_Price",
            "Marketing_Budget",
            "Duration_Hours",
            "Previous_Attendance",
            "Social_Media_Reach",
            "Event_Rating",
            "Predicted_Attendance"
        ]
    )


# =========================================================
# AI ATTENDANCE PREDICTION FUNCTION
# =========================================================

def predict_attendance(
    event_type,
    registrations,
    ticket_price,
    marketing_budget,
    venue_capacity,
    duration_hours,
    previous_attendance,
    social_media_reach,
    event_rating
):

    input_df = pd.DataFrame({
        "Event_Type": [event_type],
        "Registrations": [registrations],
        "Ticket_Price": [ticket_price],
        "Marketing_Budget": [marketing_budget],
        "Venue_Capacity": [venue_capacity],
        "Duration_Hours": [duration_hours],
        "Previous_Attendance": [previous_attendance],
        "Social_Media_Reach": [social_media_reach],
        "Event_Rating": [event_rating]
    })

    # Encode Event Type
    encoded = encoder.transform(
        input_df[["Event_Type"]]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            ["Event_Type"]
        )
    )

    # Remove text column
    numeric_df = input_df.drop(
        ["Event_Type"],
        axis=1
    )

    # Combine numerical and encoded data
    final_input = pd.concat(
        [
            numeric_df.reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

    # Match training feature order
    final_input = final_input.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict attendance
    prediction = model.predict(
        final_input
    )[0]

    # Attendance cannot exceed venue capacity
    prediction = max(
        0,
        min(
            prediction,
            venue_capacity
        )
    )

    prediction = int(
        round(prediction)
    )

    return prediction


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🎪 Event Management"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "📊 Dashboard",
        "📝 Create Event",
        "🤖 AI Attendance Prediction",
        "📈 Analytics",
        "💬 Event AI Chatbot"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "AI-Based Event Management System\n\n"
    "Machine Learning + Streamlit"
)


# =========================================================
# MAIN HEADER
# =========================================================

st.title(
    "🎪 AI-Based Event Management System"
)

st.caption(
    "Event Management and Machine Learning Based "
    "Attendance Prediction"
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.header(
        "📊 Event Dashboard"
    )

    total_events = len(
        events_df
    )

    if total_events > 0:

        total_registrations = int(
            events_df[
                "Registrations"
            ].sum()
        )

        average_attendance = int(
            events_df[
                "Predicted_Attendance"
            ].mean()
        )

        total_capacity = int(
            events_df[
                "Venue_Capacity"
            ].sum()
        )

    else:

        total_registrations = 0
        average_attendance = 0
        total_capacity = 0


    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Events",
        total_events
    )

    col2.metric(
        "Total Registrations",
        total_registrations
    )

    col3.metric(
        "Average Expected Attendance",
        average_attendance
    )

    col4.metric(
        "Total Venue Capacity",
        total_capacity
    )


    st.divider()


    # Event records
    st.subheader(
        "📋 Event Records"
    )

    if total_events > 0:

        st.dataframe(
            events_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No events available yet. "
            "Create your first event."
        )


# =========================================================
# CREATE EVENT
# =========================================================

elif page == "📝 Create Event":

    st.header(
        "📝 Create New Event"
    )

    st.write(
        "Enter event details. The AI model will "
        "predict expected attendance."
    )


    with st.form(
        "create_event_form"
    ):

        col1, col2 = st.columns(2)


        # -------------------------------------------------
        # LEFT COLUMN
        # -------------------------------------------------

        with col1:

            event_name = st.text_input(
                "Event Name"
            )

            event_type = st.selectbox(
                "Event Type",
                [
                    "Conference",
                    "Workshop",
                    "Concert",
                    "Seminar",
                    "Exhibition",
                    "Sports",
                    "Wedding",
                    "Corporate"
                ]
            )

            event_date = st.date_input(
                "Event Date"
            )

            venue = st.text_input(
                "Venue"
            )

            venue_capacity = st.number_input(
                "Venue Capacity",
                min_value=20,
                max_value=100000,
                value=500
            )

            registrations = st.number_input(
                "Expected Registrations",
                min_value=10,
                max_value=100000,
                value=300
            )


        # -------------------------------------------------
        # RIGHT COLUMN
        # -------------------------------------------------

        with col2:

            ticket_price = st.number_input(
                "Ticket Price (₹)",
                min_value=0,
                max_value=100000,
                value=500
            )

            marketing_budget = st.number_input(
                "Marketing Budget (₹)",
                min_value=0,
                max_value=10000000,
                value=20000
            )

            duration_hours = st.number_input(
                "Event Duration (Hours)",
                min_value=1,
                max_value=24,
                value=5
            )

            previous_attendance = st.number_input(
                "Previous Similar Event Attendance",
                min_value=0,
                max_value=100000,
                value=250
            )

            social_media_reach = st.number_input(
                "Social Media Reach",
                min_value=0,
                max_value=10000000,
                value=10000
            )

            event_rating = st.slider(
                "Expected Event Rating",
                min_value=3.0,
                max_value=5.0,
                value=4.0,
                step=0.1
            )


        st.divider()


        submitted = st.form_submit_button(
            "🚀 Create Event"
        )


    # -----------------------------------------------------
    # CREATE EVENT RESULT
    # -----------------------------------------------------

    if submitted:

        if event_name.strip() == "":

            st.error(
                "❌ Please enter Event Name."
            )

        elif venue.strip() == "":

            st.error(
                "❌ Please enter Venue."
            )

        elif registrations > venue_capacity:

            st.warning(
                "⚠️ Registrations are greater than "
                "venue capacity. Please check the values."
            )

        else:

            # AI prediction
            prediction = predict_attendance(
                event_type,
                registrations,
                ticket_price,
                marketing_budget,
                venue_capacity,
                duration_hours,
                previous_attendance,
                social_media_reach,
                event_rating
            )


            # Venue utilization
            utilization = (
                prediction /
                venue_capacity
            ) * 100


            # Create event record
            new_event = pd.DataFrame([
                {
                    "Event_Name": event_name,
                    "Event_Type": event_type,
                    "Date": str(event_date),
                    "Venue": venue,
                    "Venue_Capacity": venue_capacity,
                    "Registrations": registrations,
                    "Ticket_Price": ticket_price,
                    "Marketing_Budget": marketing_budget,
                    "Duration_Hours": duration_hours,
                    "Previous_Attendance": previous_attendance,
                    "Social_Media_Reach": social_media_reach,
                    "Event_Rating": event_rating,
                    "Predicted_Attendance": prediction
                }
            ])


            # Add event
            events_df = pd.concat(
                [
                    events_df,
                    new_event
                ],
                ignore_index=True
            )


            # Save event
            events_df.to_csv(
                EVENT_FILE,
                index=False
            )


            st.success(
                "✅ Event created successfully!"
            )


            # AI Prediction Result
            st.subheader(
                "🤖 AI Prediction Result"
            )


            col1, col2, col3 = st.columns(3)


            col1.metric(
                "Predicted Attendance",
                f"{prediction} people"
            )

            col2.metric(
                "Venue Capacity",
                f"{venue_capacity} people"
            )

            col3.metric(
                "Venue Utilization",
                f"{utilization:.1f}%"
            )


            # Recommendation
            if utilization > 90:

                st.warning(
                    "⚠️ Expected attendance is very high. "
                    "Consider selecting a larger venue."
                )

            elif utilization < 40:

                st.info(
                    "ℹ️ Expected attendance is relatively "
                    "low compared with venue capacity."
                )

            else:

                st.success(
                    "✅ Venue capacity looks suitable "
                    "for the predicted attendance."
                )


# =========================================================
# AI ATTENDANCE PREDICTION
# =========================================================

elif page == "🤖 AI Attendance Prediction":

    st.header(
        "🤖 AI Attendance Prediction"
    )

    st.write(
        "Use the trained Random Forest Machine Learning "
        "model to predict expected event attendance."
    )


    col1, col2 = st.columns(2)


    with col1:

        event_type = st.selectbox(
            "Event Type",
            [
                "Conference",
                "Workshop",
                "Concert",
                "Seminar",
                "Exhibition",
                "Sports",
                "Wedding",
                "Corporate"
            ]
        )

        registrations = st.number_input(
            "Registrations",
            min_value=10,
            value=500
        )

        ticket_price = st.number_input(
            "Ticket Price (₹)",
            min_value=0,
            value=500
        )

        marketing_budget = st.number_input(
            "Marketing Budget (₹)",
            min_value=0,
            value=20000
        )

        venue_capacity = st.number_input(
            "Venue Capacity",
            min_value=20,
            value=600
        )


    with col2:

        duration_hours = st.number_input(
            "Duration (Hours)",
            min_value=1,
            max_value=24,
            value=5
        )

        previous_attendance = st.number_input(
            "Previous Attendance",
            min_value=0,
            value=400
        )

        social_media_reach = st.number_input(
            "Social Media Reach",
            min_value=0,
            value=10000
        )

        event_rating = st.slider(
            "Expected Rating",
            min_value=3.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )


    st.divider()


    if st.button(
        "🔮 Predict Attendance",
        use_container_width=True
    ):

        prediction = predict_attendance(
            event_type,
            registrations,
            ticket_price,
            marketing_budget,
            venue_capacity,
            duration_hours,
            previous_attendance,
            social_media_reach,
            event_rating
        )


        utilization = (
            prediction /
            venue_capacity
        ) * 100


        st.subheader(
            "🎯 Prediction Result"
        )


        col1, col2 = st.columns(2)


        col1.metric(
            "Expected Attendance",
            f"{prediction} people"
        )

        col2.metric(
            "Venue Utilization",
            f"{utilization:.1f}%"
        )


        st.progress(
            min(
                utilization / 100,
                1.0
            )
        )


        if utilization > 90:

            st.warning(
                "⚠️ High venue utilization. "
                "Consider a larger venue."
            )

        elif utilization < 40:

            st.info(
                "ℹ️ Low expected venue utilization."
            )

        else:

            st.success(
                "✅ Venue capacity is suitable."
            )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📈 Analytics":

    st.header(
        "📈 Event Analytics"
    )


    if len(events_df) == 0:

        st.info(
            "Create events first to view analytics."
        )

    else:

        # -----------------------------------------------
        # Attendance Chart
        # -----------------------------------------------

        st.subheader(
            "Expected Attendance by Event"
        )


        attendance_chart = events_df[
            [
                "Event_Name",
                "Predicted_Attendance"
            ]
        ].set_index(
            "Event_Name"
        )


        st.bar_chart(
            attendance_chart
        )


        # -----------------------------------------------
        # Registration Comparison
        # -----------------------------------------------

        st.subheader(
            "Registrations vs Expected Attendance"
        )


        comparison_chart = events_df[
            [
                "Event_Name",
                "Registrations",
                "Predicted_Attendance"
            ]
        ].set_index(
            "Event_Name"
        )


        st.line_chart(
            comparison_chart
        )


        # -----------------------------------------------
        # Event Type Distribution
        # -----------------------------------------------

        st.subheader(
            "Event Type Distribution"
        )


        type_count = events_df[
            "Event_Type"
        ].value_counts()


        st.bar_chart(
            type_count
        )


        # -----------------------------------------------
        # Analytics Summary
        # -----------------------------------------------

        st.subheader(
            "📊 Summary"
        )


        total_events = len(
            events_df
        )

        total_registrations = int(
            events_df[
                "Registrations"
            ].sum()
        )

        average_prediction = int(
            events_df[
                "Predicted_Attendance"
            ].mean()
        )


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Total Events",
            total_events
        )

        col2.metric(
            "Total Registrations",
            total_registrations
        )

        col3.metric(
            "Average Expected Attendance",
            average_prediction
        )


# =========================================================
# EVENT AI CHATBOT
# =========================================================

elif page == "💬 Event AI Chatbot":

    event_chatbot(
        model,
        encoder,
        feature_columns
    )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "AI-Based Event Management System | "
    "Machine Learning + Streamlit"
)