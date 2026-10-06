import streamlit as st
import pandas as pd


def get_ticket_price(event_type):

    prices = {
        "Conference": 500,
        "Workshop": 300,
        "Concert": 800,
        "Seminar": 400,
        "Exhibition": 250,
        "Sports": 350,
        "Wedding": 1000,
        "Corporate": 700
    }

    return prices.get(event_type, 500)


def get_marketing_budget(registrations):

    budget = registrations * 50

    return max(5000, min(budget, 200000))


def get_venue_capacity(registrations):

    capacity = int(registrations * 1.2)

    return max(50, capacity)


def chatbot_prediction(
    model,
    encoder,
    feature_columns,
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

    # Encode event type
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

    # Combine encoded and numerical features
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

    # ML prediction
    prediction = model.predict(
        final_input
    )[0]

    prediction = max(
        0,
        min(
            prediction,
            venue_capacity
        )
    )

    prediction = int(round(prediction))

    utilization = (
        prediction / venue_capacity
    ) * 100

    return prediction, utilization


def event_chatbot(
    model,
    encoder,
    feature_columns
):

    st.header("💬 Event Planning AI Chatbot")

    st.write(
        "Tell me your event requirements and I will "
        "suggest ticket price, marketing budget, venue "
        "capacity and expected attendance."
    )

    st.divider()

    # -------------------------------------------------
    # INITIAL STATE
    # -------------------------------------------------

    if "chat_messages" not in st.session_state:

        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content":
                "👋 Hi! I am your Event Planning AI Assistant. "
                "Let's plan your event!"
            }
        ]

    if "chat_step" not in st.session_state:
        st.session_state.chat_step = 0

    if "chat_data" not in st.session_state:
        st.session_state.chat_data = {}

    # -------------------------------------------------
    # DISPLAY CHAT HISTORY
    # -------------------------------------------------

    for message in st.session_state.chat_messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

    # -------------------------------------------------
    # CURRENT QUESTION
    # -------------------------------------------------

    step = st.session_state.chat_step

    questions = [
        "What type of event are you planning? "
        "Example: Conference, Workshop, Concert",

        "How many registrations are you expecting?",

        "How many people attended your previous "
        "similar event? If this is a new event, "
        "enter 0.",

        "How many hours will the event last?",

        "What is the expected social media reach?",

        "What rating do you expect for the event? "
        "Choose between 3 and 5."
    ]

    # -------------------------------------------------
    # CHAT INPUT
    # -------------------------------------------------

    if step < len(questions):

        user_input = st.chat_input(
            questions[step]
        )

        if user_input:

            # Add user message
            st.session_state.chat_messages.append(
                {
                    "role": "user",
                    "content": user_input
                }
            )

            # -----------------------------------------
            # STEP 0 - EVENT TYPE
            # -----------------------------------------

            if step == 0:

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

                matched_type = None

                for event_type in event_types:

                    if event_type.lower() in user_input.lower():

                        matched_type = event_type
                        break

                if matched_type is None:

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content":
                            "Please enter one of these event types: "
                            "Conference, Workshop, Concert, Seminar, "
                            "Exhibition, Sports, Wedding or Corporate."
                        }
                    )

                else:

                    st.session_state.chat_data[
                        "event_type"
                    ] = matched_type

                    ticket_price = get_ticket_price(
                        matched_type
                    )

                    st.session_state.chat_data[
                        "ticket_price"
                    ] = ticket_price

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content":
                            f"Great! You selected a {matched_type}. "
                            f"I recommend an initial ticket price "
                            f"of ₹{ticket_price}. "
                            f"\n\nHow many registrations are you expecting?"
                        }
                    )

                    st.session_state.chat_step += 1

            # -----------------------------------------
            # STEP 1 - REGISTRATIONS
            # -----------------------------------------

            elif step == 1:

                try:

                    registrations = int(
                        user_input.replace(",", "")
                    )

                    if registrations <= 0:
                        raise ValueError

                    st.session_state.chat_data[
                        "registrations"
                    ] = registrations

                    marketing_budget = get_marketing_budget(
                        registrations
                    )

                    venue_capacity = get_venue_capacity(
                        registrations
                    )

                    st.session_state.chat_data[
                        "marketing_budget"
                    ] = marketing_budget

                    st.session_state.chat_data[
                        "venue_capacity"
                    ] = venue_capacity

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content":
                            f"Got it! For {registrations} registrations, "
                            f"I recommend a venue capacity of around "
                            f"{venue_capacity} people and a marketing "
                            f"budget of approximately ₹{marketing_budget:,}."
                            f"\n\nHow many people attended your previous "
                            f"similar event? If this is a new event, enter 0."
                        }
                    )

                    st.session_state.chat_step += 1

                except ValueError:

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content":
                            "Please enter a valid number. "
                            "Example: 500"
                        }
                    )

            # -----------------------------------------
            # STEP 2 - PREVIOUS ATTENDANCE
            # -----------------------------------------

            elif step == 2:

                try:

                    previous_attendance = int(
                        user_input.replace(",", "")
                    )

                    if previous_attendance < 0:
                        raise ValueError

                    st.session_state.chat_data[
                        "previous_attendance"
                    ] = previous_attendance

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content":
                            "Perfect. How many hours will "
                            "your event last?"
                        }
                    )

                    st.session_state.chat_step += 1

                except ValueError:

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content":
                            "Please enter a valid number. "
                            "Example: 5"
                        }
                    )

            # -----------------------------------------
            # STEP 3 - DURATION
            # -----------------------------------------

            elif step == 3:

                try:

                    duration = float(
                        user_input
                    )

                    if duration <= 0 or duration > 24:
                        raise ValueError

                    st.session_state.chat_data[
                        "duration_hours"
                    ] = duration

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content":
                            "Great! What is the expected "
                            "social media reach?"
                        }
                    )

                    st.session_state.chat_step += 1

                except ValueError:

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content":
                            "Please enter a number between "
                            "1 and 24 hours."
                        }
                    )

            # -----------------------------------------
            # STEP 4 - SOCIAL MEDIA
            # -----------------------------------------

            elif step == 4:

                try:

                    social_reach = int(
                        user_input.replace(",", "")
                    )

                    if social_reach < 0:
                        raise ValueError

                    st.session_state.chat_data[
                        "social_media_reach"
                    ] = social_reach

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content":
                            "Almost done! What event rating "
                            "do you expect? Enter a value from "
                            "3 to 5."
                        }
                    )

                    st.session_state.chat_step += 1

                except ValueError:

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content":
                            "Please enter a valid number. "
                            "Example: 15000"
                        }
                    )

            # -----------------------------------------
            # STEP 5 - RATING
            # -----------------------------------------

            elif step == 5:

                try:

                    rating = float(
                        user_input
                    )

                    if rating < 3 or rating > 5:
                        raise ValueError

                    st.session_state.chat_data[
                        "event_rating"
                    ] = rating

                    st.session_state.chat_step += 1

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content":
                            "🎯 All details received! "
                            "I am calculating your event plan..."
                        }
                    )

                except ValueError:

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content":
                            "Please enter a rating between "
                            "3 and 5. Example: 4.5"
                        }
                    )

            st.rerun()

    # -------------------------------------------------
    # GENERATE FINAL RESULT
    # -------------------------------------------------

    if st.session_state.chat_step >= len(questions):

        data = st.session_state.chat_data

        prediction, utilization = chatbot_prediction(
            model,
            encoder,
            feature_columns,
            data["event_type"],
            data["registrations"],
            data["ticket_price"],
            data["marketing_budget"],
            data["venue_capacity"],
            data["duration_hours"],
            data["previous_attendance"],
            data["social_media_reach"],
            data["event_rating"]
        )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content":
                "Your event plan is ready! 🎉"
            }
        )

        st.subheader(
            "🎯 AI Event Plan"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🎟️ Recommended Ticket Price",
            f"₹{data['ticket_price']:,}"
        )

        col2.metric(
            "📢 Marketing Budget",
            f"₹{data['marketing_budget']:,}"
        )

        col3.metric(
            "🏢 Recommended Venue Capacity",
            f"{data['venue_capacity']:,}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "👥 Registrations",
            f"{data['registrations']:,}"
        )

        col2.metric(
            "🤖 AI Predicted Attendance",
            f"{prediction:,}"
        )

        col3.metric(
            "📊 Venue Utilization",
            f"{utilization:.1f}%"
        )

        st.divider()

        st.subheader(
            "💰 Estimated Event Revenue"
        )

        estimated_revenue = (
            prediction * data["ticket_price"]
        )

        st.metric(
            "Expected Ticket Revenue",
            f"₹{estimated_revenue:,.0f}"
        )

        st.divider()

        st.subheader(
            "💡 AI Recommendations"
        )

        if utilization > 90:

            st.warning(
                "⚠️ Expected attendance is very high. "
                "Consider selecting a larger venue."
            )

        elif utilization < 40:

            st.info(
                "ℹ️ Expected attendance is relatively low "
                "compared with venue capacity."
            )

        else:

            st.success(
                "✅ Recommended venue capacity is suitable."
            )

        if data["marketing_budget"] < 10000:

            st.info(
                "📢 Consider increasing your marketing budget "
                "to improve event visibility."
            )

        if prediction < data["registrations"] * 0.6:

            st.warning(
                "📣 The AI prediction is significantly lower "
                "than registrations. Consider improving "
                "marketing and event engagement."
            )

        else:

            st.success(
                "🎉 Expected attendance is healthy compared "
                "with registrations."
            )

        st.divider()

        st.subheader(
            "📋 Complete Event Plan"
        )

        result_df = pd.DataFrame({
            "Event Detail": [
                "Event Type",
                "Registrations",
                "Ticket Price",
                "Marketing Budget",
                "Venue Capacity",
                "Previous Attendance",
                "Duration",
                "Social Media Reach",
                "Expected Rating",
                "AI Predicted Attendance",
                "Venue Utilization",
                "Expected Ticket Revenue"
            ],
            "Recommendation": [
                data["event_type"],
                f"{data['registrations']:,}",
                f"₹{data['ticket_price']:,}",
                f"₹{data['marketing_budget']:,}",
                f"{data['venue_capacity']:,} people",
                f"{data['previous_attendance']:,}",
                f"{data['duration_hours']} hours",
                f"{data['social_media_reach']:,}",
                f"{data['event_rating']}/5",
                f"{prediction:,} people",
                f"{utilization:.1f}%",
                f"₹{estimated_revenue:,.0f}"
            ]
        })

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        st.success(
            "🎉 Your event plan has been generated "
            "using Machine Learning + rule-based recommendations."
        )

        if st.button(
            "🔄 Plan Another Event"
        ):

            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content":
                    "👋 Sure! Let's plan another event."
                }
            ]

            st.session_state.chat_step = 0
            st.session_state.chat_data = {}

            st.rerun()