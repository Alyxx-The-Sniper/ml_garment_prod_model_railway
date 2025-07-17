# your_streamlit_app.py

import streamlit as st
import requests  # To make HTTP requests to the API

st.title("Productivity Predictor (UI)")
st.write("This interface calls a FastAPI backend to get predictions.")

# --- UI to get user input ---
# This matches the 'incentive' field in your Pydantic model
incentive_value = st.number_input(
    "Enter Incentive Amount:",
    min_value=0.0,
    max_value=150.0,
    value=50.0, # A default value
    step=1.0
)

# --- Button to trigger the API call ---
if st.button("Predict Productivity"):
    # The URL of your running FastAPI endpoint
    api_url = "http://127.0.0.1:8000/predict"

    # The data to send in the request body, matching the Pydantic model
    payload = {
        "incentive": incentive_value
    }

    try:
        # Make the POST request to the FastAPI backend
        response = requests.post(api_url, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            prediction_data = response.json() # Get the JSON response
            productivity = prediction_data['predicted_productivity']
            st.success(f"Predicted Productivity: {productivity}")
        else:
            # Show error details if something went wrong
            st.error("API call failed. See details below.")
            st.json(response.json())

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the FastAPI backend. Is it running?")

# --- Model Information and Other Work ---
st.markdown("---") # Add a separator for better readability

st.markdown(
    "**Model created by**: [Alexis Mandario](https://www.linkedin.com/in/alexis-mandario-b546881a8/)"
)

st.markdown("### ML Model & Documentation")
st.markdown("- [Model Training Procedure](https://medium.com/@kaikuh/machine-learning-code-focus-1bf13c848bc7)") # Placeholder link

st.markdown("### Others Sample of my work:")
st.markdown("- [Exploratory Data Analysis](https://medium.com/@kaikuh/data-science-statistics-and-machine-learning-part-1-663b07d42a7c)") # Placeholder link
st.markdown("- [Hypothesis Testing: Incentive vs. No Incentive](https://medium.com/@kaikuh/statistical-analysis-part-2-3-b14b87f85abd)") # Placeholder link
st.markdown("- [RAG](https://alyx-rag-sample.up.railway.app/)") # Placeholder link
st.markdown("- [Agentic AI](https://multi-agent-hvac-production.up.railway.app/") # Placeholder link
st.markdown("- [Voice time capsule](https://voice-time-capsule-production.up.railway.app/)") # Placeholder link
