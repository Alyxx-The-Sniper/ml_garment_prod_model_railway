import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, confloat

# --- 1. Pydantic Model for Input Validation ---
# This defines the structure of the JSON data your API will expect.
# inherit pydantic BaseModel
class InputData(BaseModel):
    incentive: confloat(ge=0.0, le=150.0)


# --- 2. App and Model Initialization ---
# Initialize the FastAPI app
app = FastAPI(title="Productivity Predictor API")

# Add CORS middleware to allow your Streamlit app to communicate with this API
# This is a crucial security feature for decoupled frontends.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you might restrict this to your Streamlit app's domain
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the prediction model pipeline once at startup
model = joblib.load("model_1.pkl")


# --- 3. API Endpoints ---
@app.get("/")
def read_root():
    """
    Root endpoint: A simple health check endpoint.
    Returns a JSON message indicating the API is online.
    """
    return {"status": "ok", "message": "Productivity Predictor API is running."}


@app.post("/predict")
def predict(payload: InputData):
    """
    Prediction endpoint:
    - Receives JSON data matching the InputData model.
    - FastAPI automatically validates the incoming data.
    - Runs prediction and returns the result as JSON.
    """
    # Build a one-row DataFrame from the validated JSON payload.
    # our ML model expect a pandas dataframe
    # ex. payload become dict = {'incentive': 100.0}
    df = pd.DataFrame([payload.dict()])

    # Run the pipeline’s predict method.
    prediction = float(model.predict(df)[0])

    # Return the prediction in a JSON response.
    return {"predicted_productivity": round(prediction, 3)}


# --- 4. Main entry point for running the app ---

if __name__ == "__main__":
    # Use uvicorn to run the app. It's a high-performance ASGI server.
    # The command `uvicorn main:app --reload` is typically used for development.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

