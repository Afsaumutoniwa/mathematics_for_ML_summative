from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import uvicorn
import os
# import gdown

# file_id = "1UvlAfE0nc_MWIDMt6NThw-50-vahDnO0"
# model_path = "best_hydroponic_model.pkl"

# if not os.path.exists(model_path):
#     print("📥 Downloading model from Google Drive...")
#     gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)


# Initialize FastAPI application
app = FastAPI(
    title="Hydroponic pH Prediction API",
    description="This API predicts the optimal pH level for hydroponic farming based on key soil and environmental factors.",
    version="1.0.0",
    docs_url="/swagger",  # Swagger UI Path
    redoc_url="/redoc"  # Alternative API documentation
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods including OPTIONS
    allow_headers=["*"],
)

# Define input data model with constraints
class HydroponicInput(BaseModel):
    soil_ec: float = Field(..., ge=0.0, le=3.0, description="Electrical Conductivity (dS/m), must be between 0 and 3")
    nitrogen: float = Field(..., ge=0.0, le=120.0, description="Nitrogen content (ppm), must be between 0 and 120")
    phosphorus: float = Field(..., ge=0.0, le=80.0, description="Phosphorus content (ppm), must be between 0 and 80")
    potassium: float = Field(..., ge=0.0, le=300.0, description="Potassium content (ppm), must be between 0 and 300")
    moisture: float = Field(..., ge=0.0, le=80.0, description="Moisture percentage, must be between 0 and 80")
    temperature: float = Field(..., ge=0.0, le=30.0, description="Temperature (°C), must be between 0 and 30")
    crop: int = Field(..., ge=0, le=12, description="Encoded crop type (0-12)")

# Handle CORS preflight requests explicitly
@app.options("/predict")
async def preflight():
    return {"message": "OK"}


def load_model():
    with open("best_hydroponic_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model
    
@app.get("/")
async def root():
    return {"message": "Welcome to the Hydroponic pH Prediction API! Go to /swagger for API documentation."}

@app.post("/predict")
async def predict(data: HydroponicInput):
    model = load_model()  # Load model only when needed
    input_data = np.array([[data.soil_ec, data.nitrogen, data.phosphorus,
                            data.potassium, data.moisture, data.temperature, data.crop]])
    prediction = model.predict(input_data)[0]
    return {"Predicted pH": prediction}


# Run API (for local development)
if __name__ == "__main__":    
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", 8000)))


