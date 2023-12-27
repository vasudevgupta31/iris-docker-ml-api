import pandas as pd
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from app.model.model import loaded_model
from app.model.model import __version__ as model_version


app = FastAPI()
# can be run with > uvicorn main:app --reload else can be run via docker


class ScoringItem(BaseModel):
    
    SepalLength: float
    SepalWidth: float
    PetalLength: float
    PetalWidth: float


class PredictionOut(BaseModel):
    class_label: int


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post('/predict', response_model=PredictionOut)
async def scoring_endpoint(row: ScoringItem):
    row_df = pd.DataFrame(jsonable_encoder(row), index=[0])
    prediction = loaded_model.predict(row_df.values)
    return {"class_label": prediction}
