from typing import Union, List, Dict, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tsdownsample import LTTBDownsampler
from pydantic import BaseModel
import pandas as pd
import numpy as np
import sys, os
import torch

if '/home/new_lab/test/ensemble_bae' not in sys.path:
    sys.path.append('/home/new_lab/test/ensemble_bae')
if '/home/new_lab/test/ensemble_bae/server' not in sys.path:
    sys.path.append('/home/new_lab/test/ensemble_bae/server')


from store import DataStore
import process_model as pm 

data_store = DataStore()

bvae_ens = pm.init_model() # model init

app = FastAPI()
origins = [
    "http://localhost:5173",
]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class Item(BaseModel):
    x: int
    y: float

class Epis_response(BaseModel):
    epis_upper_50: List[Item]
    epis_lower_50: List[Item]
    epis_upper_95: List[Item]
    epis_lower_95: List[Item]
    epis: List[Item]

class Anomaly_Item(BaseModel):
    x1: int
    x2: int
    anomaly_score: float

# class Mult_epis_response(BaseModel):
#     epis_1: List[Item]
#     epis_2: List[Item]
#     epis_3: List[Item]

class Mult_epis_response(BaseModel):
    epis_upper_50: Tuple[List[Item], List[Item], List[Item]]
    epis_lower_50: Tuple[List[Item], List[Item], List[Item]]
    epis_upper_95: Tuple[List[Item], List[Item], List[Item]]
    epis_lower_95: Tuple[List[Item], List[Item], List[Item]]
    epis: Tuple[List[Item], List[Item], List[Item]]

@app.get("/")
def test():
    return {"Hello": "World"}

@app.get("/api/SMAP_P1", response_model=List[Item])
def load_SMAP_P1():
   
    data: pd.Series = pd.read_csv('SMAP_P1.csv').astype("Float32")
    x = np.array(data.index)
    y = np.array(data['value'])

    pm.load_Dataset("SMAP_P1")
    data_store.add_data("SMAP_P1", y)

    return [{"x": x[i], "y": y[i]} for i in range(len(data))]

@app.get("/api/NY_Taxi", response_model=List[Item])
def load_NY_Texi():
   
    data: pd.Series = pd.read_csv('NY_Taxi.csv').astype("Float32")
    x = np.array(data.index)
    y = np.array(data['value'])

    pm.load_Dataset("NY_Taxi")
    data_store.add_data("NY_Taxi", y)

    return [{"x": x[i], "y": y[i]} for i in range(len(data))]

@app.get("/api/SWAT_PIT502", response_model=List[Item])
def load_SWAT_PIT502():
    data: pd.Series = pd.read_csv('SWAT_PIT502.csv').astype("Float32")
    x = np.array(data.index)
    y = np.array(data['value'])

    pm.load_Dataset("SWAT_PIT502")
    data_store.add_data("SWAT_PIT502", y)

    return [{"x": x[i], "y": y[i]} for i in range(len(data))]

@app.get("/api/downsample", response_model=List[Item])
def downsample(dataset="SMAP_P1"):
    y = data_store.get_data(dataset)
    x = np.arange(len(y))
    s_ds = LTTBDownsampler().downsample(y, n_out= 800)
    downsampled_y = y[s_ds]
    downsampled_x = x[s_ds]
    return [{"x": downsampled_x[i], "y": downsampled_y[i]} for i in range(len(downsampled_y))]

@app.get("/api/epis", response_model=Epis_response)
def get_epis():
    result = pm.predict_epis(bvae_ens)
    return result

@app.get("/api/mult_epis", response_model=Mult_epis_response)
def get_multi_epis(x0: float, x1: float):
    x0 = int(x0)
    x1 = int(x1)
    result = pm.predict_mult_epis(bvae_ens, index=[x0, x1])
    return result

@app.get("/api/alea", response_model=List[Item])
def get_alea():
    alea = pm.predict_alea(bvae_ens)
    return alea

@app.get("/api/anomaly", response_model=List[Anomaly_Item])
def get_anomaly():
    anomaly = pm.predict_error(bvae_ens)
    return anomaly


# if __name__ == "__main__":
#     load_SMAP_P1()
    

