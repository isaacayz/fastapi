from starlette.routing import Host
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd

#instatiate the app object
app = FastAPI()
pickle_in = open("classifier.pickle", "rb")
classifier = pickle.load(pickle_in)

#API Integration

#index
@app.get('/')
def index():
    return {'message': 'Hello, omo iya mi!'}

#welcome route
@app.get('/welcome')
def get_name(name: str):
    return {'Welcome, ', name}

#prediction route
@app.post('/predict')
def predict_banknote(data: BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])

    if(prediction[0]>0.5):
        prediction = 'Fake note'
    else:
        prediction = 'Original note'
    return {
        'prediction': prediction
    }

#run the api
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
