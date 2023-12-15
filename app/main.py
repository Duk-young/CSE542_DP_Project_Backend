import os
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
# import boto3 # S3 Library
import asyncpraw
from .consts import origins
from .routers import stream
import requests
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# dotenv initialize
load_dotenv() # Load Environment variables

# env var setups
OPENAPI_URL = os.getenv("OPENAPI_URL") 
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

# initialize the app
app = FastAPI(
    openapi_url=OPENAPI_URL,
)
# add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins.urls,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)
# include routers
app.include_router(stream.router)
# 
def download_nltk_data():
    nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
    
    if not os.path.exists(os.path.join(nltk_data_path, 'corpora/wordnet')):
        nltk.download('wordnet')
    if not os.path.exists(os.path.join(nltk_data_path, 'corpora/stopwords')):
        nltk.download('stopwords')
    if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
        nltk.download('punkt')
    if not os.path.exists(os.path.join(nltk_data_path, 'taggers/averaged_perceptron_tagger')):
        nltk.download('averaged_perceptron_tagger')
    if not os.path.exists(os.path.join(nltk_data_path, 'sentiment/vader_lexicon')):
        nltk.download('vader_lexicon')

@app.on_event("startup")
def startup_db_client():
    download_nltk_data()
    app.sia = SentimentIntensityAnalyzer()
    app.reddit = asyncpraw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )
    


@app.get("/")
async def root():
    return {"message": "Hello World"}
