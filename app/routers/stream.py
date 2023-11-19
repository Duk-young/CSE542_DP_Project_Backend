from fastapi import (
    APIRouter,
    Body,
    Request,
    HTTPException,
    status,
    File,
    UploadFile,
    Form,
)
from fastapi.responses import StreamingResponse
from ..models import test_model
import asyncpraw
import pandas as pd
import datetime
import time
from tqdm import tqdm
import os
import threading
import signal
import preprocessor as p
import nltk
from nltk.corpus import stopwords
import csv
import json
from diffprivlib.mechanisms import Laplace
from pydantic import BaseModel
# /test to reach APIs
router = APIRouter(
    prefix="/stream",
    tags=["stream"],
    responses={
        404: {"description": "Not found"},
        403: {"description": "Operation forbidden"},
    },
)

class SentimentData(BaseModel):
    sentiment_score: dict
    epsilon: float
    sensitivity_score: float

def add_ibm_laplacian_noise(data, epsilon, sensitivity):
    laplace_mech = Laplace(epsilon=epsilon, sensitivity=sensitivity)
    return laplace_mech.randomise(data)
    
# clean up text process
def preprocess_texts(text):
    result = []

    # preprocess tweet texts. removes URLs, emojis and smileys
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY)
    intmd_result = p.clean(text)
    # remove possible HTML tag leftovers
    intmd_result = intmd_result.replace("#","").replace("&amp", "").replace("\n", "")
    result = intmd_result

    return result

def get_date(created):
    return str(datetime.datetime.fromtimestamp(created))


def build_dataset(reddit, search_words='israelpalestine', items_limit=50):
    
    # Collect reddit posts
    subreddit = reddit.subreddit(search_words)
    new_subreddit = subreddit.new(limit=items_limit)
    reddit_dict = { "title":[],
                "score":[],
                "id":[], 
                "url":[],
                "comms_num": [],
                "created": [],
                "body":[],
                "author":[]}
    
    print(f"retreive new reddit posts ...")
    for post in tqdm(new_subreddit):
        reddit_dict["title"].append(post.title)
        reddit_dict["score"].append(post.score)
        reddit_dict["id"].append(post.id)
        reddit_dict["url"].append(post.url)
        reddit_dict["comms_num"].append(post.num_comments)
        reddit_dict["created"].append(post.created)
        reddit_dict["body"].append(post.selftext)
        reddit_dict["author"].append(post.author)

    for comment in tqdm(subreddit.comments(limit=100)):
        reddit_dict["title"].append("Comment")
        reddit_dict["score"].append(comment.score)
        reddit_dict["id"].append(comment.id)
        reddit_dict["url"].append("")
        reddit_dict["comms_num"].append(0)
        reddit_dict["created"].append(comment.created)
        reddit_dict["body"].append(comment.body)
        reddit_dict["author"].append(comment.author)
    reddit_df = pd.DataFrame(reddit_dict)
    print(f"new reddit posts retrieved: {len(reddit_df)}")
    reddit_df['timestamp'] = reddit_df['created'].apply(lambda x: get_date(x))

    return reddit_df
   

def update_and_save_dataset(reddit_df):   
    file_path = "reddit_israelpalestine.csv"
    if os.path.exists(file_path):
        topics_old_df = pd.read_csv(file_path)
        print(f"past reddit posts: {topics_old_df.shape}")
        topics_all_df = pd.concat([topics_old_df, reddit_df], axis=0)
        print(f"new reddit posts: {reddit_df.shape[0]} past posts: {topics_old_df.shape[0]} all posts: {topics_all_df.shape[0]}")
        topics_new_df = topics_all_df.drop_duplicates(subset = ["id"], keep='last', inplace=False)
        print(f"all reddit posts: {topics_new_df.shape}")
        topics_new_df.to_csv(file_path, index=False)
    else:
        print(f"reddit posts: {reddit_df.shape}")
        reddit_df.to_csv(file_path, index=False)

@router.get("/posts")
async def stream_posts(request: Request, community: str = 'israelpalestine'):
    subreddit = await request.app.reddit.subreddit(community)
    # threading.Thread(target=stream_posts, args=(subreddit,)).start()
    
    async def event_generator():
        global is_streaming_post
        is_streaming_post = True
        while is_streaming_post:
            async for post in subreddit.stream.submissions():
                if not is_streaming_post:
                    break
                print(post)
                sentiment_score = request.app.sia.polarity_scores(preprocess_texts(post.selftext))
                post_data = json.dumps({
                    'title': post.title,
                    'score': post.score,
                    'id': post.id,
                    'url': post.url,
                    'total_comments': post.num_comments,
                    'created': get_date(post.created),
                    'body': post.selftext,
                    'author': str(post.author),
                    'sentiment_score':sentiment_score,
                    'sentiment': 'Positive' if sentiment_score['compound'] > 0 else 'Negative'
                })
                yield f"data: {post_data}\n\n"
            await asyncio.sleep(1)
 
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/comments")
async def stream_comments(request: Request, community: str = 'israelpalestine'):
    subreddit = await request.app.reddit.subreddit(community)

    async def event_generator():
        global is_streaming_comment
        is_streaming_comment = True
        while is_streaming_comment:
            async for comment in subreddit.stream.comments():
                if not is_streaming_comment:
                    break
                print(comment)
                sentiment_score = request.app.sia.polarity_scores(preprocess_texts(comment.body))
                comment_data = json.dumps({
                    'title':"Comment",
                    'score':comment.score, 
                    'id': comment.id, 
                    'url':"", "total_comments":0, 
                    'author': str(comment.author), 
                    'body': comment.body, 
                    'created':get_date(comment.created),
                    'sentiment_score':sentiment_score,
                    'sentiment': 'Positive' if sentiment_score['compound'] > 0 else 'Negative'
                })
                yield f"data: {comment_data}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/apply-dp")
async def apply_dp(request: Request, data: SentimentData):
    print(data)
    dp_sentiment_score = {}
    for key, value in data.sentiment_score.items():
        dp_sentiment_score[key] = add_ibm_laplacian_noise(value, data.epsilon, data.sensitivity_score)
    return dp_sentiment_score

@router.post("/stop-posts")
async def stop_streaming(request: Request):
    global is_streaming_post
    is_streaming_post = False
    return {"message": "Streaming post stopped"}

@router.post("/stop-comments")
async def stop_streaming(request: Request):
    global is_streaming_comment
    is_streaming_comment = False
    return {"message": "Streaming comment stopped"}