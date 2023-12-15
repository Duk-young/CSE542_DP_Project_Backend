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
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import csv
import json
from diffprivlib.mechanisms import Laplace
from pydantic import BaseModel
import numpy as np
import random
import string
import copy
from scipy.stats import laplace
import pydp as dp 
from pydp.algorithms.laplacian import BoundedSum, Count
import math
# /test to reach APIs
router = APIRouter(
    prefix="/stream",
    tags=["stream"],
    responses={
        404: {"description": "Not found"},
        403: {"description": "Operation forbidden"},
    },
)
stop_words = set(stopwords.words('english'))
stop_words = [x.lower() for x in stop_words]

class SentimentData(BaseModel):
    sentiment_score: dict
    epsilon: float
    sensitivity_score: float

class PrivacyLossRequest(BaseModel):
    dataset: list
    epsilon: float

def add_ibm_laplacian_noise(data, epsilon, sensitivity):
    laplace_mech = Laplace(epsilon=epsilon, sensitivity=sensitivity)
    return laplace_mech.randomise(data)

def truncated_laplace_noise(sentiment_scores, epsilon, sensitivity, lower_bound=-1, upper_bound=1):
    # Scale for the Laplace distribution
    scale = sensitivity / epsilon
    # Calculate truncation points based on the current data point
    noisy_data = np.zeros_like(sentiment_scores)
    # Drawing noise from the truncated Laplace distribution
    for i in range(len(sentiment_scores)):
        val = sentiment_scores[i]
        while True:
            noise = np.random.laplace(0, scale)
            noisy_val = val + noise
            # noisy_val = add_ibm_laplacian_noise(val,epsilon,sensitivity)
            # print("noisy_val:",noisy_val)
            if lower_bound <= noisy_val <= upper_bound:
                    noisy_data[i] = noisy_val
                    break
    return noisy_data

def privacy_loss(data1, data2, mechanism, epsilon, sensitivity = 1, num_samples=100):
    """Computes privacy loss for two neighboring datasets."""
    privacy_losses = []

    for _ in range(num_samples):
        output1 = mechanism(sentiment_scores=data1, epsilon=epsilon, sensitivity=sensitivity)
        output2 = mechanism(sentiment_scores=data2, epsilon=epsilon, sensitivity=sensitivity)
        scale = 1 / epsilon
        loc = 0

        prob_dist_data1 = laplace.pdf(output1, loc=loc, scale=scale)
        prob_dist_data2 = laplace.pdf(output2, loc=loc, scale=scale)

        for o1, o2 in zip(prob_dist_data1, prob_dist_data2):
            if o2 > 0:
                p_loss = np.log(o1 / o2)
                privacy_losses.append(p_loss)
    return sorted(privacy_losses, reverse=True)

# clean up text process for sentiment analysis
def preprocess_texts(text):
    result = []

    # preprocess tweet texts. removes URLs, emojis and smileys
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY)
    intmd_result = p.clean(text)
    # remove possible HTML tag leftovers
    intmd_result = intmd_result.replace("#","").replace("&amp", "").replace("\n", "")
    result = intmd_result

    return result

# clean up text process for wordcloud
def preprocess_wordcloud(text):
    result = ""
    # convert all text to lower case
    text = text.lower()

    # preprocess texts. removes URLs, mentions, hashtags, digist, and emojis (and smileys)
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.NUMBER, p.OPT.EMOJI, p.OPT.SMILEY)
    intmd_result = p.clean(text)
    # remove possible HTML tag leftovers
    intmd_result = intmd_result.replace("&amp", "").replace("\n", "")

    
    # remove all the punctuations from text
    remove_punctuations = str.maketrans('','', string.punctuation)
    intmd_result = intmd_result.translate(remove_punctuations)

    # Initailize Lemmatizer. reduce words to their base form
    lemmatizer = WordNetLemmatizer()
    # Remove Stop words ex) a, the, his, her, etc..
    for word, tag in pos_tag(word_tokenize(intmd_result)):

        # lemmatize first
        if word not in stop_words: # comment this line if want only lemmatize
            pos = ''
            if tag.startswith("NN"):
                pos = 'n'   
            elif tag.startswith('VB'):
                pos = 'v'
            elif tag.startswith('JJ'):
                pos = 'a'
            if pos != '':
                word = lemmatizer.lemmatize(word, pos)
            result+=word+","
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
        recent_posts = subreddit.new(limit=100)
        async for post in recent_posts:
            sentiment_score = request.app.sia.polarity_scores(preprocess_texts(post.selftext))
            post_data = json.dumps({
                'title': post.title,
                'score': post.score,
                'id': post.id,
                'url': post.url,
                'total_comments': post.num_comments,
                'created': get_date(post.created),
                'body': post.selftext,
                'body_wordcloud': preprocess_wordcloud(post.selftext),
                'author': str(post.author),
                'sentiment_score':sentiment_score,
                'sentiment': 'Positive' if sentiment_score['compound'] > 0 else 'Negative'
            })
            yield f"data: {post_data}\n\n"
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
                    'body_wordcloud': preprocess_wordcloud(post.selftext),
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
        # Fetch and process the most recent 500 comments
        recent_comments = subreddit.comments(limit=500)
        async for comment in recent_comments:
            sentiment_score = request.app.sia.polarity_scores(preprocess_texts(comment.body))
            comment_data = json.dumps({
                'title':"Comment",
                'score':comment.score, 
                'id': comment.id, 
                'url':"", "total_comments":0, 
                'author': str(comment.author), 
                'body': comment.body, 
                'body_wordcloud': preprocess_wordcloud(comment.body),
                'created':get_date(comment.created),
                'sentiment_score':sentiment_score,
                'sentiment': 'Positive' if sentiment_score['compound'] > 0 else 'Negative'
            })
            yield f"data: {comment_data}\n\n"

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
                    'body_wordcloud': preprocess_wordcloud(comment.body),
                    'created':get_date(comment.created),
                    'sentiment_score':sentiment_score,
                    'sentiment': 'Positive' if sentiment_score['compound'] > 0 else 'Negative'
                })
                yield f"data: {comment_data}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/apply-dp")
async def apply_dp(request: Request, data:PrivacyLossRequest):
    dataset = data.dataset  
    epsilon = data.epsilon
    dp_applied_output = []
    for data in dataset:
        x = Count(epsilon=epsilon)
        count = x.quick_result(data)  # accepts only list as input
        dp_applied_output.append(count)
    replaced_output = [x if x >= 0 else 0 for x in dp_applied_output]

    flat_list = [sum(sublist) for sublist in dataset]
    original_prob_sum = sum(flat_list)
    dp_prob_sum = sum(replaced_output)
    original_prob = [x / original_prob_sum for x in flat_list]
    dp_prob = [x / dp_prob_sum for x in replaced_output]

    original_prob_array = np.array(original_prob)
    dp_prob_array = np.array(dp_prob)

    # Compute Total Variation Distance (TVD)
    tvd = np.sum(np.abs(original_prob_array - dp_prob_array)) / 2

    # Compute Mean Squared Error (MSE)
    mse = np.mean((original_prob_array - dp_prob_array) ** 2)
    return {"dp_histogram":replaced_output, "tvd":tvd, "mse":mse}


@router.post("/privacy-loss")
async def privacy_loss_request(request: Request, data:PrivacyLossRequest, attempts: int = 1000):
    dataset = data.dataset
    epsilon = data.epsilon
    flat_dataset = [x for xs in dataset for x in xs]
    origin_sum = sum(flat_dataset)
    result = {"inferences":[],"privacy_loss":[], 'infer_member':None, "tps":0}
    neighboring_dataset = copy.deepcopy(dataset)
    avg_noise = 0
    if len(dataset) > 1:
        while True:
            index = random.randint(0,len(neighboring_dataset)-1)
            if len(neighboring_dataset[index]) > 1:
                result["infer_member"] = neighboring_dataset[index].pop()
                break
        flat_neighboring_dataset = [x for xs in neighboring_dataset for x in xs]
        # print(neighboring_dataset)
        data1 = [sum(lst) / len(lst) for lst in dataset if lst]
        data2 = [sum(lst) / len(lst) for lst in neighboring_dataset if lst]
        data1 = [x + 1 for x in data1]
        data2 = [x + 1 for x in data2]
        result["privacy_loss"] = privacy_loss(data1=data1, data2=data2, mechanism = truncated_laplace_noise, epsilon=epsilon, sensitivity=1)
        print("works until this point")
        neighboring_sum = sum(flat_neighboring_dataset)
        for i in range(attempts):
            # dp_dataset = truncated_laplace_noise(sentiment_scores=flat_neighboring_dataset, epsilon=epsilon, sensitivity=1)
            # inference = origin_sum - sum(dp_dataset)
            dp_sum = add_ibm_laplacian_noise(neighboring_sum, epsilon, 1)
            inference = origin_sum - dp_sum
            # print("target :", result["infer_member"],"/ inference :",inference)
            if math.isclose(result['infer_member'], inference, abs_tol=0.01):
                result["tps"]+=1
            result["inferences"].append(inference)
            avg_noise += inference - result['infer_member']
    avg_noise /= attempts
    print("Epsilon:",epsilon)
    print("Target:", result['infer_member'])
    print("Positive Inferences out of 1000:", result["tps"])
    print("Average Noise:", f"{avg_noise}")
    return result


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