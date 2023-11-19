import json
# import pyspark as spark
# from pyspark import SparkConf
# from pyspark import SparkContext
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
import preprocessor as p
import nltk
from nltk.corpus import stopwords
from datetime import datetime
import csv
import pandas as pd
import os

"""
This code preprocess raw tweets dataset to be used for sentiment analysis. 
The code basically cleans up emojis and urls from text and match user-defined location
to actual location for later use of plotting sentiment statistics on world map.
"""


# download required nltk files before cleaning text
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

download_nltk_data()
print('\r Creating Location Dictionary...')

# get list of stop words
stop_words = set(stopwords.words('english'))
stop_words = [x.lower() for x in stop_words]

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

# check if tweet text is json applicable (some tweet texts are not formatted correctly)
def json_format_check(text):
    try:
        text = text.replace("\'", "\"")
        json.loads(text)
        return True
    except:
        return False
    
directory = './UkraineTweets/'
output_directory = './preprocessed/'
indexed_output_directory ='./preprocessed_indexed/'
indexed_output_filename = ''
filename_format = 'UkraineTweetsPreprocessedForBert_'
# iterate over files in the directory
total = sum([len(files) for root, dirs, files in os.walk(directory)])
count = 0
total_country_tweets = 0
output_row_dict = {}
output_by_country_row_dict = {}
for root, dirs, files in os.walk(directory):

    # Check the date of the tweets created,
    # and create output filename based on year and month
    year_month_check = ''
    for filename in files:
        output_row_num = 1
        output_by_country_row_num = 1
        output_filename = ''
        output_by_country_filename = ''
        tweet_by_country = {}
        with open(directory + filename, newline='', encoding='UTF-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            column_index = header.index('tweetcreatedts')
            dt = ''
            while(dt == ''):
                try:
                    row = next(reader)
                    date = row[column_index].split('.')[0]
                    dt = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                except StopIteration:
                    break
                except:
                    pass
            if dt == '':
                print(root +'/'+filename, '<=== Failed')
            else:
                year_month = dt.strftime('%Y_%m')
                output_filename_date_format = year_month + '.csv'
                output_by_country_filename_date_format = 'by_country_' +year_month + '.csv'
                output_filename = output_directory + filename_format + output_filename_date_format
                output_by_country_filename = output_directory + filename_format + output_by_country_filename_date_format
                if year_month not in output_row_dict:
                    output_row_dict[year_month] = 1
                if year_month not in output_by_country_row_dict:
                    output_by_country_row_dict[year_month] = 1
                output_row_num = output_row_dict[year_month]
                output_by_country_row_num = output_by_country_row_dict[year_month]
        if output_filename != '':
            count += 1

            # progress bar to check the task progress.
            printProgressBar(count, total, filename=os.path.join(directory + filename), output_filename = output_filename)

            # preprocess files with PySpark
            df = sc.read.csv(directory + filename, header=True, inferSchema=True)
            filtered_df = df.select('location','tweetcreatedts','text','hashtags','following','followers','totaltweets','retweetcount','favorite_count').where((df.language=='en'))
            filtered_df = filtered_df.rdd.filter(lambda rec: json_format_check(rec[3]) == True).map(lambda rec: match_location(rec[0]) + (rec[1], preprocess_tweets(rec[2]), [hashtag_info['text'] for hashtag_info in json.loads(rec[3].replace("\'", "\""))], rec[4], rec[5], rec[6], rec[7], rec[8], compute_reach_score(rec[4], rec[5]),compute_popularity_score(rec[7], rec[8])))
            preprocessed_output = filtered_df.collect()
            with open(output_filename, 'a', newline='', encoding='UTF-8') as f:
                writer = csv.writer(f)
                if os.path.getsize(output_filename) == 0:
                    header = ['id','city', 'state', 'country', 'lat','lon','date', 'text','hashtags','following','followers','total_tweets','retweet_count','favorite_count', 'reach_score', 'popularity_score']
                    writer.writerow(header)
                for row in preprocessed_output:
                        new_row = (output_row_num,) + row
                        writer.writerow(new_row)
                        output_row_num += 1
            output_row_dict[year_month] = output_row_num
            # #ANCHOR This is to create preprocessed tweets by country. (Tweets with undefined location will be eliminated)
            preprocessed_output = filtered_df.filter(lambda rec: rec[4] != '' and rec[5] != '').collect()
            with open(output_by_country_filename, 'a', newline='', encoding='UTF-8') as f:
                writer = csv.writer(f)
                if os.path.getsize(output_by_country_filename) == 0:
                    header = ['id','city', 'state', 'country', 'lat','lon','date', 'text','hashtags','following','followers','total_tweets','retweet_count','favorite_count', 'reach_score', 'popularity_score']
                    writer.writerow(header)
                for row in preprocessed_output:
                        new_row = (output_by_country_row_num,) + row
                        writer.writerow(new_row)
                        output_by_country_row_num += 1
            output_by_country_row_dict[year_month] = output_by_country_row_num
            # total_country_tweets += len(preprocessed_output)
            # for tweet in preprocessed_output:
            #     country = tweet[0][2]
            #     if country not in tweet_by_country:
            #         tweet_by_country[country] = [tweet]
            #     else:
            #         tweet_by_country[country].append(tweet)
            # for country in tweet_by_country.keys():
            #     country_directory = output_directory + country + '/'
            #     output_filename = country_directory + filename_format + country + output_filename_date_format
            #     # indexed_output_filename = indexed_output_directory + filename_format + country + indexed_output_filename_date_format
            #     if not os.path.exists(country_directory):
            #         os.makedirs(country_directory)
            #     with open(output_filename, 'a', newline='', encoding='UTF-8') as f:
            #         writer = csv.writer(f)
            #         writer.writerows(tweet_by_country[country])
            # # preprocessed file appended to the matching year and month output file.