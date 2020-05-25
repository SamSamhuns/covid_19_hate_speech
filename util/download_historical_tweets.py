import os
import tqdm
import json
import tweepy

import pandas as pd
from pathlib import Path


def get_all_tweets(user_id,
                   consumer_key,
                   consumer_secret,
                   access_key,
                   access_secret,
                   max_limit=200,
                   write_path='tweet_history/'):
    """
    Gets the historical tweets of a user from the current date based on the user_id str
    max_limit defines how many historical tweets to download for that user

    Consumer and Access keys must be acquired from a Twitter Developers account
    """

    os.makedirs(write_path, exist_ok=True)
    full_write_path = write_path + f'{user_id}' + '.jsonl'

    # Check for redundant downloads
    if Path(full_write_path).is_file():
        print(
            f"{user_id}.jsonl file already exists in the directory. Aborting this download")
        return

    # Twitter only allows access to a users most recent 3240 tweets with this method
    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(user_id=user_id,
                                   count=5)

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print("getting tweets before %s" % (oldest))

        # all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(user_id=user_id,
                                       count=5,
                                       max_id=oldest)
        # save most recent tweets
        alltweets.extend(new_tweets)
        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print("...%s tweets downloaded so far" % (len(alltweets)))

        if len(alltweets) > max_limit:
            break

    # Extract the json object associated with each downloaded tweet data and write to file
    json_list = [tuple_obj._json for tuple_obj in alltweets]

    with open(full_write_path, 'w', encoding='utf-8') as j_file:
        for tweet in json_list:
            j_file.write(json.dumps(tweet) + "\n")
