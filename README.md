# covid_19_hate_speech

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/77782d14ee04460e83c9e5b8f5708ffc)](https://www.codacy.com/manual/samhunsadamant/covid_19_hate_speech?utm_source=github.com&utm_medium=referral&utm_content=SamSamhuns/covid_19_hate_speech&utm_campaign=Badge_Grade)

## Data Gathering

The tweet dataset were acquired from a public dataset of tweet ids with various coronavirus related keywords from [COVID-19: The First Public Coronavirus Twitter Dataset]\(<https://github.com/echen102/COVID-19-TweetIDs>.

### Hydrating Tweets

Install all requirements from `requirements/hydration.txt`

`$ pip install requirements/hydration.txt`

Apply for a [Twitter developers account](https://developer.twitter.com/en/apply-for-access). Configure `twarc` with the developer keys from Twitter.

`$ twarc configure`

Hydration can be run using:

`$ python3 hydrate_tweets.py`

Default parameters for ZIP_TWEETS, raw tweet source dir, hydrated tweets target dir, month dir and language selections can be changed inside `hydrate_tweets.py`. By default they are:

```python
ZIP_TWEETS             = False
raw_tweet_src          = 'raw_tweet_ids/'
hydrated_tweets_target = 'hydrated_tweets/'
month_dirs             = ['2020-01', '2020-02', '2020-03']
lang_set               = set(["en", "null", None])
```

## Data Preprocessing

## Exploratory Data Analysis

## Hate Speech Classification

## Methodology

## Results

## Discussion

## Acknowledgments

## References

-   Chen E., Lerman K. and Ferrara E. (2020). COVID-19: The First Public Coronavirus Twitter Dataset. arXiv:2003.07372
