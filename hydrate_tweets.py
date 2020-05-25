# This script will walk through all the tweet id files and
# hydrate them with twarc. The line oriented JSON files will
# be placed right next to each tweet id file.
#
# Note: you will need to install twarc, tqdm, and run twarc configure
# from the command line to tell it your Twitter API keys.

import os
import gzip
import json

from tqdm import tqdm
from twarc import Twarc
from pathlib import Path

twarc = Twarc()

# data source folder, data source sub-directory folder and lang_set settings
ZIP_TWEETS = False
raw_tweet_src = 'raw_tweet_ids/'
hydrated_tweets_target = 'hydrated_tweets/'
month_dirs = ['2020-01', '2020-02', '2020-03']
lang_set = set(["en", "null", None])


def main():
    for directory in month_dirs:
        data_dir = raw_tweet_src + directory
        os.makedirs(hydrated_tweets_target +
                    directory, exist_ok=True)
        for path in Path(data_dir).iterdir():
            if path.name.endswith('.txt'):
                if ZIP_TWEETS:
                    hydrate_and_zip(path)
                else:
                    hydrate(path)


def raw_newline_count(fname):
    """
    Counts number of lines in file
    """
    def _reader_generator(reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)

    f = open(fname, 'rb')
    f_gen = _reader_generator(f.raw.read)
    return sum(buf.count(b'\n') for buf in f_gen)


def hydrate(id_file):
    print('hydrating {}'.format(id_file))

    gzip_path = id_file.with_suffix('.jsonl')
    output_fname = Path(hydrated_tweets_target +
                        gzip_path.parts[-2] + '/' + gzip_path.parts[-1])

    if output_fname.is_file():
        print('skipping json file already exists: {}'.format(output_fname))
        return

    num_ids = raw_newline_count(id_file)

    with open(output_fname, 'w') as output:
        with tqdm(total=num_ids) as pbar:
            for tweet in twarc.hydrate(id_file.open()):
                if tweet["lang"] in lang_set:
                    output.write(json.dumps(tweet) + "\n")
                    pbar.update(1)


def hydrate_and_zip(id_file):
    print('hydrating {}'.format(id_file))

    gzip_path = id_file.with_suffix('.jsonl.gz')
    output_fname = Path(hydrated_tweets_target +
                        gzip_path.parts[-2] + '/' + gzip_path.parts[-1])

    if output_fname.is_file():
        print('skipping json file already exists: {}'.format(output_fname))
        return

    num_ids = raw_newline_count(id_file)

    with gzip.open(output_fname, 'w') as output:
        with tqdm(total=num_ids) as pbar:
            for tweet in twarc.hydrate(id_file.open()):
                if tweet["lang"] in lang_set:
                    output.write(json.dumps(tweet).encode('utf8') + b"\n")
                    pbar.update(1)


if __name__ == "__main__":
    main()
