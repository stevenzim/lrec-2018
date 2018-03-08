# lrec-2018
Herein are the code books to evaluate ensemble methods on Wasseem Hatespeech and Semeval Twitter datasets published at [lrec-2018](http://lrec2018.lrec-conf.org/en/).

# Citing work
Please use information in the following bibtex entry for citation, and be sure to cite other relevant works contained within.

```
@inproceedings{Zimmerman2018Improving,
  title={Improving Hate Speech Detection with Deep Learning Ensembles},
  author={Zimmerman, Steven and Fox, Chris and Kruschwitz, Udo},
  booktitle={Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2018) },
  year={2018},
}
```
# Requirements
1. You must install all requirements in requirements file
2. [You must download and unzip  the Godin word embeddings](http://yuca.test.iminds.be:8900/fgodin/downloads/word2vec_twitter_model.tar.gz) 
and store the 'word2vec_twitter_model.bin' file in the 'resources/pre_trained_models/' directory.
3. You will need to download all tweets into Semeval and Waseem json files via the Twitter API.  A useful library is [Tweepy](http://www.tweepy.org/)

+ Use the 'tweet_id' field in each JSON file to fetch the tweet text from the Twitter API
+ Store the text into corresponding 'text' field and then save to JSON file with same name and path
+ Waseem JSON file is found in: 'resources/hate_speech_corps'
+ Semeval JSON files (5 total) are found in: 'resources/sem_eval_tweets/SemEval'
+ NOTE: Twitter authors have the right to remove their posts from Twitter, as such, it is likely you will not retrieve all tweets in these files.  It is suggested that you remove any entries with tweets not fetched from the json files before running any code.

# License
MIT License

Copyright (c) 2018 Steven E. Zimmerman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
