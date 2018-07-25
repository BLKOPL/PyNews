

```python
# 1.Each news sources has moderately negative sentiment at the time of analysis.
# 2.The most negative and most positive tweets occurred within the last 20 tweets.
# 3.It would seem that a majority of each news sources tweets' were graded with neutral compound sentiment (0).
```


```python
import tweepy
import json
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import time

from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)
```


```python
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
tweet_df = pd.DataFrame(columns = ["Source Account", "Tweet Text", "Date", "Tweets Ago", "Compound Sentiment", "Positive Sentiment", "Neutral Sentiment", "Negative Sentiment"])
```


```python
target_user = ["BBCNews", "CBSNews", "CNN", "FoxNews", "nytimes"]
indexcount = 0
comp_avg = []
```


```python
for user in target_user:
    public_tweets = api.user_timeline(user, count=100)
    tweetnumber = 0
    comp_list = []
    for tweet in public_tweets:
        search = tweet["text"]
        tweetdate = tweet["created_at"]
        compoundsent = analyzer.polarity_scores(search)["compound"]
        comp_list.append(analyzer.polarity_scores(search)["compound"])
        positivesent = analyzer.polarity_scores(search)["pos"]
        neutralsent = analyzer.polarity_scores(search)["neu"]
        negativesent = analyzer.polarity_scores(search)["neg"]
        tweet_df.set_value(indexcount, "Source Account", user)
        tweet_df.set_value(indexcount, "Tweet Text", search)
        tweet_df.set_value(indexcount, "Date", tweetdate)
        tweet_df.set_value(indexcount, "Tweets Ago", tweetnumber)
        tweet_df.set_value(indexcount, "Compound Sentiment", compoundsent)            
        tweet_df.set_value(indexcount, "Positive Sentiment", positivesent)
        tweet_df.set_value(indexcount, "Neutral Sentiment", neutralsent)
        tweet_df.set_value(indexcount, "Negative Sentiment", negativesent)
        indexcount = indexcount + 1
        tweetnumber = tweetnumber + 1
    comp_avg.append(np.mean(comp_list))
```

    /anaconda3/envs/PythonData/lib/python3.6/site-packages/ipykernel_launcher.py:13: FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead
      del sys.path[0]
    /anaconda3/envs/PythonData/lib/python3.6/site-packages/ipykernel_launcher.py:14: FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead
      
    /anaconda3/envs/PythonData/lib/python3.6/site-packages/ipykernel_launcher.py:15: FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead
      from ipykernel import kernelapp as app
    /anaconda3/envs/PythonData/lib/python3.6/site-packages/ipykernel_launcher.py:16: FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead
      app.launch_new_instance()
    /anaconda3/envs/PythonData/lib/python3.6/site-packages/ipykernel_launcher.py:17: FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead
    /anaconda3/envs/PythonData/lib/python3.6/site-packages/ipykernel_launcher.py:18: FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead
    /anaconda3/envs/PythonData/lib/python3.6/site-packages/ipykernel_launcher.py:19: FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead
    /anaconda3/envs/PythonData/lib/python3.6/site-packages/ipykernel_launcher.py:20: FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead



```python
comp_avg

```




    [-0.132012, -0.113847, 0.0036819999999999947, 0.029291, -0.03533699999999998]




```python
tweet_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Source Account</th>
      <th>Tweet Text</th>
      <th>Date</th>
      <th>Tweets Ago</th>
      <th>Compound Sentiment</th>
      <th>Positive Sentiment</th>
      <th>Neutral Sentiment</th>
      <th>Negative Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBCNews</td>
      <td>RT @BBCSport: Ex-Middlesex and England wicketk...</td>
      <td>Wed Jul 25 15:02:44 +0000 2018</td>
      <td>0</td>
      <td>-0.5574</td>
      <td>0</td>
      <td>0.833</td>
      <td>0.167</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BBCNews</td>
      <td>RT @BBCSport: Current situation... 👀\n\n#tourd...</td>
      <td>Wed Jul 25 14:39:59 +0000 2018</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BBCNews</td>
      <td>Lost wedding ring found on cricket pitch 52 ye...</td>
      <td>Wed Jul 25 14:36:04 +0000 2018</td>
      <td>2</td>
      <td>-0.3182</td>
      <td>0</td>
      <td>0.813</td>
      <td>0.187</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BBCNews</td>
      <td>Liquid water 'lake' revealed on Mars https://t...</td>
      <td>Wed Jul 25 14:06:49 +0000 2018</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BBCNews</td>
      <td>Kylie Jenner and David Beckham make Instagram ...</td>
      <td>Wed Jul 25 13:58:23 +0000 2018</td>
      <td>4</td>
      <td>0.5574</td>
      <td>0.286</td>
      <td>0.714</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BBCNews</td>
      <td>Badger cub rescued after 90ft Cornwall cliff p...</td>
      <td>Wed Jul 25 13:07:23 +0000 2018</td>
      <td>5</td>
      <td>0.4215</td>
      <td>0.259</td>
      <td>0.741</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BBCNews</td>
      <td>RT @BBCBusiness: Just under an hour to go unti...</td>
      <td>Wed Jul 25 12:15:48 +0000 2018</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BBCNews</td>
      <td>RT @BBC_HaveYourSay: Remember the 1976 heatwav...</td>
      <td>Wed Jul 25 11:59:23 +0000 2018</td>
      <td>7</td>
      <td>0.6369</td>
      <td>0.178</td>
      <td>0.822</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BBCNews</td>
      <td>Demi Lovato: Suspected overdose follows long b...</td>
      <td>Wed Jul 25 11:35:09 +0000 2018</td>
      <td>8</td>
      <td>-0.5423</td>
      <td>0</td>
      <td>0.667</td>
      <td>0.333</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BBCNews</td>
      <td>RT @bbckamal: Saving - and why we're not doing...</td>
      <td>Wed Jul 25 11:31:36 +0000 2018</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BBCNews</td>
      <td>"A lot of people are talking up inappropriatel...</td>
      <td>Wed Jul 25 11:31:31 +0000 2018</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BBCNews</td>
      <td>Sacha Baron Cohen ridicule prompts Georgia law...</td>
      <td>Wed Jul 25 11:15:18 +0000 2018</td>
      <td>11</td>
      <td>-0.4588</td>
      <td>0</td>
      <td>0.75</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BBCNews</td>
      <td>RT @BBCScienceNews: Warning over suntan lotion...</td>
      <td>Wed Jul 25 11:14:47 +0000 2018</td>
      <td>12</td>
      <td>-0.34</td>
      <td>0</td>
      <td>0.745</td>
      <td>0.255</td>
    </tr>
    <tr>
      <th>13</th>
      <td>BBCNews</td>
      <td>Man Utd boss Jose Mourinho avoids Premier Leag...</td>
      <td>Wed Jul 25 10:38:48 +0000 2018</td>
      <td>13</td>
      <td>-0.1779</td>
      <td>0</td>
      <td>0.855</td>
      <td>0.145</td>
    </tr>
    <tr>
      <th>14</th>
      <td>BBCNews</td>
      <td>RT @bbcweather: This hot spell has spurred som...</td>
      <td>Wed Jul 25 10:35:36 +0000 2018</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BBCNews</td>
      <td>Aston Hall: Psychiatric doctor in rape and cru...</td>
      <td>Wed Jul 25 10:05:48 +0000 2018</td>
      <td>15</td>
      <td>-0.8625</td>
      <td>0</td>
      <td>0.482</td>
      <td>0.518</td>
    </tr>
    <tr>
      <th>16</th>
      <td>BBCNews</td>
      <td>Swedish activist stops deportation of Afghan m...</td>
      <td>Wed Jul 25 09:57:58 +0000 2018</td>
      <td>16</td>
      <td>-0.1531</td>
      <td>0</td>
      <td>0.814</td>
      <td>0.186</td>
    </tr>
    <tr>
      <th>17</th>
      <td>BBCNews</td>
      <td>Heinz baked beans TV advert banned for second ...</td>
      <td>Wed Jul 25 09:52:25 +0000 2018</td>
      <td>17</td>
      <td>-0.4588</td>
      <td>0</td>
      <td>0.75</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>18</th>
      <td>BBCNews</td>
      <td>Tini Owens loses Supreme Court divorce fight h...</td>
      <td>Wed Jul 25 09:13:03 +0000 2018</td>
      <td>18</td>
      <td>-0.0772</td>
      <td>0.267</td>
      <td>0.37</td>
      <td>0.363</td>
    </tr>
    <tr>
      <th>19</th>
      <td>BBCNews</td>
      <td>ITV profits rise helped by Love Island https:/...</td>
      <td>Wed Jul 25 08:42:50 +0000 2018</td>
      <td>19</td>
      <td>0.7964</td>
      <td>0.542</td>
      <td>0.458</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>BBCNews</td>
      <td>Greece wildfires: British man in hospital with...</td>
      <td>Wed Jul 25 08:27:33 +0000 2018</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>BBCNews</td>
      <td>Pollwatch: The Chequers Brexit effect https://...</td>
      <td>Wed Jul 25 08:11:45 +0000 2018</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>BBCNews</td>
      <td>RT @BBCSport: The height of a legal tackle in ...</td>
      <td>Wed Jul 25 07:52:29 +0000 2018</td>
      <td>22</td>
      <td>0.4215</td>
      <td>0.155</td>
      <td>0.791</td>
      <td>0.054</td>
    </tr>
    <tr>
      <th>23</th>
      <td>BBCNews</td>
      <td>RT @BBCSport: He called time on his 50-year BB...</td>
      <td>Wed Jul 25 07:52:21 +0000 2018</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>BBCNews</td>
      <td>RT @BBCBenThompson: Could San Francisco ban wo...</td>
      <td>Wed Jul 25 07:18:57 +0000 2018</td>
      <td>24</td>
      <td>-0.2263</td>
      <td>0.111</td>
      <td>0.741</td>
      <td>0.148</td>
    </tr>
    <tr>
      <th>25</th>
      <td>BBCNews</td>
      <td>Tape reveals Trump and lawyer discussing payof...</td>
      <td>Wed Jul 25 07:17:26 +0000 2018</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>BBCNews</td>
      <td>RT @BBCBusiness: ITV profits rise helped by Lo...</td>
      <td>Wed Jul 25 07:11:38 +0000 2018</td>
      <td>26</td>
      <td>0.7964</td>
      <td>0.47</td>
      <td>0.53</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>BBCNews</td>
      <td>Louise Brown: World's first IVF baby's family ...</td>
      <td>Wed Jul 25 06:58:33 +0000 2018</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>BBCNews</td>
      <td>New evidence reveals Goodwin Sands shipwreck's...</td>
      <td>Wed Jul 25 06:56:06 +0000 2018</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>BBCNews</td>
      <td>Dozens dead in southern Syria suicide attacks ...</td>
      <td>Wed Jul 25 06:11:42 +0000 2018</td>
      <td>29</td>
      <td>-0.9136</td>
      <td>0</td>
      <td>0.299</td>
      <td>0.701</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>nytimes</td>
      <td>Evening Briefing: Here's what you need to know...</td>
      <td>Tue Jul 24 22:47:02 +0000 2018</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>471</th>
      <td>nytimes</td>
      <td>An Alabama coal executive and a lawyer were co...</td>
      <td>Tue Jul 24 22:32:03 +0000 2018</td>
      <td>71</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>472</th>
      <td>nytimes</td>
      <td>RT @NYTmag: Ticks are not new. Neither is the ...</td>
      <td>Tue Jul 24 22:18:06 +0000 2018</td>
      <td>72</td>
      <td>-0.4019</td>
      <td>0</td>
      <td>0.899</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>473</th>
      <td>nytimes</td>
      <td>The IMF now says Venezuela inflation could rea...</td>
      <td>Tue Jul 24 22:03:03 +0000 2018</td>
      <td>73</td>
      <td>0.0258</td>
      <td>0.073</td>
      <td>0.927</td>
      <td>0</td>
    </tr>
    <tr>
      <th>474</th>
      <td>nytimes</td>
      <td>RT @nytimesarts: Just before she wrote "Nanett...</td>
      <td>Tue Jul 24 21:53:02 +0000 2018</td>
      <td>74</td>
      <td>0.3612</td>
      <td>0.111</td>
      <td>0.889</td>
      <td>0</td>
    </tr>
    <tr>
      <th>475</th>
      <td>nytimes</td>
      <td>Montana's Democratic governor is suing the Tru...</td>
      <td>Tue Jul 24 21:37:06 +0000 2018</td>
      <td>75</td>
      <td>-0.5423</td>
      <td>0</td>
      <td>0.791</td>
      <td>0.209</td>
    </tr>
    <tr>
      <th>476</th>
      <td>nytimes</td>
      <td>RT @sheeraf: For a brief moment, a Facebook su...</td>
      <td>Tue Jul 24 21:23:03 +0000 2018</td>
      <td>76</td>
      <td>-0.2263</td>
      <td>0</td>
      <td>0.905</td>
      <td>0.095</td>
    </tr>
    <tr>
      <th>477</th>
      <td>nytimes</td>
      <td>Karlie Kloss, girl-coding evangelist, and Josh...</td>
      <td>Tue Jul 24 21:13:01 +0000 2018</td>
      <td>77</td>
      <td>0.4588</td>
      <td>0.167</td>
      <td>0.833</td>
      <td>0</td>
    </tr>
    <tr>
      <th>478</th>
      <td>nytimes</td>
      <td>RT @julieturkewitz: Ben Stapleton was an influ...</td>
      <td>Tue Jul 24 21:03:05 +0000 2018</td>
      <td>78</td>
      <td>0.4404</td>
      <td>0.121</td>
      <td>0.879</td>
      <td>0</td>
    </tr>
    <tr>
      <th>479</th>
      <td>nytimes</td>
      <td>RT @nytopinion: I may have multiple sclerosis,...</td>
      <td>Tue Jul 24 20:43:03 +0000 2018</td>
      <td>79</td>
      <td>0.3252</td>
      <td>0.092</td>
      <td>0.908</td>
      <td>0</td>
    </tr>
    <tr>
      <th>480</th>
      <td>nytimes</td>
      <td>Two people fell ill after eating now-recalled ...</td>
      <td>Tue Jul 24 20:33:04 +0000 2018</td>
      <td>80</td>
      <td>-0.4215</td>
      <td>0</td>
      <td>0.859</td>
      <td>0.141</td>
    </tr>
    <tr>
      <th>481</th>
      <td>nytimes</td>
      <td>A Pennsylvania state board has determined that...</td>
      <td>Tue Jul 24 20:23:05 +0000 2018</td>
      <td>81</td>
      <td>-0.0258</td>
      <td>0.211</td>
      <td>0.617</td>
      <td>0.172</td>
    </tr>
    <tr>
      <th>482</th>
      <td>nytimes</td>
      <td>RT @jswatz: Justice might be coming closer for...</td>
      <td>Tue Jul 24 20:13:02 +0000 2018</td>
      <td>82</td>
      <td>0.9223</td>
      <td>0.424</td>
      <td>0.576</td>
      <td>0</td>
    </tr>
    <tr>
      <th>483</th>
      <td>nytimes</td>
      <td>Even as Rick Gates was emerging as a subject o...</td>
      <td>Tue Jul 24 20:08:02 +0000 2018</td>
      <td>83</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>484</th>
      <td>nytimes</td>
      <td>We’re about to take you on an adventure throug...</td>
      <td>Tue Jul 24 20:03:06 +0000 2018</td>
      <td>84</td>
      <td>0.3182</td>
      <td>0.113</td>
      <td>0.887</td>
      <td>0</td>
    </tr>
    <tr>
      <th>485</th>
      <td>nytimes</td>
      <td>Millions of smart TVs in American homes are tr...</td>
      <td>Tue Jul 24 19:58:04 +0000 2018</td>
      <td>85</td>
      <td>0.4019</td>
      <td>0.119</td>
      <td>0.881</td>
      <td>0</td>
    </tr>
    <tr>
      <th>486</th>
      <td>nytimes</td>
      <td>Elliott Broidy, a top fund-raiser for Presiden...</td>
      <td>Tue Jul 24 19:52:03 +0000 2018</td>
      <td>86</td>
      <td>0.2023</td>
      <td>0.087</td>
      <td>0.913</td>
      <td>0</td>
    </tr>
    <tr>
      <th>487</th>
      <td>nytimes</td>
      <td>The Ebola outbreak that began in the Democrati...</td>
      <td>Tue Jul 24 19:43:03 +0000 2018</td>
      <td>87</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>488</th>
      <td>nytimes</td>
      <td>RT @NYTmag: James thought he was OK after the ...</td>
      <td>Tue Jul 24 19:39:24 +0000 2018</td>
      <td>88</td>
      <td>0.4466</td>
      <td>0.105</td>
      <td>0.895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>489</th>
      <td>nytimes</td>
      <td>"As a man who anticipates that my partner will...</td>
      <td>Tue Jul 24 19:33:01 +0000 2018</td>
      <td>89</td>
      <td>-0.2411</td>
      <td>0</td>
      <td>0.918</td>
      <td>0.082</td>
    </tr>
    <tr>
      <th>490</th>
      <td>nytimes</td>
      <td>Ivanka Trump, on shutting down her fashion bra...</td>
      <td>Tue Jul 24 19:31:08 +0000 2018</td>
      <td>90</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>491</th>
      <td>nytimes</td>
      <td>A police officer fired a bullet that killed a ...</td>
      <td>Tue Jul 24 19:23:05 +0000 2018</td>
      <td>91</td>
      <td>-0.8442</td>
      <td>0</td>
      <td>0.649</td>
      <td>0.351</td>
    </tr>
    <tr>
      <th>492</th>
      <td>nytimes</td>
      <td>RT @nytfood: Why has New York been inundated w...</td>
      <td>Tue Jul 24 19:13:03 +0000 2018</td>
      <td>92</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>493</th>
      <td>nytimes</td>
      <td>RT @alexburnsNYT: News: Montana @GovernorBullo...</td>
      <td>Tue Jul 24 19:01:41 +0000 2018</td>
      <td>93</td>
      <td>-0.5719</td>
      <td>0</td>
      <td>0.81</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>494</th>
      <td>nytimes</td>
      <td>Wildfires in Greece have consumed entire towns...</td>
      <td>Tue Jul 24 18:53:06 +0000 2018</td>
      <td>94</td>
      <td>-0.8481</td>
      <td>0</td>
      <td>0.709</td>
      <td>0.291</td>
    </tr>
    <tr>
      <th>495</th>
      <td>nytimes</td>
      <td>RT @nytimesbusiness: China appears to be using...</td>
      <td>Tue Jul 24 18:43:02 +0000 2018</td>
      <td>95</td>
      <td>0.0516</td>
      <td>0.098</td>
      <td>0.813</td>
      <td>0.089</td>
    </tr>
    <tr>
      <th>496</th>
      <td>nytimes</td>
      <td>The comedy-destroying, soul-affirming art of H...</td>
      <td>Tue Jul 24 18:33:06 +0000 2018</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>497</th>
      <td>nytimes</td>
      <td>Jeff Sessions repeated the phrase "lock her up...</td>
      <td>Tue Jul 24 18:23:06 +0000 2018</td>
      <td>97</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>498</th>
      <td>nytimes</td>
      <td>Review: You’ll have plenty to talk about after...</td>
      <td>Tue Jul 24 18:13:02 +0000 2018</td>
      <td>98</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>499</th>
      <td>nytimes</td>
      <td>Thousands of people were supposed to start rec...</td>
      <td>Tue Jul 24 18:03:05 +0000 2018</td>
      <td>99</td>
      <td>0.2263</td>
      <td>0.095</td>
      <td>0.905</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 8 columns</p>
</div>




```python

#export to csv for final file#export 
tweet_df.to_csv("newstweets")
```


```python
x_axis = (100, 0, 1)
bbcplot = plt.scatter(tweet_df[tweet_df["Source Account"] == "BBCNews"]["Tweets Ago"], tweet_df[tweet_df["Source Account"] == "BBCNews"]["Compound Sentiment"], marker="o", c="turquoise", edgecolors="black", linewidth=1 ,s=125, alpha=1, label="BBC")
cbsplot = plt.scatter(tweet_df[tweet_df["Source Account"] == "CBSNews"]["Tweets Ago"], tweet_df[tweet_df["Source Account"] == "CBSNews"]["Compound Sentiment"], marker="o", facecolors="green", edgecolors="black", linewidth=1, s=125, alpha=1, label="CBS")
cnnplot = plt.scatter(tweet_df[tweet_df["Source Account"] == "CNN"]["Tweets Ago"], tweet_df[tweet_df["Source Account"] == "CNN"]["Compound Sentiment"], marker="o", facecolors="red", edgecolors="black", linewidth=1,s=125, alpha=1, label="CNN")
foxplot = plt.scatter(tweet_df[tweet_df["Source Account"] == "FoxNews"]["Tweets Ago"], tweet_df[tweet_df["Source Account"] == "FoxNews"]["Compound Sentiment"], marker="o", facecolors="blue", edgecolors="black", linewidth=1,s=125, alpha=1, label="Fox")
nytplot = plt.scatter(tweet_df[tweet_df["Source Account"] == "nytimes"]["Tweets Ago"], tweet_df[tweet_df["Source Account"] == "nytimes"]["Compound Sentiment"], marker="o", facecolors="yellow", edgecolors="black", linewidth=1,s=125, alpha=1, label="New York Times")
leg = plt.legend(title="Media Sources",fontsize=12, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.setp(leg.get_title(),fontsize=14)
bbox_inches="tight"
plt.title("Sentiment Analysis of Media Tweets " + str(time.strftime("%x")), fontsize=16)
plt.xlabel("Tweets Ago", fontsize=14)
plt.ylabel("Tweet Polarity", fontsize=14)
plt.ylim(-1.1, 1.1)
plt.xlim(105, -5)
plt.savefig("scatter_analysis.png")
plt.show()

```


![png](output_9_0.png)



```python
#tweet_df.groupby('Source Account')["Compound Sentiment"].mean()
tweet_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 500 entries, 0 to 499
    Data columns (total 8 columns):
    Source Account        500 non-null object
    Tweet Text            500 non-null object
    Date                  500 non-null object
    Tweets Ago            500 non-null object
    Compound Sentiment    500 non-null object
    Positive Sentiment    500 non-null object
    Neutral Sentiment     500 non-null object
    Negative Sentiment    500 non-null object
    dtypes: object(8)
    memory usage: 55.2+ KB



```python
plt.figure(figsize=(10,7))
x_axis2 = np.arange(len(target_user))

rect1 = plt.bar(0, comp_avg[0], color='turquoise', alpha=1, align="edge", ec="black", width=1)
rect2 = plt.bar(1, comp_avg[1], color='green', alpha=1, align="edge", ec="black", width=1)
rect3 = plt.bar(2, comp_avg[2], color='red', alpha=1, align="edge", ec="black", width=1)
rect4 = plt.bar(3, comp_avg[3], color='blue', alpha=1, align="edge", ec="black", width=1)
rect5 = plt.bar(4, comp_avg[4], color='yellow', alpha=1, align="edge", ec="black", width=1)

tick_locations = [value+0.5 for value in x_axis2]
plt.grid(linestyle="dashed")
plt.xticks(tick_locations, target_user)
plt.xlim(0, 5)
plt.ylim(-.2, .05)

plt.title("Overall Media Sentiment Based on Twitter " + str(time.strftime("%x")), fontsize=20)
plt.ylabel("Tweet Polarity")

def label_negative(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., (height)-0.01,
                '-%.2f' % float(height),
                ha='center', va='bottom')


label_negative(rect1)
label_negative(rect2)
label_negative(rect3)
label_negative(rect4)
label_negative(rect5)

plt.savefig("bar_analysis.png")

plt.show()
```


![png](output_11_0.png)

