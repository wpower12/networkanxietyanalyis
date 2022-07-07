import snscrape.modules.twitter as sntwitter
import pandas as pd

SU_QUERY = 'from:jack'  # Specific User
DR_QUERY = 'its the elephant since:2020-06-01 until:2020-07-31'  # Generic Text, Date Range
SU_DR_QUERY = 'from:jack since:2020-06-01 until:2020-07-31'  # Specific User, and Date Range.

tweets_list1 = []
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(SU_QUERY).get_items()):
    if i > 50:
        break
    tweets_list1.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
tweets_df1 = pd.DataFrame(tweets_list1, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

tweets_list2 = []
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(DR_QUERY).get_items()):
    if i > 50:
        break
    tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

tweets_list3 = []
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(SU_DR_QUERY).get_items()):
    if i > 50:
        break
    tweets_list3.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
tweets_df3 = pd.DataFrame(tweets_list3, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])


print(tweets_df1.head())
print(tweets_df2.head())
print(tweets_df3.head())
