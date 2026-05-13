from src.realtime.twitter_auth import get_client

def fetch_live_tweets(query="AI", max_results=10):
    client = get_client()

    response = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=["text", "lang"]
    )

    tweets = []

    if response.data:
        for tweet in response.data:
            if tweet.lang == "en":
                tweets.append(tweet.text)

    return tweets
