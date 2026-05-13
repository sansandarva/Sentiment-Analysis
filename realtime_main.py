from src.realtime.stream_tweets import fetch_live_tweets
import time

if __name__ == "__main__":
    print("🔴 Real-Time Twitter Stream Started...\n")

    while True:
        tweets = fetch_live_tweets(query="AI", max_results=5)

        for i, tweet in enumerate(tweets, 1):
            print(f"[{i}] {tweet}\n")

        print("-" * 50)
        time.sleep(15)  # wait 15 seconds before next fetch
