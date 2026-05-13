from src.realtime.simulate_stream import simulate_real_time

if __name__ == "__main__":
    simulate_real_time(
        file_path="data/realtime_tweets.csv",
        delay=2
    )
