import os
from dotenv import load_dotenv
import tweepy

def get_client():
    # Force-load .env from PROJECT ROOT
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_path = os.path.join(base_dir, ".env")

    load_dotenv(env_path)

    bearer_token = os.getenv("BEARER_TOKEN")

    if not bearer_token:
        raise ValueError("Bearer token not found. Check .env file")

    return tweepy.Client(bearer_token=bearer_token)
