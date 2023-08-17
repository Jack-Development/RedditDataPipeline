import os
from dotenv import load_dotenv
from dataset_manager import RedditDatasetManager
from RedditDataset import RedditDataset

# Load .env variables
load_dotenv()

# Your Reddit app's client ID, client secret, and a unique user agent string
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
USER_AGENT = os.getenv('USER_AGENT')

# Create an instance of the RedditDatasetManager class
dataset_manager = RedditDatasetManager(CLIENT_ID, CLIENT_SECRET, USER_AGENT, root_dir="reddit_data/datasets/RedditDataset")

# List of Reddit usernames you want to fetch data from
reddit_users = ['JohnDoee94']

# Fetch and save the latest 200 posts (or up to) from each of the listed usernames
dataset_manager.update_dataset(reddit_users)

# Usage
root_path = "./reddit_data"
dataset = RedditDataset(root_path, split="train")
#for user_id, text in dataset:
#	print(user_id, text)
