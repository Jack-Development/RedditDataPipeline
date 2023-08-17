import praw
import prawcore

import torchtext
from torchtext.data.utils import get_tokenizer

import os
import json

class RedditDatasetManager:
	def __init__(self, client_id, client_secret, user_agent, root_dir='reddit_data'):
		# Authenticate to Reddit API
		self.reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

		self.user_dict = {}
		self.current_id = 0
		self.root_dir = root_dir
		self.posts_data = []

		if not os.path.exists(self.root_dir):
			os.makedirs(self.root_dir)

		if os.path.exists(os.path.join(self.root_dir, "UserDictionary.json")):
			self.load_data()

	def save_data(self):
		# Save user_dict
		with open(os.path.join(self.root_dir, "UserDictionary.json"), "w") as outfile:
			json.dump(self.user_dict, outfile)

		# Saving posts and comments
		for data_id, data_text in self.posts_data:
			user_id = self.fetch_user_id_by_post_id(data_id)
			user_folder = os.path.join(self.root_dir, 'train', str(user_id))
			os.makedirs(user_folder, exist_ok=True)
			
			# Distinguish between comments and posts
			if data_id.startswith("c_"):
				data_file_path = os.path.join(user_folder, f"comment_{data_id[2:]}.txt")
			else:
				data_file_path = os.path.join(user_folder, f"post_{data_id}.txt")

			# Save if not exists
			if not os.path.exists(data_file_path):
				with open(data_file_path, "w") as data_file:
					data_file.write(data_text)

	def load_data(self):
		with open(os.path.join(self.root_dir, "UserDictionary.json"), "r") as infile:
			self.user_dict = json.load(infile)

		# Update current_id
		self.current_id = max([user_data["id"] for user_data in self.user_dict.values()]) + 1

		# Load posts and comments
		for username, user_data in self.user_dict.items():
			user_folder = os.path.join(self.root_dir, 'train', str(user_data["id"]))
			if os.path.exists(user_folder):
				for file_name in os.listdir(user_folder):
					if file_name.startswith(("post_", "comment_")):
						data_id = file_name.split("_")[1].split(".")[0]
						if file_name.startswith("comment_"):
							data_id = "c_" + data_id
						with open(os.path.join(user_folder, file_name), 'r') as data_file:
							data_text = data_file.read()
							self.posts_data.append((data_id, data_text))

	def fetch_user_id(self, username):
		return self.user_dict.get(username, {}).get("id", None)

	def assign_id_to_user(self, username, data_id):
		if username not in self.user_dict:
			self.user_dict[username] = {"id": self.current_id, "posts": [data_id]}
			self.current_id += 1
		else:
			if data_id not in self.user_dict[username]["posts"]:
				self.user_dict[username]["posts"].append(data_id)
		return self.user_dict[username]["id"]

	def fetch_user_id_by_post_id(self, data_id):
		for user, data in self.user_dict.items():
			if data_id in data["posts"]:
				return data["id"]
		return None

	def update_dataset(self, usernames):
		for username in usernames:
			user_id = self.fetch_user_id(username)
			if user_id is None:
				user_id = self.assign_id_to_user(username, "placeholder")  # placeholder

			redditor = self.reddit.redditor(username)

			try:
				# Posts
				for post in redditor.submissions.new(limit=200):
					combined_text = post.title + " " + post.selftext  # Adding a space between the title and the selftext for clarity.
					if combined_text.strip() and not any(x[0] == post.id for x in self.posts_data):  # Check if combined_text has content
						self.posts_data.append((post.id, combined_text))
						self.assign_id_to_user(username, post.id)

				# Comments
				for comment in redditor.comments.new(limit=200):
					comment_id = "c_" + comment.id
					if comment.body and not any(x[0] == comment_id for x in self.posts_data):  # Check if comment has text
						self.posts_data.append((comment_id, comment.body))
						self.assign_id_to_user(username, comment_id)
			
			except prawcore.exceptions.NotFound:  # Catch the NotFound exception
				print(f"Username {username} not found. Skipping to next user.")
				continue  # Skip the current user and move to the next one
			except prawcore.exceptions.RequestException as e:  # Handle other request errors
				print(f"Unable to access posts and comments for username {username} due to {e}. Skipping to next user.")
				continue  # Skip the current user and move to the next one


		# Save collected data
		self.save_data()
