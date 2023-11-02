import requests
from bs4 import BeautifulSoup
import ray
import bittensor as bt
from db import VectorDBClient
from llm import MODEL_REGISTRY
from task import WebCrawler
import collections

# llm_name = "bert-base-uncased"
# llm_model = MODEL_REGISTRY[llm_name]()

# db_client = VectorDBClient(embed_size=llm_model.embed_size, batch_size=1)

initial_urls = ["https://cnn.com"]
max_depth = 2

# def crawl(url, depth, max_depth):
#     response = requests.get(url)
#     if response.status_code == 200:
#         # Parse the HTML content
#         soup = BeautifulSoup(response.text, "html.parser")

#         # Extract text from the webpage
#         full_text = soup.get_text()

#         # Generate BERT embeddings for the text
#         embeddings = llm_model.text_to_embedding(full_text)

#         # Insert data into Milvus
#         db_client.insert(url, full_text, embeddings)

#         # Find and crawl child links
#         # links = soup.find_all("a")
#         # for link in links:
#         #     child_url = link.get("href")
#         #     if child_url and child_url.startswith("http") or child_url and child_url.startswith("https"):
#         #         # print(child_url)
#         #         crawl(child_url, depth + 1, max_depth)

# crawl(inital_url, 0, 10)

# class Crawler:
#     def __init__(self):
#         self.initial_urls = ["https://cnn.com"]
#         self.max_depth = 2
    
#     def crawl(self, url, depth, max_depth, links_cache, llm_model, db_client):
#         if depth > max_depth:
#             return

#         response = requests.get(url)
#         if response.status_code == 200:
#             # Parse the HTML content
#             soup = BeautifulSoup(response.text, "html.parser")

#             # Extract text from the webpage
#             full_text = soup.get_text()

#             # Generate BERT embeddings for the text
#             embeddings = llm_model.text_to_embedding(full_text)

#             # Insert data into Milvus
#             db_client.insert(url, full_text, embeddings)

#             # Find and crawl child links
#             links = soup.find_all("a")
#             for link in links:
#                 child_url = link.get("href")
#                 if child_url and child_url.startswith("http") or child_url and child_url.startswith("https"):
#                     # print(child_url)
#                     # crawl(child_url, depth + 1, max_depth)


# llm_name = "bert-base-uncased"
# llm_model = MODEL_REGISTRY[llm_name]()

# db_client = VectorDBClient(embed_size=llm_model.embed_size, batch_size=1)

ray.init(address="auto")

# Instantiate Ray worker code
crawler = WebCrawler.remote()

# Use a set for deduplication
seen_urls = set()

# Use a deque as a queue
url_queue = collections.deque(initial_urls)


print("Starting to crawl...")
while url_queue:
    current_url = url_queue.popleft()
    if current_url in seen_urls:
        continue  # Skip this URL because it's already been crawled

    seen_urls.add(current_url)  # Mark this URL as seen
    new_links = ray.get(crawler.crawl.remote(current_url, 0, max_depth))

    # Process new links - deduplicate and add to the queue
    if new_links:
        for link in new_links:
            if link not in seen_urls:
                url_queue.append(link)
# print('new_links', new_links)
# # Wait for all tasks to complete
# print("Done crawling.")
# ray.shutdown()