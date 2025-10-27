import pymongo
from dotenv import load_dotenv 
import os
load_dotenv()

mongo_url = os.getenv("MONGODB_URL")
mongo_db = os.getenv("MONGODB_DATABASE")

client = pymongo.MongoClient(mongo_url)
db = client[mongo_db]

col = db["crawl_data"]

x = col.find_one()

print(x)