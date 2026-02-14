from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")

client = MongoClient(MONGO_URI)
db = client["MobileDB"]

conversations_col = db["conversations"]
messages_col = db["messages"]
