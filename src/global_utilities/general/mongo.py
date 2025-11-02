

import sys
import os
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URI = os.getenv("MONGO_DB_URI")

def get_mongo_client(db_name="cannondale_bikes_db", collection_name="bikes_collection"):
    client = MongoClient(MONGO_DB_URI)
    db = db_name
    collection_name = collection_name
    collection = client[db][collection_name]
    return client, db, collection_name, collection


