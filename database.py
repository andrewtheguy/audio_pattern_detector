
import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()  # take environment variables from .env.

uri = os.environ['MONGODB_URI']

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

def get_database():
    db = client.get_database('audio_offset_finder')
    return db
    #return client['audio_offset_finder_v1']    

db=get_database()

collection=db['audios_v1']

# Define the keys and index options
keys = [("show_name", 1), ("show_date", 1)]  # Replace key1 and key2 with your actual field names
index_options = {"unique": True}

# Create the index
collection.create_index(keys, **index_options)

if __name__ == '__main__':
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    collection.update_one(
        {"show_name": "test1", "show_date": "20220414"},
        {"$set": {"segments": [
    [
        "00:02:22",
        "00:10:04"
    ],
    [
        "00:19:11",
        "00:40:04"
    ],
    [
        "00:45:49",
        "01:06:04"
    ],
    [
        "01:16:11",
        "01:36:04"
    ]
]}},upsert=True
    )
