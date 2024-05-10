
import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from andrew_utils import get_md5sum_file

from utils import get_ffprobe_info

load_dotenv()  # take environment variables from .env.

uri = os.environ['MONGODB_URI']

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))


def get_database():
    db = client.get_database('audio_offset_finder')
    return db
    # return client['audio_offset_finder_v1']


def get_files_collection():
    db = get_database()
    return db['files_v1']


def get_segments_collection():
    db = get_database()
    return db['segments_v1']


db = get_database()


def setup_collections():
    segments_collection = get_segments_collection()

    # Define the keys and index options
    keys = [("show_name", 1), ("show_episode", 1)]
    index_options = {"unique": True}

    # Create the index
    segments_collection.create_index(keys, **index_options)

    # files
    files_collection = get_files_collection()

    # Define the keys and index options
    keys = [("md5", 1), ("file_size", 1)]
    index_options = {"unique": True}

    # Create the index
    files_collection.create_index(keys, **index_options)


def scan_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".m4a"):
                
                file_path=os.path.abspath(os.path.join(root, file))
                
                # Get the file size in bytes
                size_in_bytes = os.path.getsize(file_path)
                # Get the MD5 hash of the file
                md5_hash = get_md5sum_file(file_path)
                

                files_collection = get_files_collection()
                key = {"md5": md5_hash, "file_size": size_in_bytes}
                if(files_collection.find_one(key) is None):
                    metadata = get_ffprobe_info(file_path)
                    files_collection.update_one(
                        key,
                        {"$set": {"metadata": metadata}},
                        upsert=True
                    )
                files_collection.update_one(
                    key,
                    {'$addToSet': { 'files': file_path } }
                );

def save_timestamps_to_db(show_name, show_episode, segments):
    segments_collection = get_segments_collection()
    segments_collection.update_one(
        {"show_name": show_name, "show_episode": show_episode},
        {"$set": {"segments": segments},
        }, upsert=True
    )

def save_debug_info_to_db(show_name, show_episode, debug_info):
    segments_collection = get_segments_collection()
    segments_collection.update_one(
        {"show_name": show_name, "show_episode": show_episode},
        {"$set": {"debug_info": debug_info},
        }, upsert=True
    )

def find_episode_segments_in_db(show_name, show_episode):
    segments_collection = get_segments_collection()
    return segments_collection.find_one({"show_name": show_name, "show_episode": show_episode})

setup_collections()

if __name__ == '__main__':
    #scan_directory("./tmp")
    segments_collection = get_segments_collection()
    print(segments_collection.find_one({"show_name": "test1","show_episode":"20220414"}))
