import argparse
from datetime import datetime,timezone
import glob
import hashlib
import json
import logging
import os
import re
import boto3
from botocore.client import Config
import botocore
from jinja2 import Environment, FileSystemLoader, select_autoescape
from scrape import extract_prefix
from dotenv import load_dotenv
from andrew_utils import get_md5sum_file
logger = logging.getLogger(__name__)

load_dotenv()  # take environment variables from .env.

base_endpoint = "https://podcasts.andrewtheguy.workers.dev"

bucket_name = "podcasts"

s3 = boto3.client('s3',
    endpoint_url='https://s3.filebase.com',
    aws_access_key_id=os.environ['FIREBASE_ACCESS_KEY'],
    aws_secret_access_key=os.environ['FIREBASE_SECRET_ACCESS_KEY'],
    region_name='us-east-1',
    config=boto3.session.Config(signature_version='s3v4'))

def get_existing_object_metadata(s3,bucket_name, key):
    # check if key exists
    try:
        response = s3.head_object(Bucket=bucket_name, Key=key)
        logger.info(f"key exists, skip publishing")
        return response
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return None
        else:
            raise  # Re-raise other errors

def publish_to_firebase(file,path):

    # check if key exists

    response = get_existing_object_metadata(s3,bucket_name=bucket_name, key=path)
    if response:
        return response['ResponseMetadata']['HTTPHeaders']['x-amz-meta-cid']

    with open(file, 'rb') as f:
        response = s3.put_object(Body=f, Bucket=bucket_name, Key=path)

    logger.info(response)
    
    logger.info(f"published to {path}")
    return response['ResponseMetadata']['HTTPHeaders']['x-amz-meta-cid']

import requests

def upload_cloudflare(data):
  """
  Makes an API call using the requests library with JSON data and Bearer authentication.

  Args:
    url (str): The URL of the API endpoint.
    data (dict): The JSON data to send in the request body.
    bearer_token (str): The Bearer authentication token.

  Returns:
    requests.Response: The response object from the API call.
  """
  bearer_token = os.environ['PODCAST_UPLOAD_API_KEY']
  headers = {
      'Authorization': f'Bearer {bearer_token}',
      'Content-Type': 'application/json'  # Set Content-Type for JSON data
  }
  url=f"{base_endpoint}/upload"
  response = requests.post(url, headers=headers, json=data)
  logger.info(response.text)
  response.raise_for_status()
  return response

def publish_podcast(folder,title,inputs,dest_dir):

    last_build_date = None
    episodes = []
    for obj in inputs:
        #logger.info(obj['date'])
        date = datetime.strptime(obj['date'],'%Y%m%d')
        #logger.info('date',date)


        filename = os.path.basename(obj['file'])
        ext=os.path.splitext(filename)[-1]
        if ext.lower()=='.m4a':
            file_type = 'audio/m4a'
        else:
            file_type = 'application/octet-stream'

        enclosure = {'file_len': obj['file_len'], "file_type": file_type}

        if not last_build_date:
            last_build_date = date
        elif date > last_build_date:
            last_build_date = date


        datestr = date.strftime("%a, %d %b %Y %H:%M:%S %z")

        #filename, file_extension = os.path.splitext(obj['file'])

        link = 'https://ipfs.filebase.io/ipfs/'+obj['cid']


        episodes.append({
            'title': obj['title'],
            'link': link,
            'hash_md5': obj['hash_md5'],
            'author': "Various",
            'date': datestr,
            'enclosure': enclosure,
        })

    if not last_build_date:
         last_build_date = datetime.now(timezone.utc)

    #last_build_date = now

    env = Environment(
        loader=FileSystemLoader(os.path.dirname(os.path.realpath(__file__))),
        autoescape=select_autoescape(['html', 'xml', 'jinja'])
    )
    template = env.get_template("feed_template.xml.jinja")

    feed = template.render(channel={"title": title, "link": "https://www.andrewtheguy.com"}, episodes=episodes, last_build_date=last_build_date.strftime("%a, %d %b %Y %H:%M:%S %z"))
    feed_file = f'{folder}/feed.xml'
    with open(feed_file, 'w') as f:
        f.write(feed)
    # no need to be super secure for now    
    salt="fdsgdfgfdgfdgfdgdfdsfsd"
    title_clean=re.sub('[^0-9a-zA-Z]+', '*', title)
    suffix = hashlib.md5(f"{title_clean}{salt}".encode("utf-8")).hexdigest()
    remote_name = f"{title_clean}_{suffix}"
    data = {"key":remote_name,"xml":feed}
    upload_cloudflare(data)
    print(f"uploaded feed to cloudflare as {base_endpoint}/feeds/{remote_name}.xml")

    paginator = s3.get_paginator('list_objects_v2')

    file_del = []
    files_keep = [os.path.basename(obj['file']) for obj in inputs]
    for page in paginator.paginate(Bucket=bucket_name, Prefix=dest_dir):
        if "Contents" in page:
            for object_info in page["Contents"]:
                file_name = object_info["Key"]
                if file_name.endswith("/"):  # Filter out folders
                    continue
                check,_,f = file_name.rpartition('/')
                f=None if len(check)==0 else f
                if f not in files_keep:
                    file_del.append(file_name)

    for key in file_del:
        logger.info(f"deleting {key}")
        s3.delete_object(Bucket=bucket_name, Key=key)

def extract_folder(path):
  """
  Extracts folder from the given path, regardless of trailing slashes.

  Args:
    path (str): The path to extract from.

  Returns:
    str: folder
  """
  # Remove trailing slashes if present
  clean_path = path.rstrip('/')
  # Get the base name (last component) of the path
  basename = os.path.basename(clean_path)
  return basename


def publish(folder):
    folder = folder.rstrip('/')
    result_file = f"{folder}/results.json"
    results = []
    # only keep last 3
    m4a_files = sorted(glob.glob(os.path.join(folder,"*.m4a")))[-3:]
    if len(m4a_files)==0:
        raise ValueError("no m4a files found")
    dest_dir = extract_folder(folder)
    if len(dest_dir)==0:
        raise ValueError("folder name is empty")
    for file in m4a_files:
        cid = publish_to_firebase(file,f"{dest_dir}/{os.path.basename(file)}")
        prefix,date=extract_prefix(os.path.basename(file))
        logger.info(file)
        logger.info(date)
        results.append({"cid":cid,"title":f"{prefix}{date}",
                        "date":date,
                        "hash_md5":get_md5sum_file(file),
                        "file_len":os.stat(file).st_size,
                        "file": file})
    results = sorted(results, key=lambda d: d['date'])   
    with open(result_file,"w") as f:
        json.dump(results,f)
    channel = dest_dir    
    # only publish the last 3 episodes
    publish_podcast(folder,channel,results,dest_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')     
    args = parser.parse_args()
    publish(args.folder)
