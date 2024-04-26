import argparse
from datetime import datetime,timezone
import glob
import hashlib
import json
import os
import re
import boto3
from botocore.client import Config
import botocore
from jinja2 import Environment, FileSystemLoader, select_autoescape
from scrape import extract_prefix
from dotenv import load_dotenv
from andrew_utils import get_md5sum_file

load_dotenv()  # take environment variables from .env.

base_endpoint = "https://podcasts.andrewtheguy.workers.dev"

def publish_to_firebase(file,path):
 
    s3 = boto3.client('s3',
        endpoint_url='https://s3.filebase.com',
        aws_access_key_id=os.environ['FIREBASE_ACCESS_KEY'],
        aws_secret_access_key=os.environ['FIREBASE_SECRET_ACCESS_KEY'],
        region_name='us-east-1',
        config=boto3.session.Config(signature_version='s3v4'))
        
  
    bucket_name = "podcasts"

    # check if key exists
    try:
        response = s3.head_object(Bucket=bucket_name, Key=path)
        print(f"key exists, skip publishing")
        return response['ResponseMetadata']['HTTPHeaders']['x-amz-meta-cid']
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            key_exists=False
        else:
            raise  # Re-raise other errors

    with open(file, 'rb') as f:
        response = s3.put_object(Body=f, Bucket=bucket_name, Key=path)

    print(response)
    
    print(f"published to {path}")
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
  print(response.text)
  response.raise_for_status()
  return response

def publish_podcast(folder,title,inputs):

    last_build_date = None
    episodes = []
    for obj in inputs:
        #print(obj['date'])
        date = datetime.strptime(obj['date'],'%Y%m%d')
        #print('date',date)


        filename = os.path.split(os.path.basename(obj['file']))[-1]
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
    result_file = f"{folder}/results.json"
    results = []
    m4a_files = glob.glob(os.path.join(folder,"*.m4a"))
    if len(m4a_files)==0:
        raise ValueError("no m4a files found")
    dest_dir = os.path.basename(args.folder)
    for file in m4a_files:
        cid = publish_to_firebase(file,f"{dest_dir}/{os.path.basename(file)}")
        prefix,date=extract_prefix(os.path.basename(file))
        print(file)
        print(date)
        results.append({"cid":cid,"title":f"{prefix}{date}",
                        "date":date,
                        "hash_md5":get_md5sum_file(file),
                        "file_len":os.stat(file).st_size,
                        "file": file})
    results = sorted(results, key=lambda d: d['date'])   
    with open(result_file,"w") as f:
        json.dump(results,f)
    channel = dest_dir    
    publish_podcast(folder,channel,results)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')     
    args = parser.parse_args()
    publish(args.folder)
