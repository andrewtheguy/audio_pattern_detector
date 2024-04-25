import argparse
import glob
import os
import boto3
from botocore.client import Config

def test_bucket(bucket):
        s3 = boto3.resource('s3',
                             endpoint_url='https://s3.filebase.com',
        aws_access_key_id="6C2E7A9AFE5313A2F836",
        aws_secret_access_key="GyU3AVhIG8qRcT5j3i1BduKEC0S74OYMRBwg6Wdm",
                             config=Config(signature_version='s3v4'))
        b = s3.Bucket(bucket)
        objects = b.objects.all()
        for obj in objects:
            print(obj.key)

        

def publish_to_firebase(file,path):
 
    s3 = boto3.client('s3',
        endpoint_url='https://s3.filebase.com',
        aws_access_key_id="6C2E7A9AFE5313A2F836",
        aws_secret_access_key="GyU3AVhIG8qRcT5j3i1BduKEC0S74OYMRBwg6Wdm",
        config=boto3.session.Config(signature_version='s3v4'))
        
  
    bucket_name = "podcasts"

    with open(file, 'rb') as f:
        response = s3.put_object(Body=f, Bucket=bucket_name, Key=path)

    print(response)
    
    print(f"published to {path}")


def command():
     test_bucket("podcasts")
    #publish_to_firebase('/Users/it3/Documents/after.png',"podcasts/after.png")
    # parser = argparse.ArgumentParser()
    # parser.add_argument('folder')     
    # args = parser.parse_args()
    # m4a_files = glob.glob(f"{args.folder}/*.m4a")
    # dest_dir = os.path.basename(args.folder)
    # for file in m4a_files:
    #     publish_to_firebase(file,f"{dest_dir}/{os.path.basename(file)}")
    

    
if __name__ == '__main__':
    #print(url_ok("https://rthkaod3-vh.akamaihd.net/i/m4a/radio/archive/radio1/happydaily/m4a/20240417.m4a/index_0_a.m3u8"))
    
    #upload_file("./tmp/out.pcm","/test5/5.pcm",skip_if_exists=True)
    
    #exit(1)
    #pair=[]
    #process(pair)
    #print(pair)
    command()
