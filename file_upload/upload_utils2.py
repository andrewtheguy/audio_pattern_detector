import json
import logging
import subprocess

#from webdav4.client import Client
logger = logging.getLogger(__name__)

#rclone_config_file = "./rclone.conf"
rclone_backend = "remote"

# https://github.com/rclone/rclone/blob/master/bin/config.py
# https://github.com/rclone/rclone/blob/2257c03391be526011743e2fe80fc7c6b1e23179/docs/content/docs.md for exit codes
def stat(remote_path):
    cmd = ["rclone", '--config', '/notfound', "lsjson", f"{rclone_backend}:{remote_path}"]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        # 3 - Directory not found, 4 - File not found
        if e.returncode in [3, 4]:
            return None
        raise e

def remote_exists(remote_path):
    return stat(remote_path) is not None

def download_file(remote_path,local_path):
    print("downloading",remote_path,"to",local_path)
    cmd = ["rclone", '--config', '/notfound', "copyto", f"{rclone_backend}:{remote_path}", local_path]
    subprocess.run(cmd, check=True)

def upload_file(file,dest_path,skip_if_exists=False):
    if skip_if_exists and stat(dest_path):
        print(f'upload_file: file {dest_path} already exists,skipping')
        return
    print("uploading",file,"to",dest_path)
    cmd = ["rclone","-v", '--config', '/notfound', "copyto", file, f"{rclone_backend}:{dest_path}"]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    upload_file("test data.txt","/chilo/chjafhtfhfga2/ggggg/chafa2.txt", skip_if_exists=True)