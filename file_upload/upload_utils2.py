# stackoverflow's approach to create directories in sftp
# https://stackoverflow.com/questions/14819681/upload-files-using-sftp-in-python-but-create-directories-if-path-doesnt-exist
# def is_sftp_dir_exists(sftp, path):
#     try:
#         sftp.stat(path)
#         return True
#     except Exception:
#         return False


# def create_sftp_dir(sftp, path):
#     try:
#         sftp.mkdir(path)
#     except IOError as exc:
#         if not is_sftp_dir_exists(sftp, path):
#             raise exc


# def create_sftp_dir_recursive(sftp, path):
#     parts = deque(Path(path).parts)

#     to_create = Path()
#     while parts:
#         to_create /= parts.popleft()
#         create_sftp_dir(sftp, str(to_create))

import logging
import os
import paramiko
from webdav4.client import Client
logger = logging.getLogger(__name__)

from upload_utils.sftp import create_remote_sftp_dir_recursively,file_exists

def upload_file(file,dest_path,skip_if_exists=False):
    # create ssh client 
    with paramiko.SSHClient() as ssh_client:
        # remote server credentials
        host = "10.22.33.20"
        username = "andrew"
        password = "qwertasdfg"
        port = '2022'
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=host,port=port,username=username,password=password, look_for_keys=False)

        # create an SFTP client object
        with ssh_client.open_sftp() as sftp:
            if skip_if_exists:
                try:
                    sftp.stat(dest_path)
                    logger.info(f'file {dest_path} already exists,skipping')
                    return
                except IOError:
                    logger.info(f"file {dest_path} doesn't exist, uploading")
                    #return
            logger.info(f"uploading {file} to {dest_path}")
            create_remote_sftp_dir_recursively(sftp_client=sftp, remote_dir=os.path.dirname(dest_path))
            sftp.put(file,dest_path)


def upload_file_with_webdav(file,dest_path,skip_if_exists=False):
    client = Client("http://10.22.33.20:9080", auth=("andrew", "qwertasdfg"))
    
    if(skip_if_exists and client.exists(dest_path)):
        logger.info(f"webdav: file {dest_path} already exists,skipping")
        return

    dir=os.path.dirname(dest_path)
    if not client.exists(dir):
        client.mkdir(dir)
    logger.info(f"uploading {file} to {dest_path}")
#    client.mkdir(dir)
    client.upload_file(file,dest_path,overwrite=True)

if __name__ == '__main__':
    with paramiko.SSHClient() as ssh_client:
        # remote server credentials
        host = "10.22.33.20"
        username = "andrew"
        password = "qwertasdfg"
        port = '2022'
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=host,port=port,username=username,password=password, look_for_keys=False)

        # create an SFTP client object
        with ssh_client.open_sftp() as sftp:
            #create_remote_sftp_dir_recursively(sftp_client=sftp, remote_dir="/chafa/trimmed/chilo")
            print(file_exists(sftp_client=sftp, remote_path="/chafa/trimmed/chilo"))