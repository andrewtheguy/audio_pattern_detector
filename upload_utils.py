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

import os
import paramiko
from webdav4.client import Client

def create_remote_dir_recursively(sftp_client, remote_dir):
    """
    Creates a directory and its parents recursively on the SFTP server,
    always relative to the root directory.
    """
    if not isinstance(sftp_client,paramiko.SFTPClient):
        raise ValueError("sftp_client must be an instance of paramiko.SFTPClient")
    
    if remote_dir.startswith('/'):  # Handle absolute paths
        path_components = remote_dir.split('/')[1:]  # Split and remove leading '/'
    else:  # don't support path not starting with '/' to prevent mistakes
        raise ValueError("Only absolute paths are supported")
        #path_components = remote_dir.split('/')

    current_dir = '/'  # Start from the root directory
    for component in path_components:
        if component == '':  # Skip empty components
            continue
        current_dir = os.path.join(current_dir, component)
        try:
            sftp_client.stat(current_dir)
        except IOError:
            sftp_client.mkdir(current_dir)


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
                    print(f'file {dest_path} already exists,skipping')
                    return
                except IOError:
                    print(f"file {dest_path} doesn't exist, uploading")
                    #return
            print(f"uploading {file} to {dest_path}")
            create_remote_dir_recursively(sftp_client=sftp, remote_dir=os.path.dirname(dest_path))
            sftp.put(file,dest_path)


def upload_file_with_webdav(file,dest_path,skip_if_exists=False):
    client = Client("http://10.22.33.20:9080", auth=("andrew", "qwertasdfg"))
    
    if(skip_if_exists and client.exists(dest_path)):
        print(f"webdav: file {dest_path} already exists,skipping")
        return

    dir=os.path.dirname(dest_path)
    if not client.exists(dir):
        client.mkdir(dir)
    print(f"uploading {file} to {dest_path}")
#    client.mkdir(dir)
    client.upload_file(file,dest_path,overwrite=True)