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