import argparse
import os
import subprocess
import feedparser

onedrive_disallowed_chars = ['"', '*', ':', '<', '>', '?', '/', '\\', '|']

def download_file_with_curl(url, filename):
  """Downloads a file using the curl command.

  Args:
    url: The URL of the file to download.
    filename: The local filename to save the file as.
  """
  try:
    # Construct the curl command
    curl_command = ["curl", "-o", filename, url]

    # Run the command and capture the output
    process = subprocess.run(
        curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Check for errors
    if process.returncode != 0:
      error_message = process.stderr.decode("utf-8").strip()
      raise RuntimeError(f"Curl command failed: {error_message}")

    print(f"File '{filename}' downloaded successfully using curl.")
  except (subprocess.CalledProcessError, RuntimeError) as e:
    print(f"Error downloading file: {e}")

def get_extension(type):
    if type in ['audio/mpeg', 'audio/mp3']:
        return 'mp3'
    else:
        raise ValueError(f"Unknown type: {type}")

def parse_feed(rss_feed):
    episode_urls = []
    feed = feedparser.parse(rss_feed)

    for entry in feed.entries:
        #print(entry.title)
        title = entry.title
        for char in onedrive_disallowed_chars:
            title = title.replace(char, '_')
        #print(title)
        #continue
        # Different feeds might use different keys for the audio URL.
        # Here are some common ones:
        if 'enclosures' in entry:
            for enclosure in entry.enclosures:
                type = enclosure.type
                url = enclosure.url
                if type.startswith('audio/'):
                    #print(enclosure.url)
                    episode_urls.append({'filename':f'{title}.{get_extension(type)}','url':url})
                    break
        elif 'links' in entry:
            for link in entry.links:
                type = link.type
                url = link.href
                if link.rel == 'enclosure' and type.startswith('audio/'):
                    #print(link.href)
                    episode_urls.append({'filename':f'{title}.{get_extension(type)}','url':url})
                    break

    return episode_urls

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rss-feed', required=True, type=str, help='rss feed')
    parser.add_argument('--dest-folder', required=True, type=str, help='rss file')
    args = parser.parse_args()
    entries=parse_feed(args.rss_feed)
    dest_folder = os.path.abspath(args.dest_folder)
    os.makedirs(dest_folder, exist_ok=True)
    # adding sequence assumming old files don't get removed automatically
    # usually only for podcasts that stopped updating
    for i, entry in enumerate(reversed(entries)):
        filename = os.path.join(f"{dest_folder}",f"{str(i+1).zfill(3)}_{entry['filename']}")
        print(f"Downloading {entry['url']} to {filename}")
        download_file_with_curl(entry['url'], filename)
        print(f"Downloaded {entry['url']}")

