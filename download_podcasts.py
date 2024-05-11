import argparse
import contextlib
import os
import sqlite3
import subprocess
import feedparser

onedrive_disallowed_chars = ['"', '*', ':', '<', '>', '?', '/', '\\', '|']

def download_file_with_curl(url, filename):

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
        guid=entry.guid
        for char in onedrive_disallowed_chars:
            title = title.replace(char, '_')
        found_one = None
        if 'enclosures' in entry:
            for enclosure in entry.enclosures:
                type = enclosure.type
                url = enclosure.url
                if type.startswith('audio/'):
                    #print(enclosure.url)
                    found_one = {'filename':f'{title}.{get_extension(type)}','url':url,'guid':guid}
                    break
        # elif 'links' in entry:
        #     for link in entry.links:
        #         type = link.type
        #         url = link.href
        #         if link.rel == 'enclosure' and type.startswith('audio/'):
        #             #print(link.href)
        #             episode_urls.append({'filename':f'{title}.{get_extension(type)}','url':url,'guid':guid})
        #             break
        if found_one:
            episode_urls.append(found_one)
        else:
            raise RuntimeError(f"Could not find an audio file in the entry: {entry}")
    return episode_urls

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rss-feed', required=True, type=str, help='rss feed')
    parser.add_argument('--dest-folder', required=True, type=str, help='rss file')
    args = parser.parse_args()
    entries=parse_feed(args.rss_feed)
    print(entries)
    #exit(0) 
    dest_folder = os.path.abspath(args.dest_folder)
    os.makedirs(dest_folder, exist_ok=True)
    with sqlite3.connect(os.path.join(dest_folder,"podcasts.db")) as conn:
        #c = conn.cursor()
        conn.execute('''CREATE TABLE if not exists podcast_tracker_v1
                    (guid text PRIMARY KEY,
                    done integer NOT NULL
                    )''')
        # adding sequence assumming old files don't get removed automatically
        # usually only for podcasts that stopped updating
        for i, entry in enumerate(reversed(entries)):
            filename = os.path.join(f"{dest_folder}",f"{str(i+1).zfill(3)}_{entry['filename']}")
            guid = entry['guid']

            #guid='https://podcast.rthk.hk/podcast/media/readyourmind/554_1312060911_96364.mp3'
            sqlite_select_query = """SELECT guid,done from podcast_tracker_v1 where guid = ?"""
            with contextlib.closing(conn.cursor()) as cursor:
                cursor.execute(sqlite_select_query,(guid,))
                print("Fetching single row")
                record = cursor.fetchone()
                if record:
                    guid=record[0]
                    print("record[1]",record[1])
                    done=record[1] != 0
                    if done:
                        print(f"Skipping {filename} because it's already downloaded")
                        continue
                print(f"Downloading {entry['url']} to {filename}")
                download_file_with_curl(entry['url'], filename)
                print(f"Downloaded {entry['url']}")
                conn.execute("INSERT OR IGNORE INTO podcast_tracker_v1 (guid,done) VALUES (?, ?)", (guid, True));  
                conn.commit()
   

