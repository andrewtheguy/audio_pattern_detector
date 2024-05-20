import logging
from apscheduler.schedulers.blocking import BlockingScheduler

from scrape_rthk import download_and_scrape

logging.basicConfig(level=logging.INFO)
#logging.getLogger('scrape').setLevel(logging.INFO)
logging.getLogger('apscheduler').setLevel(logging.INFO)
if __name__ == '__main__':
    scheduler = BlockingScheduler()
    scheduler.add_job(download_and_scrape, 'interval', hours=1)
    # add a job to run immediately
    scheduler.add_job(download_and_scrape)
    scheduler.start()