import argparse
import logging
from apscheduler.schedulers.blocking import BlockingScheduler

import scrape_rthk
import scrape_standard

logging.basicConfig(level=logging.INFO)
#logging.getLogger('scrape').setLevel(logging.INFO)
logging.getLogger('apscheduler').setLevel(logging.INFO)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('podcast')
    args = parser.parse_args()
    scheduler = BlockingScheduler()
    if args.podcast == 'rthk':
        scheduler.add_job(scrape_rthk.process_podcasts, 'interval', hours=1)
        # add a job to run immediately
        scheduler.add_job(scrape_rthk.process_podcasts)
    elif args.podcast == 'am1430':
        scheduler.add_job(scrape_standard.process_podcasts, 'interval', hours=1)
        # add a job to run immediately
        scheduler.add_job(scrape_standard.process_podcasts)
    scheduler.start()