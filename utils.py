import datetime


def second_to_time(seconds):
    return "{:0>8}".format(str(datetime.timedelta(seconds=seconds)))