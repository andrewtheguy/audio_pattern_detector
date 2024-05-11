from scrape import get_sec

print(get_sec('1:00:00')) # 3600

print(get_sec('0:00:01')) # 1

print(get_sec('0:00:01.006')) # 1.006

print(get_sec('0:00:01.006')) # 1.006
