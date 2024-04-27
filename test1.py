

from scrape import extract_prefix


print(extract_prefix("testagain2022041"))  # None
print(extract_prefix("testagain20220414"))  # returns ('testagain', '20220414')
print(extract_prefix("testagain220220414"))  # returns ('testagain2', '20220414')
print(extract_prefix("testagain220220414hahanada"))  # returns ('testagain2', '20220414')