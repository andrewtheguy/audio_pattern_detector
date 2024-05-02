import re


# return a tuple of prefix and date
# happydaily20220430 will return ("happydaily","20220430")
# test220220430 will return ("test2","20220430")
def extract_prefix(text):
  match = re.match(r"(.*\d{8,})", text)
  return (match.group(1)[:-8],match.group(1)[-8:]) if match else (None,None)
