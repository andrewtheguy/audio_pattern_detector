from publish import upload_cloudflare

data = {"key": "test", "xml": '''<xml>test</xml>'''}
upload_cloudflare(data)