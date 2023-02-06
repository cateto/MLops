import requests
from fake_headers import Headers

url = "https://de-api.eco.astro.com.my/feed/api/v1/articles/405316?site=awani"
header = Headers().generate()
print(header)

output = requests.get(url, headers=header).text
print(output)