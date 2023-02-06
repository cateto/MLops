import requests
from fake_headers import Headers

url = "https://www.sinardaily.my/article/189752/malaysia/national/forest-reserves-in-selangor-not-affected-by-ecrl-project"
header = Headers().generate()
print(header)

output = requests.get(url, headers=header).text
print(output)