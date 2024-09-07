#Wikipedia example of using conjecture

import requests
from conjecture.Judge import Judge

WIKIPEDIA_PAGE_NAME = "YouTube"

response = requests.get(
     'https://en.wikipedia.org/w/api.php',
     params={
         'action': 'query',
         'format': 'json',
         'titles': WIKIPEDIA_PAGE_NAME,
         'prop': 'extracts',
         'exintro': True,
         'explaintext': True,
     }
 ).json()

page = next(iter(response['query']['pages'].values()))

data = page['extract']

judge = Judge("unsloth/mistral-7b-instruct-v0.3-bnb-4bit",data.split("."))

judge.assess()