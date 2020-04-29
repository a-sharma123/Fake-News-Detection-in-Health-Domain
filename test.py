import json
from textblob import TextBlob
import re
finput1 = open('./FakeHealth/dataset/engagements/HealthRelease.json', 'r')
finput2 = open('./FakeHealth/dataset/reviews/HealthRelease.json', 'r')
finput3 = open('./FakeHealth/dataset/content/HealthRelease/news_reviews_00000.json')

o = json.load(finput3)
for k, v in o.items():
    print(k,':',v)


