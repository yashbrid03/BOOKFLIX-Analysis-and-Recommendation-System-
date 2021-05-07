import pandas as pd
import numpy as np
from textblob import TextBlob
import re

df = pd.read_csv("Reviews.csv")

# Cleaning the reviews
# Creating a function called clean. removing hyperlink, #, RT, @mentions


def clean(x):
    x = re.sub(r'^RT[\s]+', '', x)
    x = re.sub(r'https?:\/\/.*[\r\n]*', '', x)
    x = re.sub(r'#', '', x)
    x = re.sub(r'@[A-Za-z0-9]+', '', x)
    return x


df['ReviewContent'] = df['ReviewContent'].apply(clean)

# Calculating Polarity and Subjectivity


def polarity(x): return TextBlob(x).sentiment.polarity
def subjectivity(x): return TextBlob(x).sentiment.subjectivity


df['polarity'] = df['ReviewContent'].apply(polarity)
df['subjectivity'] = df['ReviewContent'].apply(subjectivity)


# Converting Polarity into Positive, Negative and Neutral and storing it in analysis column
def ratio(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


df['analysis'] = df['polarity'].apply(ratio)
Analysis = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
df.analysis = [Analysis[item] for item in df.analysis]

df.to_csv(r'D:\yash\projects\yash\sentiments.csv', index=False)
