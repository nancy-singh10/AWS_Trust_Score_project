import pandas as pd
import json

# Kaggle
df_kaggle = pd.read_csv("data/train.csv")

# Flipkart
with open("data/flipkart_data.json") as f:
    flipkart = json.load(f)

df_flipkart = pd.json_normalize(
    flipkart, 
    "reviews", 
    ["product", "price", "rating", "category"]
)

# Reddit
with open("data/reddit_data.json") as f:
    reddit = json.load(f)

df_reddit = pd.DataFrame(reddit)

print("Loaded all data")
import re

def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df_kaggle["clean"] = df_kaggle["Text"].apply(clean)
df_flipkart["clean"] = df_flipkart["text"].apply(clean)
df_reddit["clean"] = df_reddit["text"].apply(clean)
