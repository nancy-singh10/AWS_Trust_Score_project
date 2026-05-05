import pandas as pd
import json

print("🚀 Starting Data Pipeline...")

# =========================
# 1. LOAD FLIPKART CSV
# =========================
flipkart = pd.read_csv("data/train.csv")

# Clean column names
flipkart.columns = flipkart.columns.str.lower()

# Convert numeric fields
flipkart['rating'] = pd.to_numeric(flipkart['rating'], errors='coerce')
flipkart['actprice1'] = pd.to_numeric(flipkart['actprice1'], errors='coerce')

# Rename columns
flipkart = flipkart.rename(columns={
    'actprice1': 'price',
    'norating1': 'total_ratings',
    'noreviews1': 'total_reviews'
})

# Add missing fields
flipkart['review_text'] = ""
flipkart['source'] = 'flipkart'

flipkart_df = flipkart[[
    'title', 'price', 'rating',
    'total_reviews', 'total_ratings',
    'review_text', 'maincateg', 'source'
]]

print("✅ Flipkart CSV Loaded:", flipkart_df.shape)


# =========================
# 2. LOAD AMAZON REVIEWS
# =========================
amazon = pd.read_csv("data/Reviews.csv")

amazon = amazon.rename(columns={
    'Score': 'rating',
    'Text': 'review_text',
    'Summary': 'title'
})

amazon['price'] = None
amazon['total_reviews'] = None
amazon['total_ratings'] = None
amazon['maincateg'] = None
amazon['source'] = 'amazon'

amazon_df = amazon[[
    'title', 'price', 'rating',
    'total_reviews', 'total_ratings',
    'review_text', 'maincateg', 'source'
]]

print("✅ Amazon Data Loaded:", amazon_df.shape)


# =========================
# 3. LOAD REDDIT DATA
# =========================
with open("data/reddit_data.json") as f:
    reddit = json.load(f)

reddit_df = pd.DataFrame(reddit)

reddit_df = reddit_df.rename(columns={
    'text': 'review_text',
    'query': 'title'
})

reddit_df['price'] = None
reddit_df['rating'] = None
reddit_df['total_reviews'] = None
reddit_df['total_ratings'] = None
reddit_df['maincateg'] = None
reddit_df['source'] = 'reddit'

reddit_df = reddit_df[[
    'title', 'price', 'rating',
    'total_reviews', 'total_ratings',
    'review_text', 'maincateg', 'source'
]]

print("✅ Reddit Data Loaded:", reddit_df.shape)


# =========================
# 4. LOAD FLIPKART JSON
# =========================
with open("data/flipkart_data.json") as f:
    fk_json = json.load(f)

fk_json_df = pd.DataFrame(fk_json)

# Rename columns
fk_json_df = fk_json_df.rename(columns={
    'product': 'title',
    'category': 'maincateg'
})

# Clean price (₹82,900 → 82900)
fk_json_df['price'] = fk_json_df['price'].astype(str)\
    .str.replace('₹', '', regex=False)\
    .str.replace(',', '', regex=False)

fk_json_df['price'] = pd.to_numeric(fk_json_df['price'], errors='coerce')

# Clean rating
fk_json_df['rating'] = pd.to_numeric(fk_json_df['rating'], errors='coerce')

# Handle reviews list → text
fk_json_df['review_text'] = fk_json_df['reviews'].apply(
    lambda x: " ".join(x) if isinstance(x, list) else ""
)

# Add missing columns
fk_json_df['total_reviews'] = 0
fk_json_df['total_ratings'] = 0
fk_json_df['source'] = 'flipkart_json'

fk_json_df = fk_json_df[[
    'title', 'price', 'rating',
    'total_reviews', 'total_ratings',
    'review_text', 'maincateg', 'source'
]]

print("✅ Flipkart JSON Loaded:", fk_json_df.shape)


# =========================
# 5. MERGE ALL DATA
# =========================
final_df = pd.concat([
    flipkart_df,
    amazon_df,
    reddit_df,
    fk_json_df
], ignore_index=True)

print("🔥 Final Dataset Shape:", final_df.shape)


# =========================
# 6. BASIC CLEANING
# =========================
final_df.drop_duplicates(inplace=True)

# Fill missing text
final_df['review_text'] = final_df['review_text'].fillna("")

print("✅ Data Cleaning Done")


# =========================
# 7. SAVE FINAL DATASET
# =========================
final_df.to_csv("data/final_dataset.csv", index=False)

print("🎉 Final dataset saved as data/final_dataset.csv")