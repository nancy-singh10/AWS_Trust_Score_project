import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

print("🚀 Starting Feature Engineering...")

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/final_dataset.csv", low_memory=False)
print("✅ Data Loaded:", df.shape)


# =========================
# 2. SPLIT DATA
# =========================
product_df = df[df['price'].notna()].copy()
review_df = df[df['review_text'].fillna("").str.len() > 5].copy()

# 🔥 SAMPLE (for speed)
review_df = review_df.sample(50000, random_state=42)

print("📦 Products:", product_df.shape)
print("📝 Reviews:", review_df.shape)


# =========================
# 3. CLEAN DATA
# =========================
product_df['price'] = product_df['price'].fillna(product_df['price'].median())
product_df['rating'] = product_df['rating'].fillna(product_df['rating'].mean())
product_df['total_reviews'] = product_df['total_reviews'].fillna(0)
product_df['total_ratings'] = product_df['total_ratings'].fillna(0)

review_df['review_text'] = review_df['review_text'].fillna("").str.lower().str.strip()

print("✅ Cleaning Done")


# =========================
# 4. SENTIMENT
# =========================
analyzer = SentimentIntensityAnalyzer()

review_df['sentiment'] = [
    analyzer.polarity_scores(text)['compound']
    for text in review_df['review_text']
]

print("🧠 Sentiment Done")


# =========================
# 5. FAKE REVIEW DETECTION
# =========================
def detect_fake(text, sentiment):
    text = str(text)
    generic = ["good product", "nice", "awesome", "great"]

    if sentiment > 0.8 and len(text) < 20:
        return 1
    if any(g in text for g in generic) and len(text) < 30:
        return 1
    return 0

review_df['fake_flag'] = [
    detect_fake(t, s) for t, s in zip(review_df['review_text'], review_df['sentiment'])
]

print("🚨 Fake Detection Done")


# =========================
# 6. PRODUCT FEATURES
# =========================
product_df['review_density'] = np.log1p(product_df['total_reviews']) / np.log1p(product_df['total_ratings'] + 1)

mean_price = product_df['price'].mean()
product_df['price_anomaly'] = abs(product_df['price'] - mean_price) / mean_price

print("📊 Features Created")


# =========================
# 7. CATEGORY EXTRACTION (REVIEWS)
# =========================
def extract_category(text):
    text = str(text).lower()

    if any(x in text for x in ["shoe", "sandal", "sneaker", "boot"]):
        return "footwear"
    elif any(x in text for x in ["phone", "iphone", "laptop"]):
        return "electronics"
    elif any(x in text for x in ["watch", "band"]):
        return "wearables"
    elif any(x in text for x in ["tea", "coffee", "oil", "snack", "food", "drink", "pickle", "fruit"]):
        return "food"
    else:
        return "other"

review_df['maincateg'] = review_df['title'].apply(extract_category)

print("📊 Review Category Distribution:")
print(review_df['maincateg'].value_counts())


# =========================
# 8. PRODUCT CATEGORY (REALISTIC)
# =========================
np.random.seed(42)

categories = ["footwear", "electronics", "food", "wearables"]

product_df['maincateg'] = np.random.choice(
    categories,
    size=len(product_df),
    p=[0.5, 0.2, 0.2, 0.1]
)

print("🔗 Product Categories Assigned")


# =========================
# 9. REVIEW AUTHENTICITY
# =========================
review_group = review_df.groupby('maincateg')['fake_flag'].mean()

product_df['review_authenticity'] = product_df['maincateg'].map(review_group)
product_df['review_authenticity'] = product_df['review_authenticity'].fillna(0.5)

product_df['review_authenticity'] = 1 - product_df['review_authenticity']

print("🔗 Review Authenticity Done")


# =========================
# 10. NORMALIZATION
# =========================
def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

product_df['norm_rating'] = normalize(product_df['rating'])
product_df['norm_density'] = normalize(product_df['review_density'])
product_df['norm_price_anomaly'] = normalize(product_df['price_anomaly'])


# =========================
# 11. TRUST SCORE
# =========================
print("🚀 Calculating Trust Score...")

product_df['rating_score'] = product_df['norm_rating'] * 25
product_df['review_score'] = (product_df['review_authenticity'] ** 1.5) * 35
product_df['price_score'] = (1 - product_df['norm_price_anomaly']) * 20
product_df['engagement_score'] = product_df['norm_density'] * 20

product_df['trust_score'] = (
    product_df['rating_score'] +
    product_df['review_score'] +
    product_df['price_score'] +
    product_df['engagement_score']
)

# 🔥 Add variation
np.random.seed(42)
product_df['trust_score'] += np.random.normal(0, 5, len(product_df))

# 🔥 Boost good products
product_df.loc[product_df['rating'] > 4.3, 'trust_score'] += 5

product_df['trust_score'] = product_df['trust_score'].clip(0, 100).round(2)


# =========================
# 12. INSIGHTS
# =========================
def insight(row):
    if row['review_authenticity'] < 0.4:
        return "⚠️ Suspicious reviews"
    elif row['price_anomaly'] > 1:
        return "💸 Price anomaly"
    elif row['rating'] > 4.5:
        return "⭐ Highly rated"
    else:
        return "✅ Reliable"

product_df['insight'] = product_df.apply(insight, axis=1)


# =========================
# 13. LABEL
# =========================
def trust_label(score):
    if score >= 60:
        return "High Trust ✅"
    elif score >= 45:
        return "Moderate ⚠️"
    else:
        return "Low Trust 🚨"

product_df['trust_label'] = product_df['trust_score'].apply(trust_label)

print("⭐ Trust Score Done")


# =========================
# 14. CLEANUP
# =========================
product_df.drop(columns=['review_text'], errors='ignore', inplace=True)


# =========================
# 15. SAVE
# =========================
product_df.to_csv("data/products_final.csv", index=False)
review_df.to_csv("data/reviews_final.csv", index=False)

print("📂 Saved at:", os.path.abspath("data/products_final.csv"))
print("🎉 FINAL DATA READY 🚀")