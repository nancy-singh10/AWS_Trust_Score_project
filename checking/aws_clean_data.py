# import pandas as pd

# print("🚀 Creating AWS-ready dataset...")

# # =========================
# # 1. LOAD DATA
# # =========================
# df = pd.read_csv("data/products_final.csv")

# print("Original data:", df.shape)


# # =========================
# # 2. SELECT BEST CATEGORIES
# # =========================
# selected_categories = ["footwear", "electronics", "food"]

# df = df[df['maincateg'].isin(selected_categories)]

# print("After category filter:", df.shape)


# # =========================
# # 3. BALANCE LABELS
# # =========================
# high = df[df['trust_label'] == "High Trust ✅"]
# moderate = df[df['trust_label'] == "Moderate ⚠️"]
# low = df[df['trust_label'] == "Low Trust 🚨"]

# # 🔥 Adjust size here (500–1500 ideal)
# n = 10000

# high = high.sample(min(len(high), n), random_state=42)
# moderate = moderate.sample(min(len(moderate), n), random_state=42)
# low = low.sample(min(len(low), n), random_state=42)

# df_balanced = pd.concat([high, moderate, low])

# print("Balanced dataset:", df_balanced.shape)


# # =========================
# # 4. CLEAN IMPORTANT COLUMNS
# # =========================
# columns_needed = [
#     "title",
#     "price",
#     "rating",
#     "maincateg",
#     "trust_score",
#     "trust_label",
#     "insight"
# ]

# df_final = df_balanced[columns_needed]


# # =========================
# # 5. SAVE FINAL DATASET
# # =========================
# df_final.to_csv("data/aws_ready_dataset.csv", index=False)

# print("📂 Saved at: data/aws_ready_dataset2.csv")
# print("🎉 DONE! Your AWS dataset2 is ready 🚀")
import pandas as pd

print("🚀 Creating AWS-ready dataset (LARGE - 6000)...")

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/products_final.csv")

print("Original data:", df.shape)


# =========================
# 2. SELECT CATEGORIES
# =========================
selected_categories = ["footwear", "electronics", "food"]
df = df[df['maincateg'].isin(selected_categories)]

print("After category filter:", df.shape)


# =========================
# 3. BALANCE LABELS (6000 EACH)
# =========================
high = df[df['trust_label'] == "High Trust ✅"]
moderate = df[df['trust_label'] == "Moderate ⚠️"]
low = df[df['trust_label'] == "Low Trust 🚨"]

n = 6000  # 🔥 Larger dataset

high = high.sample(min(len(high), n), random_state=42)
moderate = moderate.sample(min(len(moderate), n), random_state=42)
low = low.sample(min(len(low), n), random_state=42)

df_balanced = pd.concat([high, moderate, low])

print("Balanced dataset:", df_balanced.shape)


# =========================
# 4. CLEAN DATA
# =========================
columns_needed = [
    "title",
    "price",
    "rating",
    "maincateg",
    "trust_score",
    "trust_label",
    "insight"
]

df_final = df_balanced[columns_needed]

# Clean for AWS
df_final = df_final.dropna()
df_final = df_final.drop_duplicates()

# Convert types (important for Athena)
df_final["price"] = pd.to_numeric(df_final["price"], errors="coerce")
df_final["rating"] = pd.to_numeric(df_final["rating"], errors="coerce")


# =========================
# 5. SAVE SECOND DATASET
# =========================
output_path = "data/aws_ready_dataset_large.csv"
df_final.to_csv(output_path, index=False)

print(f"📂 Saved at: {output_path}")
print("🎉 LARGE dataset (6000/class) ready 🚀")