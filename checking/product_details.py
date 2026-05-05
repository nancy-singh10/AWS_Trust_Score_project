import pandas as pd

df = pd.read_csv("data/products_final.csv")

print("🔍 BASIC INFO")
print(df.info())

print("\n📊 NULL VALUES")
print(df.isnull().sum())

print("\n📊 UNIQUE CATEGORIES")
print(df['maincateg'].value_counts())

print("\n📊 TRUST SCORE DISTRIBUTION")
print(df['trust_score'].describe())

print("\n📊 LABEL DISTRIBUTION")
print(df['trust_label'].value_counts())