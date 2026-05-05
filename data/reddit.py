import json

reddit_data = [
    {"query": "boat headphones", "text": "sound is good but build quality is poor"},
    {"query": "iphone", "text": "too expensive for features"},
    {"query": "nike shoes", "text": "not durable for long run"},
    {"query": "smartwatch", "text": "battery drains very fast"},
    {"query": "headphones", "text": "overhyped product not worth money"}
]

with open("reddit_data.json", "w") as f:
    json.dump(reddit_data, f, indent=4)

print("✅ Sample Reddit data created")