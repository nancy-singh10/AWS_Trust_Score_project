import requests
import json

queries = [
    "boat headphones review",
    "iphone review",
    "nike shoes quality",
    "smartwatch review"
]

headers = {
    "User-Agent": "Mozilla/5.0"
}

all_data = []

for query in queries:
    print(f"Searching: {query}")

    url = f"https://www.reddit.com/search.json?q={query}&limit=10"

    try:
        res = requests.get(url, headers=headers)

        if res.status_code != 200:
            print("❌ Failed:", res.status_code)
            continue

        data = res.json()

        posts = data["data"]["children"]

        for post in posts:
            title = post["data"]["title"]
            text = post["data"].get("selftext", "")

            combined = title + " " + text

            if len(combined) > 20:
                all_data.append({
                    "query": query,
                    "text": combined
                })

    except Exception as e:
        print("Error:", e)

# SAVE
with open("reddit_data.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=4, ensure_ascii=False)

print("✅ Reddit data collected successfully")