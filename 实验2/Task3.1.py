import requests

# 手动拼接URL（避免参数覆盖）
search_url = "https://www.nature.com/articles/s42256-024-00944-1"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Referer": "https://www.nature.com/",
    "Accept-Language": "en-US,en;q=0.9"
}

response = requests.get(search_url, headers=headers)
response.encoding = "utf-8"

with open("page3.1.txt", "w", encoding="utf-8") as f:
    f.write(response.text)