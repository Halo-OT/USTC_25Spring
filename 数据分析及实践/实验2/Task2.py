from bs4 import BeautifulSoup
import json
import re

# 读取page1.txt内容
with open('page1.txt', 'r', encoding='utf-8') as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')

# 初始化期刊字典列表
journal_data = []

# 提取所有论文条目
articles = soup.select('li.app-article-list-row__item article.c-card')

for article in articles:
    # 提取标题和URL
    title_tag = article.find('h3', class_='c-card__title').find('a')
    title = title_tag.text.strip() if title_tag else "No title"
    url = title_tag['href'] if title_tag else "No URL"

    # 提取作者列表
    authors = []
    author_list = article.select('[itemprop="creator"] [itemprop="name"]')
    for author in author_list:
        authors.append(author.text.strip())

    # 提取简介
    desc_tag = article.find('div', {'data-test': 'article-description'})
    description = desc_tag.text.strip() if desc_tag else "no description"

    # 提取类型和日期
    meta_section = article.find('div', class_='c-meta')
    article_type = meta_section.find('span', class_='c-meta__type').text.strip() if meta_section else "Unknown"
    
    # 提取期刊信息
    journal_tag = meta_section.find('div', {'data-test': 'journal-title-and-link'})
    journal_name = journal_tag.text.strip() if journal_tag else "Unknown"
    
    # 提取卷宗信息
    volume_tag = meta_section.find('div', {'data-test': 'volume-and-page-info'})
    volume_page_info = volume_tag.text.strip() if volume_tag else ""

    # 按期刊分类
    target_journal = None
    for j in journal_data:
        if j['journal'] == journal_name:
            target_journal = j
            break
            
    if not target_journal:
        target_journal = {
            "journal": journal_name,
            "papers": []
        }
        journal_data.append(target_journal)

    # 构建论文条目
    paper_entry = {
        "title": title,
        "url": url,
        "authors": authors,
        "description": description,
        "type": article_type,
        "volume_page_info": volume_page_info
    }
    
    target_journal['papers'].append(paper_entry)

# 统计各期刊论文数量
journal_counts = {j['journal']: len(j['papers']) for j in journal_data}
print("各期刊论文数量统计:")
for journal, count in journal_counts.items():
    print(f"- {journal}: {count}篇")

# 保存到JSON文件
with open('nature_llm.json', 'w', encoding='utf-8') as f:
    json.dump(journal_data, f, indent=2, ensure_ascii=False)

print("\n数据已保存至 nature_llm.json 文件")