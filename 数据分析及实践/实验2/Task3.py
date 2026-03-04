import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from retrying import retry


# 基础配置
BASE_URL = "https://www.nature.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL,
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
}
TIMEOUT = 30  # 增加超时时间至30秒
MAX_RETRIES = 3  # 最大重试次数

def retry_on_timeout(exception):
    """定义重试条件"""
    return isinstance(exception, requests.exceptions.Timeout)

@retry(stop_max_attempt_number=MAX_RETRIES, retry_on_exception=retry_on_timeout)
def safe_request(url):
    """带重试机制的请求函数"""
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    return response

def extract_authors(soup):
    """从HTML中提取作者列表（最新策略）"""
    authors = []
    
    # 方法1：通过data-test属性精准定位
    author_tags = soup.select('a[data-test="author-name"]')
    if author_tags:
        authors = [tag.text.strip() for tag in author_tags]
        return authors
    
    # 方法2：备用选择器（应对结构微调）
    author_tags = soup.select('.c-article-author-list__item a[href^="#auth"]')
    if author_tags:
        authors = [tag.text.strip() for tag in author_tags]
        return authors
    
    # 方法3：结构化数据兜底
    try:
        ld_json = json.loads(soup.find('script', type='application/ld+json').string)
        return [a["name"] for a in ld_json["mainEntity"]["author"]]
    except Exception:
        pass
    
    return authors

def update_article_authors(paper):
    """更新单篇文章的作者信息"""
    try:
        full_url = urljoin(BASE_URL, paper["url"])
        print(f"\n正在处理: {paper['title'][:30]}...")
        
        # 发送请求（带重试机制）
        response = safe_request(full_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 提取最新作者列表
        new_authors = extract_authors(soup)
        original_count = len(paper["authors"])  # 记录原始数量
        
        # 智能更新逻辑
        if len(new_authors) > original_count:
            paper["authors"] = new_authors
            print(f"作者数量增加: {original_count} → {len(new_authors)} 位")
        elif len(new_authors) == original_count:
            if paper["authors"] != new_authors:
                paper["authors"] = new_authors
                print(f"作者内容修正: 保持 {original_count} 位（更新排序/拼写）")
            else:
                print(f"作者无变化: {original_count} 位")
        else:
            print(f"警告：获取到更少作者 {original_count} → {len(new_authors)} 位，保留原始数据")
            
        return paper
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        return paper        

def process_journal_data(journal_data):
    """处理整个期刊数据"""
    for journal in journal_data:
        print(f"\n{'='*30}")
        print(f"开始处理期刊: {journal['journal']}")
        print(f"{'='*30}")
        
        for i, paper in enumerate(journal["papers"], 1):
            journal["papers"][i-1] = update_article_authors(paper)
            time.sleep(2)  # 增加请求间隔
    
    return journal_data

def main():
    # 加载初始数据
    with open('nature_llm.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 执行更新
    updated_data = process_journal_data(data)
    
    # 保存结果
    with open('nature_llm_final.json', 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
    
    print("\n最终数据已保存至 nature_llm_final.json")

if __name__ == "__main__":
    main()