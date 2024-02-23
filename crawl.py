import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# 这是你的起始URL
root_url = "https://help.easyar.cn/EasyAR%20Mega/index.html"

def extract_hrefs(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 提取所有带有href的<a>标签，同时只保留有两个"/"的href（即三层目录）
    a_tags = soup.find_all('a', class_='reference internal', href=True)
    hrefs = [tag['href'] for tag in a_tags if tag['href'].count('/') == 2]

    return hrefs

def extract_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 提取所有<h1>，<h2>，<h3>和<p>标签的文本
    relevant_tags = soup.find_all(['h1', 'h2', 'h3', 'p'])

    # 按照标签的顺序拼接起来
    texts = [tag.get_text(strip=True) for tag in relevant_tags if not (tag.name == 'p' and ('Copyright' in tag.get_text() or '开发中心' in tag.get_text()))]

    return ["参考链接: " + url] + texts

# 创建存放下载文件的目录
os.makedirs('downloaded_texts', exist_ok=True)

# 合法的href
valid_hrefs = extract_hrefs(root_url)

for href in valid_hrefs:
    url = urljoin(root_url, href)
    text_content = extract_text(url)

    # 保存到文本文件
    filename = f'{href.rstrip(".html")}.txt'
    filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
    filepath = os.path.join('downloaded_texts', filename)

    with open(filepath, 'w', encoding='utf-8') as file:
        file.write("\n".join(text_content))