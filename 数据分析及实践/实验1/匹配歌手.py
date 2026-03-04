import re
import sys

# 读取输入，直到遇到两个连续的换行符
html = sys.stdin.read().split('\n\n')[0]

# 使用正则表达式匹配singer和歌名，使用re.DOTALL处理跨行情况
pattern = re.compile(r'<a.*?singer="(.*?)".*?>\s*(.*?)\s*</a>', re.DOTALL)
matches = pattern.findall(html)

print(matches)
