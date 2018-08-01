# encoding=utf-8
import re
pattern = r'(\d)'
target = 'this 12is34'
# obj = re.(pattern, target)
print(re.sub(pattern, lambda x: "#" * len(x.group()), target))
print(re.search(pattern, target))
print(re.search(pattern, target))
print(re.search(pattern, target))
print(re.findall(pattern, target))
# print(obj.group())
