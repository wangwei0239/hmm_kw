# encoding=utf-8
import jieba

jieba.add_word("软间隔")
jieba.add_word("线性不可分")
result = jieba.cut("老师你说的SVM包括线性可分和线性不可分，软间隔那些都要15分钟内吗")

print(list(result))
#
# from pytrie import SortedStringTrie as trie
# t = trie(双黄连=0, 曲米新乳膏=1)
# print(t)
# # print(t.keys(prefix='我有双黄连和曲米新乳膏'))
# print(t.longest_prefix('我有双黄连和曲米新乳膏'))

from pypinyin import pinyin
print(pinyin("中心"))