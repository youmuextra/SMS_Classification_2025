# 从Jupyter Notebook转换而来
# 源文件: word2vec_train.ipynb

# 单元格 1
# coding; utf-8
"""
将从网络上下载的xml格式的wiki百科训练语料转为txt格式
"""

# from gensim.corpora import WikiCorpus
#
# if __name__ == '__main__':
#
#     print('主程序开始...')
#
#     input_file_name = 'data/enwiki-20250920-pages-articles-multistream.xml.bz2'
#     output_file_name = 'data/wiki.en.txt'
#     print('开始读入wiki数据...')
#     # input_file = WikiCorpus(input_file_name, lemmatize=False, dictionary={})
#     input_file = WikiCorpus(input_file_name, dictionary={})
#     print('wiki数据读入完成！')
#     output_file = open(output_file_name, 'w', encoding="utf-8")
#
#     print('处理程序开始...')
#     count = 0
#     for text in input_file.get_texts():
#         output_file.write(' '.join(text) + '\n')
#         count = count + 1
#         if count % 10000 == 0:
#             print('目前已处理%d条数据' % count)
#         if count == 500000:
#             break
#     print('处理程序结束！')
#
#     output_file.close()
#     print('主程序结束！')

# 单元格 2
# import zhconv
#
# print('主程序执行开始...')
#
# input_file_name = 'data/wiki.en.txt'
# output_file_name = 'data/wiki.en.simple.txt'
# input_file = open(input_file_name, 'r', encoding='utf-8')
# output_file = open(output_file_name, 'w', encoding='utf-8')
#
# print('开始读入繁体文件...')
# lines = input_file.readlines()
# print('读入繁体文件结束！')
#
# print('转换程序执行开始...')
# count = 1
# for line in lines:
#     output_file.write(zhconv.convert(line, 'zh-hans'))
#     count += 1
#     if count % 10000 == 0:
#         print('目前已转换%d条数据' % count)
# print('转换程序执行结束！')
#
# print('主程序执行结束！')

# 单元格 3
# coding:utf-8
import jieba

print('主程序执行开始...')

input_file_name = 'data/wiki.en.simple.txt'
output_file_name = 'data/wiki.en.simple.separate.txt'
input_file = open(input_file_name, 'r', encoding='utf-8')
output_file = open(output_file_name, 'w', encoding='utf-8')

print('开始读入数据文件...')
lines = input_file.readlines()
print('读入数据文件结束！')

print('分词程序执行开始...')
count = 1
for line in lines:
    # jieba分词的结果是一个list，需要拼接，但是jieba把空格回车都当成一个字符处理
    output_file.write(' '.join(jieba.cut(line.split('\n')[0])) + '\n')
    count += 1
    if count % 10000 == 0:
        print('目前已分词%d条数据' % count)
print('分词程序执行结束！')

print('主程序执行结束！')

# 单元格 4
# coding:utf-8
import re

print('主程序执行开始...')

input_file_name = 'data/wiki.en.simple.separate.txt'
output_file_name = 'data/wiki_en.txt'
input_file = open(input_file_name, 'r', encoding='utf-8')
output_file = open(output_file_name, 'w', encoding='utf-8')

print('开始读入数据文件...')
lines = input_file.readlines()
print('读入数据文件结束！')

print('分词程序执行开始...')
count = 1
cn_reg = '^[\u4e00-\u9fa5a-zA-Z]+$'

for line in lines:
    line_list = line.split('\n')[0].split(' ')
    line_list_new = []
    for word in line_list:
        if re.search(cn_reg, word):
            line_list_new.append(word)
    print(line_list_new)
    output_file.write(' '.join(line_list_new) + '\n')
    count += 1
    if count % 10000 == 0:
        print('目前已分词%d条数据' % count)
print('分词程序执行结束！')

print('主程序执行结束！')

# 单元格 5
# coding:utf-8
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == "__main__":
    input_file_name = 'data/wiki_en.txt'
    model_file_name = 'wiki/wiki_en.model'
    print('转换过程开始...')
    model = Word2Vec(LineSentence(input_file_name),
                     vector_size=100,  # 词向量长度为400
                     window=5,
                     min_count=5,
                     workers=multiprocessing.cpu_count())
    print('转换过程结束！')
    print('开始保存模型...')
    model.save(model_file_name)
    print('模型保存结束！')

