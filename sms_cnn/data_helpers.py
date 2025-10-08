import numpy as np
import jieba
import zhconv
import re
import io
import pickle

def clean_str(string):
    """
    短信文本预处理
    1.去掉短信文本中的特殊字符用空格代替
    2.繁体字转简体字
    """
    #去掉特殊字符 保留汉字与英文字母
    string = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]+', " ", string)
    #return string.strip().lower()
    
    #繁体字转换为简体字
    string = zhconv.convert(string.strip(), 'zh-hans')
    string.strip().lower()
    return string
def filter_stopword(sentences, stopwords_file):
    '''
    停用词过滤
    '''
    # 读停用词
    stopwords = []
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line)>0:
                stopwords.append(line.strip())
    # 过滤停用词
    sentences_new = []
    for sentence in sentences:
        words = []
        for word in sentence:
            if word not in stopwords:
                words.append(word)
        sentences_new.append(words)
    return sentences_new
def cut_sentence(sentence):
    words = []
    cut = jieba.cut(sentence)
    for word in cut:
        if word != " ":
            words.append(word)
    return words

def cut_sentences(sentences):
    '''
    对句子进行中文分词
    返回分词后的句子，和最长的句子长度
    '''
    return [list(cut_sentence(sentence)) for sentence in sentences]

def load_data(dataFile):
    """
    从数据文件中加载数据，并进行数据清洗
    返回清洗后的文本和标签.
    """
    x_text = []
    y = []
    with open(dataFile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip('\n')
            label, text = line.split('\t')
            x_text.append(clean_str(text))
            if label == 'ham' or label == '0':
                y.append('0')
            if label == 'spam' or label == '1':
                y.append('1')
    return [x_text, y]




# def batch_iter(data, batch_size, num_epochs, shuffle=True):
#     """
#     为数据集生成批处理迭代器
#     """
#     data = np.array(data)
#     data_size = len(data)
#     num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
#     for epoch in range(num_epochs):
#         # 每轮对数据进行重新洗牌
#         if shuffle:
#             shuffle_indices = np.random.permutation(np.arange(data_size))
#             shuffled_data = data[shuffle_indices]
#         else:
#             shuffled_data = data
#         for batch_num in range(num_batches_per_epoch):
#             start_index = batch_num * batch_size
#             end_index = min((batch_num + 1) * batch_size, data_size)
#             yield shuffled_data[start_index:end_index]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    为数据集生成批处理迭代器
    data: 列表，每个元素是 (features, label) 的元组
    """
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1

    for epoch in range(num_epochs):
        # 每轮对数据进行重新洗牌
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = [data[i] for i in shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            batch_data = shuffled_data[start_index:end_index]

            # 分离特征和标签
            x_batch = np.array([item[0] for item in batch_data])
            y_batch = np.array([item[1] for item in batch_data])

            yield x_batch, y_batch

def saveDict(input_dict, dict_file):
    with open(dict_file, 'wb') as f:
        pickle.dump(input_dict, f) 

def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict
    

if __name__ == "__main__":
    dataFile = "data/sms.txt"

    load_data(dataFile)
    
