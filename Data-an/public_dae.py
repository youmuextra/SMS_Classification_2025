import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import pickle
from collections import Counter

# 设置中文字体（用于可视化）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置数据路径
DATA_PATH = r'D:\\PythonProject\\data\\'  # 数据文件夹路径
RAW_FILE = ("ham_data.csv")  # 原始数据文件名
PROCESSED_FILE = "processed_spam_messages.csv"  # 处理后的数据文件名
FEATURES_FILE = "text_features.pkl"  # 特征文件

# 创建输出目录
os.makedirs(DATA_PATH, exist_ok=True)

print(f"数据路径: {os.path.abspath(DATA_PATH)}")


def load_data(file_path):
    """加载数据文件"""
    try:
        # 尝试不同的编码格式
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用 {encoding} 编码加载数据")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise Exception("无法找到合适的编码格式")

        return df
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        print("请确保文件路径正确，当前工作目录:", os.getcwd())
        return None


class ChineseTextPreprocessor:
    def __init__(self, stopwords_path=None):
        """初始化中文文本预处理器"""
        self.stopwords = set()

        # 加载停用词表
        if stopwords_path and os.path.exists(stopwords_path):
            self.load_stopwords(stopwords_path)
        else:
            # 使用内置的常见中文停用词
            self.load_builtin_stopwords()

        # 初始化jieba
        jieba.initialize()

    def load_stopwords(self, file_path):
        """从文件加载停用词表"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.stopwords = set([line.strip() for line in f])
            print(f"已加载 {len(self.stopwords)} 个停用词")
        except Exception as e:
            print(f"加载停用词失败: {e}")
            self.load_builtin_stopwords()

    def load_builtin_stopwords(self):
        """加载内置常见中文停用词"""
        common_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
            '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '我们',
            '你们', '他们', '这个', '那个', '这些', '那些', '这样', '那样', '然后', '但是', '因为', '所以', '如果',
            '虽然', '可以', '应该', '已经', '还是', '通过', '进行', '以及', '或者', '并且'
        }
        self.stopwords = common_stopwords
        print(f"使用内置停用词表，共 {len(self.stopwords)} 个词")

    def clean_chinese_text(self, text):
        """清洗中文文本"""
        if not isinstance(text, str):
            return ""

        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # 移除邮箱
        text = re.sub(r'\S*@\S*\s?', '', text)

        # 移除电话号码
        text = re.sub(r'[\+]?[0-9]{1,3}?[-\s]?[0-9]{1,4}?[-\s]?[0-9]{1,4}?[-\s]?[0-9]{1,9}', '', text)

        # 移除特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)

        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def segment_text(self, text, use_stopwords=True, use_pos_filter=False):
        """中文分词"""
        text = self.clean_chinese_text(text)

        if use_pos_filter:
            # 使用词性标注，只保留名词、动词、形容词
            words = pseg.cut(text)
            words = [word for word, flag in words if flag.startswith(('n', 'v', 'a'))]
        else:
            # 普通分词
            words = jieba.cut(text)

        # 过滤停用词
        if use_stopwords:
            words = [word for word in words if word not in self.stopwords and len(word) > 1]

        return ' '.join(words)

    def preprocess_dataframe(self, df, text_column='message'):
        """预处理整个DataFrame"""
        print("开始中文文本预处理...")

        # 创建清洗后的文本列
        df['cleaned_text'] = df[text_column].apply(
            lambda x: self.segment_text(x, use_stopwords=True, use_pos_filter=False)
        )

        # 统计信息
        original_samples = len(df)
        df = df[df['cleaned_text'].str.len() > 0]  # 移除空文本
        cleaned_samples = len(df)

        print(f"预处理完成: {original_samples} -> {cleaned_samples} 个样本")
        print(f"过滤了 {original_samples - cleaned_samples} 个空文本")

        return df


def explore_chinese_data(df, text_column='message', label_column='label'):
    """探索中文数据特征"""
    print("=== 数据探索 ===")

    # 基本统计
    print(f"总样本数: {len(df)}")
    print(f"标签分布:\n{df[label_column].value_counts()}")

    # 文本长度分析
    df['text_length'] = df[text_column].str.len()
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 标签分布
    df[label_column].value_counts().plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('标签分布')

    # 文本长度分布
    df['text_length'].hist(bins=50, ax=axes[0, 1])
    axes[0, 1].set_title('原始文本长度分布')

    # 词汇数量分布
    df['word_count'].hist(bins=50, ax=axes[1, 0])
    axes[1, 0].set_title('清洗后词汇数量分布')

    # 标签间的文本长度比较
    df.boxplot(column='text_length', by=label_column, ax=axes[1, 1])
    axes[1, 1].set_title('各标签文本长度比较')

    plt.suptitle('数据探索分析')
    plt.tight_layout()
    plt.show()

    # 文本统计
    print(f"\n文本长度统计:")
    print(df['text_length'].describe())
    print(f"\n词汇数量统计:")
    print(df['word_count'].describe())

    return df


def chinese_text_feature_analysis(df, text_column='cleaned_text'):
    """执行中文文本特征分析 - 修复后的函数"""
    print("开始中文文本特征分析...")

    # 1. 基本统计
    print("\n=== 基本文本统计 ===")
    df['text_length'] = df[text_column].str.len()
    df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))

    print(f"总样本数: {len(df)}")
    print(f"平均文本长度: {df['text_length'].mean():.2f}")
    print(f"平均词汇数量: {df['word_count'].mean():.2f}")
    print(f"最大文本长度: {df['text_length'].max()}")
    print(f"最小文本长度: {df['text_length'].min()}")

    # 2. 词汇分析
    print("\n=== 词汇分析 ===")
    all_words = []
    for text in df[text_column]:
        if isinstance(text, str):
            words = text.split()
            all_words.extend(words)

    word_freq = Counter(all_words)
    print(f"总词汇数: {len(all_words)}")
    print(f"独特词汇数: {len(word_freq)}")
    print(f"最常出现的10个词汇:")
    for word, count in word_freq.most_common(10):
        print(f"  {word}: {count}次")

    # 3. 可视化
    print("\n生成可视化图表...")
    plt.figure(figsize=(15, 5))

    # 文本长度分布
    plt.subplot(1, 3, 1)
    plt.hist(df['text_length'], bins=30, alpha=0.7)
    plt.xlabel('文本长度')
    plt.ylabel('频次')
    plt.title('文本长度分布')

    # 词汇数量分布
    plt.subplot(1, 3, 2)
    plt.hist(df['word_count'], bins=30, alpha=0.7, color='orange')
    plt.xlabel('词汇数量')
    plt.ylabel('频次')
    plt.title('词汇数量分布')

    # 词频分布
    plt.subplot(1, 3, 3)
    top_words = [word for word, count in word_freq.most_common(20)]
    top_counts = [count for word, count in word_freq.most_common(20)]
    plt.barh(range(len(top_words)), top_counts)
    plt.yticks(range(len(top_words)), top_words)
    plt.xlabel('出现频次')
    plt.title('前20个高频词汇')

    plt.tight_layout()
    plt.show()

    print("中文文本特征分析完成!")


def vectorize_chinese_text(df, text_column='cleaned_text'):
    """中文文本向量化"""
    print("=== 文本向量化 ===")

    # 1. TF-IDF 向量化
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),  # 包含单个词和双词组合
        token_pattern=r'(?u)\b\w+\b'  # 中文分词后已经是空格分隔
    )

    tfidf_features = tfidf_vectorizer.fit_transform(df[text_column])
    print(f"TF-IDF特征矩阵形状: {tfidf_features.shape}")

    # 2. 词袋模型
    count_vectorizer = CountVectorizer(
        max_features=3000,
        ngram_range=(1, 2)
    )

    count_features = count_vectorizer.fit_transform(df[text_column])
    print(f"词袋模型特征矩阵形状: {count_features.shape}")

    # 获取特征名称
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print(f"\n前20个TF-IDF特征: {feature_names[:20]}")

    return {
        'tfidf_features': tfidf_features,
        'count_features': count_features,
        'tfidf_vectorizer': tfidf_vectorizer,
        'count_vectorizer': count_vectorizer
    }


def generate_wordcloud(df, label_column='label', text_column='cleaned_text'):
    """生成词云图"""
    print("=== 生成词云 ===")

    labels = df[label_column].unique()

    fig, axes = plt.subplots(1, len(labels), figsize=(15, 6))
    if len(labels) == 1:
        axes = [axes]

    for i, label in enumerate(labels):
        # 获取该标签下的所有文本
        text_data = ' '.join(df[df[label_column] == label][text_column])

        if len(text_data) > 0:
            # 生成词云
            wordcloud = WordCloud(
                font_path='simhei.ttf',  # 中文字体
                width=400,
                height=300,
                background_color='white',
                max_words=100
            ).generate(text_data)

            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'"{label}" 类词云')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'"{label}" 类无数据',
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def save_processed_data(df, text_features, save_dir=DATA_PATH):
    """保存处理后的数据"""
    print("=== 保存处理结果 ===")

    # 1. 保存处理后的DataFrame
    processed_file = os.path.join(save_dir, PROCESSED_FILE)
    df.to_csv(processed_file, index=False, encoding='utf-8')
    print(f"处理后的数据已保存: {processed_file}")

    # 2. 保存特征和向量化器
    features_file = os.path.join(save_dir, FEATURES_FILE)
    with open(features_file, 'wb') as f:
        pickle.dump(text_features, f)
    print(f"特征数据已保存: {features_file}")

    # 3. 保存预处理报告
    report_file = os.path.join(save_dir, "preprocessing_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("中文垃圾短信数据预处理报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"处理时间: {pd.Timestamp.now()}\n")
        f.write(f"原始样本数: {len(df)}\n")
        f.write(f"处理后样本数: {len(df)}\n")
        f.write(f"特征维度: {text_features['tfidf_features'].shape[1]}\n")
        f.write(f"标签分布:\n{df['label'].value_counts().to_string()}\n")

    print(f"预处理报告已保存: {report_file}")

    return {
        'processed_data': processed_file,
        'features': features_file,
        'report': report_file
    }


def inspect_processed_results(saved_files):
    """查看处理结果"""
    print("\n=== 如何查看处理结果 ===")

    for file_type, file_path in saved_files.items():
        print(f"\n{file_type.upper()} 文件: {file_path}")

        if os.path.exists(file_path):
            if file_type == 'processed_data':
                # 查看处理后的数据
                df_processed = pd.read_csv(file_path, encoding='utf-8')
                print("处理后的数据前5行:")
                print(df_processed[['message', 'cleaned_text', 'label']].head())

            elif file_type == 'report':
                # 查看报告
                with open(file_path, 'r', encoding='utf-8') as f:
                    print("预处理报告内容:")
                    print(f.read())

            elif file_type == 'features':
                # 查看特征信息
                with open(file_path, 'rb') as f:
                    features = pickle.load(f)
                    print("特征矩阵形状:", features['tfidf_features'].shape)
                    print("特征数量:", len(features['tfidf_vectorizer'].get_feature_names_out()))
        else:
            print("文件不存在")


def complete_chinese_spam_preprocessing(data_file, text_column='message', label_column='label'):
    """完整的中文垃圾短信预处理流程"""

    # 1. 加载数据
    df = load_data(data_file)
    if df is None:
        return None

    # 2. 初始化预处理器
    preprocessor = ChineseTextPreprocessor()

    # 3. 文本预处理
    df = preprocessor.preprocess_dataframe(df, text_column)

    # 4. 数据探索
    df = explore_chinese_data(df, text_column, label_column)

    # 5. 中文文本特征分析 - 修复后的函数调用
    chinese_text_feature_analysis(df)

    # 6. 文本向量化
    text_features = vectorize_chinese_text(df)

    # 7. 可视化
    generate_wordcloud(df, label_column)

    # 8. 保存结果
    saved_files = save_processed_data(df, text_features)

    # 9. 查看结果
    inspect_processed_results(saved_files)

    return df, text_features, saved_files


# 运行完整流程
if __name__ == "__main__":
    data_file = os.path.join(DATA_PATH, RAW_FILE)

    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请确保已将您的CSV文件放置在正确路径")
    else:
        results = complete_chinese_spam_preprocessing(data_file)

        if results:
            df_processed, features, files = results
            print("\n🎉 预处理完成! 所有文件已保存到指定目录")

            # 显示保存的文件路径
            print("\n生成的文件:")
            for file_type, file_path in files.items():
                print(f"  {file_type}: {file_path}")
        else:
            print("\n❌ 预处理失败，请检查数据文件")