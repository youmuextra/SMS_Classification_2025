# encoding: utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import word2vec_helpers

from text_cnn import TextCNN
from sklearn.model_selection import train_test_split


# 命令行参数
# =======================================================

#加载数据参数
tf.compat.v1.flags.DEFINE_float("test_sample_percentage", .1, "用于验证的数据百分比")
tf.compat.v1.flags.DEFINE_string("stopwords_file", "./data/stopwords.txt", "停用词文件地址")
tf.compat.v1.flags.DEFINE_integer("num_labels", 2, "数据类别数（默认：2）")

#数据地址
# 理论上支持中英两种语言
# tf.compat.v1.flags.DEFINE_string("data_file", "./data/data_cn.txt", "短信数据文件地址") # 中文
# tf.compat.v1.flags.DEFINE_string("data_file", "./data/sms5w.txt", "短信数据文件地址") # 中文
# tf.compat.v1.flags.DEFINE_string("data_file", "./data/en_SMS_spam.csv", "短信数据文件地址") # 中文
tf.compat.v1.flags.DEFINE_string("data_file", "./data/SMSSpamCollection.txt", "短信数据文件地址")   # 英文

#模型超参数
tf.compat.v1.flags.DEFINE_integer("embedding_dim", 100, "词向量维度(默认: 100)")
tf.compat.v1.flags.DEFINE_string("filter_sizes", "3,4,5", "卷积核尺寸(默认: '3,4,5')")
tf.compat.v1.flags.DEFINE_integer("num_filters", 128, "每种卷积核个数 (默认: 128)")
tf.compat.v1.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout保持概率 (默认: 0.5)")
tf.compat.v1.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2正则化lambda (默认: 0.0)")
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.01, "学习率 (默认: 0.01)")

#训练参数
tf.compat.v1.flags.DEFINE_integer("batch_size", 128, "批量大小(默认: 1000)")
tf.compat.v1.flags.DEFINE_integer("num_epochs", 20, "训练时段数 (默认: 20)")
tf.compat.v1.flags.DEFINE_integer("evaluate_every", 100, "多少步之后在测试集上评估模型 (默认: 100)")
tf.compat.v1.flags.DEFINE_integer("checkpoint_every", 100, "每多少步后保存检查点 (默认: 100)")
tf.compat.v1.flags.DEFINE_integer("num_checkpoints", 5, "要存储的检查点数(默认: 5)")

#词向量模型参数
# tf.compat.v1.flags.DEFINE_string("word2vec_model_file", "/home/hustlib/PycharmProjects/sms_cnn/wiki/wiki.model", "词向量模型文件路径")
tf.compat.v1.flags.DEFINE_string("word2vec_model_file", "/home/hustlib/PycharmProjects/sms_cnn/wiki/wiki_en.model", "词向量模型文件路径")


# 解析命令行参数
FLAGS = tf.compat.v1.flags.FLAGS
FLAGS.flag_values_dict()
print("\n参数:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
print("")

# 准备模型和摘要输出目录
# =======================================================

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("模型与摘要保存路径 {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# 数据预处理
# =======================================================

# 加载数据
print("加载数据集...")
x_text, y = data_helpers.load_data(FLAGS.data_file)

#中文分词
print("分词...")
sentences = data_helpers.cut_sentences(x_text)

#停用词过滤
#print("停用词过滤...")
#sentences = data_helpers.filter_stopword(sentences, FLAGS.stopwords_file)

#计算最大句子长度
maxSentenceLen = max([len(sentence) for sentence in sentences])

# 获取词向量
print("获取词向量...")
x = np.array(word2vec_helpers.get_vectors(sentences, FLAGS.word2vec_model_file, maxSentenceLen))
y = np.array([[1,0] if label=='0' else [0,1] for label in y])
print("x.shape = {}".format(x.shape))
print("y.shape = {}".format(y.shape))

# 保存参数，测试数据时需要用到
training_params_file = os.path.join(out_dir, 'training_params.pickle')
params = {'num_labels' : FLAGS.num_labels, 'max_sentence_len' : maxSentenceLen}
data_helpers.saveDict(params, training_params_file)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=FLAGS.test_sample_percentage,  # 测试集比例
    random_state=42,  # 随机种子，确保可重复性
    shuffle=True,     # 是否打乱数据
    stratify=y        # 保持训练集和测试集的类别分布一致
)

print("训练/测试数据分离: {:d}/{:d}".format(len(y_train), len(y_test)))

# 训练
# =======================================================

with tf.Graph().as_default():
    session_conf = tf.compat.v1.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    sess = tf.compat.v1.Session(config = session_conf)
    with sess.as_default():
        cnn = TextCNN(text_len = x_train.shape[1], 
                      num_classes = y_train.shape[1], 
                      embedding_size = FLAGS.embedding_dim,
                      conv_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                      num_convs = FLAGS.num_filters,
                      l2_reg_lambda = FLAGS.l2_reg_lambda)
        # 定义训练程序
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # 检查点目录
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # 初始化所有变量
        sess.run(tf.compat.v1.global_variables_initializer())
        def train_step(x_batch, y_batch):
            """
            一个简单的测试步骤
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy, f1 = sess.run([train_op, global_step, cnn.loss, cnn.accuracy, cnn.f1_score], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, f1 {:g}".format(time_str, step, loss, accuracy, f1))

        def test_step(x_batch, y_batch):
            """
            在一个测试集上评估模型
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 0.5
            }
            # step, loss, accuracy, f1 = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.f1_score], feed_dict)
            loss, accuracy, f1 = sess.run([cnn.loss, cnn.accuracy, cnn.f1_score], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}, acc {:g}, f1 {:g}".format(time_str, step, loss, accuracy, f1))
            print("EVA {}: loss {:g}, acc {:g}, f1 {:g}".format(time_str, loss, accuracy, f1))

        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # 对每一批数据进行循环训练
        for batch in batches:
            # x_batch, y_batch = zip(*batch)
            x_batch, y_batch = batch
            train_step(x_batch, y_batch)
            current_step = tf.compat.v1.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\评估:")
                test_step(x_test, y_test)
                print("......")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("保存模型检查点到 {}\n".format(path))
                
        print("训练完成")
        print("\评估:")

        test_step(x_test, y_test)
        # test_batches = data_helpers.batch_iter(list(zip(x_test, y_test)), batch_size=100, num_epochs=1)
        # for batch in test_batches:
        #     x_batch, y_batch = batch
        #     acc, f1 = sess.run([cnn.accuracy, cnn.f1_score],
        #                        feed_dict={
        #         cnn.input_x: x_batch,
        #         cnn.input_y: y_batch,
        #         cnn.dropout_keep_prob: 1.0
        #     })
        #     break
        # time_str = datetime.datetime.now().isoformat()
        # print("EVA {}: acc: {:g}, f1: {:g}".format(time_str, acc, f1))

        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("保存模型检查点到 {}\n".format(path))
