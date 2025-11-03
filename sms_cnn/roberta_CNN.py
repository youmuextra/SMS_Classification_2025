import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from transformers import TFAutoModel, AutoTokenizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, classification_report, accuracy_score

# 模型和参数定义
MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
MAX_LEN = 64  # 短信最大长度
NUM_FILTERS = 128  # 每个卷积核尺寸的滤波器数量
KERNEL_SIZES = [3, 4, 5]  # 卷积核的尺寸（即 N-gram 的长度）
DROPOUT_RATE = 0.2

# 加载 Tokenizer 和 TF 版本的 RoBERTa 模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# output_hidden_states=False 是默认设置，模型会输出最后一层的 hidden state
roberta_model = TFAutoModel.from_pretrained(MODEL_NAME)
HIDDEN_SIZE = roberta_model.config.hidden_size  # 获取RoBERTa的隐藏层维度 (768)

def build_roberta_cnn_model():
    # 定义模型输入 (与Tokenizer的输出对应)
    input_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')

    # RoBERTa 特征提取层
    # 将输入传入 RoBERTa 模型
    # training=False: 初始时通常冻结 RoBERTa，在微调阶段再解冻
    roberta_output = roberta_model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    # 取最后一层的 Token 隐藏状态序列
    # Shape: (None, MAX_LEN, HIDDEN_SIZE) -> (Batch Size, 64, 768)
    sequence_output = roberta_output.last_hidden_state

    # CNN 局部特征学习层
    cnn_branches = []

    # 遍历不同尺寸的卷积核
    for kernel_size in KERNEL_SIZES:
        # Conv1D 层：在 RoBERTa 序列上滑动，捕获局部模式
        # Input Shape: (None, 64, 768)
        # Output Shape: (None, 64 - kernel_size + 1, NUM_FILTERS)
        conv = Conv1D(
            filters=NUM_FILTERS,
            kernel_size=kernel_size,
            activation='relu',
            name=f'conv_{kernel_size}'
        )(sequence_output)

        # Global Max Pooling 1D 层：从每个卷积结果中提取最显著的特征
        # Input Shape: (None, sequence_length, NUM_FILTERS)
        # Output Shape: (None, NUM_FILTERS) -> (Batch Size, 128)
        pool = GlobalMaxPooling1D(name=f'gmp_{kernel_size}')(conv)
        cnn_branches.append(pool)

    # Concatenate 层：将所有不同尺寸卷积核提取的特征拼接起来
    # Output Shape: (None, len(KERNEL_SIZES) * NUM_FILTERS) -> (Batch Size, 3 * 128 = 384)
    concatenated_features = Concatenate(name='cnn_concat')(cnn_branches)
    # 分类头
    # Dropout 层：防止过拟合
    x = Dropout(DROPOUT_RATE, name='dropout_layer')(concatenated_features)
    # 全连接层（可以添加更多层进行特征整合）
    x = Dense(128, activation='relu', name='dense_1')(x)
    x = Dropout(DROPOUT_RATE, name='dropout_2')(x)
    # 输出层：使用 Sigmoid 进行二分类
    output_tensor = Dense(1, activation='sigmoid', name='output_layer')(x)
    # 构建 Keras 模型并编译
    model = Model(inputs=[input_ids, attention_mask], outputs=output_tensor)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),  # RoBERTa微调推荐的小学习率
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def tokenize_data(texts):
    # 使用 RoBERTa Tokenizer 对文本进行编码
    return tokenizer(
        texts,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

# 进行具体的模型训练
SMS_DATA = []
datafile = "data/train.txt"
switch = 0
with open(datafile, "r", encoding='utf-8') as f:
    for line in f:
        unit = []
        line = line.strip('\n')
        num, label, text = line.split('\t', maxsplit=2)
        unit.append(text)
        if label == '1' or label == 1 or label == "垃圾短信类":
            unit.append(1)
        else:
            unit.append(0)
        if switch == 1:
            SMS_DATA.append(unit)
        switch = 1
texts, labels = zip(*SMS_DATA)
labels = np.array(labels).astype(np.float32)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"训练集大小: {len(train_texts)}")
print(f"验证集大小: {len(val_texts)}")

# 确保 tokenizer 已经加载 (继承自前面的步骤)
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# MAX_LEN = 64

# 编码训练集和验证集
train_encodings = tokenize_data(list(train_texts))
val_encodings = tokenize_data(list(val_texts))

# 准备模型所需的输入字典
train_inputs = {
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask']
}
val_inputs = {
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask']
}

print(f"编码后的 Input IDs 形状: {train_inputs['input_ids'].shape}")

model = build_roberta_cnn_model()

# 浅层训练 (冻结 RoBERTa 权重)
print("\n浅层训练（冻结 RoBERTa）")

# 冻结 RoBERTa 的参数，只训练 CNN 和 Dense 分类头
roberta_model.trainable = False
# 重新编译模型以应用训练状态的改变 (这里使用更传统的学习率)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# 训练配置
BATCH_SIZE = 8 # 短信任务通常可以采用小批次
SHALLOW_EPOCHS = 3

history_shallow = model.fit(
    x=train_inputs,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=SHALLOW_EPOCHS,
    validation_data=(val_inputs, val_labels),
    verbose=1
)

# 微调训练 (解冻 RoBERTa 权重)
print("\n微调训练（解冻 RoBERTa）")

# 解冻 RoBERTa 的参数，对整个模型进行微调
roberta_model.trainable = True
# 使用非常小的学习率进行微调
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

FINE_TUNE_EPOCHS = 2

history_fine_tune = model.fit(
    x=train_inputs,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=(val_inputs, val_labels),
    verbose=1
)

# 预测验证集 将概率转换为类别 (阈值 0.5)
val_predictions = model.predict(val_inputs)
predicted_classes = (val_predictions > 0.5).astype(int).flatten()

print("\n--- 验证集预测结果 ---")
print(f"真实标签: {val_labels}")
print(f"预测类别: {predicted_classes}")

# 计算评估指标
report = classification_report(val_labels, predicted_classes, target_names=['Normal (0)', 'Spam (1)'])
f1 = f1_score(val_labels, predicted_classes)
accuracy = accuracy_score(val_labels, predicted_classes)

# 输出结果
print("\n" + "="*50)
print("============== EVA ==============")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\n--- Classification Report ---")
print(report)
print("------------------------------------------")
print(f"真实标签:      {val_labels}")
print(f"预测类别:      {predicted_classes}")
print("==========================================\n")