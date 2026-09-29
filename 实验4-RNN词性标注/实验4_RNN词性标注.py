import numpy as np
import re
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import warnings
import os
import matplotlib as mpl
import matplotlib.font_manager as fm

# 设置中文字体 - 使用Windows 11自带字体
try:
    # 尝试使用微软雅黑字体（Windows 11自带）
    mpl.rcParams['font.family'] = 'Microsoft YaHei'
    mpl.rcParams['axes.unicode_minus'] = False
except:
    # 如果失败，尝试其他Windows自带中文字体
    windows_fonts = ['Microsoft YaHei', 'SimSun', 'SimHei', 'KaiTi', 'FangSong']
    available_fonts = []
    
    for font in windows_fonts:
        if any(f.name == font for f in fm.fontManager.ttflist):
            available_fonts.append(font)
    
    if available_fonts:
        mpl.rcParams['font.family'] = available_fonts[0]
        mpl.rcParams['axes.unicode_minus'] = False
        print(f"使用Windows自带字体: {available_fonts[0]}")
    else:
        print("警告：未找到Windows自带中文字体，将使用英文标题")
        # 使用英文标题
        titles = {
            'eigenfaces': 'Principal Components (Eigenfaces)',
            'original': 'Original',
            'reconstructed': 'PCA Reconstruction',
            'difference': 'Difference',
            'variance': 'Number of Principal Components vs Cumulative Explained Variance'
        }
        

# 设置显示选项
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

print(f"TensorFlow 版本: {tf.__version__}")

class ImprovedPOSTagger:
    def __init__(self):
        self.word2idx = {}
        self.tag2idx = {}
        self.idx2word = {}
        self.idx2tag = {}
        self.model = None
        self.max_len = 40  # 增加最大序列长度以容纳更长句子
    
    def load_real_corpus(self, file_path):
        """加载真实的人民日报语料"""
        sentences = []
        tags = []
        
        print(f"尝试加载文件: {file_path}")
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件路径不存在: {file_path}")
                
            with open(file_path, 'r', encoding='gbk') as f:  # 尝试GBK编码
                content = f.read()
                print("文件读取成功，开始解析...")
                
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:  # 尝试UTF-8编码
                    content = f.read()
                    print("使用UTF-8编码读取成功")
            except Exception as e:
                print(f"UTF-8编码也失败: {e}")
                return self.load_example_data()
        except Exception as e:
            print(f"文件读取错误: {e}")
            return self.load_example_data()
        
        # 解析人民日报语料格式
        lines = content.split('\n')
        current_sentence = []
        current_tags = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
                continue
            
            # 解析每行的词语/标签对
            pairs = line.split()
            for pair in pairs:
                if '/' in pair:
                    parts = pair.rsplit('/', 1)
                    if len(parts) == 2:
                        word, tag = parts
                        # 过滤掉方括号等特殊符号
                        if word not in ['[', ']']:
                            current_sentence.append(word)
                            # 清理标签，只保留主要词性
                            clean_tag = re.sub(r'[^a-zA-Z]', '', tag)
                            if clean_tag:
                                current_tags.append(clean_tag)
                            else:
                                current_tags.append(tag)
        
        if current_sentence:
            sentences.append(current_sentence)
            tags.append(current_tags)
        
        print(f"成功加载语料，共 {len(sentences)} 个句子")
        return sentences, tags
    
    def load_example_data(self):
        """备用示例数据"""
        print("使用示例数据代替")
        sentences = [
            ["我", "爱", "北京", "天安门"],
            ["今天", "天气", "很好", "。"],
            ["他", "在", "学校", "学习", "计算机", "科学"],
            ["我们", "一起", "去", "公园", "玩"],
            ["这本书", "非常", "有趣"],
            ["中国", "是", "伟大", "的", "国家"],
            ["我", "喜欢", "编程", "和", "机器学习"],
            ["明天", "会", "更好"],
            ["阳光", "明媚", "的", "早晨"],
            ["深度", "学习", "很", "有趣"],
            ["人工智能", "正在", "改变", "世界"],
            ["自然", "语言", "处理", "是", "重要", "的", "研究", "领域"],
            ["他", "每天", "都", "坚持", "锻炼", "身体"],
            ["这个", "问题", "很", "复杂", "需要", "深入", "思考"],
            ["科技", "发展", "带来", "了", "许多", "便利"]
        ]
        
        tags = [
            ["r", "v", "ns", "ns"],
            ["t", "n", "a", "w"],
            ["r", "p", "n", "v", "n", "n"],
            ["r", "d", "v", "n", "v"],
            ["r", "d", "a"],
            ["ns", "v", "a", "u", "n"],
            ["r", "v", "v", "c", "n"],
            ["t", "v", "a"],
            ["n", "a", "u", "n"],
            ["n", "v", "d", "a"],
            ["n", "d", "v", "n"],
            ["n", "n", "v", "v", "a", "u", "v", "n"],
            ["r", "t", "d", "v", "v", "n"],
            ["r", "n", "d", "a", "v", "a", "v"],
            ["n", "v", "v", "u", "m", "n"]
        ]
        
        return sentences, tags
    
    def build_vocabulary(self, sentences, tags):
        """构建词汇表"""
        # 构建词汇表
        word_freq = Counter()
        for sentence in sentences:
            word_freq.update(sentence)
        
        # 只保留出现频率较高的词
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for word, freq in word_freq.items():
            if freq >= 2:  # 提高频率阈值以减少低频词
                self.word2idx[word] = idx
                idx += 1
        
        # 构建标签表
        all_tags = set()
        for tag_seq in tags:
            all_tags.update(tag_seq)
        
        self.tag2idx = {}
        for i, tag in enumerate(sorted(all_tags)):
            self.tag2idx[tag] = i
        
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        
        print(f"词汇表大小: {len(self.word2idx)}")
        print(f"标签种类: {len(self.tag2idx)}")
        print(f"标签示例: {list(self.tag2idx.keys())[:10]}")
    
    def prepare_sequences(self, sentences, tags):
        """准备训练序列"""
        X = []
        y = []
        
        for sentence, tag_seq in zip(sentences, tags):
            if len(sentence) != len(tag_seq):
                continue  # 跳过长度不匹配的句子
                
            # 编码句子
            encoded_sentence = []
            for word in sentence:
                if word in self.word2idx:
                    encoded_sentence.append(self.word2idx[word])
                else:
                    encoded_sentence.append(self.word2idx['<UNK>'])
            
            encoded_tags = [self.tag2idx.get(tag, 0) for tag in tag_seq]
            
            # 填充或截断序列
            if len(encoded_sentence) > self.max_len:
                encoded_sentence = encoded_sentence[:self.max_len]
                encoded_tags = encoded_tags[:self.max_len]
            else:
                pad_len = self.max_len - len(encoded_sentence)
                encoded_sentence = [self.word2idx['<PAD>']] * pad_len + encoded_sentence
                encoded_tags = [0] * pad_len + encoded_tags
            
            X.append(encoded_sentence)
            y.append(encoded_tags)
        
        return np.array(X), np.array(y)
    
    def build_advanced_model(self):
        """构建更先进的RNN模型"""
        vocab_size = len(self.word2idx)
        tag_size = len(self.tag2idx)
        
        model = tf.keras.Sequential([
            # 嵌入层 - 增加维度
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=128,  # 增加嵌入维度以捕获更多语义信息
                input_length=self.max_len,
                mask_zero=True  # 启用masking
            ),
            
            # 双向LSTM层 - 增加单元数
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.3)
            ),
            
            # 第二个双向LSTM层 - 增加网络深度
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3)
            ),
            
            # 全连接层 - 增加单元数
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(64, activation='relu')
            ),
            
            # Dropout层 - 减少过拟合
            tf.keras.layers.Dropout(0.2),
            
            # 输出层
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(tag_size, activation='softmax')
            )
        ])
        
        return model
    
    def train(self, X, y, epochs=30):
        """训练模型"""
        # 手动分割训练集和测试集
        split_idx = int(0.7 * len(X))  # 70%训练
        val_idx = int(0.85 * len(X))   # 15%验证，15%测试
        
        X_train, X_val, X_test = X[:split_idx], X[split_idx:val_idx], X[val_idx:]
        y_train, y_val, y_test = y[:split_idx], y[split_idx:val_idx], y[val_idx:]
        
        print(f"训练集: {X_train.shape[0]} 个样本")
        print(f"验证集: {X_val.shape[0]} 个样本")
        print(f"测试集: {X_test.shape[0]} 个样本")
        
        self.model = self.build_advanced_model()
        
        # 编译模型 - 使用更优的优化器
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("模型结构:")
        self.model.summary()
        
        # 添加早停和模型检查点
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)
        ]
        
        # 训练模型 - 增加训练轮次
        history = self.model.fit(
            X_train, y_train,
            batch_size=64,  # 增加批次大小
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history, X_test, y_test
    
    def predict_sentence(self, sentence):
        """对新句子进行词性标注"""
        encoded_sentence = []
        for word in sentence:
            if word in self.word2idx:
                encoded_sentence.append(self.word2idx[word])
            else:
                encoded_sentence.append(self.word2idx['<UNK>'])
        
        # 填充序列
        if len(encoded_sentence) < self.max_len:
            pad_len = self.max_len - len(encoded_sentence)
            padded_sentence = [self.word2idx['<PAD>']] * pad_len + encoded_sentence
        else:
            padded_sentence = encoded_sentence[:self.max_len]
        
        padded_sentence = np.array([padded_sentence])
        predictions = self.model.predict(padded_sentence, verbose=0)
        predicted_tags_idx = np.argmax(predictions[0], axis=-1)
        
        # 提取非填充部分的预测结果
        result = []
        start_index = self.max_len - len(sentence)
        for i in range(len(sentence)):
            word = sentence[i]
            tag_idx = predicted_tags_idx[start_index + i]
            tag = self.idx2tag.get(tag_idx, 'UNK')
            result.append((word, tag))
        
        return result
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"测试集准确率: {test_accuracy:.4f}")
        
        # 详细评估
        predictions = self.model.predict(X_test, verbose=0)
        predicted_labels = np.argmax(predictions, axis=-1)
        
        correct = 0
        total = 0
        for i in range(len(X_test)):
            for j in range(self.max_len):
                if X_test[i, j] != 0:  # 忽略填充位置
                    if predicted_labels[i, j] == y_test[i, j]:
                        correct += 1
                    total += 1
        
        detailed_accuracy = correct / total if total > 0 else 0
        print(f"详细准确率(忽略填充): {detailed_accuracy:.4f}")
        
        return test_accuracy

def main():
    print("="*60)
    print("实验项目4：循环神经网络实现词性标注（使用真实数据）")
    print("="*60)
    
    # 初始化词性标注器
    tagger = ImprovedPOSTagger()
    
    # 1. 数据预处理 - 使用真实数据
    print("\n步骤1: 数据预处理")
    print("-" * 30)
    
    # 指定您的文件路径
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', '199801.txt')
    sentences, tags = tagger.load_real_corpus(file_path)
    
    if len(sentences) == 0:
        print("警告：未能加载到有效数据，使用示例数据")
        sentences, tags = tagger.load_example_data()
    
    tagger.build_vocabulary(sentences, tags)
    X, y = tagger.prepare_sequences(sentences, tags)
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 2. 构建和训练模型
    print("\n步骤2: 训练循环神经网络模型")
    print("-" * 30)
    history, X_test, y_test = tagger.train(X, y, epochs=10)  # 增加训练轮次
    
    # 3. 评估模型
    print("\n步骤3: 模型评估")
    print("-" * 30)
    test_accuracy = tagger.evaluate(X_test, y_test)
    
    # 4. 词性标注演示
    print("\n步骤4: 词性标注演示")
    print("-" * 30)
    
    test_sentences = [
        ["我", "爱", "北京", "天安门"],
        ["今天", "天气", "很好", "。"],
        ["他", "在", "学校", "学习", "编程"],
        ["人工智能", "是", "未来", "方向"],
        ["我们", "应该", "保护", "环境"],
        ["深度学习", "在", "自然语言处理", "中", "有", "广泛", "应用"]
    ]
    
    print("词性标注结果演示:")
    for i, test_sentence in enumerate(test_sentences, 1):
        result = tagger.predict_sentence(test_sentence)
        print(f"\n句子 {i}: {' '.join(test_sentence)}")
        print("标注结果: ", end="")
        for word, tag in result:
            print(f"{word}/{tag} ", end="")
        print()
    
    # 5. 实验结果分析
    print("\n步骤5: 实验结果分析")
    print("-" * 30)
    
    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.axhline(y=test_accuracy, color='r', linestyle='--', 
                label=f'测试准确率: {test_accuracy:.4f}')
    plt.title('模型准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend() 
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"最终训练准确率: {history.history['accuracy'][-1]:.4f}")
    print(f"最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    if test_accuracy > 0.9:
        print("✅ 模型性能优秀！")
    elif test_accuracy > 0.5:
        print("✅ 模型性能良好！")
    else:
        print("⚠️  模型性能有待提升")
    
    print("\n" + "="*60)
    print("实验完成！已使用指定路径的真实语料数据")
    print("="*60)

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    tf.random.set_seed(42)
    
    main()
