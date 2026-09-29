import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import matplotlib as mpl
import matplotlib.font_manager as fm

# 设置环境变量以关闭TensorFlow的日志信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误和警告信息

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

# 1. 数据加载函数 - 直接读取本地CIFAR-10数据集
def load_cifar10_from_local(data_dir):
    """
    从本地路径加载CIFAR-10数据集
    参数data_dir: 数据集所在目录，如 'C:/Users/aa/.keras/datasets/cifar-10-batches-py'
    """
    # 检查路径是否存在
    if not os.path.exists(data_dir):
        # 尝试使用TensorFlow内置方法加载（会从默认路径查找）
        print("指定路径不存在，尝试从TensorFlow默认路径加载...")
        return tf.keras.datasets.cifar10.load_data()
    
    train_images = []
    train_labels = []
    
    # 加载5个训练批次
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f'data_batch_{i}')
        try:
            with open(batch_path, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                # 转换字节键为字符串键
                if b'data' in data:
                    images = data[b'data']
                    labels = data[b'labels']
                else:
                    images = data['data']
                    labels = data['labels']
                
                # 重塑图像维度 (10000, 3, 32, 32) -> (10000, 32, 32, 3)
                images = images.reshape((len(images), 3, 32, 32)).transpose(0, 2, 3, 1)
                train_images.append(images)
                train_labels.extend(labels)
        except Exception as e:
            print(f"加载批次 {i} 时出错: {e}")
            return tf.keras.datasets.cifar10.load_data()
    
    # 合并训练数据
    train_images = np.vstack(train_images)
    train_labels = np.array(train_labels)
    
    # 加载测试集
    test_path = os.path.join(data_dir, 'test_batch')
    try:
        with open(test_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            if b'data' in data:
                test_images = data[b'data']
                test_labels = data[b'labels']
            else:
                test_images = data['data']
                test_labels = data['labels']
            
            test_images = test_images.reshape((len(test_images), 3, 32, 32)).transpose(0, 2, 3, 1)
            test_labels = np.array(test_labels)
    except Exception as e:
        print(f"加载测试集时出错: {e}")
        return tf.keras.datasets.cifar10.load_data()
    
    return (train_images, train_labels), (test_images, test_labels)

# 2. 指定本地数据集路径
CIFAR10_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cifar-10-python')

# 加载数据
print("正在从本地路径加载CIFAR-10数据集...")
(train_images, train_labels), (test_images, test_labels) = load_cifar10_from_local(CIFAR10_PATH)

# 3. 数据预处理
# 归一化像素值到[0,1]范围
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# 类别名称
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print("数据加载完成!")
print(f"训练集形状: {train_images.shape}")
print(f"测试集形状: {test_images.shape}")
print(f"类别数量: {len(class_names)}")

# 4. 检查是否存在已训练的模型
MODEL_PATH = 'cifar10_cnn_model.h5'
MODEL_CHECKPOINT_PATH = 'cifar10_best_model.h5'

def load_or_create_model(model_path):
    """加载已有模型或创建新模型"""
    if os.path.exists(model_path):
        print(f"找到已训练的模型: {model_path}")
        choice = input("是否加载已有模型继续训练? (y/n): ").lower()
        if choice == 'y':
            try:
                # 加载模型
                model = tf.keras.models.load_model(model_path)
                print("模型加载成功!")
                
                # 显示模型信息
                print(f"模型结构:")
                model.summary()
                
                # 检查模型在测试集上的当前性能
                test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
                print(f"当前模型在测试集上的准确率: {test_accuracy:.4f}")
                
                return model, True
            except Exception as e:
                print(f"模型加载失败: {e}")
                print("将创建新模型...")
                return create_cnn_model(), False
        else:
            print("将创建新模型...")
            return create_cnn_model(), False
    else:
        print("未找到已训练的模型，将创建新模型...")
        return create_cnn_model(), False

# 5. 构建卷积神经网络模型
def create_cnn_model():
    """创建CNN模型"""
    model = models.Sequential([
        # 第一个卷积块
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第二个卷积块
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第三个卷积块
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # 全连接层
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10)  # 输出层，10个类别
    ])
    
    return model

# 6. 创建学习率调度器（用于继续训练时调整学习率）
def create_lr_scheduler(initial_lr=0.0001):
    """创建学习率调度器"""
    def lr_schedule(epoch, lr):
        # 每10个epoch将学习率减半
        if epoch > 0 and epoch % 10 == 0:
            return lr * 0.5
        return lr
    return tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# 7. 创建早停回调
def create_early_stopping():
    """创建早停回调防止过拟合"""
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

# 8. 创建模型检查点
def create_model_checkpoint():
    """创建模型检查点回调"""
    return tf.keras.callbacks.ModelCheckpoint(
        MODEL_CHECKPOINT_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )

# 主程序
def main():
    # 加载或创建模型
    model, is_loaded = load_or_create_model(MODEL_PATH)
    
    # 编译模型（如果是新模型或需要重新编译）
    if not is_loaded:
        # 对于新模型，使用初始学习率
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
    else:
        # 对于已加载的模型，使用更小的学习率进行微调
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(
            optimizer=optimizer,
            loss=model.loss,
            metrics=['accuracy']
        )
    
    # 显示模型结构
    print("模型结构:")
    model.summary()
    
    # 设置回调函数
    callbacks = [
        create_early_stopping(),
        create_model_checkpoint(),
        create_lr_scheduler(0.0001 if is_loaded else 0.001)  # 根据是否加载模型设置初始学习率
    ]
    
    # 询问用户训练参数
    try:
        if is_loaded:
            # 如果是继续训练，初始epoch为30（假设之前训练了30个epoch）
            initial_epoch = 30
            additional_epochs = int(input("请输入要继续训练的轮数 : ") or "30")
        else:
            initial_epoch = 0
            additional_epochs = int(input("请输入要训练的轮数 : ") or "30")
    except:
        additional_epochs = 30
        initial_epoch = 30 if is_loaded else 0
    
    print(f"开始训练，从第 {initial_epoch} 个epoch开始，共训练 {additional_epochs} 个epoch...")
    
    # 训练模型
    history = model.fit(
        train_images, train_labels,
        batch_size=64,
        epochs=initial_epoch + additional_epochs,
        initial_epoch=initial_epoch,
        validation_data=(test_images, test_labels),
        callbacks=callbacks,
        verbose=1
    )
    
    # 模型评估
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"\n最终测试集准确率: {test_accuracy:.4f}")
    print(f"最终测试集损失: {test_loss:.4f}")
    
    # 保存最终模型
    model.save(MODEL_PATH)
    print(f"模型已保存为 '{MODEL_PATH}'")
    
    # 如果有检查点模型，加载最佳模型进行最终评估
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        best_model = tf.keras.models.load_model(MODEL_CHECKPOINT_PATH)
        best_test_loss, best_test_accuracy = best_model.evaluate(test_images, test_labels, verbose=0)
        print(f"最佳模型测试集准确率: {best_test_accuracy:.4f}")
        print(f"最佳模型测试集损失: {best_test_loss:.4f}")
    
    # 可视化训练过程
    def plot_training_history(history, initial_epoch=0):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 调整epoch计数
        epochs = range(initial_epoch + 1, initial_epoch + len(history.history['accuracy']) + 1)
        
        # 准确率曲线
        ax1.plot(epochs, history.history['accuracy'], label='训练准确率')
        ax1.plot(epochs, history.history['val_accuracy'], label='验证准确率')
        ax1.set_title('模型准确率')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # 损失曲线
        ax2.plot(epochs, history.history['loss'], label='训练损失')
        ax2.plot(epochs, history.history['val_loss'], label='验证损失')
        ax2.set_title('模型损失')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 显示训练历史
    plot_training_history(history, initial_epoch)
    
    # 进行预测并显示结果
    def plot_predictions(model, images, labels, class_names, num_images=10):
        """显示预测结果"""
        # 创建包含softmax的模型用于预测
        probability_model = tf.keras.Sequential([
            model, 
            layers.Softmax()
        ])
        
        # 进行预测
        predictions = probability_model.predict(images[:num_images])
        
        # 绘制结果
        plt.figure(figsize=(12, 8))
        for i in range(num_images):
            plt.subplot(2, 5, i + 1)
            plt.imshow(images[i])
            plt.xticks([])
            plt.yticks([])
            
            predicted_label = np.argmax(predictions[i])
            true_label = labels[i]
            
            color = 'blue' if predicted_label == true_label else 'red'
            
            plt.xlabel(f"{class_names[predicted_label]} {100*np.max(predictions[i]):.0f}%\n({class_names[true_label]})", 
                      color=color)
        plt.tight_layout()
        plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 显示预测结果
    print("显示预测结果...")
    plot_predictions(model, test_images, test_labels, class_names)
    
    # 计算每个类别的准确率
    def per_class_accuracy(model, test_images, test_labels, class_names):
        """计算每个类别的准确率"""
        # 进行预测
        predictions = model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 计算每个类别的准确率
        class_correct = [0] * 10
        class_total = [0] * 10
        
        for i in range(len(test_labels)):
            label = test_labels[i]
            class_total[label] += 1
            if predicted_classes[i] == label:
                class_correct[label] += 1
        
        # 打印结果
        print("\n每个类别的准确率:")
        for i in range(10):
            accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f"{class_names[i]:12s}: {accuracy:.2f}%")
    
    # 显示每个类别的准确率
    per_class_accuracy(model, test_images, test_labels, class_names)
    
    print("\n训练完成！")

# 运行主程序
if __name__ == "__main__":
    main()
