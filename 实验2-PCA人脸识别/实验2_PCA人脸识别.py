import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import warnings
import os
from scipy.linalg import eigh
import matplotlib as mpl
import matplotlib.font_manager as fm

# 设置中文字体 - 使用Windows 11自带字体
try:
    mpl.rcParams['font.family'] = 'Microsoft YaHei'
    mpl.rcParams['axes.unicode_minus'] = False
except:
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
        titles = {
            'eigenfaces': 'Principal Components (Eigenfaces)',
            'original': 'Original',
            'reconstructed': 'PCA Reconstruction',
            'difference': 'Difference',
            'variance': 'Number of Principal Components vs Cumulative Explained Variance'
        }

# 过滤弃用警告
warnings.filterwarnings('ignore', category=DeprecationWarning)

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None
    
    def fit(self, X):
        """PCA算法实现[1,9](@ref)"""
        # 步骤1: 对所有样本进行中心化
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 步骤2: 计算样本的协方差矩阵
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # 步骤3: 对协方差矩阵做特征值分解
        eigenvalues, eigenvectors = eigh(covariance_matrix)
        
        # 步骤4: 取最大的d'个特征值对应的特征向量
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.components = eigenvectors[:, :self.n_components]
        
        # 计算方差解释率
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance
        
        return self
    
    def transform(self, X):
        """将数据投影到主成分空间"""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def inverse_transform(self, X_transformed):
        """将降维后的数据重构回原始空间"""
        return np.dot(X_transformed, self.components.T) + self.mean

def load_and_preprocess_data():
    """加载并预处理人脸数据"""
    print("加载人脸数据集...")
    faces = fetch_olivetti_faces(data_home=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    X = faces.data
    y = faces.target
    images = faces.images
    
    print(f"数据集形状: {X.shape}")
    print(f"图像形状: {images.shape[1:]}")
    
    return X, y, images

def demonstrate_pca(X, images):
    """演示PCA降维效果"""
    # 使用PCA降维
    n_components = 100
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_transformed = pca.transform(X)
    
    print(f"原始数据维度: {X.shape[1]}")
    print(f"降维后维度: {X_transformed.shape[1]}")
    print(f"前10个主成分的方差解释率: {pca.explained_variance_ratio[:10]}")
    print(f"累计方差解释率: {np.sum(pca.explained_variance_ratio):.3f}")
    
    # 显示特征脸
    show_eigenfaces(pca.components, images.shape[1:])
    
    # 显示重构效果
    show_reconstruction(X, images, pca)
    
    return pca, X_transformed

def show_eigenfaces(components, image_shape, n_faces=16):
    """显示特征脸"""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.ravel()
    
    for i in range(min(n_faces, components.shape[1])):
        eigenface = components[:, i].reshape(image_shape)
        axes[i].imshow(eigenface, cmap='gray')
        axes[i].set_title(f'主成分 {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('主成分（特征脸）')
    plt.tight_layout()
    plt.show()

def show_reconstruction(X, images, pca, n_examples=5):
    """显示原始图像和重构图像的对比"""
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4*n_examples))
    
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    indices = np.random.choice(len(X), n_examples, replace=False)
    
    for i, idx in enumerate(indices):
        # 原始图像
        axes[i, 0].imshow(images[idx], cmap='gray')
        axes[i, 0].set_title('原始图像')
        axes[i, 0].axis('off')
        
        # 降维重构
        transformed = pca.transform(X[idx:idx+1])
        reconstructed = pca.inverse_transform(transformed)
        reconstructed_image = reconstructed.reshape(images.shape[1:])
        
        axes[i, 1].imshow(reconstructed_image, cmap='gray')
        axes[i, 1].set_title('PCA重构')
        axes[i, 1].axis('off')
        
        # 差异
        diff = np.abs(images[idx] - reconstructed_image)
        axes[i, 2].imshow(diff, cmap='hot')
        axes[i, 2].set_title('差异')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def face_recognition_demo(X, y, pca):
    """人脸识别演示 - 增加了最小距离和投影形状的输出[1,6,7](@ref)"""
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 投影到主成分空间
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # 输出投影形状信息[1](@ref)
    print("\n=== 投影形状信息 ===")
    print(f"训练集投影形状: {X_train_pca.shape}")
    print(f"测试集投影形状: {X_test_pca.shape}")
    print(f"每个样本的主成分数量: {X_train_pca.shape[1]}")
    
    # 简单的最近邻分类[6,7](@ref)
    correct = 0
    min_distances = []  # 存储所有测试样本的最小距离
    all_distances = []  # 存储所有距离信息用于分析
    
    print("\n=== 最小距离统计 ===")
    for i, test_sample in enumerate(X_test_pca):
        # 计算与所有训练样本的欧氏距离[6](@ref)
        distances = np.linalg.norm(X_train_pca - test_sample, axis=1)
        min_distance = np.min(distances)
        nearest_idx = np.argmin(distances)
        predicted_label = y_train[nearest_idx]
        
        min_distances.append(min_distance)
        all_distances.append(distances)
        
        # 每10个样本输出一次详细信息
        if i % 10 == 0:
            print(f"测试样本 {i}: 最小距离 = {min_distance:.4f}, 预测标签 = {predicted_label}, 真实标签 = {y_test[i]}")
        
        if predicted_label == y_test[i]:
            correct += 1
    
    # 计算准确率
    accuracy = correct / len(X_test)
    
    # 距离统计分析[7](@ref)
    min_distances = np.array(min_distances)
    print(f"\n=== 距离统计详细信息 ===")
    print(f"最小距离统计:")
    print(f"  平均值: {np.mean(min_distances):.4f}")
    print(f"  标准差: {np.std(min_distances):.4f}")
    print(f"  最小值: {np.min(min_distances):.4f}")
    print(f"  最大值: {np.max(min_distances):.4f}")
    print(f"  中位数: {np.median(min_distances):.4f}")
    
    # 显示前5个测试样本的详细距离信息
    print(f"\n=== 前5个测试样本的详细距离信息 ===")
    for i in range(min(5, len(X_test))):
        test_sample_distances = all_distances[i]
        min_dist_idx = np.argmin(test_sample_distances)
        print(f"测试样本 {i}:")
        print(f"  最小距离: {min_distances[i]:.4f}")
        print(f"  最近邻索引: {min_dist_idx}")
        print(f"  最近邻真实标签: {y_train[min_dist_idx]}")
        print(f"  预测结果: {'正确' if y_train[min_dist_idx] == y_test[i] else '错误'}")
        print(f"  距离统计 - 均值: {np.mean(test_sample_distances):.4f}, 标准差: {np.std(test_sample_distances):.4f}")
    
    print(f"\n人脸识别准确率: {accuracy:.3f}")
    
    # 绘制最小距离分布图
    plt.figure(figsize=(10, 6))
    plt.hist(min_distances, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(min_distances), color='red', linestyle='dashed', linewidth=1, label=f'平均距离: {np.mean(min_distances):.4f}')
    plt.xlabel('最小欧氏距离')
    plt.ylabel('频数')
    plt.title('测试样本与训练集最近邻的最小距离分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return accuracy, min_distances

def main():
    """主函数"""
    print("=== PCA人脸识别实验 ===")
    
    # 加载数据
    X, y, images = load_and_preprocess_data()
    
    # PCA演示
    pca, X_transformed = demonstrate_pca(X, images)
    
    # 人脸识别演示（现在返回准确率和距离信息）
    accuracy, min_distances = face_recognition_demo(X, y, pca)
    
    # 方差解释率曲线
    plt.figure(figsize=(10, 6))
    explained_variance = np.cumsum(pca.explained_variance_ratio)
    plt.plot(explained_variance)
    plt.xlabel('主成分数量')
    plt.ylabel('累计方差解释率')
    plt.title('主成分数量 vs 累计方差解释率')
    plt.grid(True)
    plt.show()
    
    # 最终总结输出
    print("\n=== 实验总结 ===")
    print(f"使用的总主成分数量: {pca.n_components}")
    print(f"投影后特征维度: {pca.n_components} (原始维度: {X.shape[1]})")
    print(f"降维比例: {(1 - pca.n_components/X.shape[1])*100:.1f}%")
    print(f"人脸识别准确率: {accuracy:.3f}")
    print(f"测试样本最小距离统计:")
    print(f"  平均最小距离: {np.mean(min_distances):.4f} ± {np.std(min_distances):.4f}")
    
    print("实验完成！")

if __name__ == "__main__":
    main()
