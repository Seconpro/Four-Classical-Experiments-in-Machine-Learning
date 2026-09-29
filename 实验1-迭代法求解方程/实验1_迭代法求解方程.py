import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

def f(x):
    """定义原函数"""
    return x**5 + x**4 + np.exp(x) - 11*x + 1

def df(x):
    """定义导数函数"""
    return 5*x**4 + 4*x**3 + np.exp(x) - 11

def newton_method(f, df, x0, tol=1e-8, max_iter=1000):
    """牛顿迭代法"""
    x = x0
    iterations = 0
    history = [x0]
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-12:  # 防止除零
            break
            
        x_new = x - fx / dfx
        history.append(x_new)
        
        if abs(x_new - x) < tol and abs(fx) < tol:
            break
            
        x = x_new
        iterations += 1
        
    return x_new, iterations, history

def hill_climbing_method(f, x0, step_size=0.1, tol=1e-6, max_iter=10000):
    """爬山法求解方程根"""
    x = x0
    best_x = x0
    best_f = abs(f(x0))
    history = [x0]
    
    for i in range(max_iter):
        # 尝试左右移动
        x_left = x - step_size
        x_right = x + step_size
        
        f_left = abs(f(x_left))
        f_right = abs(f(x_right))
        f_current = abs(f(x))
        
        # 选择函数值绝对值最小的方向
        if f_left < f_current and f_left <= f_right:
            x = x_left
        elif f_right < f_current and f_right <= f_left:
            x = x_right
        else:
            # 如果两个方向都不好，减小步长
            step_size *= 0.5
            continue
            
        history.append(x)
        
        if abs(f(x)) < tol:
            break
            
    return x, len(history), history

def find_roots():
    """寻找三个根"""
    # 通过函数图像分析大致根的位置
    x_vals = np.linspace(-3, 3, 1000)
    y_vals = f(x_vals)
    
    # 寻找函数变号的区间
    roots_intervals = []
    for i in range(len(x_vals)-1):
        if y_vals[i] * y_vals[i+1] < 0:
            roots_intervals.append((x_vals[i], x_vals[i+1]))
    
    print("找到的可能根区间:", roots_intervals)
    
    # 使用牛顿迭代法求解
    print("\n=== 牛顿迭代法结果 ===")
    newton_roots = []
    initial_guesses = [-2.0, 0.5, 2.0]  # 根据函数图像选择的初始值
    
    for i, x0 in enumerate(initial_guesses):
        root, iterations, history = newton_method(f, df, x0)
        newton_roots.append(root)
        print(f"根 {i+1}: x = {root:.8f}")
        print(f"函数值: f(x) = {f(root):.2e}")
        print(f"迭代次数: {iterations}")
        print(f"验证: f({root:.6f}) = {f(root):.2e}")
        print("-" * 40)
    
    # 使用爬山法求解
    print("\n=== 爬山法结果 ===")
    hill_roots = []
    for i, x0 in enumerate(initial_guesses):
        root, iterations, history = hill_climbing_method(f, x0)
        hill_roots.append(root)
        print(f"根 {i+1}: x = {root:.8f}")
        print(f"函数值: f(x) = {f(root):.2e}")
        print(f"迭代次数: {iterations}")
        print(f"验证: f({root:.6f}) = {f(root):.2e}")
        print("-" * 40)
    
    return newton_roots, hill_roots

def plot_function_and_roots():
    """绘制函数图像和根的位置"""
    x_vals = np.linspace(-2.5, 2.5, 1000)
    y_vals = f(x_vals)
    
    plt.figure(figsize=(12, 8))
    
    # 绘制函数图像
    plt.subplot(2, 1, 1)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x^5 + x^4 + e^x - 11x + 1')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('函数图像')
    plt.legend()
    
    # 绘制局部放大图
    plt.subplot(2, 1, 2)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('局部放大图（根附近）')
    plt.ylim(-5, 5)
    
    plt.tight_layout()
    plt.savefig('function_plot.png')  # 保存图像而不是直接显示
    print("函数图像已保存为 'function_plot.png'")

# 执行求解
if __name__ == "__main__":
    # 绘制函数图像
    plot_function_and_roots()
    
    # 求解方程
    newton_roots, hill_roots = find_roots()
    
    print("\n=== 结果总结 ===")
    print("牛顿迭代法求得的根:", [f"{r:.6f}" for r in newton_roots])
    print("爬山法求得的根:", [f"{r:.6f}" for r in hill_roots])
    
    # 将结果保存到文件
    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write("=== 结果总结 ===\n")
        f.write("牛顿迭代法求得的根: " + ", ".join([f"{r:.6f}" for r in newton_roots]) + "\n")
        f.write("爬山法求得的根: " + ", ".join([f"{r:.6f}" for r in hill_roots]) + "\n")
    print("结果已保存为 'results.txt'")
