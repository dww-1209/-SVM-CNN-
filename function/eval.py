import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
def plot_confusion_matrix(y_true, y_pred, idx2label, normalize=True, cmap='Blues',feature_name='CNN'):
    """
    计算并绘制混淆矩阵。
    
    参数：
      - y_true:   shape=(n_samples,) 的真实整数标签数组
      - y_pred:   shape=(n_samples,) 的预测整数标签数组
      - idx2label: dict，将整数标签映射到类别名称
      - normalize: bool，是否对混淆矩阵按行进行归一化（每行加和为 1）
      - cmap:      str 或 Colormap，对应 matplotlib 的配色方案
    """
    path = os.path.join('confusion_matrix',feature_name)
    # 1. 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(idx2label.keys()), normalize='true' if normalize else None)

    # 2. 借用大模型得知，可以使用 ConfusionMatrixDisplay 绘制
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[idx2label[i] for i in sorted(idx2label.keys())])
    fig, ax = plt.subplots(figsize=(16,16))
    disp.plot(include_values=True, cmap=cmap, ax=ax, xticks_rotation=45)
    title = feature_name + "  Normalized Confusion Matrix" 
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path)
    print(f'已将{feature_name}模型的混淆矩阵写入：{path}.png')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D  # 用于 3D 绘图
import os

def plot_confusion_matrix_3d(y_true, y_pred, idx2label, normalize=True, feature_name='CNN'):
    """
    绘制 3D 柱状图形式的混淆矩阵。
    
    参数：
    - y_true: array-like, 真实标签
    - y_pred: array-like, 预测标签
    - idx2label: dict，整数标签到名称的映射
    - normalize: bool，是否行归一化
    - feature_name: str，用于保存文件命名
    """
    labels = list(idx2label.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true' if normalize else None)
    cm = np.array(cm)

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')

    # 网格坐标
    x_len = len(labels)
    xpos, ypos = np.meshgrid(np.arange(x_len), np.arange(x_len), indexing='ij')

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    dx = dy = 0.8  # 宽度
    dz = cm.flatten()

    # 颜色映射（可以按值高低设定色彩深浅）
    colors = plt.cm.jet(dz / np.max(dz))

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')

    # 设置坐标轴标签
    ax.set_xlabel('Estimated Category', labelpad=10)
    ax.set_ylabel('True Category', labelpad=10)
    ax.set_zlabel('Proportion' if normalize else 'Count')

    # 设置坐标轴刻度为类名
    tick_labels = [idx2label[i] for i in labels]
    ax.set_xticks(np.arange(x_len) + dx / 2)
    ax.set_yticks(np.arange(x_len) + dy / 2)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_yticklabels(tick_labels, rotation=0)
    ax.set_title(f'{feature_name} Confusion Matrix (3D)', pad=20)

    # 保存图像
    save_path = os.path.join('confusion_matrix_3d', feature_name)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'3D 混淆矩阵已保存至：{save_path}')



