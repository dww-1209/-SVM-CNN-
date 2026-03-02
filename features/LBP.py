#经查询资料，纹理特征的形式多种多样，这里我们采用比较常见的LBP特征：局部二值模式（LBP）
#每个像素与周围像素比较，得到二值码，再统计码的直方图。
import numpy as np
from skimage import color
from skimage.feature import local_binary_pattern

def extract_lbp_features(images, 
                         radius=3, 
                         n_points=None, 
                         method='uniform'):
    """
    对输入灰度图 img 提取 LBP 直方图特征。

    参数：
    - img:       灰度图，二维 numpy 数组，uint8 或浮点
    - radius:    LBP 半径，默认 3（邻域大小 = 2*radius + 1）
    - n_points:  采样点数，默认 None 表示 8*radius
    - method:    LBP 计算方法，常用 'uniform', 'default', 'ror' 等

    返回：
    - hist:      归一化后的 LBP 直方图，一维 numpy 数组
    """
    feature = []
    # 默认采样点数 = 8 × radius
    if n_points is None:
        n_points = radius * 8
    for img in images :
        img = color.rgb2gray(img)
        # 1）计算 LBP 图，结果是每个像素对应的 LBP 码
        lbp = local_binary_pattern(img, P=n_points, R=radius, method=method)
        # lbp 的取值范围：
        # - 如果 method='uniform'，取值 [0, n_points + 1]
        # - 如果 method='default'，取值 [0, 2**n_points - 1]

        # 2）统计直方图
        # bins = n_points + 2 for 'uniform' 模式，其它模式按 2**n_points
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True       # 统计所有位置上出现的各个 LBP 码出现频率，并归一化。归一化直方图，使之和为 1
        )
        feature.append(hist)
    return np.stack(feature)