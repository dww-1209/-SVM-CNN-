import numpy as np
from skimage import color
from skimage.feature import graycomatrix,graycoprops
import numpy as np
from sklearn.decomposition import PCA



def quantize_gray(img, levels=16):
    """把 uint8 灰度图 img 量化到 [0, levels-1]"""
    bins = np.linspace(0, 256, levels+1, endpoint=True) #划分区间：用 levels 个区间将灰度 [0,255] 均分。
    img_q = np.digitize(img, bins) - 1                  #映射像素：把每个像素落在哪个区间，就映射为该区间编号。
    return img_q.astype(np.uint8)


def GLCM_features(images, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16):
    """
    对一组 RGB 图像列表，提取 GLCM Haralick 特征。
    返回值：X，shape = (n_images, n_features), 其中
      n_features = len(distances) * len(angles) * n_props
      n_props    = 5 (contrast, dissimilarity, homogeneity, energy, correlation)
    """
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    n_props = len(props)
    n_dist = len(distances)
    n_ang  = len(angles)

    features = []
    for img in images:
        # 1) 转成灰度 float [0,1] → 再映成 [0,255] uint8
        gray = color.rgb2gray(img)        # 结果范围 [0,1]
        gray = (gray * 255).astype(np.uint8)

        # 2) 量化灰度到 [0, levels-1]
        img_q = quantize_gray(gray, levels=levels)

        # 3) 计算灰度共生矩阵
        #    返回的 GLCM.shape = (levels, levels, len(distances), len(angles))
        glcm = graycomatrix(img_q,
                            distances=distances,
                            angles=angles,
                            levels=levels,
                            symmetric=True,
                            normed=True)

        # 4) 对每个属性、每个距离、每个角度，提取对应值
        feat_vector = []
        for prop in props:
            # greycoprops 会返回一个 shape=(len(distances), len(angles)) 的矩阵
            vals = graycoprops(glcm, prop)
            # 将其“拉平”成一维：一共 n_dist * n_ang 个值
            feat_vector.extend(vals.flatten().tolist())

        features.append(feat_vector)

    X = np.array(features, dtype=np.float32)
    return X