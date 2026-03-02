import numpy as np


def extract_rgb_histogram(images, bins=256):
    """
    提取RGB三通道的颜色直方图，进行归一化操作后并拼接成一个一维向量。
    """
    result = []
    for image in images:
        hist_r, _ = np.histogram(image[:, :, 0], bins=bins, range=(0, 256))
        hist_g, _ = np.histogram(image[:, :, 1], bins=bins, range=(0, 256))
        hist_b, _ = np.histogram(image[:, :, 2], bins=bins, range=(0, 256))


        hist_r = hist_r / hist_r.sum()
        hist_g = hist_g / hist_g.sum()
        hist_b = hist_b / hist_b.sum()

        feature_vec= np.concatenate([hist_r, hist_g, hist_b])
        result.append(feature_vec)
    return np.stack(result)
