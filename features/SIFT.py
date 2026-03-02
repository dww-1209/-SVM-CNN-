import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
def sift_feature(images, max_features=100):
    sift = cv2.SIFT_create()
    feature = []
    for image in images :
        if isinstance(image, Image.Image):
            image = image.convert("L")
            image = np.array(image)
        elif isinstance(image, np.ndarray) and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None:
            descriptors = np.zeros((max_features, 128), dtype=np.float32)
        elif descriptors.shape[0] > max_features:
            descriptors = descriptors[:max_features]
        else:
            pad = np.zeros((max_features - descriptors.shape[0], 128), dtype=np.float32)
            descriptors = np.vstack([descriptors, pad])
            
        feature_vector = descriptors.flatten()
        feature.append(feature_vector)
    X = np.stack(feature)
    return X


